"""
audit_counters.py
=================
Re-score every checkpoint under a runs directory in the HONEST generalization
currency, replacing weighted far-OOD Delta%.

Why: the weighted |D:H| metric can be gamed by error cancellation -- a
miscalibrated continue-coin (assigning long strings more likelihood than the
true grammar, negative coin excess) can pay for counting errors (positive
counter bits) on exactly the lengths that dominate the weighted average.
A network was observed scoring far-OOD = -0.6% while its exact counting broke
at n=10, inside its own training range. Per-length counter bits cannot be
gamed this way: the b-run after the first b, plus '#', is deterministic, its
optimum is 0 bits, and every bit spent there is a counting error.

Metrics per checkpoint:
  exact_n        largest probed n such that counter bits <= fail_bits for ALL
                 probed n' <= n (the exact counting range)
  extrap         exact_n - n_train_max (positive = counts beyond training)
  c25/c50/c100   counter bits at n = 25 / 50 / 100 (approximation quality)
  coin50         coin excess at n = 50 (negative = overconfident coin, the
                 signature of the cancellation exploit)
  greedy_n       largest probed n such that the ARGMAX prediction is correct at
                 every counter position for all n' <= n, i.e. the network would
                 emit the right string under greedy decoding. Insensitive to
                 slow calibration drift, which exact_n punishes twice as hard on
                 a^n b^n c^n (2n forced tokens per string vs n). Report both.
  maxtok50       largest single-token counter bits at n = 50 (> 1 bit means
                 some position is a genuine coin flip, not drift)
  exact_n_<run>  the same exact-range criterion applied to ONE letter run's
                 bits (a^n b^n c^n: exact_n_b, exact_n_c). Locates the failure:
                 exact_n_b >> exact_n_c means the b-count works and the c-count
                 (re-using n a second time) is what breaks. Absent for a^n b^n.
  <run>50        that run's bits at n = 50
  H_bits / DH_bits / mdl_total
                 the two-part code of the checkpoint under the natural code:
                 |H| = L0 x (log2 N_quantizable + log2(2m)), m = levels per sign (1 ternary, 127 int8); |D:H| = total bits on the
                 run's OWN training set (1e-10 smoothing, same as ga_polish),
                 and their sum. Plot mdl_total against greedy_n to see whether
                 the objective SGD and the polish minimise has anything to say
                 about length generalization.
  pb1/pb20/pb100/pb300
                 P(b | <bos> a^k): the coin. The grammar is memoryless, so the
                 shortest description of the coin is a constant 0.30. Drift in
                 k (via <bos> dilution under NoPE) is fitted sample noise.
  coin_dev_ood   max over probed k > n_train_max of |P(b | a^k) - p|: 0 means
                 the coin extrapolates as the TRUE constant; ~0.7 means it has
                 flipped to 'always stop'. (Replaces a max-min measure, which a
                 coin saturated at 1.0 fooled.) Together with L0 this is the
                 compression-vs-distribution plot.
  coin_flat_ood  max - min over the same k (kept: 0 with a large coin_dev_ood
                 = flat at the wrong constant)
  L0, zeros      sparsity of the checkpoint (for L0-vs-generalization plots)
  L0_bias        how many of the surviving weights are gridded biases (--bias runs)
  p_ref          the p the coin is scored against: config["polish"]["p_end"] for
                 a checkpoint ga_polish.py scored on other data, else p_geom. (A
                 network polished at p_end=0.1 has a flat coin at 0.1; scoring
                 it against 0.3 produced the spurious coin_dev 0.19-0.25 of the
                 earlier audits.) coin50 is referenced to the same p_ref.

Every run contributes its final checkpoint (model.pt, ckpt=final), the
near-OOD-selected one when present (model_best_near.pt, ckpt=best_near, with
its step in ckpt_step), and any ga_polish.py output (model_polished*.pt,
ckpt=polished / polished_guarded / polished_guarded_greedy), so trained,
selected and MDL-polished networks sit in one table.

Usage:
  python audit_counters.py                    # audits runs/**/model.pt (+ best_near)
  python audit_counters.py --root runs/fixed --max_n 200
Writes analysis/audit.csv and prints the league table sorted by exact_n.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os

import torch
import torch.nn.functional as F

from grammar import BOS, PAD, collate, log2_prob_of_n, make_train_set, task_from_config
from model import LoopedTransformer
from quantization import QuantConfig, quantize

LN2 = math.log(2.0)


def probe_ns(max_n: int) -> list[int]:
    return list(range(1, min(41, max_n + 1))) + list(range(45, max_n + 1, 5))


@torch.no_grad()
def counter_profile(model, ns, device, p, task, batch: int = 32):
    """{n: (coin_excess_bits, counter_bits, {run: bits}, greedy_ok, max_tok_bits)}
    per probed length. greedy_ok: argmax == target at every counter position."""
    out = {}
    for i in range(0, len(ns), batch):
        chunk = ns[i : i + batch]
        inp, tgt = collate([task.string_for_n(n) for n in chunk], device)
        logp = F.log_softmax(model(inp).float(), dim=-1)
        pr = logp.exp().clamp_min(1e-10)
        tb = -(pr.log() / LN2).gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        tb = tb * (tgt != PAD)
        am = logp.argmax(-1)
        for j, n in enumerate(chunk):
            row = tb[j]
            lo, hi = task.counter_span(n)
            coin = float(row[:lo].sum()) + log2_prob_of_n(n, p)
            counter = float(row[lo:hi].sum())
            runs = {name: float(row[a:b].sum())
                    for name, (a, b) in task.counter_run_spans(n).items()}
            greedy_ok = bool((am[j, lo:hi] == tgt[j, lo:hi]).all())
            max_tok = float(row[lo:hi].max())
            out[n] = (coin, counter, runs, greedy_ok, max_tok)
    return out


COIN_KS = (1, 5, 10, 20, 50, 100, 200, 300)


@torch.no_grad()
def train_bits(model, seqs, device, batch: int = 128) -> float:
    """Total -log2 q over a set of strings (sum, 1e-10 smoothing)."""
    total = 0.0
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        inp, tgt = collate([seqs[i] for i in idx], device)
        lp = F.log_softmax(model(inp).float(), dim=-1)
        p = lp.exp().clamp_min(1e-10)
        tb = -(p.log() / LN2).gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        total += float((tb * (tgt != PAD)).sum())
    return total


@torch.no_grad()
def coin_profile(model, task, device, ks=COIN_KS):
    """{k: P(b | <bos> a^k)} at the last position."""
    a_tok, b_tok = task.letters[0], task.letters[1]
    out = {}
    for k in ks:
        inp = torch.tensor([[BOS] + [a_tok] * k], device=device)
        out[k] = float(F.softmax(model(inp).float(), dim=-1)[0, -1, b_tok])
    return out


def audit_one(ckpt_path: str, device, max_n: int, fail_bits: float) -> dict | None:
    try:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ck["config"]
        task = task_from_config(cfg)
        qcfg = QuantConfig(mode=cfg["quant"], scale=cfg["scale"],
                           fixed_delta=cfg["fixed_delta"])
        model = LoopedTransformer.from_config(cfg, task.vocab_size, qcfg).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
    except Exception as e:
        print(f"[skip] {ckpt_path}: {e}")
        return None

    train = make_train_set(cfg["train_size"], seed=cfg["seed"],
                           p=cfg["p_geom"], task=task)
    n_train_max = train.n_max
    dh_bits = train_bits(model, train.seqs, device)      # always the run's OWN training set
    pol = cfg.get("polish") or {}
    p_ref = float(pol.get("p_end") or cfg["p_geom"])     # the generator the coin should match
    ns = probe_ns(max_n)
    prof = counter_profile(model, ns, device, p_ref, task)

    def exact_range(get):
        e = 0
        for n in ns:
            if get(n) > fail_bits:
                break
            e = n
        return e

    exact_n = exact_range(lambda n: prof[n][1])
    greedy_n = exact_range(lambda n: 0.0 if prof[n][3] else 1.0)   # first argmax error
    run_names = list(task.counter_run_spans(1))
    per_run = {}
    if len(run_names) > 1:                       # only meaningful for k >= 3
        for name in run_names:
            per_run[f"exact_n_{name}"] = exact_range(lambda n, s=name: prof[n][2][s])
            per_run[f"{name}50"] = (round(prof[50][2][name], 2) if 50 in prof
                                    else float("nan"))

    sp = model.sparsity_report()
    m_lv = getattr(model.qcfg, "n_levels_per_sign", None) or 1      # 1 ternary, 127 int8
    h_bits = sp["L0_nonzero_count"] * (math.log2(model.param_report()["quantizable_params"])
                                       + math.log2(2 * m_lv))
    coin = coin_profile(model, task, device)
    ood_ks = [k for k in coin if k > n_train_max]
    coin_flat = (max(coin[k] for k in ood_ks) - min(coin[k] for k in ood_ks)
                 if ood_ks else float("nan"))
    coin_dev = (max(abs(coin[k] - p_ref) for k in ood_ks)
                if ood_ks else float("nan"))
    l0_bias = sum(int((quantize(w, model.qcfg) != 0).sum())
                  for n, w in model.quantizable_named_weights() if n.endswith(".bias"))
    fname = os.path.basename(ckpt_path)
    if fname == "model_best_near.pt":
        ckpt_kind = "best_near"
    elif fname.startswith("model_polished"):
        ckpt_kind = fname[len("model_"):-len(".pt")]      # polished[_guarded[_greedy]]
    elif fname.startswith("polish"):
        ckpt_kind = fname[:-len(".pt")]                    # ga_polish --out files
    else:
        ckpt_kind = "final"
    ckpt_step = ck.get("step", cfg["steps"])

    def at(n):
        return round(prof[n][1], 2) if n in prof else float("nan")

    return {
        "run": os.path.dirname(ckpt_path),
        "ckpt": ckpt_kind, "ckpt_step": ckpt_step,
        "group": os.path.normpath(os.path.dirname(ckpt_path)).split(os.sep)[-2]
                 if os.sep in os.path.normpath(os.path.dirname(ckpt_path)) else "",
        "task": task.name,
        "attn": cfg.get("attn", "softmax"), "norm": cfg.get("norm", "ln"),
        "bias": cfg.get("bias", "none"), "p_geom": cfg["p_geom"], "p_ref": p_ref,
        "polish_hbits": pol.get("hbits"), "polish_guard_coin": pol.get("guard_coin"),
        "n_loops": cfg["n_loops"], "lr": cfg.get("lr"),
        "diverged_at": ck.get("diverged_at_step"),
        "quant": cfg["quant"], "scale": cfg["scale"],
        "weight_decay": cfg["weight_decay"], "seed": cfg["seed"],
        "steps": cfg["steps"], "train_size": cfg["train_size"],
        "batch_size": cfg.get("batch_size"),
        "n_train_max": n_train_max,
        "exact_n": exact_n,
        "extrap": exact_n - n_train_max,
        "greedy_n": greedy_n,
        "greedy_extrap": greedy_n - n_train_max,
        "maxtok50": round(prof[50][4], 2) if 50 in prof else float("nan"),
        "c25": at(25), "c50": at(50), "c100": at(100),
        "coin50": round(prof[50][0], 2) if 50 in prof else float("nan"),
        **per_run,
        "H_bits": round(h_bits), "DH_bits": round(dh_bits),
        "mdl_total": round(h_bits + dh_bits),
        "pb1": round(coin[1], 3), "pb20": round(coin[20], 3),
        "pb100": round(coin[100], 3), "pb300": round(coin[300], 3),
        "coin_dev_ood": round(coin_dev, 3),
        "coin_flat_ood": round(coin_flat, 3),
        "L0": sp["L0_nonzero_count"],
        "L0_bias": l0_bias,
        "sumsq": int(round(sp["L2sq_of_quantized"] / cfg["fixed_delta"] ** 2))
                 if cfg["scale"] == "fixed" else None,
        "zero_frac": round(sp["hard_zero_frac"], 3),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="runs")
    ap.add_argument("--max_n", type=int, default=150)
    ap.add_argument("--fail_bits", type=float, default=0.5)
    ap.add_argument("--out", default="analysis/audit.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted(glob.glob(os.path.join(args.root, "**", "model.pt"),
                             recursive=True)
                   + glob.glob(os.path.join(args.root, "**", "model_best_near.pt"),
                               recursive=True)
                   + glob.glob(os.path.join(args.root, "**", "model_polished*.pt"),
                               recursive=True)
                   + glob.glob(os.path.join(args.root, "**", "polish*.pt"),
                               recursive=True))
    if not ckpts:
        raise SystemExit(f"no model.pt found under {args.root}/")
    print(f"auditing {len(ckpts)} checkpoints on {device} "
          f"(probing n=1..{args.max_n}, fail > {args.fail_bits} counter bits)")

    rows = []
    for i, c in enumerate(ckpts, 1):
        r = audit_one(c, device, args.max_n, args.fail_bits)
        if r:
            rows.append(r)
            extra = "".join(f" {k}={v}" for k, v in r.items()
                            if k.startswith("exact_n_"))
            tag = ("" if r["ckpt"] == "final"
                   else f" [best_near@{r['ckpt_step']}]" if r["ckpt"] == "best_near"
                   else f" [{r['ckpt']}]")
            print(f"[{i}/{len(ckpts)}] {r['run']}{tag}: exact_n={r['exact_n']} "
                  f"greedy_n={r['greedy_n']} "
                  f"(train<={r['n_train_max']}, extrap={r['extrap']:+d}/"
                  f"{r['greedy_extrap']:+d})"
                  f"{extra} c50={r['c50']} maxtok50={r['maxtok50']} "
                  f"coin50={r['coin50']} pb1/20/300={r['pb1']}/{r['pb20']}/"
                  f"{r['pb300']} coin_dev={r['coin_dev_ood']} L0={r['L0']} "
                  f"mdl={r['H_bits']}+{r['DH_bits']}={r['mdl_total']}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = list(rows[0].keys())          # union: anbn rows lack the run columns
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    rows.sort(key=lambda r: (-r["exact_n"], r["c50"] if r["c50"] == r["c50"] else 1e9))
    print(f"\n=== league table (top 15 of {len(rows)}) -> {args.out} ===")
    run_cols = [k for k in fields if k.startswith("exact_n_")]
    print(f"{'exact_n':>7} {'greedy_n':>8} {'extrap':>7} "
          + "".join(f"{c:>10}" for c in run_cols)
          + f" {'c50':>8} {'maxtok50':>8} {'coin50':>8} {'L0':>6}  run")
    for r in rows[:15]:
        print(f"{r['exact_n']:>7} {r['greedy_n']:>8} {r['extrap']:>+7} "
              + "".join(f"{r.get(c, ''):>10}" for c in run_cols)
              + f" {r['c50']:>8} {r['maxtok50']:>8} {r['coin50']:>8} "
              f"{r['L0']:>6}  {r['run']}")


if __name__ == "__main__":
    main()
