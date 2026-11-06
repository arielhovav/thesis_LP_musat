"""
ln_ablate.py
============
How much of a network's function is carried by its LayerNorm gains and
biases -- the 6*d unquantized scalars that |H| never counts?

1. PRICE. Counts the LN scalars that can affect the output: for each LayerNorm,
   the dims its CONSUMER reads (ln1 -> Wq/Wk/Wv columns, ln2 -> ff_in columns,
   ln_f -> head columns). Note a read-but-never-written dim still carries
   (-mu/sigma)*g + b after normalization, so "read" is the right criterion.
   Prints the count and what it would add to |H| at 8 / 16 / 32 bits each.

2. ABLATIONS, each scored with the audit's own metrics (greedy_n, exact_n,
   coin_dev_ood, |D:H| on the run's training set):
     original
     gains->mean     every LN gain vector replaced by its mean (1 scalar per LN)
     bias->0         every LN bias zeroed
     both            gains->mean and bias->0 (3 scalars left in total)
     ln_f gain->mean only the final gain flattened (is it just a temperature?)
   If 'both' keeps the counter, the LN subsidy is ~3 scalars (negligible);
   if the counter dies, per-dim LN parameters are part of the program and must
   be priced.

Usage:
  python ln_ablate.py --ckpt runs/rule_settled/nope_fixed_wd3_seed100/model_polished_guarded_greedy.pt
"""
from __future__ import annotations

import argparse
import copy
import math

import torch

from audit_counters import coin_profile, counter_profile, probe_ns, train_bits
from grammar import make_train_set, task_from_config
from model import LoopedTransformer
from quantization import QuantConfig, quantize


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    task = task_from_config(cfg)
    qcfg = QuantConfig(mode=cfg["quant"], scale=cfg["scale"], fixed_delta=cfg["fixed_delta"])
    model = LoopedTransformer.from_config(cfg, task.vocab_size, qcfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg, task


def score(model, cfg, task, device, max_n, fail_bits):
    ns = probe_ns(max_n)
    prof = counter_profile(model, ns, device, cfg["p_geom"], task)

    def rng(get):
        e = 0
        for n in ns:
            if get(n) > fail_bits:
                break
            e = n
        return e

    train = make_train_set(cfg["train_size"], seed=cfg["seed"], p=cfg["p_geom"], task=task)
    coin = coin_profile(model, task, device)
    ood = [k for k in coin if k > train.n_max]
    return {
        "greedy_n": rng(lambda n: 0.0 if prof[n][3] else 1.0),
        "exact_n": rng(lambda n: prof[n][1]),
        "coin_dev": max(abs(coin[k] - cfg["p_geom"]) for k in ood) if ood else float("nan"),
        "DH_bits": train_bits(model, train.seqs, device),
    }


@torch.no_grad()
def read_dims(model):
    q = {n: quantize(w, model.qcfg) for n, w in model.quantizable_named_weights()}
    any_col = lambda *names: torch.stack([(q[n] != 0).any(dim=0) for n in names]).any(dim=0)
    return {
        "block.ln1": any_col("block.attn.wq", "block.attn.wk", "block.attn.wv"),
        "block.ln2": any_col("block.ff_in"),
        "ln_f": any_col("head"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_n", type=int, default=300)
    ap.add_argument("--fail_bits", type=float, default=0.5)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, task = load(args.ckpt, device)
    lns = {"block.ln1": model.block.ln1, "block.ln2": model.block.ln2, "ln_f": model.ln_f}
    if not all(isinstance(m, torch.nn.LayerNorm) and m.elementwise_affine for m in lns.values()):
        raise SystemExit("this checkpoint has no affine LayerNorm; nothing to ablate")

    L0 = model.sparsity_report()["L0_nonzero_count"]
    n_q = model.param_report()["quantizable_params"]
    h_bits = L0 * (math.log2(n_q) + 1)
    rd = read_dims(model)
    n_ln = 0
    print(f"{args.ckpt}\n  L0 = {L0}, |H| (natural code) = {h_bits:.0f} bits\n")
    print("1. LN scalars that can reach the output (gain + bias on dims the consumer reads):")
    for name, mask in rd.items():
        k = int(mask.sum())
        n_ln += 2 * k
        g = lns[name].weight[mask.to(lns[name].weight.device)]
        print(f"  {name:10s} consumer reads {k:2d} dims -> {2*k:3d} scalars; "
              f"gain on read dims: min {g.min():.2f} mean {g.mean():.2f} max {g.max():.2f}")
    for b in (8, 16, 32):
        print(f"  at {b:2d} bits/scalar: +{n_ln*b:5d} bits  "
              f"(|H| {h_bits:.0f} -> {h_bits + n_ln*b:.0f}, x{(h_bits + n_ln*b)/h_bits:.2f})")

    def variant(gmean=(), bzero=()):
        m = copy.deepcopy(model)
        ms = {"block.ln1": m.block.ln1, "block.ln2": m.block.ln2, "ln_f": m.ln_f}
        with torch.no_grad():
            for n in gmean:
                ms[n].weight.fill_(float(ms[n].weight.mean()))
            for n in bzero:
                ms[n].bias.zero_()
        return m

    all_ln = tuple(lns)
    variants = {
        "original": variant(),
        "gains->mean": variant(gmean=all_ln),
        "bias->0": variant(bzero=all_ln),
        "both": variant(gmean=all_ln, bzero=all_ln),
        "ln_f gain->mean": variant(gmean=("ln_f",)),
    }
    print(f"\n2. ablations (probing n = 1..{args.max_n}):")
    print(f"  {'variant':18s} {'greedy_n':>8} {'exact_n':>7} {'coin_dev':>8} {'|D:H|':>8}")
    for name, m in variants.items():
        s = score(m, cfg, task, device, args.max_n, args.fail_bits)
        print(f"  {name:18s} {s['greedy_n']:>8} {s['exact_n']:>7} "
              f"{s['coin_dev']:>8.3f} {s['DH_bits']:>8.0f}")


if __name__ == "__main__":
    main()
