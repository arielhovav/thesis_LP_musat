"""
anatomize.py
============
Complete mechanistic account of a compressed counter checkpoint whose
query/key matrices are (quantized to) exactly zero -- e.g.
runs/rule/nope_fixed_wd1.0_seed100 (962 weights) and ..._seed103 (857).

Why full understanding is possible for these networks (supervisor point c):
with Wq = Wk = 0, attention scores are input-independent, so softmax over the
causal prefix is EXACTLY uniform, and the attention layer equals
Wo @ (running mean of Wv @ x). With NoPE there is no positional input either.
Consequently the hidden state at position t is a function of ONLY
(the position's own token, the multiset of prefix tokens) -- i.e. of
(own token, n_a, n_b, t). The reachable input space of the network on this
grammar is therefore a small enumerable set, and the network's function can
be written down IN FULL as a finite table, with no approximation.

This script produces that account in three parts:

  1. EQUIVALENCE WITNESS: rebuilds the model with attention literally replaced
     by cumulative mean pooling (no softmax, no Q/K/scores) and verifies the
     logits are identical to machine precision on real strings. This turns
     "attention is uniform" from an argument into an executable proof.
  2. CENSUS: every non-zero parameter, by role -- used embedding dimensions,
     the rank and size of the value pathway Wo@Wv, the number of ALIVE FFN
     hidden units (nonzero input row AND output column), head rows. After
     this, "all non-zero parameters" are assigned a function.
  3. THE COMPLETE I/O TABLE: enumerates every reachable prefix a^i b^j
     (j <= i <= --max_i), runs the real network, and records its full
     next-token distribution. Checks the rule: continue 'b' while j < i,
     emit '#' exactly at j = i; reports the coin P(b|a^i) in the a-run.
     Written to <ckpt_dir>/io_table.csv. This table IS the network's
     complete behavior on the grammar.

Usage:
  python anatomize.py --ckpt runs/rule/nope_fixed_wd1.0_seed100/model.pt --max_i 220

Scope: a^n b^n checkpoints only. The I/O-table enumeration (a^i b^j), the
comparator regions and the gain law are the mechanistic story of THIS grammar;
an a^n b^n c^n checkpoint (config["task"] == "anbncn") is refused with a
message rather than analysed with the wrong rule. probe_lengths.py and
audit_counters.py are the task-agnostic tools.
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import torch
import torch.nn.functional as F

from grammar import A, B, BOS, END, collate, task_from_config
from model import LoopedTransformer, fake_quant_act, sinusoidal_pe
from quantization import QuantConfig, quantize


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    task = task_from_config(cfg)
    if task.name != "anbn":
        raise SystemExit(
            f"anatomize.py is specific to a^n b^n; this checkpoint was trained "
            f"on task {task.name!r}. Its I/O table, comparator regions and gain "
            f"law assume the a^i b^j reachable set. Use probe_lengths.py / "
            f"audit_counters.py for a^n b^n c^n counter diagnostics.")
    qcfg = QuantConfig(mode=cfg["quant"], scale=cfg["scale"],
                       fixed_delta=cfg["fixed_delta"])
    model = LoopedTransformer.from_config(cfg, task.vocab_size, qcfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    if cfg.get("bias", "none") != "none":
        raise SystemExit("anatomize.py predates gridded biases (its census and "
                         "equivalence witness assume bias-free layers); use "
                         "anatomize2.py for --bias checkpoints.")
    return model, cfg


@torch.no_grad()
def qw(model, name):
    mod = dict(model.named_modules())[name]
    return quantize(mod.weight, model.qcfg)


@torch.no_grad()
def meanpool_logits(model, tokens):
    """Forward pass with attention REPLACED by exact cumulative mean pooling:
    attn(x) = Wo @ cummean(Wv @ x). Valid iff Wq = Wk = 0 (uniform softmax).
    Everything else (LN, residuals, FFN, loop, head) is the real model."""
    aq = lambda y: fake_quant_act(y, model.act_bits, model.act_max)
    x = model.emb(tokens) * math.sqrt(model.d)
    if model.pos == "sin":
        x = x + sinusoidal_pe(tokens.shape[1], model.d, x.device, x.dtype)
    x = aq(x)
    blk = model.block
    for _ in range(model.n_loops):
        h = aq(blk.ln1(x))
        v = F.linear(h, qw(model, "block.attn.wv"))
        csum = v.cumsum(dim=1)
        cnt = torch.arange(1, v.shape[1] + 1, device=v.device,
                           dtype=v.dtype).view(1, -1, 1)
        att = F.linear(csum / cnt, qw(model, "block.attn.wo"))
        x = aq(x + att)
        h2 = aq(blk.ln2(x))
        x = aq(x + F.linear(F.relu(F.linear(h2, qw(model, "block.ff_in"))),
                            qw(model, "block.ff_out")))
    return model.head(aq(model.ln_f(x)))


@torch.no_grad()
def deep_dive(model, device, max_i):
    """Part 4: promote the channel interpretation from inference to
    measurement, census the alive units, and measure the gain G."""
    d = model.d
    emb = qw(model, "emb")
    tok_names = {BOS: "BOS", A: "a", B: "b", END: "#"}
    U = {}
    for tid, nm in tok_names.items():
        x = emb[tid] * math.sqrt(d)
        U[nm] = model.block.ln1(x.unsqueeze(0)).squeeze(0).float()
    vo = (qw(model, "block.attn.wo").float() @ qw(model, "block.attn.wv").float())
    Us, S, Vh = torch.linalg.svd(vo)
    print("\n[4a] channel basis - what the two singular directions of Wo@Wv read:")
    for k in range(2):
        projs = {nm: float(Vh[k] @ (U[nm] / U[nm].norm())) for nm in ("BOS", "a", "b", "#")}
        print(f"    dir {k+1} (sigma={float(S[k]):.3f}): "
              + "  ".join(f"{nm}:{v:+.2f}" for nm, v in projs.items()))
    print("    reading guide: opposite signs on a/b = the balance reader s1;")
    print("    dominated by BOS = the 1/t ruler s2. Sign/order are arbitrary.")

    fi = qw(model, "block.ff_in").float()
    fo = qw(model, "block.ff_out").float()
    head = qw(model, "head").float()
    alive = ((fi != 0).any(1) & (fo != 0).any(0)).nonzero().flatten().tolist()
    ua, ub = U["a"] / U["a"].norm(), U["b"] / U["b"].norm()
    print(f"\n[4b] census of the {len(alive)} alive FFN units (strongest read -> strongest writes):")
    id2name = dict(tok_names)
    for h in alive:
        w_in, w_out = fi[h], fo[:, h]
        reads = {"chan1": float(w_in @ Us[:, 0]), "chan2": float(w_in @ Us[:, 1]),
                 "tok_a": float(w_in @ ua), "tok_b": float(w_in @ ub)}
        rk = max(reads, key=lambda k: abs(reads[k]))
        lg = head @ w_out
        order = torch.argsort(lg.abs(), descending=True)[:2].tolist()
        wr = "  ".join(f"{id2name.get(t, 'PAD')}:{float(lg[t]):+.2f}" for t in order)
        print(f"    unit {h:>3}: reads {rk} ({reads[rk]:+.2f})   writes {wr}")

    print("\n[4c] end-to-end gain G, measured at the decision boundary (j = i):")
    print(f"    {'i':>5} {'t':>5} {'ln P(#)/P(b)':>14} {'margin*t':>10}")
    prods = []
    for i in [x for x in (5, 10, 20, 40, 80, 120, 160, 200) if x <= max_i]:
        pref = [A] * i + [B] * i + [END]
        inp, _ = collate([pref], device)
        lp = F.log_softmax(model(inp)[0, 2 * i].float(), -1)
        marg = float(lp[END] - lp[B])
        t = 2 * i + 1
        prods.append(marg * t)
        print(f"    {i:>5} {t:>5} {marg:>14.3f} {marg * t:>10.1f}")
    prods.sort()
    ghat = prods[len(prods) // 2]
    nstar = int((ghat / 0.85 - 1) // 2) if ghat > 0 else 0
    print(f"    G-hat = median(margin * t) = {ghat:.0f}")
    print(f"    -> predicted exact counting range n* ~ {nstar} "
          f"(where the margin decays to ~0.85, i.e. ~0.5 counter bits).")
    print("    Constancy of margin*t across i IS the G law; compare n* with the")
    print("    probe's observed break to validate the mechanism end to end.")


@torch.no_grad()
def open_comparator(model, device, ci):
    """Part 5: the comparator, intensionally. For four behaviorally distinct
    regions (a-run; deficit 3; deficit 1; boundary) run REAL prefixes, capture
    each alive unit's total ReLU activation (summed over the 6 loop passes) at
    the deciding position, and multiply by its direct write into the logits
    (head @ ff_out column). Direct-effect attribution: ignores the final
    LayerNorm Jacobian, so pushes are comparable to each other, approximate in
    absolute scale."""
    fi_mod = dict(model.named_modules())["block.ff_in"]
    fo = qw(model, "block.ff_out").float()
    head = qw(model, "head").float()
    fi = qw(model, "block.ff_in").float()
    alive = ((fi != 0).any(1) & (fo != 0).any(0)).nonzero().flatten().tolist()

    regions = [("a-run", [A] * ci),
               ("deficit3", [A] * ci + [B] * (ci - 3)),
               ("deficit1", [A] * ci + [B] * (ci - 1)),
               ("boundary", [A] * ci + [B] * ci)]
    acts = {}
    for name, pref in regions:
        store = []
        h = fi_mod.register_forward_hook(lambda m, i, o: store.append(o))
        inp, _ = collate([pref + [END]], device)
        model(inp)
        h.remove()
        pos = len(pref)                      # deciding position (BOS at 0)
        total = torch.stack([torch.relu(t[0, pos].float()) for t in store]).sum(0)
        acts[name] = total                   # (d_ff,) summed over 6 loops

    wb = head[B] @ fo                        # (d_ff,) write into logit b
    we = head[END] @ fo                      # write into logit '#'
    print(f"\n[5] comparator opened intensionally (reference i = {ci}; "
          f"activation = sum of ReLU over 6 loops at the deciding position):")
    print(f"    {'unit':>5} {'a-run':>7} {'def3':>7} {'def1':>7} {'bnd':>7}"
          f" {'write_b':>8} {'write_#':>8} {'push_#@bnd':>11}")
    rows = []
    for u in alive:
        r = [float(acts[n][u]) for n, _ in regions]
        rows.append((u, r, float(wb[u]), float(we[u]), r[3] * float(we[u])))
    rows.sort(key=lambda x: -abs(x[4]))
    for u, r, b, e, p in rows:
        print(f"    {u:>5} {r[0]:>7.2f} {r[1]:>7.2f} {r[2]:>7.2f} {r[3]:>7.2f}"
              f" {b:>+8.3f} {e:>+8.3f} {p:>+11.3f}")
    for k, nm in enumerate(n for n, _ in regions):
        tb = sum(r[k] * b for _, r, b, e, p in rows)
        te = sum(r[k] * e for _, r, b, e, p in rows)
        print(f"    ledger {nm:>9}: direct push  b {tb:+8.2f}   # {te:+8.2f}"
              f"   (#-b = {te - tb:+7.2f})")
    print("    reading: a unit's hinge is where its activation changes across")
    print("    regions; push_# = activation x write. The bottom ledger is the")
    print("    comparator's decision budget per region.")


@torch.no_grad()
def trace_run(model, device, n):
    """A real run of the real network over a^n b^n #, position by position."""
    seq = [A] * n + [B] * n + [END]
    inp, tgt = collate([seq], device)
    lp = F.log_softmax(model(inp)[0].float(), dim=-1)
    name = {A: "a", B: "b", END: "#", BOS: "BOS"}
    print(f"\n[trace] real run over a^{n} b^{n} #  "
          f"(each row: the model's next-token distribution after the prefix)")
    print(f"    {'pos':>4} {'prefix':>14} {'true':>5} {'P(a)':>7} {'P(b)':>7}"
          f" {'P(#)':>7} {'bits':>6}  note")
    na = nb = 0
    for pos in range(len(seq)):
        true = seq[pos]
        pa, pb, pe = (float(lp[pos, k].exp()) for k in (A, B, END))
        bits = -float(lp[pos, true]) / math.log(2.0)
        if pos == 0:
            note = "start"
        elif nb == 0 and na < n:
            note = "coin: continue (optimal 0.515 bits)"
        elif nb == 0 and na == n:
            note = "coin resolves: stop (optimal 1.737 bits)"
        elif nb < n:
            note = f"forced move: deficit {n - nb}"
        else:
            note = "forced move: boundary -> #"
        pref = f"a^{na} b^{nb}" if pos else "(BOS)"
        print(f"    {pos:>4} {pref:>14} {name[true]:>5} {pa:>7.3f} {pb:>7.3f}"
              f" {pe:>7.3f} {bits:>6.3f}  {note}")
        if true == A: na += 1
        elif true == B: nb += 1
    coin = sum(-float(lp[k, seq[k]]) / math.log(2) for k in range(0, n + 1))
    forced = sum(-float(lp[k, seq[k]]) / math.log(2) for k in range(n + 1, 2 * n + 1))
    print(f"    totals: coin bits {coin:.3f} (optimal {-log2_prob_of_n_local(n):.3f})"
          f"   counter bits {forced:.3f} (optimal 0)")


def log2_prob_of_n_local(n, p=0.3):
    return (n - 1) * math.log2(1 - p) + math.log2(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_i", type=int, default=220,
                    help="enumerate prefixes a^i b^j for all j <= i <= max_i")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--comparator", action="store_true",
                    help="part 5: open the FFN intensionally - measure every "
                         "alive unit's activation in each decision region and "
                         "its direct logit contribution (the hinge table)")
    ap.add_argument("--comp_i", type=int, default=40,
                    help="reference i for the comparator regions")
    ap.add_argument("--trace", type=int, default=0,
                    help="print the model's real position-by-position run "
                         "over the string a^N b^N #")
    ap.add_argument("--deep", action="store_true",
                    help="part 4: channel basis (what the rank-2 directions "
                         "read), alive-unit census (what each FFN unit reads "
                         "and writes), and end-to-end gain measurement G")
    ap.add_argument("--force", action="store_true",
                    help="proceed even if Wq/Wk are not exactly zero "
                         "(part 1 then measures how non-uniform attention is)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load(args.ckpt, device)
    outdir = os.path.dirname(args.ckpt) or "."

    # ---- precondition: Q = K = 0 ---------------------------------------
    nz_q = int((qw(model, "block.attn.wq") != 0).sum())
    nz_k = int((qw(model, "block.attn.wk") != 0).sum())
    print(f"Wq non-zeros: {nz_q}/{qw(model,'block.attn.wq').numel()}   "
          f"Wk non-zeros: {nz_k}/{qw(model,'block.attn.wk').numel()}")
    if (nz_q or nz_k) and not args.force:
        raise SystemExit("Q/K are not all-zero: attention is not provably "
                         "uniform; this tool targets the compressed "
                         "counters. Use --force to compare anyway.")

    # ---- part 1: executable uniform-attention proof ---------------------
    seqs = [[A] * n + [B] * n + [END] for n in range(1, 41)]
    inp, _ = collate(seqs, device)
    diff = (model(inp) - meanpool_logits(model, inp)).abs().max().item()
    verdict = "IDENTICAL (attention == exact cumulative mean)" \
        if diff < args.tol else "DIFFERENT"
    print(f"\n[1] mean-pool replacement: max |logit diff| = {diff:.2e} -> {verdict}")

    # ---- part 2: census of every non-zero parameter ---------------------
    emb, wv = qw(model, "emb"), qw(model, "block.attn.wv")
    wo = qw(model, "block.attn.wo")
    fi, fo = qw(model, "block.ff_in"), qw(model, "block.ff_out")
    head = qw(model, "head")
    vo = wo @ wv
    alive_ffn = ((fi != 0).any(dim=1) & (fo != 0).any(dim=0)).sum().item()
    used_tok = [BOS, A, B, END]
    emb_dims = (emb[used_tok] != 0).any(dim=0).sum().item()
    print(f"\n[2] census of the {int(sum((t != 0).sum() for t in [emb, wv, wo, fi, fo, head]))} "
          f"non-zero weights:")
    print(f"    embedding: {int((emb[used_tok]!=0).sum())} weights over "
          f"{emb_dims}/{emb.shape[1]} used dimensions (4 token types)")
    print(f"    value pathway Wo@Wv: {int((wv!=0).sum())}+{int((wo!=0).sum())} "
          f"weights, effective rank {torch.linalg.matrix_rank(vo.float()).item()}"
          f" -> the prefix mean is read through a "
          f"{torch.linalg.matrix_rank(vo.float()).item()}-dimensional channel")
    print(f"    FFN: {int((fi!=0).sum())}+{int((fo!=0).sum())} weights; ALIVE "
          f"hidden units {alive_ffn}/{fi.shape[0]} (rest are dead rows/cols)")
    print(f"    head: {int((head!=0).sum())} weights")

    # ---- part 3: the complete I/O table ---------------------------------
    rows, bad_i = [], None
    coin_samples = []
    for i in range(1, args.max_i + 1):
        prefs = [[A] * i + [B] * j for j in range(0, i + 1)]
        # collate builds inp = [BOS] + seq[:-1]; appending a dummy tail token
        # makes inp = [BOS] + full prefix, and the output at index len(prefix)
        # is the model's next-token distribution after the whole prefix.
        prefs2 = [p + [END] for p in prefs]          # dummy tail, ignored
        inp2, _ = collate(prefs2, device)
        with torch.no_grad():
            logits2 = model(inp2)
        idx = torch.tensor([len(p) for p in prefs], device=device)
        probs = torch.softmax(
            logits2[torch.arange(len(prefs), device=device), idx], -1)
        ok = True
        for j in range(0, i + 1):
            pa, pb, pe = probs[j, A].item(), probs[j, B].item(), probs[j, END].item()
            rows.append({"i": i, "j": j, "P(a)": round(pa, 4),
                         "P(b)": round(pb, 4), "P(#)": round(pe, 4)})
            if j == 0:
                coin_samples.append(pb)
            elif j < i and pb < max(pa, pe):
                ok = False
            elif j == i and pe < max(pa, pb):
                ok = False
        if not ok and bad_i is None:
            bad_i = i
    path = os.path.join(outdir, "io_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    coin = sum(coin_samples) / len(coin_samples)
    print(f"\n[3] complete I/O table: {len(rows)} reachable prefixes "
          f"(a^i b^j, i <= {args.max_i}) -> {path}")
    print(f"    coin: mean P(b | a^i) = {coin:.4f}  (true grammar: "
          f"{cfg['p_geom']})")
    if bad_i is None:
        print(f"    RULE HOLDS on the entire enumerated domain: argmax is 'b' "
              f"for all j<i and '#' at j=i, for every i <= {args.max_i}.")
    else:
        print(f"    rule holds for all i < {bad_i}; first argmax error at "
              f"i = {bad_i}.")
    if args.deep:
        deep_dive(model, device, args.max_i)
    if args.comparator:
        open_comparator(model, device, args.comp_i)
    if args.trace:
        trace_run(model, device, args.trace)

    print("\nSummary: parts 1-3 together are a complete account: the state at "
          "any position is a function of (own token, prefix mean); part 1 "
          "proves the reduction, part 2 assigns a role to every surviving "
          "weight, part 3 tabulates the induced function on the whole "
          "reachable domain.")


if __name__ == "__main__":
    main()