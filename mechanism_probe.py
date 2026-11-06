"""
mechanism_probe.py
==================
Two read-outs of HOW a NoPE checkpoint counts, for networks whose attention is
content-selective (Wq/Wk populated, so anatomize.py's prefix-mean reduction
does not apply). Task-aware (a^n b^n and a^n b^n c^n).

1. Coin profile.  P(next = b | <bos> a^k) for k = 1..--max_k. The true grammar
   is memoryless (0.30 at every k). Under NoPE the only k-dependent signal in
   an all-a prefix is the dilution of <bos> in the causal prefix (1/(k+1)),
   so any drift here is a learned, non-stationary coin. A coin that decays to
   ~0 at large k is exactly what produces the large NEGATIVE coin excess /
   negative far-OOD Delta% seen on some checkpoints (the cancellation exploit
   audit_counters.py describes): the counter generalizes, the distribution
   does not.

2. Attention census.  For the string of --n, at each loop iteration and head,
   the attention distribution from a set of diagnostic query positions (last
   a, middle b, last b, [middle c, last c]) is summed by SOURCE TOKEN CLASS
   (<bos>, a, b, c) and compared with uniform-over-prefix. A head whose mass
   sits on the a's while another's sits on the c's is a two-counter
   implementation; mass equal to the class's share of the prefix (the 'unif'
   column) means the head is effectively uniform, i.e. prefix-mean.
   'ent/max' is attention entropy over the prefix as a fraction of log2(prefix
   length): 1.00 = perfectly uniform, ~0 = one-hot.

Usage:
  python mechanism_probe.py --ckpt runs/anbncn/loops_nope/ternary_L3_seed101/model.pt
  python mechanism_probe.py --ckpt ... --n 60 --max_k 300 --queries last_a,mid_b,last_b,mid_c,last_c
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from grammar import BOS, collate, task_from_config
from model import LoopedTransformer, fake_quant_act, sinusoidal_pe
from quantization import QuantConfig


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    task = task_from_config(cfg)
    qcfg = QuantConfig(mode=cfg["quant"], scale=cfg["scale"],
                       fixed_delta=cfg["fixed_delta"])
    model = LoopedTransformer.from_config(cfg, task.vocab_size, qcfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg, task


# ----------------------------------------------------------------- coin ------
@torch.no_grad()
def coin_profile(model, task, device, ks):
    """P(b | <bos> a^k) at the last position, for each k (batched by length)."""
    a_tok, b_tok = task.letters[0], task.letters[1]
    out = {}
    for k in ks:
        inp = torch.tensor([[BOS] + [a_tok] * k], device=device)
        p = F.softmax(model(inp).float(), dim=-1)[0, -1]
        out[k] = (float(p[b_tok]), float(p[a_tok]))
    return out


# ------------------------------------------------------------ attention ------
@torch.no_grad()
def attention_census(model, task, device, n, query_names):
    """Returns {(loop, head, query_name): (mass_by_class dict, unif dict, ent_frac)}.
    Re-implements LoopedTransformer.forward / Block.forward step by step so the
    per-head attention matrices can be read (F.scaled_dot_product_attention
    does not expose them)."""
    seq = task.string_for_n(n)
    inp, _ = collate([seq], device)                 # (1, L), L = k*n + 1
    L = inp.shape[1]
    # class of each INPUT position: 0 = <bos>, then the letter runs
    classes = ["<bos>"] + [task.id2tok[t] for t in seq[:-1]]

    # diagnostic query positions in INPUT index space
    # input index i holds string token i-1; last a is at n, last b at 2n, ...
    pos = {"last_a": n, "mid_b": n + max(1, n // 2), "last_b": 2 * n}
    if task.k >= 3:
        pos["mid_c"] = 2 * n + max(1, n // 2)
        pos["last_c"] = 3 * n
    queries = {q: pos[q] for q in query_names if q in pos}

    blk = model.block
    attn = blk.attn
    h, d = attn.h, model.d
    dh = d // h
    m_ln1, m_attn_resid, m_ln2, m_ffn_resid = blk.act_max

    x = model.emb(inp) * math.sqrt(d)
    if model.pos == "sin":
        x = x + sinusoidal_pe(L, d, x.device, x.dtype)
    x = fake_quant_act(x, model.act_bits, model.act_max[0])

    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    results = {}
    for t in range(model.n_loops):
        if t >= model.attn_loops:          # FFN-only iteration (--attn_loops)
            h2 = fake_quant_act(blk.ln2(x), blk.act_bits, m_ln2)
            x = fake_quant_act(x + blk.ff_out(F.relu(blk.ff_in(h2))),
                               blk.act_bits, m_ffn_resid)
            continue
        h1 = fake_quant_act(blk.ln1(x), blk.act_bits, m_ln1)
        q = attn.wq(h1).view(1, L, h, dh).transpose(1, 2)   # (1,h,L,dh)
        k = attn.wk(h1).view(1, L, h, dh).transpose(1, 2)
        v = attn.wv(h1).view(1, L, h, dh).transpose(1, 2)
        mode = getattr(attn, "attn", "softmax")
        if mode == "softmax":
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
            scores = scores.masked_fill(~causal, float("-inf"))
            A = F.softmax(scores, dim=-1)                   # (1,h,L,L)
        elif mode == "linear":
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
            A = scores.masked_fill(~causal, 0.0)            # unnormalized weights
        else:                                               # sum: all-ones causal
            A = causal.to(q.dtype).expand(1, h, L, L).clone()
        for head in range(h):
            for qname, qi in queries.items():
                row = A[0, head, qi, : qi + 1]               # prefix incl. self
                mass, unif = {}, {}
                for c in dict.fromkeys(classes):            # ordered unique
                    idx = [i for i in range(qi + 1) if classes[i] == c]
                    if idx:
                        mass[c] = float(row[idx].sum())
                        unif[c] = len(idx) / (qi + 1)
                ent = float(-(row * (row.clamp_min(1e-12)).log2()).sum())
                results[(t, head, qname)] = (mass, unif, ent / math.log2(qi + 1))
        out = (A @ v).transpose(1, 2).contiguous().view(1, L, d)
        x = fake_quant_act(x + attn.wo(out), blk.act_bits, m_attn_resid)
        h2 = fake_quant_act(blk.ln2(x), blk.act_bits, m_ln2)
        x = fake_quant_act(x + blk.ff_out(F.relu(blk.ff_in(h2))),
                           blk.act_bits, m_ffn_resid)
    return results, list(dict.fromkeys(classes)), queries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=40,
                    help="string length for the attention census")
    ap.add_argument("--max_k", type=int, default=300,
                    help="longest a-run for the coin profile")
    ap.add_argument("--queries", default="last_a,mid_b,last_b,mid_c,last_c")
    ap.add_argument("--skip_coin", action="store_true")
    ap.add_argument("--skip_attn", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, task = load(args.ckpt, device)
    print(f"task={task.name} quant={cfg['quant']}/{cfg['scale']} "
          f"pos={cfg.get('pos', 'sin')} d={cfg['d_model']} heads={cfg['n_heads']} "
          f"loops={cfg['n_loops']} wd={cfg['weight_decay']}")

    if not args.skip_coin:
        ks = list(range(1, 21)) + list(range(25, args.max_k + 1, 25))
        prof = coin_profile(model, task, device, ks)
        p_true = cfg["p_geom"]
        print(f"\n=== coin profile: P(b | <bos> a^k)   (true grammar: {p_true:.2f} "
              f"at every k; excess bits/token = log2({p_true:.2f}/P) on a stop, "
              f"log2({1-p_true:.2f}/P(a)) on a continue) ===")
        print(f"{'k':>5} {'P(b)':>7} {'P(a)':>7} {'bits/a-token vs true':>22}")
        for k in ks:
            pb, pa = prof[k]
            excess_a = -math.log2(max(pa, 1e-12)) - (-math.log2(1 - p_true))
            print(f"{k:>5} {pb:7.3f} {pa:7.3f} {excess_a:+22.3f}")
        print("(a negative last column = overconfident 'continue' coin; it is "
              "what drives far-OOD Delta% below zero while the counter is exact)")

    if not args.skip_attn:
        qnames = [q.strip() for q in args.queries.split(",") if q.strip()]
        res, classes, queries = attention_census(model, task, device, args.n, qnames)
        print(f"\n=== attention census on {task.name} string with n={args.n} "
              f"(seq len {task.seq_len(args.n)}; attn={cfg.get('attn', 'softmax')}) ===")
        if cfg.get("attn", "softmax") != "softmax":
            print("NOTE: unnormalized attention -- 'mass' is the SUM of weights on each "
                  "class (counts, not fractions); 'unif' and ent/max are not meaningful.")
        print("mass = attention mass on each source class from the query position; "
              "unif = the class's share of the causal prefix (what a uniform head "
              "would give); ent/max = entropy / log2(prefix len), 1.0 = uniform")
        hdr = f"{'loop':>4} {'head':>4} {'query':>7} " + "".join(
            f"{c:>8}{'(unif)':>7}" for c in classes) + f" {'ent/max':>8}"
        print(hdr)
        for t in range(model.attn_loops):
            for head in range(model.block.attn.h):
                for qname in queries:
                    mass, unif, ent = res[(t, head, qname)]
                    line = f"{t:>4} {head:>4} {qname:>7} "
                    for c in classes:
                        if c in mass:
                            line += f"{mass[c]:8.3f}{unif[c]:7.2f}"
                        else:
                            line += f"{'-':>8}{'':>7}"
                    print(line + f" {ent:8.2f}")
                print()
        print("reading: a head with ent/max ~1 and mass ~ unif is a prefix-mean head "
              "(anatomize.py's regime); a head with mass concentrated on one class "
              "far above its unif share is counting that class by attention; a head "
              "with ent/max ~0 is a pointer (attends to one position).")


if __name__ == "__main__":
    main()
