"""
growth_probe.py
===============
How fast does the residual stream grow with string length?

With --attn sum and --norm none, each loop adds a causal prefix SUM. If what a
loop writes is read again by Wv on the next loop, the state picks up terms of
order t^k after k loops (iterated cumulative sums). A clean accumulator keeps
growth at most linear in t (k ~ 1). This script measures the exponent.

For each probed n it runs the full string a^n b^n (+ c^n) #, records
max|x| over positions after every loop iteration and the logit spread at the
last position, then fits the log-log slope against sequence length over
n >= --fit_from. Slope ~0: bounded (softmax-like); ~1: one accumulation;
~k > 1: iterated sums, the counter cannot extrapolate.

Usage:
  python growth_probe.py --ckpt runs/sum_settled/sum_noln_wd1.0_seed104/model.pt
  python growth_probe.py --ckpt runs/sum_settled/softmax_noln_wd1.0_seed102/model.pt
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

from grammar import collate, task_from_config
from model import LoopedTransformer
from quantization import QuantConfig


def load(path, device, n_loops=None):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    task = task_from_config(cfg)
    qcfg = QuantConfig(mode=cfg["quant"], scale=cfg["scale"], fixed_delta=cfg["fixed_delta"])
    model = LoopedTransformer.from_config(cfg, task.vocab_size, qcfg, act_bits=0, n_loops=n_loops or cfg["n_loops"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg, task


@torch.no_grad()
def trace(model, inp):
    """max|x| after each loop (over all positions), and the final logits."""
    blk = model.block
    x = model.emb(inp) * math.sqrt(model.d)
    per_loop = []
    for i in range(model.n_loops):
        if i < model.attn_loops:
            x = x + blk.attn(blk.ln1(x))
        x = x + blk.ff_out(F.relu(blk.ff_in(blk.ln2(x))))
        per_loop.append(float(x.abs().max()))
    logits = model.head(model.ln_f(x))
    return per_loop, logits[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ns", default="5,10,15,20,25,30,40,50,75,100,150,200,300")
    ap.add_argument("--fit_from", type=int, default=20)
    ap.add_argument("--n_loops", type=int, default=None)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, task = load(args.ckpt, device, args.n_loops)
    ns = [int(s) for s in args.ns.split(",")]
    print(f"{args.ckpt}\n  attn={cfg.get('attn','softmax')} norm={cfg.get('norm','ln')} "
          f"loops={model.n_loops} (trained {cfg['n_loops']})")
    hdr = f"{'n':>5} {'len':>5} " + "".join(f"{'loop'+str(i+1):>11}" for i in range(model.n_loops)) \
          + f" {'logit_spread':>13} {'max|logit|':>11}"
    print(hdr)
    rows = []
    for n in ns:
        inp, _ = collate([task.string_for_n(n)], device)
        # model.forward caches quantized weights; replicate that for the manual trace
        for m in model._qmods:
            from quantization import quantize_ste
            m._qw = quantize_ste(m.weight, m.qcfg)
        try:
            per_loop, logits = trace(model, inp)
        finally:
            for m in model._qmods:
                m._qw = None
        last = logits[-1]
        spread = float(last.max() - last.min())
        mx = float(logits.abs().max())
        rows.append((n, inp.shape[1], per_loop, spread, mx))
        print(f"{n:>5} {inp.shape[1]:>5} " + "".join(f"{v:11.3g}" for v in per_loop)
              + f" {spread:13.3g} {mx:11.3g}")

    fit = [r for r in rows if r[0] >= args.fit_from]
    if len(fit) >= 3:
        L = np.log([r[1] for r in fit])
        print(f"\nlog-log slope vs sequence length (n >= {args.fit_from}):")
        for i in range(model.n_loops):
            y = np.log([max(r[2][i], 1e-12) for r in fit])
            print(f"  after loop {i+1}: max|x| ~ t^{np.polyfit(L, y, 1)[0]:.2f}")
        y = np.log([max(r[4], 1e-12) for r in fit])
        print(f"  max|logit|      ~ t^{np.polyfit(L, y, 1)[0]:.2f}")
        print("(~0 bounded, ~1 one accumulation, >1 iterated sums: the state grows "
              "faster than any fixed comparator can track)")


if __name__ == "__main__":
    main()
