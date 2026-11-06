"""
precision_mdl.py
================
How many bits of PRECISION does a network actually need? The description length
that transfers to dense floating-point networks (Hinton & van Camp 1993; the
"Hinton/Hochreiter" row of Musat's Table 1) is not the non-zero count but the
number of bits per weight below which the function changes. This script
measures it directly:

  1. global curve   re-quantize every quantizable tensor onto a uniform b-bit
                    grid over [-M, M] (M = the checkpoint's own rail) for
                    b = 1..16 and report |D:H| on the training set at each b;
                    bits_needed = smallest b with |D:H| <= (1 + tol) x baseline.
  2. per tensor     the same, one tensor at a time with the rest at full
                    precision: which parts of the program carry the bits.
  3. codes          |H| under a precision code, L0 x (log2 N + bits_needed),
                    next to the natural code L0 x (log2 N + log2 2m), and the
                    two-part totals with |D:H| at that precision.

On a polished int8 counter with levels at the rail, bits_needed is small (the
program is sign + a few magnitudes); on an SGD-trained dense net it is what
weight decay did or did not buy. For the digital +-1 nets bits_needed = 1.

Usage:
  python precision_mdl.py --ckpt runs/bias/int8_wd1.0_p0.3_seed101/polish_natural_s0.pt
  python precision_mdl.py --ckpt runs/bias/int8_wd1.0_p0.3_seed101/model.pt --tol 0.05
  python precision_mdl.py --ckpt a.pt b.pt c.pt --csv analysis/precision.csv
"""
from __future__ import annotations

import argparse
import csv
import math

import torch

from audit_counters import train_bits
from grammar import make_train_set, task_from_config
from model import LoopedTransformer
from quantization import QuantConfig, quantize


def load_on_grid(path, device):
    """Model with its weights already PROJECTED onto the checkpoint's grid and no
    further quantization in the forward pass, so coarsening below is exact."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    task = task_from_config(cfg)
    qcfg = QuantConfig(mode=cfg["quant"], scale=cfg["scale"], fixed_delta=cfg["fixed_delta"])
    m = LoopedTransformer.from_config(cfg, task.vocab_size, QuantConfig("none")).to(device)
    m.load_state_dict(ck["state_dict"])
    with torch.no_grad():
        for _, w in m.quantizable_named_weights():
            w.copy_(quantize(w, qcfg))
    m.eval()
    return m, cfg, task, qcfg


def coarsen(w, bits, M):
    """Uniform signed b-bit grid over [-M, M]: 2^(b-1) - 1 levels per sign (b = 1: sign x M)."""
    if bits <= 1:
        return torch.sign(w) * M
    m = 2 ** (bits - 1) - 1
    d = M / m
    return torch.clamp(torch.round(w / d), -m, m) * d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--tol", type=float, default=0.02, help="relative |D:H| slack that still counts as 'same function'")
    ap.add_argument("--max_bits", type=int, default=16)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for path in args.ckpt:
        m, cfg, task, qcfg = load_on_grid(path, device)
        train = make_train_set(cfg["train_size"], seed=cfg["seed"], p=cfg["p_geom"], task=task)
        base = train_bits(m, train.seqs, device)
        named = list(m.quantizable_named_weights())
        orig = {n: w.detach().clone() for n, w in named}
        M = max(float(w.detach().abs().max()) for _, w in named) or 1.0
        L0 = sum(int((w != 0).sum()) for _, w in named)
        N = m.param_report()["quantizable_params"]
        mlv = qcfg.n_levels_per_sign or 1
        print(f"\n{path}\n  task={task.name} quant={cfg['quant']} L0={L0} rail M={M:.4g}  |D:H| at the checkpoint's grid = {base:.1f}")

        # 1. global curve
        print(f"  {'bits':>4} {'|D:H|':>9} {'delta':>9}")
        curve, needed = {}, None
        for b in range(1, args.max_bits + 1):
            with torch.no_grad():
                for n, w in named:
                    w.copy_(coarsen(orig[n], b, M))
            dh = train_bits(m, train.seqs, device)
            curve[b] = dh
            print(f"  {b:>4} {dh:9.1f} {dh - base:+9.1f}")
            if needed is None and dh <= base * (1 + args.tol):
                needed = b
            if needed is not None and b >= needed + 2:
                break
        with torch.no_grad():
            for n, w in named:
                w.copy_(orig[n])
        if needed is None:
            needed = args.max_bits
        print(f"  bits needed (global, tol {args.tol:.0%}): {needed}")

        # 2. per tensor
        per = {}
        for n, w in named:
            if not (orig[n] != 0).any():
                continue
            Mt = float(orig[n].abs().max())
            for b in range(1, args.max_bits + 1):
                with torch.no_grad():
                    w.copy_(coarsen(orig[n], b, Mt))
                dh = train_bits(m, train.seqs, device)
                if dh <= base * (1 + args.tol):
                    per[n] = b
                    break
            else:
                per[n] = args.max_bits
            with torch.no_grad():
                w.copy_(orig[n])
        print("  bits needed per tensor (rest at full precision): "
              + ", ".join(f"{n.replace('block.', '')}={b}" for n, b in per.items()))

        # 3. codes
        addr = math.log2(N)
        h_nat = L0 * (addr + math.log2(2 * mlv))
        h_prec = L0 * (addr + needed)
        print(f"  |H| natural = {h_nat:.0f}  (+|D:H| {base:.0f} = {h_nat + base:.0f})   "
              f"|H| precision = L0 x (log2 N + {needed}) = {h_prec:.0f}  (+|D:H| {curve[needed]:.0f} = {h_prec + curve[needed]:.0f})")
        rows.append({"ckpt": path, "task": task.name, "quant": cfg["quant"], "L0": L0, "rail_M": round(M, 4),
                     "DH_base": round(base, 1), "bits_needed": needed, "DH_at_needed": round(curve[needed], 1),
                     "H_natural": round(h_nat), "H_precision": round(h_prec),
                     **{f"bits_{n.replace('block.', '')}": b for n, b in per.items()}})
    if args.csv:
        fields = []
        for r in rows:
            fields += [k for k in r if k not in fields]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
        print(f"\n-> {args.csv}")


if __name__ == "__main__":
    main()
