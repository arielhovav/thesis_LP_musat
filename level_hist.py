"""
level_hist.py
=============
Histogram of |grid level| k = |w| / delta over the quantizable weights of one
or more checkpoints (replaces the one-liner of handoff sec. 8.3; counts
gridded bias vectors too, and reports them separately). Also prints L0,
sum(k^2) and the natural-code / SGD-objective ratio sum(k^2)/L0, which is 1 on
ternary and is the factor by which SGD's L2 over-prices a magnitude solution.

Usage:
  python level_hist.py runs/int8_settled/*/polish_*.pt
"""
import collections
import sys

import torch

from quantization import QuantConfig, quantize

for path in sys.argv[1:]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    delta = cfg["fixed_delta"] if cfg.get("scale") == "fixed" else None
    if delta is None:
        print(f"{path}: not a fixed grid (scale={cfg.get('scale')}); skipped")
        continue
    qcfg = QuantConfig(mode=cfg["quant"], scale="fixed", fixed_delta=delta)
    # the state_dict holds LATENT weights for a trained net (ga_polish writes
    # exact grid values): project onto the grid exactly as the forward pass does
    sd = {k: quantize(v, qcfg) for k, v in ck["state_dict"].items()}
    w = torch.cat([(v / delta).round().abs().flatten() for k, v in sd.items()
                   if v.dim() == 2]).to(torch.int64)
    b = [(v / delta).round().abs().flatten() for k, v in sd.items()
         if v.dim() == 1 and k.endswith(".bias") and not k.startswith("ln")
         and not ".ln" in k]
    b = torch.cat(b).to(torch.int64) if b else torch.zeros(0, dtype=torch.int64)
    allk = torch.cat([w, b])
    c = collections.Counter(allk[allk != 0].tolist())
    l0 = int((allk != 0).sum())
    ssq = int((allk ** 2).sum())
    pol = cfg.get("polish")
    print(f"{path}\n  quant={cfg['quant']} delta={delta} bias={cfg.get('bias', 'none')}"
          + (f" polished@p_end={pol['p_end']} hbits={pol['hbits']}" if pol else ""))
    print(f"  L0 = {l0} (weights {int((w != 0).sum())}, biases {int((b != 0).sum())})   "
          f"sum(k^2) = {ssq}   sum(k^2)/L0 = {ssq / max(l0, 1):.1f}   max level = {max(c) if c else 0}")
    print("  levels:", {k: c[k] for k in sorted(c)})
    if int((b != 0).sum()):
        cb = collections.Counter(b[b != 0].tolist())
        print("  bias levels:", {k: cb[k] for k in sorted(cb)})
