"""
regrid.py
=========
Put an existing checkpoint onto another fixed grid: every quantizable weight is
rounded to the nearest multiple of the new delta and clipped to the new range,
and the config's quant / fixed_delta are rewritten so every other script
(ga_polish, audit_counters, anatomize2, precision_mdl) treats it as a native
checkpoint of that grid.

  python regrid.py --ckpt golden11.pt --quant int6 --delta 0.205 --out golden11_int6.pt
  python regrid.py --ckpt golden11.pt --quant int4 --delta 0.907 --out golden11_int4M.pt
  python regrid.py --ckpt golden_digital15_deep_polish.pt --quant int8 --delta 1.0 --out golden_digital14_int8.pt

Prints the non-zero count before and after and the largest rounding error, so
a coarse grid that cannot hold the program is visible immediately (a weight
rounded to zero, or a comparator ratio that moved).
"""
import argparse

import torch

from quantization import INT_LEVELS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--quant", required=True, choices=list(INT_LEVELS))
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = INT_LEVELS[a.quant]
    sd = ck["state_dict"]
    before = after = 0
    worst = 0.0
    for k, v in sd.items():
        if not v.is_floating_point() or v.dim() == 0:
            continue
        q = torch.clamp(torch.round(v / a.delta), -m, m) * a.delta
        before += int((v != 0).sum()); after += int((q != 0).sum())
        worst = max(worst, float((q - v).abs().max()))
        sd[k] = q
    cfg["quant"], cfg["fixed_delta"], cfg["scale"] = a.quant, a.delta, "fixed"
    cfg.setdefault("polish", {})
    cfg["polish"] = {**cfg["polish"], "regrid_from": a.ckpt}
    torch.save(ck, a.out)
    print(f"{a.ckpt} -> {a.out}: {a.quant} delta={a.delta} M={m * a.delta:.3g}; non-zeros {before} -> {after}; "
          f"largest rounding change {worst:.4g} ({'lossless' if worst < 1e-9 else 'check the function with audit_counters'})")


if __name__ == "__main__":
    main()