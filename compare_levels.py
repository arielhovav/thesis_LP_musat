"""
compare_levels.py
=================
Step 3 of handoff section 0: compare the fixed-precision levels. Reads one or
more audit_counters.py CSVs (e.g. analysis/audit_bias.csv, and the earlier
audit_anbn_int8.csv / audit_anbn_settled.csv for the bias-free baselines) and
prints, per (quant, bias, weight_decay, p_geom) cell and per checkpoint kind:

  n          seeds (or polish restarts) in the cell
  L0         median L0 (and the min..max range)
  gen        how many of the n generalize: greedy_n >= --gen_n AND
             coin_dev_ood <= --coin_eps (stationary coin against p_ref)
  exact/greedy   median exact_n / greedy_n
  coin_dev   median coin_dev_ood
  mdl        median mdl_total under the natural code, |H| + |D:H| on the
             run's own training set (polished networks scored on another
             generator carry their own |D:H| in the polish log; mdl_total here
             is always on the TRAINING set, so pend-polished rows are not
             directly comparable on this column -- see polish_sumsq below)
  sumsq      median sum(k^2) = ||theta||^2 / delta^2 (the SGD objective's
             magnitude term; = L0 on ternary)
  L0_bias    median number of surviving bias weights

Then the two headline comparisons the chapter needs, per level:
  trained L0 -> polished L0 (the gap predicted to grow with precision), and
  the generalization count at p=0.3 vs p=0.1.

Usage:
  python compare_levels.py analysis/audit_bias.csv [more.csv ...] --gen_n 100
  python compare_levels.py analysis/audit_bias.csv --md > analysis/levels.md
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def med(xs):
    xs = [x for x in xs if x == x]
    return st.median(xs) if xs else float("nan")


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                r["_src"] = p
                rows.append(r)
    return rows


def kind(r):
    c = r["ckpt"]
    if c == "final" or c == "best_near":
        return c
    return c                                   # polish_natural_s0, polish_sgd_s1, ...


def arm(r):
    """polish file -> arm name without the restart index."""
    c = r["ckpt"]
    if c in ("final", "best_near"):
        return c
    if c.startswith("polish_"):
        base = c[len("polish_"):]
        return base.rsplit("_s", 1)[0] if "_s" in base else base
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audits", nargs="+")
    ap.add_argument("--gen_n", type=int, default=100,
                    help="greedy_n needed to count as 'generalizes'")
    ap.add_argument("--coin_eps", type=float, default=0.05,
                    help="max coin_dev_ood allowed for 'generalizes'")
    ap.add_argument("--md", action="store_true", help="markdown table output")
    a = ap.parse_args()
    rows = load(a.audits)
    for r in rows:
        for k in ("L0", "L0_bias", "sumsq", "exact_n", "greedy_n", "coin_dev_ood",
                  "mdl_total", "H_bits", "DH_bits", "n_train_max", "weight_decay",
                  "p_geom", "p_ref"):
            r[k] = fnum(r.get(k))
        r["bias"] = r.get("bias", "none") or "none"
        r["_arm"] = arm(r)
        r["_gen"] = (r["greedy_n"] >= a.gen_n) and (r["coin_dev_ood"] <= a.coin_eps)

    cells = defaultdict(list)
    for r in rows:
        cells[(r["quant"], r["bias"], r["weight_decay"], r["p_geom"], r["_arm"])].append(r)

    qorder = {"ternary": 0, "int4": 1, "int8": 2}
    aorder = {"final": 0, "best_near": 1}
    keys = sorted(cells, key=lambda k: (qorder.get(k[0], 9), k[1], k[2], -k[3],
                                        aorder.get(k[4], 5), k[4]))
    hdr = ["quant", "bias", "wd", "p", "ckpt", "n", "L0 med (min..max)", "gen",
           "exact", "greedy", "coin_dev", "mdl", "sumsq", "L0_bias"]
    lines = []
    for k in keys:
        rs = cells[k]
        l0 = [r["L0"] for r in rs]
        lines.append([k[0], k[1], f"{k[2]:g}", f"{k[3]:g}", k[4], str(len(rs)),
                      f"{med(l0):.0f} ({min(l0):.0f}..{max(l0):.0f})",
                      f"{sum(r['_gen'] for r in rs)}/{len(rs)}",
                      f"{med([r['exact_n'] for r in rs]):.0f}",
                      f"{med([r['greedy_n'] for r in rs]):.0f}",
                      f"{med([r['coin_dev_ood'] for r in rs]):.3f}",
                      f"{med([r['mdl_total'] for r in rs]):.0f}",
                      f"{med([r['sumsq'] for r in rs]):.0f}",
                      f"{med([r['L0_bias'] for r in rs]):.0f}"])
    if a.md:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
        for l in lines:
            print("| " + " | ".join(l) + " |")
    else:
        w = [max(len(h), *(len(l[i]) for l in lines)) for i, h in enumerate(hdr)]
        print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
        for l in lines:
            print("  ".join(v.ljust(w[i]) for i, v in enumerate(l)))
    print(f"\ngen = greedy_n >= {a.gen_n} and coin_dev_ood <= {a.coin_eps} (coin scored "
          "against p_ref: the polish generator when there is one, else p_geom)")

    # headline 1: trained L0 -> polished L0 per level
    print("\n== trained L0 -> natural-polish L0 (median over seeds/restarts), per level ==")
    for q in ("ternary", "int4", "int8"):
        for b in sorted({k[1] for k in keys}):
            tr = [r["L0"] for k in keys if k[0] == q and k[1] == b and k[4] == "final"
                  for r in cells[k]]
            po = [r["L0"] for k in keys if k[0] == q and k[1] == b and k[4] == "natural"
                  for r in cells[k]]
            if tr or po:
                print(f"  {q:8s} bias={b:6s} trained {med(tr):7.0f}  ->  polished {med(po):7.0f}"
                      f"   (ratio {med(tr) / med(po) if po and med(po) else float('nan'):.1f}x)")
    # headline 2: generalization count at p=0.3 vs p=0.1, trained nets
    print("\n== seeds that generalize (final checkpoints), p=0.3 vs p=0.1, per level and wd ==")
    for q in ("ternary", "int4", "int8"):
        for b in sorted({k[1] for k in keys}):
            for wd in sorted({k[2] for k in keys}):
                g = {}
                for p in (0.3, 0.1):
                    rs = cells.get((q, b, wd, p, "final"), [])
                    g[p] = f"{sum(r['_gen'] for r in rs)}/{len(rs)}" if rs else "-"
                if any(v != "-" for v in g.values()):
                    print(f"  {q:8s} bias={b:6s} wd={wd:<4g}  p=0.3: {g[0.3]:>5}   p=0.1: {g[0.1]:>5}")


if __name__ == "__main__":
    main()
