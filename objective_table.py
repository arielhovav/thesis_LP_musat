"""
objective_table.py
==================
Abudy et al.'s Table 2 ("Regularizer | |H| | Train |D:H| | Test |D:H| | dTrain | dTest")
for our golden-matrix runs: for every starting network (a golden net on a grid)
and every objective it was polished under (natural, Elias, L1, L2), the
description length of the result and how far its data cost strays from the
analytical optimum.

Columns
  |H| nat      bits under the natural code (common currency across objectives)
  |H| own      bits under the objective's own price (natural / Elias / L1 / L2 at the
               natural-equivalent rate), i.e. what the search was minimising
  L0           non-zero weights
  Train D:H    bits on the 500 training strings; dTrain% = deviation from the optimum
               (the true generator's code length of those strings)
  Test D:H     expected bits per string over the exhaustive set n = 1..70, weighted by
               the generator p(1-p)^(n-1): Abudy's test |D:H| (optimum 2.94 at p = 0.3)
  OOD D:H      the same over n = 71..150, where the training data has no support
  greedy       greedy-correct length from the audit (if the audit CSV is given)

Usage:
  python objective_table.py --root runs/golden_matrix --p 0.3 --md analysis/objective_table_p0.3.md
  python objective_table.py --root runs/golden_matrix --p 0.1 --md analysis/objective_table_p0.1.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re

import torch
import torch.nn.functional as F

from grammar import collate, make_train_set, task_from_config
from model import LoopedTransformer
from quantization import INT_LEVELS, QuantConfig

LN2 = math.log(2.0)
T_RUN = 3884   # tokens the polish prices were set on (the run's p = 0.3 training set)


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]; task = task_from_config(cfg)
    qcfg = QuantConfig(mode=cfg["quant"], scale=cfg["scale"], fixed_delta=cfg["fixed_delta"])
    m = LoopedTransformer.from_config(cfg, task.vocab_size, qcfg).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, cfg, task


@torch.no_grad()
def string_bits(m, task, n, device):
    inp, tgt = collate([task.string_for_n(n)], device)
    lp = F.log_softmax(m(inp).float(), -1)
    mask = tgt >= 0
    return float(-(lp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)[mask]).sum()) / LN2


def optimal_string_bits(task, n, p):
    # the generator's code length of the string: continue the a-run n-1 times, then stop
    return -math.log2(p) - (n - 1) * math.log2(1 - p)


def n_of(task, seq):
    """n from the string length: <bos> a^n b^n # has 2n+2 tokens, a^n b^n c^n # has 3n+2."""
    runs = 3 if task.name == "anbncn" else 2
    return max(1, (len(seq) - 2) // runs)


@torch.no_grad()
def expected_bits(m, task, p, lo, hi, device):
    tot, opt, w = 0.0, 0.0, 0.0
    for n in range(lo, hi + 1):
        pn = p * (1 - p) ** (n - 1)
        tot += pn * string_bits(m, task, n, device); opt += pn * optimal_string_bits(task, n, p); w += pn
    return tot / w, opt / w


@torch.no_grad()
def train_bits(m, task, cfg, p, device):
    tr = make_train_set(cfg["train_size"], seed=cfg["seed"], p=p, task=task)
    tot, opt = 0.0, 0.0
    for s in tr.seqs:
        inp, tgt = collate([s], device)
        lp = F.log_softmax(m(inp).float(), -1); mask = tgt >= 0
        tot += float(-(lp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)[mask]).sum()) / LN2
    # optimum: the generator's length of each training string, from its n
    for s in tr.seqs:
        opt += optimal_string_bits(task, n_of(task, s), p)
    return tot, opt


def prices(m, cfg):
    delta = cfg["fixed_delta"]; mlv = INT_LEVELS[cfg["quant"]]
    N = m.param_report()["quantizable_params"]; addr = math.log2(N); nat = addr + math.log2(2 * mlv)
    L0 = 0; s1 = 0; s2 = 0; elias_v = 0.0
    for _, w in m.quantizable_named_weights():
        k = torch.round(w.detach() / delta).abs().flatten(); k = k[k != 0]
        L0 += int(k.numel()); s1 += int(k.sum()); s2 += int((k ** 2).sum())
        if k.numel():
            lg = torch.log2(k.double()); elias_v += float((lg + 2 * torch.log2(lg + 1) + 2).sum())
    return {"L0": L0, "natural": L0 * nat, "elias": L0 * addr + elias_v,
            "l1": s1 * nat, "sgd": s2 * nat, "bits_per_weight": nat}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="runs/golden_matrix")
    ap.add_argument("--p", type=float, default=0.3, help="which generator's cells to tabulate (0.3 or 0.1)")
    ap.add_argument("--audit", default="analysis/audit_golden_matrix.csv")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    greedy = {}
    if os.path.exists(a.audit):
        for r in csv.DictReader(open(a.audit)):
            greedy[(r.get("run", ""), r.get("ckpt", ""))] = r.get("greedy_n", "")
    codes = ["natural", "elias", "l1", "sgd"]
    label = {"natural": "MDL, flat (natural)", "elias": "MDL, Elias", "l1": "L1 (natural rate)", "sgd": "L2 (natural rate)"}
    starts = {}
    for d in sorted(glob.glob(os.path.join(a.root, f"*_p{a.p}"))):
        mo = re.match(r"(.*)_(natural|elias|l1|sgd)_p[\d.]+$", os.path.basename(d))
        if mo: starts.setdefault(mo.group(1), {})[mo.group(2)] = d
    rows = []
    for start, cells in starts.items():
        first = next(iter(cells.values()))
        m, cfg, task = load(os.path.join(first, "model.pt"), device)
        p = a.p
        def row(name, m, cfg, run, ck):
            pr = prices(m, cfg)
            trb, tro = train_bits(m, task, cfg, p, device)
            teb, teo = expected_bits(m, task, p, 1, 70, device)
            oob, ooo = expected_bits(m, task, p, 71, 150, device)
            g = greedy.get((run, ck), "")
            return [start, name, pr["L0"], round(pr["natural"]), round(pr[code_of(name)]) if code_of(name) else "",
                    round(trb, 1), f"{100 * (trb - tro) / tro:+.1f}", round(teb, 3), f"{100 * (teb - teo) / teo:+.1f}",
                    round(oob, 2), f"{100 * (oob - ooo) / ooo:+.1f}", g]
        def code_of(name):
            for c in codes:
                if label[c] == name: return c
            return None
        rows.append(row("(Golden)", m, cfg, first, "final"))
        for c in codes:
            if c in cells:
                mm, cc, _ = load(os.path.join(cells[c], f"polish_{c}.pt"), device)
                rows.append(row(label[c], mm, cc, cells[c], f"polish_{c}"))
    hdr = ["start", "objective", "L0", "|H| nat", "|H| own", "Train D:H", "dTrain%", "Test D:H (n<=70)", "dTest%",
           "OOD D:H (71-150)", "dOOD%", "greedy"]
    lines = ["| " + " | ".join(hdr) + " |", "| " + " | ".join("---" for _ in hdr) + " |"]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    out = "\n".join(lines)
    print(out)
    if a.md:
        os.makedirs(os.path.dirname(a.md) or ".", exist_ok=True)
        hb = -(a.p * math.log2(a.p) + (1 - a.p) * math.log2(1 - a.p)) / a.p
        open(a.md, "w", encoding="utf-8").write(f"# Objectives from the golden networks, generator p = {a.p}\n\n"
            f"Optimum test |D:H| = H_b(p)/p = {hb:.3f} bits per string; |H| in bits; L1/L2 priced at the natural-equivalent rate.\n\n" + out + "\n")
        print("->", a.md)


if __name__ == "__main__":
    main()