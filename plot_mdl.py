"""
plot_mdl.py
===========
The two compression-chapter figures, straight from audit_counters.py output.

  fig_mdl_vs_generalization.png
      x = mdl_total (|H| + |D:H| on the run's own training set, log scale)
      y = greedy_n  (length to which greedy decoding is correct)
      one point per checkpoint, marker by checkpoint kind (final / best_near /
      guarded polish / unguarded polish), colour by weight decay. The claim
      the figure carries: the two-part code SGD and the polish minimise has
      no relation to length generalization at any scale of the code.

  fig_dh_plateau.png
      x = L0 (log), y = DH_bits, trained checkpoints only, with the analytical
      optimum (train_size x H(Geom(p))) as a line: the whole wd dose-response
      from 0 to 1 happens at constant training fit.

Usage:
  python plot_mdl.py analysis/audit_anbn_settled.csv [analysis/audit_anbncn_settled.csv] --out analysis
"""
from __future__ import annotations

import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KIND_MARKER = {"final": "o", "best_near": "^", "polished_guarded": "s",
               "polished_guarded_greedy": "s", "polished": "s"}


def kind_of(row):
    k = row["ckpt"]
    if k.startswith("polish_unguarded") or k == "polished_unguarded":
        return "unguarded polish", "x"
    if k.startswith("polish") or k.startswith("polished"):
        return "guarded polish", "s"
    if k == "best_near":
        return "best near-OOD", "^"
    return "final", "o"


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("greedy_n", "L0", "H_bits", "DH_bits", "mdl_total", "train_size"):
            if k in r and r[k] not in ("", None):
                r[k] = float(r[k])
        r["weight_decay"] = float(r.get("weight_decay", 0))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audits", nargs="+")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--p", type=float, default=0.3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    h_geom = (-a.p * math.log2(a.p) - (1 - a.p) * math.log2(1 - a.p)) / a.p

    fig, axes = plt.subplots(1, len(a.audits), figsize=(6.5 * len(a.audits), 4.8), squeeze=False)
    fig2, axes2 = plt.subplots(1, len(a.audits), figsize=(6.5 * len(a.audits), 4.8), squeeze=False)
    for ax, ax2, path in zip(axes[0], axes2[0], a.audits):
        rows = load(path)
        task = rows[0].get("task", "")
        wds = sorted({r["weight_decay"] for r in rows})
        cmap = plt.get_cmap("viridis", max(len(wds), 2))
        seen = set()
        for r in rows:
            if "mdl_total" not in r or r["mdl_total"] == "":
                continue
            label, marker = kind_of(r)
            c = cmap(wds.index(r["weight_decay"]))
            lab = label if label not in seen else None
            seen.add(label)
            ax.scatter(r["mdl_total"], r["greedy_n"], marker=marker, color=c,
                       s=38, edgecolor="k", linewidth=0.4, alpha=0.85, label=lab)
        ax.set_xscale("log")
        ax.set_xlabel("|H| + |D:H| on the training set  (bits, natural code)")
        ax.set_ylabel("greedy_n  (longest n decoded correctly)")
        ax.set_title(f"{task}: two-part code vs length generalization\ncolour = weight decay, "
                     f"{', '.join(f'{w:g}' for w in wds)}")
        ax.axhline(max(r.get("n_train_max", 0) and float(r["n_train_max"]) for r in rows),
                   color="k", lw=0.6, ls=":", label="training n_max")
        ax.legend(fontsize=7, loc="upper right")

        tr = [r for r in rows if r["ckpt"] == "final" and "DH_bits" in r]
        for r in tr:
            ax2.scatter(r["L0"], r["DH_bits"], color=cmap(wds.index(r["weight_decay"])),
                        s=38, edgecolor="k", linewidth=0.4)
        n_train = float(tr[0]["train_size"]) if tr and tr[0].get("train_size") else 500
        ax2.axhline(n_train * h_geom, color="k", lw=0.8, ls="--",
                    label=f"analytical optimum {n_train:g} x {h_geom:.3f} = {n_train * h_geom:.0f} bits")
        ax2.set_xscale("log")
        ax2.set_xlabel("L0 (surviving weights)")
        ax2.set_ylabel("|D:H| on the training set (bits)")
        ax2.set_title(f"{task}: training fit along the wd dose-response (final checkpoints)")
        ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_mdl_vs_generalization.png"), dpi=150)
    fig2.tight_layout(); fig2.savefig(os.path.join(a.out, "fig_dh_plateau.png"), dpi=150)
    print("->", os.path.join(a.out, "fig_mdl_vs_generalization.png"))
    print("->", os.path.join(a.out, "fig_dh_plateau.png"))


if __name__ == "__main__":
    main()