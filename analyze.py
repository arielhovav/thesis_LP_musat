"""
analyze.py
==========
Aggregate every runs/**/results.json into a summary table and the plots that
answer each question:

  summary.csv               one row per run: final Delta%, sparsity, L0, ...
  grokking_<group>.png      train/test/OOD Delta% and zero-fraction vs step
                            (log x). Memorization-then-jump shows up as train
                            Delta% dropping early while test/OOD Delta% stays
                            high, then falls abruptly later. Sudden vs linear
                            is read directly off these curves.
  size_sweep.png            final test Delta% vs quantizable parameter count,
                            one line per quant regime (log x). Linear-ish vs
                            step-like improvement with size is read here.
  sparsity_final.png        final hard-zero fraction and L0 per regime --
                            Musat's "quantisation strengthens the bias".
  effective_complexity.png  test Delta% vs ||th||^2 log ||th||^2 across all
                            runs (his third prediction: this, not raw count,
                            should track generalization).

Usage:  python analyze.py [--runs runs] [--out analysis] [--task anbn|anbncn]

Tasks: every results.json carries config["task"] (missing => 'anbn', the runs
from before the flag existed). Runs sweep.py writes for a^n b^n c^n live under
runs/anbncn/<group>/..., and that task directory level is skipped when
deriving <group>, so grokking_quant.png etc. keep their names. Use --task to
restrict an analysis to one language (recommended: the two languages have
different quantizable parameter counts and should not be pooled in the size
or sparsity plots), or point --runs at runs/anbncn directly:

    python analyze.py --runs runs --task anbncn --out analysis_anbncn
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grammar import TASK_NAMES

COLORS = {"none": "tab:gray", "bf16": "tab:blue", "int16": "tab:cyan", "int12": "tab:olive",
          "int8": "tab:green", "int6": "tab:brown", "int4": "tab:orange", "ternary": "tab:red"}


def load_runs(root, task=None):
    runs = []
    for path in sorted(glob.glob(os.path.join(root, "**", "results.json"),
                                 recursive=True)):
        with open(path) as f:
            r = json.load(f)
        r["_task"] = r["config"].get("task", "anbn")
        if task is not None and r["_task"] != task:
            continue
        r["_path"] = os.path.dirname(path)
        parts = os.path.relpath(r["_path"], root).split(os.sep)
        # sweep.py writes non-anbn tasks under runs/<task>/<group>/<run>:
        # drop the task level so <group> means the same thing for every task.
        if parts[0] in TASK_NAMES and len(parts) > 1:
            parts = parts[1:]
        r["_group"] = parts[0]
        runs.append(r)
    return runs


def write_summary(runs, out):
    rows = []
    for r in runs:
        c, fin, sp = r["config"], r["final"], r["final_sparsity"]
        rows.append({
            "run": os.path.relpath(r["_path"]),
            "task": r["_task"],
            "group": r["_group"], "quant": c["quant"], "d_model": c["d_model"],
            "n_loops": c["n_loops"], "weight_decay": c["weight_decay"],
            "steps": c["steps"], "seed": c["seed"],
            "params_total": r["param_report"]["total_params"],
            "params_quantizable": r["param_report"]["quantizable_params"],
            "train_delta_pct": fin["train_delta_pct"],
            "test_delta_pct": fin["test_delta_pct"],
            "ood_delta_pct": fin["ood_delta_pct"],
            # new slices; older pilot runs fall back to the combined OOD
            "near_ood_delta_pct": fin.get("near_ood_delta_pct", float("nan")),
            "far_ood_delta_pct": fin.get("far_ood_delta_pct",
                                         fin["ood_delta_pct"]),
            "hard_zero_frac": sp["hard_zero_frac"],
            "soft_zero_frac": sp["soft_zero_frac"],
            "L0_nonzero": sp["L0_nonzero_count"],
            "L2sq_quantized": sp["L2sq_of_quantized"],
            "eff_complexity": sp["effective_complexity_L2sq_logL2sq"],
        })
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return rows


def plot_grokking(runs, out):
    by_group = defaultdict(list)
    for r in runs:
        by_group[r["_group"]].append(r)
    for group, rs in by_group.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for r in rs:
            q = r["config"]["quant"]
            h = r["history"]
            steps = [max(e["step"], 1) for e in h]
            c = COLORS.get(q, None)
            axes[0].plot(steps, [e["test_delta_pct"] for e in h],
                         color=c, label=f"{q} test")
            axes[0].plot(steps, [e["train_delta_pct"] for e in h],
                         color=c, ls="--", alpha=0.5)
            axes[0].plot(steps,
                         [e.get("far_ood_delta_pct", e["ood_delta_pct"]) for e in h],
                         color=c, ls=":", alpha=0.7)
            axes[1].plot(steps, [e["hard_zero_frac"] for e in h], color=c, label=q)
        axes[0].set(xscale="log", xlabel="training step",
                    ylabel=r"$\Delta$% from optimal |D:H|",
                    title=f"{group}: generalization vs training length\n"
                          "(solid=test, dashed=train, dotted=far-OOD)")
        axes[0].axhline(0, color="k", lw=0.5)
        axes[0].set_yscale("symlog")
        axes[1].set(xscale="log", xlabel="training step",
                    ylabel="fraction of quantized weights == 0",
                    title="sparsity vs training length")
        for ax in axes:
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"grokking_{group}.png"), dpi=150)
        plt.close(fig)


def plot_size(runs, out):
    pts = defaultdict(list)
    for r in runs:
        if r["_group"] != "size":
            continue
        pts[r["config"]["quant"]].append(
            (r["param_report"]["quantizable_params"],
             r["final"]["test_delta_pct"]))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for q, xy in pts.items():
        xy.sort()
        ax.plot([x for x, _ in xy], [y for _, y in xy], "o-",
                color=COLORS.get(q), label=q)
    ax.set(xscale="log", yscale="symlog", xlabel="quantizable parameters",
           ylabel=r"final test $\Delta$%", title="network size sweep")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "size_sweep.png"), dpi=150)
    plt.close(fig)


def plot_sparsity(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    order = ["none", "bf16", "int16", "int12", "int8", "int6", "int4", "ternary"]
    for ax, key, ttl in [(axes[0], "hard_zero_frac", "final hard-zero fraction"),
                         (axes[1], "L0_nonzero", "final L0 (surviving params)")]:
        for q in order:
            vals = [r[key] for r in rows if r["quant"] == q]
            if vals:
                ax.bar(q, sum(vals) / len(vals), color=COLORS[q])
        ax.set_title(ttl)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sparsity_final.png"), dpi=150)
    plt.close(fig)


def plot_eff_complexity(rows, out):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for r in rows:
        ax.scatter(max(r["eff_complexity"], 1e-3), r["test_delta_pct"],
                   color=COLORS.get(r["quant"]), s=18)
    for q, c in COLORS.items():
        ax.scatter([], [], color=c, label=q)
    ax.set(xscale="log", yscale="symlog",
           xlabel=r"$\|\theta\|_2^2 \log \|\theta\|_2^2$ (quantized)",
           ylabel=r"final test $\Delta$%",
           title="effective complexity vs generalization (Musat Sec. 5.3)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "effective_complexity.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--task", default=None, choices=list(TASK_NAMES),
                    help="only analyze runs of this language (default: all)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = load_runs(a.runs, a.task)
    if not runs:
        raise SystemExit(f"no results.json found under {a.runs}/"
                         + (f" for task {a.task}" if a.task else ""))
    tasks = sorted({r["_task"] for r in runs})
    if len(tasks) > 1:
        print(f"note: pooling runs from {tasks}; pass --task to separate them")
    rows = write_summary(runs, a.out)
    plot_grokking(runs, a.out)
    plot_size(runs, a.out)
    plot_sparsity(rows, a.out)
    plot_eff_complexity(rows, a.out)
    print(f"{len(runs)} runs -> {a.out}/summary.csv + plots")