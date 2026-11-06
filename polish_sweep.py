"""
polish_sweep.py
===============
Runs the polish arms of handoff section 0 over every trained checkpoint under a
runs directory (default runs/bias), in parallel, resumable. Per model.pt:

  natural   unguarded ga_polish, natural code, |D:H| scored on the run's OWN
            generator (p_geom of the run)            -> polish_natural_s{k}.pt
  pend0.1   for arms trained at p_geom = 0.3 only: the same, scored on 500
            strings from p_end = 0.1 (the generator under which MDL selected
            the counter, handoff sec. 6)               -> polish_pend0.1_s{k}.pt
  sgd       --hbits sgd: the TRUE SGD objective, sum(k^2) priced at
            wd*T*delta^2/ln2 per squared level, own generator. This is the
            sec. 9 experiment that was never actually run (the old files
            labelled sgdL2 used the flat 14.01-bit price). Header must read
            "bits x sum(k^2)"; this script checks the log and flags it.
                                                       -> polish_sgd_s{k}.pt
  coin      optional (--coin EPS): natural code + --guard_coin EPS, own
            generator: the cheapest network with a STATIONARY coin
                                                       -> polish_coin{EPS}_s{k}.pt

k runs over --restarts search seeds (restart spread at p=0.3 was ~400 bits, so
use >= 2). Every output file name starts with "polish", which is what
audit_counters.py globs, so

  python audit_counters.py --root runs/bias --max_n 300 --out analysis/audit_bias.csv
  python compare_levels.py analysis/audit_bias.csv

tabulate trained and polished networks per (grid, wd, p_end) in one go.

Usage:
  python polish_sweep.py --root runs/bias --gens 4000 --pop 24 --restarts 2 --parallel 3
  python polish_sweep.py --root runs/bias --arms natural --restarts 1 --dry
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def jobs_for(ckpt: str, args) -> list[dict]:
    with open(os.path.join(os.path.dirname(ckpt), "results.json")) as f:
        cfg = json.load(f)["config"]
    out = []
    d = os.path.dirname(ckpt)
    for k in range(args.restarts):
        base = ["--ckpt", ckpt, "--gens", str(args.gens), "--pop", str(args.pop),
                "--seed", str(k)]
        if "natural" in args.arms:
            out.append({"out": f"{d}/polish_natural_s{k}.pt", "argv": base, "arm": "natural"})
        if "pend0.1" in args.arms and abs(cfg["p_geom"] - 0.3) < 1e-9:
            out.append({"out": f"{d}/polish_pend0.1_s{k}.pt",
                        "argv": base + ["--polish_pend", "0.1"], "arm": "pend0.1"})
        if "sgd" in args.arms:
            out.append({"out": f"{d}/polish_sgd_s{k}.pt",
                        "argv": base + ["--hbits", "sgd"], "arm": "sgd"})
        if args.coin is not None:
            out.append({"out": f"{d}/polish_coin{args.coin:g}_s{k}.pt",
                        "argv": base + ["--guard_coin", str(args.coin)], "arm": "coin"})
    for j in out:
        j["argv"] = j["argv"] + ["--out", j["out"]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="runs/bias")
    ap.add_argument("--ckpt_name", default="model.pt",
                    help="which checkpoint of each run to polish (model.pt or model_best_near.pt)")
    ap.add_argument("--arms", default="natural,pend0.1,sgd",
                    help="comma-separated subset of natural,pend0.1,sgd")
    ap.add_argument("--coin", type=float, default=None, help="also run the --guard_coin EPS arm")
    ap.add_argument("--gens", type=int, default=4000)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--dry", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--force", action="store_true", help="redo arms whose output exists")
    args = ap.parse_args()
    args.arms = {a.strip() for a in args.arms.split(",") if a.strip()}

    ckpts = sorted(glob.glob(os.path.join(args.root, "**", args.ckpt_name), recursive=True))
    jobs = [j for c in ckpts for j in jobs_for(c, args)]
    if not args.force:
        jobs = [j for j in jobs if not os.path.exists(j["out"])]
    print(f"{len(ckpts)} checkpoints, {len(jobs)} polish jobs"
          + (" (dry run)" if args.dry else ""))
    if args.dry:
        for j in jobs:
            print(" ", sys.executable, "ga_polish.py", " ".join(j["argv"]))
        return

    queue, running, failed, done = list(jobs), [], [], 0
    t0 = time.time()
    while queue or running:
        while queue and len(running) < args.parallel:
            j = queue.pop(0)
            log = open(j["out"][:-3] + ".log", "w")
            p = subprocess.Popen([sys.executable, os.path.join(HERE, "ga_polish.py")] + j["argv"],
                                 stdout=log, stderr=subprocess.STDOUT)
            running.append((p, j, log, time.time()))
            print(f"[start] {j['out']}  ({len(running)} running, {len(queue)} queued)")
        still = []
        for p, j, log, ts in running:
            if p.poll() is None:
                still.append((p, j, log, ts))
                continue
            log.close()
            done += 1
            ok = p.returncode == 0
            if ok and j["arm"] == "sgd":
                head = open(j["out"][:-3] + ".log").read(2000)
                if "bits x sum(k^2)" not in head:
                    ok = False
                    print(f"  !! {j['out']}: --hbits sgd did not print 'bits x sum(k^2)'; "
                          "this ga_polish.py is the pre-patch version")
            tag = "done" if ok else "FAIL"
            if not ok:
                failed.append(j["out"])
            print(f"[{tag} {done}/{len(jobs)}] {j['out']}  ({time.time() - ts:.0f}s)")
        running = still
        time.sleep(1.0)
    print(f"finished {done} jobs in {(time.time() - t0) / 60:.1f} min"
          + (f"; FAILED: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
