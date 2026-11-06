"""
sweep.py
========
Runs the experiment grids, each testable independently, optionally with a
process pool so several runs share the GPU.

Task selection
--------------
  python sweep.py quant                   # a^n b^n  -> runs/quant/...  (unchanged)
  python sweep.py quant --task anbncn     # a^n b^n c^n -> runs/anbncn/quant/...

--task injects `task` into every job and re-roots its outdir under
runs/<task>/ for any task other than the default a^n b^n, so the two
experiments never share a directory and every grid below is reusable as-is
for the new language. Note that a^n b^n c^n strings are 3n+1 tokens, so the
same --steps costs ~1.5-2x the wall time of the a^n b^n run.

  python sweep.py quant      # Exp A: all 5 precision regimes, fixed size
  python sweep.py grokking   # Exp B: one long run per regime; the eval
                             #        HISTORY of a single long run is the whole
                             #        training-length curve
  python sweep.py size       # Exp C: d_model in {8,16,32,64,128} x regimes
  python sweep.py wd         # weight-decay dose-response (per-tensor scale)
  python sweep.py fixed      # THE CRUX: wd dose-response under a FIXED grid,
                             #        where decay is an L0/MDL penalty
  python sweep.py confirm    # 2x2 {none,ternary}x{wd 0,1} -- run with >= 5
                             #        seeds to power the pilot surprises
  python sweep.py memorize   # grokking hunt: small train sets (50/100/200)
  python sweep.py bias --seeds 100 101 102 --parallel 3
                             # handoff sec. 0: {ternary,int4,int8} x wd {0.3,1,3}
                             #   x p_end {0.3,0.1}, no-LN NoPE softmax fixed grid,
                             #   gridded biases (--bias ffh). 54 runs at 3 seeds.
  python sweep.py bias_ctrl  # the matching bias-free cells at wd 1
  python sweep.py grid --seeds 100 101 102 --parallel 3
                             # M fixed at 6.35, delta shrinking: ternary..int16 (36 runs)
  python sweep.py all        # (does not include bias / bias_ctrl)

Parallelism
-----------
At d_model=32 a single training process uses ~1% of a modern GPU and well
under 1 GB of VRAM, so total throughput scales almost linearly with the number
of concurrent processes (until the CPU cores driving the kernel launches
saturate -- 3-4 is a good number on a gaming laptop).

  python sweep.py fixed --seeds 100 101 102 --parallel 3

launches up to 3 runs at once. Each run's live output goes to
<outdir>/train.log; this process prints compact [start]/[done] lines.
--parallel 1 (the default) preserves the old behavior: runs execute
sequentially in-process with their logs streamed to the console.

--resume skips any run whose results.json already exists, so an interrupted
sweep can be re-launched safely. It is OFF by default on purpose: after a
config change (e.g. the batch-size standardization) a bare re-run should
regenerate results, not silently keep stale ones.
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time

from grammar import TASK_NAMES
from train import get_parser, main as train_main

QUANTS = ["none", "bf16", "int8", "int4", "ternary"]


# ----------------------- experiment grids -> job lists ------------------------
def exp_quant(steps, seeds):
    return [{"quant": q, "steps": steps, "seed": s,
             "outdir": f"runs/quant/{q}_seed{s}"}
            for q, s in itertools.product(QUANTS, seeds)]


def exp_grokking(steps, seeds):
    # Long runs; the log-spaced eval history = the training-length curve.
    return [{"quant": q, "steps": steps, "seed": s,
             "outdir": f"runs/grokking/{q}_seed{s}"}
            for q, s in itertools.product(QUANTS, seeds)]


def exp_size(steps, seeds):
    sizes = [(8, 2, 32), (16, 4, 64), (32, 4, 128), (64, 8, 256), (128, 8, 512)]
    return [{"d_model": d, "n_heads": h, "d_ff": f, "quant": q, "steps": steps,
             "seed": s, "outdir": f"runs/size/{q}_d{d}_seed{s}"}
            for (d, h, f), q, s in itertools.product(sizes, QUANTS, seeds)]


def exp_wd(steps, seeds):
    return [{"weight_decay": wd, "quant": q, "steps": steps, "seed": s,
             "outdir": f"runs/wd/{q}_wd{wd}_seed{s}"}
            for wd, q, s in itertools.product([0.0, 0.1, 0.3, 1.0, 3.0],
                                              ["ternary", "none"], seeds)]


def exp_fixed(steps, seeds):
    """THE CRUX: weight decay under a FIXED grid, where the Lp collapse makes
    it an L0/MDL penalty. Readouts: (1) sparsity should now show a clean wd
    dose-response (per-tensor scaling showed none); (2) does the late OOD
    blow-up seen at wd>0 under per-tensor scaling disappear or invert?"""
    return [{"quant": q, "scale": "fixed", "fixed_delta": 0.05,
             "weight_decay": wd, "steps": steps, "seed": s,
             "outdir": f"runs/fixed/{q}_wd{wd}_seed{s}"}
            for q, wd, s in itertools.product(["ternary", "int4"],
                                              [0.0, 0.3, 1.0, 3.0], seeds)]


def exp_confirm(steps, seeds):
    """Power up the two pilot surprises ('wd=0 beats wd>0', 'ternary retains
    OOD best') on just the decisive cells. Run with >= 5 seeds."""
    return [{"quant": q, "weight_decay": wd, "steps": steps, "seed": s,
             "outdir": f"runs/confirm/{q}_wd{wd}_seed{s}"}
            for q, wd, s in itertools.product(["none", "ternary"],
                                              [0.0, 1.0], seeds)]


def exp_memorize(steps, seeds):
    """Grokking hunt: 500 geometric strings may be too rich to memorize, which
    would explain the absence of a memorize-then-jump phase. Shrink the train
    set; if delayed generalization exists in this task family, it appears here."""
    return [{"train_size": ts, "quant": q, "steps": steps, "seed": s,
             "outdir": f"runs/memorize/{q}_ts{ts}_seed{s}"}
            for ts, q, s in itertools.product([50, 100, 200],
                                              ["ternary", "none"], seeds)]


def exp_tail(steps, seeds):
    """THE THEORETICALLY DECISIVE EXPERIMENT. At p=0.3, exact counting past
    n~15 buys ~nothing in |D:H| on the training distribution, so a compressed
    APPROXIMATE counter may be the genuine MDL optimum for this data. Thicken
    the tail (smaller p => longer strings carry real probability mass => exact
    counting starts paying in |D:H|). If exact_n grows substantially as p
    drops, MDL was optimizing exactly what it was given and the 'failure' was
    the data distribution; if not, the failure is in the optimizer or the
    architecture. Audit these with --max_n 300 (train n_max grows with 1/p)."""
    conds = [("ternary", "per_tensor", 0.0),   # best-counter condition
             ("ternary", "fixed", 1.0)]        # MDL-pressure condition
    return [{"quant": q, "scale": sc, "fixed_delta": 0.05, "weight_decay": wd,
             "p_geom": p, "steps": steps, "seed": s,
             "outdir": f"runs/tail/{q}_{sc}_wd{wd}_p{p}_seed{s}"}
            for (q, sc, wd), p, s in itertools.product(
                conds, [0.3, 0.15, 0.05], seeds)]


def exp_loops(steps, seeds):
    """Musat's 'looped depth helps low-K targets' prediction, and a mechanism
    check: 6 iterations of a 32-dim block may simply not express exact
    counting at large n. Depth is free in parameters, so any exact_n gain here
    is pure compute, not description length. wd=0 isolates the depth effect
    in the best-counter regime. NOTE: step time scales ~linearly with loops."""
    return [{"n_loops": nl, "quant": q, "weight_decay": 0.0, "steps": steps,
             "seed": s, "outdir": f"runs/loops/{q}_L{nl}_seed{s}"}
            for nl, q, s in itertools.product([3, 6, 12, 24],
                                              ["ternary", "none"], seeds)]


def exp_pos(steps, seeds):
    """NoPE ablation. Absolute PEs are the prime suspect for the collapse at
    n~110 (test positions far outside the training range); NoPE adds 0 params
    (same |H| accounting) and often length-generalizes better on algorithmic
    tasks. Run at wd=0, the best-counter regime."""
    return [{"pos": p, "quant": q, "weight_decay": 0.0, "steps": steps,
             "seed": s, "outdir": f"runs/pos/{q}_{p}_seed{s}"}
            for p, q, s in itertools.product(["sin", "nope"],
                                             ["ternary", "none"], seeds)]


def exp_rule(steps, seeds):
    """Follow-ups to the NoPE discovery (ternary+NoPE+wd0 counted exactly to
    n=140-155 trained on n<=25 -- the first rule-grade extrapolation).
    Two questions:
    (a) compression vs rule, revisited: under NoPE, does fixed-grid weight
        decay still trade counting range away, or can the rule-grade solution
        be compressed? (nope x ternary x fixed grid, wd 0.3/1/3)
    (b) ceiling push: NoPE x heavy tail (p=0.05) -- the tail runs that reached
        exact_n 75-95 used sin PE, the very confound NoPE removed. Audit these
        with --max_n 400 or more; exact_n may hit the probe limit."""
    jobs = [{"pos": "nope", "quant": "ternary", "scale": "fixed",
             "fixed_delta": 0.05, "weight_decay": wd, "steps": steps,
             "seed": s, "outdir": f"runs/rule/nope_fixed_wd{wd}_seed{s}"}
            for wd, s in itertools.product([0.3, 1.0, 3.0], seeds)]
    jobs += [{"pos": "nope", "quant": "ternary", "weight_decay": 0.0,
              "p_geom": 0.05, "steps": steps, "seed": s,
              "outdir": f"runs/rule/nope_tail_p0.05_seed{s}"}
             for s in seeds]
    return jobs


def exp_ceiling(steps, seeds):
    """Test of the TWO-CEILING model. Data so far: sin PE caps exact counting
    near the training length regardless of tail mass (ceiling 1, removable by
    NoPE); under NoPE, exact_n tops out at ~150-200 at d=32 even when training
    data itself reaches n~170 (ceiling 2). Hypothesis: ceiling 2 is the analog
    resolution limit of the uniform-attention prefix-proportion counter (the
    discriminative signal shrinks like 1/L), so under NoPE exact_n should GROW
    with d_model. Also hunts for wider versions of the 962-weight artifact
    (nope + fixed grid + wd1 at d=64)."""
    sizes = [(16, 4, 64), (64, 8, 256), (128, 8, 512)]   # d32 exists in runs/pos
    jobs = [{"pos": "nope", "quant": "ternary", "weight_decay": 0.0,
             "d_model": d, "n_heads": h, "d_ff": f, "steps": steps, "seed": s,
             "outdir": f"runs/ceiling/nope_d{d}_seed{s}"}
            for (d, h, f), s in itertools.product(sizes, seeds)]
    jobs += [{"pos": "nope", "quant": "ternary", "scale": "fixed",
              "fixed_delta": 0.05, "weight_decay": 1.0,
              "d_model": 64, "n_heads": 8, "d_ff": 256, "steps": steps,
              "seed": s, "outdir": f"runs/ceiling/nope_fixed_wd1_d64_seed{s}"}
             for s in seeds]
    return jobs


def exp_grokinit(steps, seeds):
    """Directed grokking hunt (supervisor point a). Grokking requires a regime
    where memorization is found before the generalizing circuit forms --
    classically: fp32, small train set, weight decay > 0, and a LARGE
    initialization norm (omnigrok, Liu et al.). Our earlier null used
    init_scale=1; this sweeps the init norm. If grokking exists for a^n b^n,
    it appears here as a late test-Delta%% drop long after train-Delta%% goes
    negative (read grokking_grokinit.png)."""
    return [{"quant": "none", "weight_decay": 1.0, "train_size": 200,
             "init_scale": a, "steps": steps, "seed": s,
             "outdir": f"runs/grokinit/a{a}_seed{s}"}
            for a, s in itertools.product([1.0, 4.0, 8.0], seeds)]


def exp_pos_full(steps, seeds):
    """Fill the quantization x NoPE grid (Table 1's missing cells): bf16, int8
    and int4 under NoPE at wd=0. Outdirs match the existing runs/pos naming, so
    audit_counters merges them with the ternary/none cells automatically. Shows
    WHERE between fp32 and ternary the discreteness effect turns on."""
    return [{"pos": "nope", "quant": q, "weight_decay": 0.0, "steps": steps,
             "seed": s, "outdir": f"runs/pos/{q}_nope_seed{s}"}
            for q, s in itertools.product(["bf16", "int8", "int4"], seeds)]


def exp_act(steps, seeds):
    """Activation precision - the 'limit t' experiment (last Musat fidelity
    gap). The analog counter's signal is ~1/t, so counting to n needs
    ~log2(t) bits of ACTIVATION precision; the weights never do.

    Range calibration is PER QUANTIZATION POINT, not one shared act_max:
    measure_act_range.py showed the six points (emb+pe, ln1_out,
    post_attn_resid, ln2_out, post_ffn_resid, final_ln) can differ by >10x
    in natural magnitude, and WHICH point is large depends on the solution
    (the dense counter's gain lives in the residual stream, ~124; the
    compressed counter's lives in final_ln, ~27, while its own residual
    stays under 5). One shared range either clips the large point or wastes
    all the resolution on points that didn't need it - v1 of this sweep hit
    both failure modes (global act_max=8 clipped everything; global
    act_max=160 gave LN outputs a step of 1.25, larger than their whole
    natural range, causing a spurious cliff at 8 bits). Order below matches
    measure_act_range.py's print order: (emb+pe, ln1, attn_resid, ln2,
    ffn_resid, final_ln), each with ~25-40% headroom over the measured max.
    RECALIBRATE these two lists if you change architecture, wd, or regime -
    they are specific to ternary_nope_seed102 (dense) and
    nope_fixed_wd1.0_seed100 (compressed)."""
    dense_act_max = "6,7,160,6,160,9"          # measured 4.4,4.7,111,3.9,124,6.3
    compressed_act_max = "1,12,5,8,6,35"       # measured 0.28,8.6,3.5,5.7,4.0,26.6
    jobs = [{"pos": "nope", "quant": "ternary", "weight_decay": 0.0,
             "act_bits": b, "act_max_list": dense_act_max, "steps": steps,
             "seed": s, "outdir": f"runs/act/nope_b{b}_seed{s}"}
            for b, s in itertools.product([16, 8, 6, 4], seeds)]
    jobs += [{"pos": "nope", "quant": "ternary", "scale": "fixed",
              "fixed_delta": 0.05, "weight_decay": 1.0, "act_bits": b,
              "act_max_list": compressed_act_max, "steps": steps, "seed": s,
              "outdir": f"runs/act/nope_fixed_wd1_b{b}_seed{s}"}
             for b, s in itertools.product([16, 8, 6, 4], seeds)]
    return jobs


FROZEN_NOLN = {"pos": "nope", "norm": "none", "attn": "softmax", "scale": "fixed",
               "fixed_delta": 0.05, "batch_size": 256, "lr": 1e-3, "lr_decay_frac": 0.25,
               "d_model": 32, "d_ff": 128, "n_loops": 6}
# n_heads is deliberately NOT set here: it takes train.py's default (4), like
# every earlier grid in this file. The handoff's description "single-head
# attention" refers to the 32x32 matrices; if results.json of the
# runs/int8_settled/* runs says n_heads = 1, add "n_heads": 1 above so the
# bias sweep is matched to them.


def exp_bias(steps, seeds):
    """The quantization sweep of handoff section 0, with gridded, counted
    biases: every fixed-precision level Musat names x weight decay x two
    training generators, under the no-LN, no-PE, softmax, fixed-grid
    architecture. Predictions are in predictions_bias_sweep.md; write nothing
    there after looking at results. Polish with polish_sweep.py, audit with
      python audit_counters.py --root runs/bias --max_n 300 --out analysis/audit_bias.csv
    and tabulate with compare_levels.py."""
    return [{**FROZEN_NOLN, "bias": "ffh", "quant": q, "weight_decay": wd, "p_geom": p,
             "steps": steps, "seed": s,
             "outdir": f"runs/bias/{q}_wd{wd}_p{p}_seed{s}"}
            for q, wd, p, s in itertools.product(["ternary", "int4", "int8"],
                                                 [0.3, 1.0, 3.0], [0.3, 0.1], seeds)]


def exp_bias_ctrl(steps, seeds):
    """Bias-free controls for the cells that do not exist yet: the int8_settled
    runs cover (int8, wd 1, p 0.3) without biases; this adds the other two
    grids at wd 1 and the p=0.1 generator, so 'bias vs no bias' can be read at
    matched wd/p on every grid."""
    return [{**FROZEN_NOLN, "bias": "none", "quant": q, "weight_decay": 1.0, "p_geom": p,
             "steps": steps, "seed": s,
             "outdir": f"runs/bias_ctrl/{q}_wd1.0_p{p}_seed{s}"}
            for q, p, s in itertools.product(["ternary", "int4", "int8"], [0.3, 0.1], seeds)]


GRID_M = 6.35   # the int8 delta=0.05 rail; every grid below shares it, only delta changes
GRID_DELTA = {"ternary": 6.35, "int4": 6.35 / 7, "int6": 6.35 / 31, "int8": 0.05,
              "int12": 6.35 / 2047, "int16": 6.35 / 32767}


def exp_grid(steps, seeds):
    """Grid dose-response at FIXED dynamic range: M = 6.35 on every grid, so
    M/delta runs 1, 7, 31, 127, 2047, 32767 (Musat's slack (M/delta)^2 from 1
    to 1e9). The bias sweep varied M at fixed delta = 0.05; this varies delta at
    fixed M, which is the question "does a finer grid make L2 behave like L0":
    prediction: SGD trains equally well on all of them, the trained -> polished
    L0 gap and the L2-vs-L0 inversion grow with (M/delta)^2, the tilt tolerance
    (1/n_max) does not change. The ternary cell here is ternary at M = 6.35,
    NOT the delta = 0.05 ternary of the earlier sweeps."""
    return [{**FROZEN_NOLN, "bias": "ffh", "quant": q, "fixed_delta": GRID_DELTA[q],
             "weight_decay": 1.0, "p_geom": p, "steps": steps, "seed": s,
             "outdir": f"runs/grid/{q}_M6.35_p{p}_seed{s}"}
            for q, p, s in itertools.product(["ternary", "int4", "int6", "int8", "int12", "int16"],
                                             [0.3, 0.1], seeds)]


# --------------------------------- execution ----------------------------------
def run_inprocess(overrides: dict):
    args = get_parser().parse_args([])   # defaults
    for k, v in overrides.items():
        setattr(args, k, v)
    train_main(args)


def job_cmd(overrides: dict) -> list[str]:
    """Build the `python train.py --k v ...` command for a job. Override keys
    match train.py's argparse dests, so this is a generic translation."""
    train_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    cmd = [sys.executable, train_py]
    for k, v in overrides.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def execute(jobs: list[dict], parallel: int, resume: bool):
    if resume:
        kept = []
        for j in jobs:
            if os.path.exists(os.path.join(j["outdir"], "results.json")):
                print(f"[skip] {j['outdir']} (results.json exists)")
            else:
                kept.append(j)
        jobs = kept
    total = len(jobs)
    if total == 0:
        print("nothing to run")
        return

    if parallel <= 1:
        for i, j in enumerate(jobs, 1):
            print(f"=== [{i}/{total}] {j['outdir']} ===")
            run_inprocess(j)
        return

    queue, running, failed = list(jobs), [], []
    done, t_start = 0, time.time()
    print(f"{total} runs, up to {parallel} in parallel; "
          f"per-run output -> <outdir>/train.log")
    try:
        while queue or running:
            while queue and len(running) < parallel:
                j = queue.pop(0)
                os.makedirs(j["outdir"], exist_ok=True)
                log = open(os.path.join(j["outdir"], "train.log"), "w")
                p = subprocess.Popen(job_cmd(j), stdout=log,
                                     stderr=subprocess.STDOUT)
                running.append((p, j, log, time.time()))
                print(f"[start] {j['outdir']}   "
                      f"({len(running)} running, {len(queue)} queued)")
            still = []
            for p, j, log, t0 in running:
                rc = p.poll()
                if rc is None:
                    still.append((p, j, log, t0))
                    continue
                log.close()
                done += 1
                dt = time.time() - t0
                if rc == 0:
                    print(f"[done {done}/{total}] {j['outdir']}   ({dt:.0f}s)")
                else:
                    failed.append(j["outdir"])
                    print(f"[FAIL {done}/{total}] {j['outdir']} exit={rc} "
                          f"-> see {j['outdir']}/train.log")
            running = still
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\ninterrupted -- terminating child runs...")
        for p, j, log, _ in running:
            p.terminate()
            log.close()
        raise SystemExit(1)
    mins = (time.time() - t_start) / 60
    msg = f"finished {done}/{total} in {mins:.1f} min"
    if failed:
        msg += f"; FAILED: {failed}"
    print(msg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["quant", "grokking", "size", "wd", "fixed",
                                      "confirm", "memorize", "tail", "loops",
                                      "pos", "pos_full", "rule", "ceiling", "grokinit", "act",
                                      "bias", "bias_ctrl", "grid", "all"])
    ap.add_argument("--steps", type=int, default=None,
                    help="default: 30k (quant/size/wd/confirm), 60k (fixed), "
                         "100k (grokking/memorize)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[100])
    ap.add_argument("--parallel", type=int, default=1,
                    help="max concurrent training processes (3-4 is a good "
                         "number on one GPU at these model sizes)")
    ap.add_argument("--resume", action="store_true",
                    help="skip runs whose results.json already exists")
    ap.add_argument("--task", default="anbn", choices=list(TASK_NAMES),
                    help="formal language; anything but 'anbn' is written "
                         "under runs/<task>/ so existing runs are untouched")
    a = ap.parse_args()

    jobs: list[dict] = []
    if a.which in ("quant", "all"):
        jobs += exp_quant(a.steps or 30000, a.seeds)
    if a.which in ("grokking", "all"):
        jobs += exp_grokking(a.steps or 100000, a.seeds)
    if a.which in ("size", "all"):
        jobs += exp_size(a.steps or 30000, a.seeds)
    if a.which in ("wd", "all"):
        jobs += exp_wd(a.steps or 30000, a.seeds)
    if a.which in ("fixed", "all"):
        jobs += exp_fixed(a.steps or 60000, a.seeds)
    if a.which in ("confirm", "all"):
        jobs += exp_confirm(a.steps or 30000, a.seeds)
    if a.which in ("memorize", "all"):
        jobs += exp_memorize(a.steps or 100000, a.seeds)
    if a.which in ("tail", "all"):
        jobs += exp_tail(a.steps or 30000, a.seeds)
    if a.which in ("loops", "all"):
        jobs += exp_loops(a.steps or 30000, a.seeds)
    if a.which in ("pos", "all"):
        jobs += exp_pos(a.steps or 30000, a.seeds)
    if a.which in ("pos_full", "all"):
        jobs += exp_pos_full(a.steps or 30000, a.seeds)
    if a.which in ("rule", "all"):
        jobs += exp_rule(a.steps or 30000, a.seeds)
    if a.which in ("ceiling", "all"):
        jobs += exp_ceiling(a.steps or 30000, a.seeds)
    if a.which in ("grokinit", "all"):
        jobs += exp_grokinit(a.steps or 100000, a.seeds)
    if a.which in ("act", "all"):
        jobs += exp_act(a.steps or 30000, a.seeds)
    if a.which == "bias":                       # not in 'all': run it on purpose
        jobs += exp_bias(a.steps or 30000, a.seeds)
    if a.which == "bias_ctrl":
        jobs += exp_bias_ctrl(a.steps or 30000, a.seeds)
    if a.which == "grid":
        jobs += exp_grid(a.steps or 30000, a.seeds)

    if a.task != "anbn":
        for j in jobs:
            j["task"] = a.task
            assert j["outdir"].startswith("runs/"), j["outdir"]
            j["outdir"] = f"runs/{a.task}/" + j["outdir"][len("runs/"):]

    execute(jobs, a.parallel, a.resume)