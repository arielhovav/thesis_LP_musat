"""
train.py
========
Train ONE looped transformer on a counting language (a^n b^n by default, or
a^n b^n c^n with --task anbncn) under one quantization regime, logging
everything needed for the grokking / size / sparsity analyses.

Design choices that matter for the hypotheses being tested:

  * FINITE, FIXED training set (--train_size, default 500 strings, as in
    Abudy et al.'s batch of 500). A finite memorizable corpus is a
    precondition for observing a memorization phase followed by delayed
    generalization (grokking). Batches are sampled from this fixed set.
  * AdamW with DECOUPLED weight decay applied ONLY to the quantizable weight
    matrices (LayerNorm parameters get no decay, standard practice). Under a
    grid regime, decay on the latent weights is the force that produces exact
    zeros -- this is the regularizer whose Kolmogorov interpretation Musat's
    theorem is about.
  * Evaluation on a log-spaced schedule (dense early, sparse late) so the
    grokking transition is visible on a log-x plot.
  * Every eval logs: train/test/OOD Delta%, hard/soft sparsity, L0, ||th||_2^2
    of the quantized net, and the effective complexity ||th||^2 log ||th||^2
    (Musat, Section 5.3, third prediction).

Outputs in --outdir:
  results.json         config + full history + final reports
  model.pt             final checkpoint (step = --steps)
  model_best_near.pt   the eval checkpoint with the smallest |near-OOD Delta%|
                       (the validation slice; selecting on it is allowed by the
                       protocol, selecting on far-OOD is not). Written unless
                       --no_save_best. Fixed-grid QAT with decay flips weights
                       across delta/2 until the very end, so the final snapshot
                       is noisy; this checkpoint and --lr_decay_frac are the
                       two remedies. audit_counters.py scores both files.
  structure.txt        non-zero quantized weights (only if the net got small enough)

Example:
  python train.py --quant ternary --d_model 32 --steps 30000 \
                  --weight_decay 1.0 --outdir runs/ternary_d32
  python train.py --task anbncn --quant ternary --outdir runs/anbncn/ternary_d32

The task is stored in the checkpoint/results config, so every downstream tool
(evaluate, probe_lengths, audit_counters, ga_polish, measure_act_range) picks
the right vocabulary and counter span automatically. Configs from before the
--task flag existed have no such key and resolve to 'anbn'. The vocabulary of
a^n b^n c^n has one extra token ('c'), so its embedding and head each carry
d_model more quantizable parameters than the a^n b^n model of the same width.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from grammar import (PAD, TASK_NAMES, collate, get_task, make_exhaustive_test_set,
                     make_train_set)
from evaluate import full_eval
from model import ATTN_MODES, BIAS_MODES, NORM_MODES, LoopedTransformer, normalize_act_max
from quantization import QuantConfig


def eval_schedule(total_steps: int, dense_until: int = 200, per_decade: int = 12):
    """Log-spaced eval steps: every 25 steps early, then ~12 per decade."""
    pts = set(range(0, min(dense_until, total_steps) + 1, 25))
    s = dense_until
    while s < total_steps:
        s = int(math.ceil(s * 10 ** (1.0 / per_decade)))
        pts.add(min(s, total_steps))
    pts.add(total_steps)
    return sorted(pts)


def build(args, device):
    task = get_task(args.task)
    qcfg = QuantConfig(mode=args.quant, scale=args.scale, fixed_delta=args.fixed_delta)
    args.act_max = list(normalize_act_max(   # resolve once, store back on args
        [float(x) for x in args.act_max_list.split(",")] if args.act_max_list
        else args.act_max))
    model = LoopedTransformer(task.vocab_size, args.d_model, args.n_heads, args.d_ff,
                              args.n_loops, qcfg, pos=args.pos,
                              act_bits=args.act_bits,
                              act_max=args.act_max, attn=args.attn,
                              norm=args.norm, attn_loops=args.attn_loops,
                              bias=args.bias).to(device)
    # NOTE: the decay group below is built from quantizable_named_weights(),
    # which yields the gridded biases too, so --bias parameters are decayed
    # (they live on the grid and are priced by |H| like any other weight).
    decay, no_decay = [], []
    quant_ids = {id(w) for _, w in model.quantizable_named_weights()}
    for p in model.parameters():
        (decay if id(p) in quant_ids else no_decay).append(p)
    if args.init_scale != 1.0:
        # Omnigrok (Liu et al.): large initialization norm induces the lazy /
        # memorization-first regime in which delayed generalization (grokking)
        # is observed. No-op for per-tensor ternary (scale-invariant grid).
        with torch.no_grad():
            for _, w in model.quantizable_named_weights():
                w.mul_(args.init_scale)
    if args.init_ckpt:
        # Abudy et al. Experiment 3 in this class: start gradient descent AT a
        # known solution (a golden / polished network) and watch what the
        # regularizer does to it. The checkpoint's state_dict is loaded after
        # init_scale; the architecture must match the flags given here.
        ck0 = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ck0["state_dict"], strict=False)
        if missing or unexpected:
            raise SystemExit(f"--init_ckpt architecture mismatch: missing {missing}, "
                             f"unexpected {unexpected}")
        print(f"initialized from {args.init_ckpt} "
              f"(L0 = {model.sparsity_report()['L0_nonzero_count']} on this grid)")
    groups = [{"params": decay, "weight_decay": args.weight_decay}]
    if no_decay:                       # empty under --norm none (no LayerNorm)
        groups.append({"params": no_decay, "weight_decay": 0.0})
    # --optimizer decides what "weight decay" means:
    #   adamw   decoupled decay (Loshchilov & Hutter): w <- w - lr*wd*w, then an Adam
    #           step of size ~lr regardless of gradient size. Not the negative log of
    #           a fixed prior; imposes a per-weight magnitude ceiling |w| ~ c/wd.
    #   sgd_l2  SGD + momentum with COUPLED L2: gradient of loss + (wd/2)||w||^2.
    #           This is MAP estimation under the Gaussian prior of Musat sec. 5.1,
    #           i.e. "real L2": the objective ga_polish.py --hbits sgd prices.
    #   adam_l2 Adam with the L2 term added to the gradient (preconditioned by the
    #           second moment: a per-weight prior with a learned scale).
    # (torch's SGD(weight_decay=) and Adam(weight_decay=) are the coupled forms;
    #  AdamW's is decoupled.)
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.98))
    elif args.optimizer == "sgd_l2":
        opt = torch.optim.SGD(groups, lr=args.lr, momentum=0.9)
    elif args.optimizer == "adam_l2":
        opt = torch.optim.Adam(groups, lr=args.lr, betas=(0.9, 0.98))
    else:
        raise SystemExit(f"unknown --optimizer {args.optimizer}")
    return model, opt


def main(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.attn in ("sum", "linear") and args.act_bits > 0:
        # Earlier versions refused this combination ("an accumulator cannot live
        # in a bounded activation alphabet"). That is wrong for a comparator:
        # the causal sum is recomputed from the prefix at every position, so it
        # may saturate at +-act_max during the a-run and still pass through zero
        # exactly where n_a = n_b. golden_digital.py is a 15-weight ternary
        # (delta = M = 1) counter in exactly this regime, greedy-correct to 1000.
        # What IS true: the a-run count itself is lost once it clips, so a
        # network that needs n_a (not n_a - n_b) cannot be built this way.
        print("note: --attn sum with --act_bits > 0: the prefix sum is clipped to "
              f"+-{args.act_max}; only differences near zero survive (the digital-"
              "comparator regime; see golden_digital.py)")
    task = get_task(args.task)
    train = make_train_set(args.train_size, seed=args.seed, p=args.p_geom, task=task)
    test = make_exhaustive_test_set(train.n_max, extra=args.test_extra,
                                    p=args.p_geom, task=task)
    model, opt = build(args, device)

    preport = model.param_report()
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[{args.outdir}] device={device} task={task.name} attn={args.attn} norm={args.norm} "
          f"bias={args.bias} quant={args.quant} d={args.d_model} loops={args.n_loops} "
          f"params={preport['total_params']} "
          f"(quantizable={preport['quantizable_params']}, of which "
          f"{preport['quantizable_bias_slots']} bias slots)")
    print(f"  train n_max={train.n_max}; test n=1..{train.n_max + args.test_extra}; "
          f"optimal test |D:H| = {test.optimal_bits:.4f} bits")

    # ---- performance: build every training batch ONCE, on the device -------
    # The whole train set is padded and moved to the GPU a single time; each
    # step just indexes rows. Padding to the global max length is exact: pads
    # sit AFTER the real tokens, causal attention never lets real positions
    # see them, and the loss masks them, so results are unchanged.
    all_inp, all_tgt = collate(train.seqs, device)
    seq_lens = np.array([len(s) for s in train.seqs])   # to trim batch padding

    run_model = model
    if args.compile:
        try:
            run_model = torch.compile(model)
            print("  torch.compile enabled (first steps will be slow while compiling)")
        except Exception as e:  # e.g. no Triton on Windows
            print(f"  torch.compile unavailable ({e}); running eager")

    rng = np.random.default_rng(args.seed + 1)   # same stream as before: batch
    history = []                                 # composition is reproducible
    evals = set(eval_schedule(args.steps))
    t0 = time.time()
    best = {"score": float("inf"), "step": None}
    best_path = os.path.join(args.outdir, "model_best_near.pt")
    decay_start = int(round(args.steps * (1.0 - args.lr_decay_frac)))

    def lr_at(step):
        """Constant lr, then cosine to 0 over the final --lr_decay_frac of steps
        so the fixed grid settles (no-op when the fraction is 0)."""
        if args.lr_decay_frac <= 0 or step <= decay_start:
            return args.lr
        t = (step - decay_start) / max(1, args.steps - decay_start)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * t))

    def log_eval(step):
        rec = {"step": step, **full_eval(run_model, train, test, device)}
        sp = model.sparsity_report()
        rec.update({
            "hard_zero_frac": sp["hard_zero_frac"],
            "soft_zero_frac": sp["soft_zero_frac"],
            "L0": sp["L0_nonzero_count"],
            "L2sq_q": sp["L2sq_of_quantized"],
            "eff_complexity": sp["effective_complexity_L2sq_logL2sq"],
        })
        history.append(rec)
        score = abs(rec["near_ood_delta_pct"])
        if not args.no_save_best and score == score and score < best["score"]:
            best["score"], best["step"] = score, step
            torch.save({"state_dict": {k: v.detach().cpu()
                                       for k, v in model.state_dict().items()},
                        "config": vars(args), "step": step,
                        "selected_on": "min |near_ood_delta_pct|"}, best_path)
        print(f"  step {step:>7d} | train D%={rec['train_delta_pct']:+7.2f} "
              f"| test D%={rec['test_delta_pct']:+8.2f} "
              f"| near D%={rec['near_ood_delta_pct']:+8.2f} "
              f"| far D%={rec['far_ood_delta_pct']:+9.2f} "
              f"| zeros={rec['hard_zero_frac']:.3f} | L0={rec['L0']}")
        model.train()

    log_eval(0)
    model.train()
    diverged_at, spikes, ema = None, [], None
    last_step = 0
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(train.seqs), size=args.batch_size)
        sel = torch.as_tensor(idx, device=device)
        L = int(seq_lens[idx].max())                    # trim padding: pads sit
        inp, tgt = all_inp[sel, :L], all_tgt[sel, :L]   # after real tokens, so
        logits = run_model(inp)                          # this is exact
        loss = F.cross_entropy(logits.view(-1, task.vocab_size), tgt.view(-1),
                               ignore_index=PAD)
        lv = float(loss.detach())
        if not math.isfinite(lv):
            # Stop BEFORE backward: the weights are still finite, so model.pt
            # is the last good state and results.json records the divergence.
            diverged_at = step
            print(f"  !! non-finite loss at step {step}; stopping (diverged)")
            break
        if ema is not None and step > 1000 and lv > 2.0 * ema and len(spikes) < 200:
            spikes.append([step, round(lv, 4), round(ema, 4)])
        ema = lv if ema is None else 0.99 * ema + 0.01 * lv
        last_step = step
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if args.lr_decay_frac > 0:
            cur = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = cur
        opt.step()
        if step in evals:
            log_eval(step)

    if diverged_at is not None and (not history or history[-1]["step"] != last_step):
        log_eval(last_step)
    final_sparsity = model.sparsity_report()
    result = {
        "diverged_at_step": diverged_at,
        "n_loss_spikes_2x": len(spikes),
        "loss_spikes_2x": spikes,
        "config": vars(args),
        "param_report": preport,
        "test_optimal_bits": test.optimal_bits,
        "n_train_max": train.n_max,
        "history": history,
        "final": history[-1],
        "final_sparsity": final_sparsity,
        "best_near_step": best["step"],
        "best_near_abs_delta_pct": best["score"] if best["step"] is not None else None,
        "wall_time_sec": time.time() - t0,
    }
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    torch.save({"state_dict": model.state_dict(), "config": vars(args),
                "step": last_step, "diverged_at_step": diverged_at},
               os.path.join(args.outdir, "model.pt"))
    if best["step"] is not None:
        print(f"  best near-OOD checkpoint: step {best['step']} "
              f"(|near D%| = {best['score']:.2f}) -> model_best_near.pt")
    if final_sparsity["L0_nonzero_count"] <= args.structure_max_l0:
        with open(os.path.join(args.outdir, "structure.txt"), "w") as f:
            f.write(model.dump_nonzero_structure())
    if spikes:
        print(f"  {len(spikes)} loss spikes (>2x running mean) after step 1000")
    print(f"done in {result['wall_time_sec']:.0f}s -> {args.outdir}/results.json")
    return result


def get_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    # task
    ap.add_argument("--task", default="anbn", choices=list(TASK_NAMES),
                    help="formal language: 'anbn' (a^n b^n, default -- the "
                         "original experiment) or 'anbncn' (a^n b^n c^n). "
                         "Recorded in the checkpoint config.")
    # model
    ap.add_argument("--d_model", type=int, default=32)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=128)
    ap.add_argument("--n_loops", type=int, default=6)
    ap.add_argument("--attn_loops", type=int, default=None,
                    help="run attention only on the first k loop iterations "
                         "(later ones FFN-only, same tied weights, 0 extra "
                         "params). With --attn sum: one accumulation, then "
                         "amplification. Default: every loop.")
    ap.add_argument("--act_bits", type=int, default=0,
                    help="activation quantization bits (0 = fp32, default); "
                         "grid step = 2*act_max/2^bits at every quantized point")
    ap.add_argument("--act_max", type=float, default=8.0,
                    help="clip range, broadcast to all 6 points unless "
                         "--act_max_list is given")
    ap.add_argument("--act_max_list", default="",
                    help="comma-separated 6 floats, ordered emb+pe,ln1_out,"
                         "post_attn_resid,ln2_out,post_ffn_resid,final_ln "
                         "(see measure_act_range.py); overrides --act_max")
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd_l2", "adam_l2"],
                    help="adamw (decoupled decay, the default and what every run so far used), "
                         "sgd_l2 (SGD+momentum with coupled L2 = Musat's Gaussian-prior objective), "
                         "adam_l2 (Adam with coupled L2). Recorded in the config.")
    ap.add_argument("--init_ckpt", default=None,
                    help="start from this checkpoint's weights instead of a random "
                         "init (Abudy-style 'train from the golden network'); the "
                         "flags must describe the same architecture and grid")
    ap.add_argument("--init_scale", type=float, default=1.0,
                    help="multiply initial quantizable weights (omnigrok-style "
                         "grokking recipe uses 4-8 with fp32 + wd>0)")
    ap.add_argument("--attn", default="softmax", choices=list(ATTN_MODES),
                    help="softmax (default; with Wq=Wk=0 a prefix MEAN), linear "
                         "(unnormalized dot-product, a linear RNN) or sum (the "
                         "causal cumulative sum: an exact accumulator, Abudy's "
                         "counting unit in Musat's regime). Recorded in the config.")
    ap.add_argument("--norm", default="ln", choices=list(NORM_MODES),
                    help="ln (pre-LayerNorm, default) or none (Musat's block as "
                         "written; required for --attn sum to stay an accumulator)")
    ap.add_argument("--bias", default="none", choices=list(BIAS_MODES),
                    help="gridded, counted biases: none (default; every existing "
                         "run), ffh (ff_in, ff_out, head) or ffh_vo (also Wv, Wo). "
                         "Same grid, same STE, same weight decay, same |H| price "
                         "as the weights; recorded in the config.")
    ap.add_argument("--pos", default="sin", choices=["sin", "nope"],
                    help="positional encoding: parameter-free sinusoidal (default) or none (NoPE ablation); both add 0 params")
    # quantization
    ap.add_argument("--quant", default="ternary",
                    choices=["none", "bf16", "int16", "int12", "int8", "int6", "int4", "ternary"])
    ap.add_argument("--scale", default="per_tensor", choices=["per_tensor", "fixed"])
    ap.add_argument("--fixed_delta", type=float, default=0.05)
    # data
    ap.add_argument("--train_size", type=int, default=500)
    ap.add_argument("--p_geom", type=float, default=0.3)
    ap.add_argument("--test_extra", type=int, default=50)
    # optimization
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch_size", type=int, default=256,
                    help="FROZEN CONFIG default. Pilot runs before the batch "
                         "standardization used 64; don't mix the two in one "
                         "analysis.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_decay_frac", type=float, default=0.0,
                    help="cosine-decay the lr to 0 over this final fraction of "
                         "steps (0 = constant lr, the original behaviour). "
                         "0.2-0.3 lets a fixed grid settle before the snapshot.")
    ap.add_argument("--no_save_best", action="store_true",
                    help="do not write model_best_near.pt")
    ap.add_argument("--weight_decay", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--outdir", default="runs/run0")
    # performance / output
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model (needs Triton; may not work "
                         "on Windows). Numerics may differ at fp rounding level.")
    ap.add_argument("--structure_max_l0", type=int, default=2000,
                    help="write structure.txt when the final non-zero count "
                         "is at most this")
    return ap


if __name__ == "__main__":
    main(get_parser().parse_args())