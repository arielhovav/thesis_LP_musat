"""
probe_lengths.py
================
Per-length, UNWEIGHTED evaluation of a checkpoint, splitting each string's
bits into its two computationally different parts.

Why this exists: the standard test |D:H| weights strings by their true
probability P(n) ~ 0.3 * 0.7^(n-1), so lengths beyond n ~ 60 carry ~zero
weight and the weighted metric literally cannot distinguish a counter that
works to n=70 from one that works to n=225. This probe evaluates each length
individually and, per length, separates:

  coin bits     targets 1..n (the a-run continue/stop decisions).
                Optimal total = -log2 P(n). Excess here grows ~linearly in n
                even for a PERFECT counter whose 0.7/0.3 coin is slightly
                miscalibrated -- it is NOT evidence of counting failure.
  counter bits  targets n+1..k*n (the deterministic remainder after the first
                b, plus '#'): b^(n-1) # for a^n b^n, b^(n-1) c^n # for
                a^n b^n c^n. Optimal = 0 bits exactly. ANY bits here are
                counting errors. This is the clean counter test. The task is
                read from the checkpoint config (see grammar.Task).

A working counter shows counter bits ~ 0 up to some break length; the probe
reports the first n where counter bits exceed --fail_bits (default 0.5).

Ablation (--zero): zero out whole weight matrices before evaluating, to test
mechanistic hypotheses read off the sparsity structure. E.g. if Wq/Wk are
nearly empty, attention should be ~uniform (a prefix-averaging counter);
then `--zero wq,wk` should leave counter bits unchanged:

  python probe_lengths.py --ckpt runs/fixed/int4_wd3.0_seed101/model.pt --max_n 300
  python probe_lengths.py --ckpt ... --max_n 100 --zero wq,wk
  python probe_lengths.py --ckpt ... --max_n 100 --zero wo        # should break it

Module suffixes for --zero: emb, wq, wk, wv, wo, ff_in, ff_out, head.
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from grammar import PAD, collate, log2_prob_of_n, make_train_set, task_from_config
from model import LoopedTransformer
from quantization import QuantConfig

LN2 = math.log(2.0)


@torch.no_grad()
def token_bits(model, seq: list[int], device, smooth: float = 1e-10):
    """(bits per target token, argmax-correct per target) for one string
    (paper's 1e-10 smoothing)."""
    inp, tgt = collate([seq], device)
    logp = F.log_softmax(model(inp).float(), dim=-1)
    p = logp.exp().clamp_min(smooth)
    nll2 = -(p.log() / LN2)
    tb = nll2.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    ok = (logp.argmax(-1) == tgt)
    return (tb * (tgt != PAD))[0].cpu(), ok[0].cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_n", type=int, default=300)
    ap.add_argument("--zero", default="",
                    help="comma-separated module suffixes to zero out before "
                         "evaluating (ablation), e.g. wq,wk")
    ap.add_argument("--fail_bits", type=float, default=0.5,
                    help="counter bits above this = counting failure")
    ap.add_argument("--n_loops", type=int, default=None,
                    help="run the SAME weights with this many loop iterations "
                         "instead of the trained value (tests whether the loop "
                         "acts as an amplifier: more loops = more gain)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]
    task = task_from_config(cfg)
    qcfg = QuantConfig(mode=cfg["quant"], scale=cfg["scale"],
                       fixed_delta=cfg["fixed_delta"])
    n_loops = args.n_loops if args.n_loops is not None else cfg["n_loops"]
    model = LoopedTransformer.from_config(cfg, task.vocab_size, qcfg, n_loops=n_loops).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    if args.n_loops is not None:
        print(f"[loops] running with n_loops={n_loops} (trained with {cfg['n_loops']})")

    if args.zero:
        wanted = [s.strip() for s in args.zero.split(",") if s.strip()]
        hit = set()
        with torch.no_grad():
            for name, w in model.quantizable_named_weights():
                base = name[:-len(".bias")] if name.endswith(".bias") else name
                short = base.split(".")[-1]          # a module's bias is zeroed with it
                if short in wanted:
                    w.zero_()
                    hit.add(short)
                    print(f"[ablate] zeroed {name} ({w.numel()} params)")
        missed = set(wanted) - hit
        if missed:
            raise SystemExit(f"--zero: no module matched {sorted(missed)}; "
                             "valid: emb, wq, wk, wv, wo, ff_in, ff_out, head")

    n_train_max = make_train_set(cfg["train_size"], seed=cfg["seed"],
                                 p=cfg["p_geom"], task=task).n_max
    ns = list(range(1, min(41, args.max_n + 1)))
    ns += list(range(45, args.max_n + 1, 5))

    run_names = list(task.counter_run_spans(1))   # e.g. ['b'] or ['b', 'c']
    split = len(run_names) > 1
    print(f"task = {task.name}; trained up to n = {n_train_max}; counter bits "
          f"optimal = 0 exactly; failure threshold = {args.fail_bits} bits")
    hdr = f"{'n':>5} {'seqlen':>6} {'coin_excess':>12} {'counter_bits':>13}"
    if split:
        hdr += "".join(f" {name + '_run_bits':>11}" for name in run_names)
    hdr += f" {'max_tok':>8} {'greedy':>6}"
    print(hdr)
    break_n = None
    greedy_break = None
    break_run = {name: None for name in run_names}
    for n in ns:
        tb, ok = token_bits(model, task.string_for_n(n), device)
        lo, hi = task.counter_span(n)
        greedy_ok = bool(ok[lo:hi].all())
        max_tok = float(tb[lo:hi].max())
        if not greedy_ok and greedy_break is None:
            greedy_break = n
        coin = float(tb[:lo].sum())                # targets 0..n (a-run + 1st b)
        counter = float(tb[lo:hi].sum())           # deterministic remainder + '#'
        coin_excess = coin - (-log2_prob_of_n(n, cfg["p_geom"]))
        per_run = {name: float(tb[a:b].sum())
                   for name, (a, b) in task.counter_run_spans(n).items()}
        for name, v in per_run.items():
            if v > args.fail_bits and break_run[name] is None:
                break_run[name] = n
        mark = ""
        if counter > args.fail_bits and break_n is None:
            break_n = n
            mark = "   <-- COUNTER FAILS"
        line = f"{n:>5} {task.seq_len(n):>6} {coin_excess:12.3f} {counter:13.3f}"
        if split:
            line += "".join(f" {per_run[name]:11.3f}" for name in run_names)
        line += f" {max_tok:8.3f} {'ok' if greedy_ok else 'WRONG':>6}"
        print(line + mark)

    print()
    if greedy_break is None:
        print(f"greedy decoding emits the correct string at every probed n "
              f"(argmax right at every counter position up to n = {args.max_n})")
    else:
        print(f"greedy decoding first emits a wrong symbol at n = {greedy_break}")
    if split:
        for name in run_names:
            b = break_run[name]
            print(f"{name}-run: " + (f"counter bits exceed {args.fail_bits} first at n = {b}"
                                     if b is not None else
                                     f"counter bits <= {args.fail_bits} at every probed n"))
        print("(if the b-run holds while the c-run breaks, the network has ONE "
              "counter (n_a - n_b) and no second readout n_a - n_c)")
    if break_n is None:
        print(f"counter bits stayed <= {args.fail_bits} up to n = {args.max_n} "
              f"(trained on n <= {n_train_max}): the counting mechanism "
              f"generalizes across the whole probed range.")
    else:
        print(f"counter breaks at n = {break_n} "
              f"(trained on n <= {n_train_max}, so it extrapolates "
              f"{break_n - n_train_max - 1} lengths beyond training).")


if __name__ == "__main__":
    main()
