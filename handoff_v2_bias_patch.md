# Handoff v2 (20 Oct 2026): bias patch applied, pricing patch re-applied, sweep wired up

Supplement to `handoff_mdl_counter_thesis.md` (19 Oct). Sections 1-10 of that
document stand. This file covers what the 20 Oct session did against its
section 0, and exactly what is left to run. Read section A first.

## A. State of the repo you received vs. what you got back

The uploaded repo was NOT the patched one: `ga_polish.py` had no `--guard_coin`,
no `Genome.sumsq()`, `--hbits sgd` still printed `|H| = L0 x 14.01 bits
(SGD-implied ...)`, and `audit_counters.py` did not read `config["polish"]`.
`pricing_patch.zip` itself was not among the files, so its contents were
re-implemented from section 8.2 of the handoff (same semantics, slightly
richer `config["polish"]`). Consequence, unchanged from before: **every file
labelled sgdL2 on disk is a flat-14.01-bit natural-code run; the sum(k^2)
experiment has still not been run.** `polish_sweep.py` now refuses to count an
sgd arm as done unless its log header reads `bits x sum(k^2)`.

Apply the attached `bias_patch_2026-10-20.zip` over the repo (it contains all
files in this list; nothing else changed). Checkpoints on disk all load
unchanged (`cfg.get("bias", "none")`), verified on a config with the key removed.

## B. What was done (section 0, steps 1 and the plumbing for 2-4)

### B.1 Gridded, counted biases (step 1) -- section 8.4 as specified
- `model.py`: `QuantLinear(..., bias=)` with the bias on the same (delta, M)
  grid, same STE, cached per forward like the weight. `--bias {none|ffh|ffh_vo}`:
  `ffh` = ff_in (128), ff_out (32), head (5); `ffh_vo` adds Wv, Wo (64). Wq/Wk get
  none (a key bias cancels in the softmax; a query bias is a constant offset).
  Biases start at exactly 0. `quantizable_named_weights()` yields them as
  `"<module>.bias"`, so L0, |H|, sparsity, `sumsq`, the GA genome, the census
  and `dump_nonzero_structure` count them with no further changes. N = 12,608
  -> 12,773 under `ffh` (log2 N: 13.622 -> 13.641, +0.02 bits/weight as predicted).
- `LoopedTransformer.from_config(cfg, vocab_size, qcfg, **overrides)`: every
  loader in the repo (evaluate, audit_counters, ga_polish, anatomize, anatomize2,
  probe_lengths, growth_probe, ln_ablate, measure_act_range, mechanism_probe,
  ablate_qk) now builds through it. **Side effect to know:** `act_bits`/`act_max`
  are now taken from the checkpoint config in ALL tools. Before, evaluate /
  audit / ga_polish / probe_lengths / ablate_qk / ln_ablate silently built
  fp32-activation models even for `runs/act/*` checkpoints. Nothing in sections
  3-7 used `act_bits > 0`, so no number there changes; re-auditing `runs/act`
  would.
- `train.py`: `--bias` flag, stored in config; biases are in the decay group
  automatically (the group is built from `quantizable_named_weights()`).
- `ga_polish.py`: genome and mutations handle 1-D tensors (already did; verified);
  dead-unit pruning zeroes the `ff_in.bias` of a dead unit and treats a
  bias-only unit as alive (a constant unit).
- `anatomize2.py`: census rows for bias vectors, `L0 (of which k bias)`, live
  hidden units include bias-only units, ff_out bias counts as writing a residual
  dim; program readout prints `+ b` terms in every hidden-unit and logit
  equation, plus a line for what `ff_out.bias` writes into the residual. Abudy
  structural code prices a bias as log2(#target units) + 2 bits (no source).
- `probe_lengths.py --zero ff_in` also zeroes `ff_in.bias`; `ablate_qk.py`
  zeroes Wq/Wk biases if present (there are none under either mode).
  `anatomize.py` (the old a^n b^n-only tool) refuses `--bias` checkpoints and
  points to anatomize2.
- `level_hist.py` replaces the 8.3 one-liner; it projects latent weights onto
  the grid first (the one-liner did not, so it mis-reported trained ternary
  nets) and prints `sum(k^2)/L0`.

### B.2 Pricing patch (section 8.2), re-implemented
- `ga_polish.py --hbits sgd`: |H| = (wd*T*delta^2/ln2) x sum(k^2); header
  `|H| = <x> bits x sum(k^2)`; the final line also prints the natural-code price
  of the network it found. `--guard_coin EPS` rejects children with
  |P(b|a^k) - p| > EPS at k in audit_counters.COIN_KS (p = the scoring
  generator, i.e. `--polish_pend` if given). Output `config["polish"]` =
  {p_end, train_size, hbits, hbits_value, price (L0|sumsq), guard, guard_mode,
  guard_bits, guard_coin, seed, gens, pop, final_total_bits, final_L0,
  final_sumsq, from_ckpt}.
- `audit_counters.py`: `p_ref` = `config["polish"]["p_end"]` or `p_geom`;
  `coin_dev_ood`, `coin_flat_ood` and `coin50` are referenced to it (this is
  the fix for the spurious 0.19-0.25 coin_dev in section 9). New columns:
  `bias, p_geom, p_ref, polish_hbits, polish_guard_coin, L0_bias, sumsq`.
  The old pend checkpoints lack `config["polish"]`; re-running their polish, or
  a one-off `ck["config"]["polish"] = {"p_end": 0.1}` patch, fixes their rows.

### B.3 Sweep plumbing (steps 2-3)
- `sweep.py bias`: {ternary,int4,int8} x wd {0.3,1,3} x p_end {0.3,0.1} x seeds,
  `--bias ffh`, frozen no-LN config, -> `runs/bias/{q}_wd{wd}_p{p}_seed{s}`.
  54 runs at 3 seeds. `sweep.py bias_ctrl`: the bias-free cells at wd 1 for both
  generators (the int8/wd1/p0.3 one exists in runs/int8_settled).
  **n_heads is left at train.py's default (4), like every earlier grid.** The
  handoff text says "single-head"; check `n_heads` in
  `runs/int8_settled/*/results.json` and set it in `FROZEN_NOLN` if it is 1.
- `polish_sweep.py`: per model.pt runs natural / pend0.1 (p=0.3 arms only) /
  sgd / optional `--coin EPS` arms, `--restarts k` search seeds, parallel,
  resumable, outputs `polish_<arm>_s<k>.pt` + `.log` (audit globs `polish*.pt`).
- `compare_levels.py`: per (quant, bias, wd, p, ckpt-kind) table from audit
  CSVs: median L0 (range), #generalizing (greedy_n >= 100 and coin_dev <= 0.05),
  exact/greedy medians, coin_dev, mdl_total, sum(k^2), L0_bias; then the two
  headline blocks (trained->polished L0 per level; generalization count p=0.3 vs
  p=0.1 per level and wd). `--md` for the thesis.
- `predictions_bias_sweep.md`: P1-P8 pre-registered with the readout for each
  and the three falsifiers of the section-10 sentence. Do not edit after
  results exist.

### B.4 Smoke tests passed (CPU, tiny runs under runs/smoke, not shipped)
train (bias/none) -> all four polish arms -> audit -> compare_levels ->
anatomize2 (bias terms printed) -> probe_lengths --zero -> ablate_qk ->
evaluate -> growth_probe -> mechanism_probe -> measure_act_range -> ln_ablate
(refuses, no LN) -> level_hist; old-style config without `bias` key loads;
`ffh_vo` builds and forwards; dead-unit pruning with biases unit-tested.

## C. What is left to run (the actual experiment), in order

```
# 0. apply the zip; confirm the header
python ga_polish.py --ckpt runs/int8_settled/softmax_noln_int8_wd1.0_seed100/model.pt --gens 1 --pop 1 --hbits sgd --out /tmp/x.pt | head -1
#    -> must print "|H| = 14.01 bits x sum(k^2)"

# 1. the sum(k^2) experiment that was never run (section 9), on the existing nets
python ga_polish.py --ckpt runs/int8_settled/softmax_noln_int8_wd1.0_seed100/model.pt --gens 4000 --pop 24 --hbits sgd --out runs/int8_settled/softmax_noln_int8_wd1.0_seed100/polish_sgd_s0.pt
python ga_polish.py --ckpt runs/int8_settled/softmax_noln_int8_wd1.0_seed100/model.pt --gens 4000 --pop 24 --hbits sgd --polish_pend 0.1 --out runs/int8_settled/softmax_noln_int8_wd1.0_seed100/polish_sgd_pend0.1_s0.pt

# 2. the two one-liners from section 9 that were written but not run
python ablate_qk.py --ckpt runs/int8_settled/softmax_noln_int8_wd1.0_seed100/polish_guarded_bits15_2.pt
python ga_polish.py --ckpt runs/sum_settled/sum_noln_A1_wd1.0_seed102/model.pt --gens 2000 --pop 24 --guard_coin 0.05 --out runs/sum_settled/sum_noln_A1_wd1.0_seed102/polish_coin0.05_s0.pt

# 3. the sweep (GPU; ~54 runs x 30k steps; 3 in parallel)
python sweep.py bias --seeds 100 101 102 --parallel 3
python sweep.py bias_ctrl --seeds 100 101 102 --parallel 3        # optional controls
python polish_sweep.py --root runs/bias --gens 4000 --pop 24 --restarts 2 --parallel 3
python audit_counters.py --root runs/bias --max_n 300 --out analysis/audit_bias.csv
python audit_counters.py --root runs/bias_ctrl --max_n 300 --out analysis/audit_bias_ctrl.csv
python compare_levels.py analysis/audit_bias.csv analysis/audit_bias_ctrl.csv analysis/audit_anbn_int8.csv --md > analysis/levels.md
python level_hist.py runs/bias/int8_*/model.pt runs/bias/int8_*/polish_natural_s0.pt

# 4. anatomize the smallest polished network per grid (step 4)
python anatomize2.py --ckpt <smallest int8 polish> --max_i 60
python anatomize2.py --ckpt <smallest int4 polish> --max_i 60
python anatomize2.py --ckpt <smallest ternary polish> --max_i 60 --skip_neurons   # large nets: census + table only
```
Then fill the outcome column of `predictions_bias_sweep.md` and relate the
int4 circuit to the 16/28/32-weight int8 circuits of handoff section 5 (the
question P1 poses: do gains of 3-10 reappear on int4, i.e. is replication the
substitute for the missing levels 8-127?).

## D. Caveats carried forward / new
- Everything in handoff section 9 still holds except the audit coin_dev artifact
  (fixed going forward) and the missing `ablate_qk` run (still missing).
- Choice of which layers get biases is a design decision not an experiment:
  `ffh` is the minimal set the circuits in section 5 would use (constant into
  the comparator ReLU, constant into the residual, constant logit offset).
  `ffh_vo` is there if an int4 circuit turns out to want a constant in the
  value pathway; it was not run.
- The p=0.1 training arm changes the training distribution, not only the
  scoring one, so its trained networks are not comparable to section 4's
  numbers; the p=0.3 arm polished at p_end=0.1 is the comparable row (as in
  section 6).
- Nothing was run at scale in this session (no GPU); all numbers above P1-P8
  are predictions.
