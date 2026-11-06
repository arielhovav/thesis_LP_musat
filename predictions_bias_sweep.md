# Pre-registered predictions: quantization sweep with gridded biases

Written 20 Oct 2026, before `python sweep.py bias` was launched. Do not edit
after results exist; add a dated "outcome" column instead.

Design (frozen): a^n b^n, NoPE, `--norm none --attn softmax`, d=32, d_ff=128,
6 loops, batch 256, lr 1e-3, 30k steps, `--lr_decay_frac 0.25`, `--scale fixed
--fixed_delta 0.05`, `--bias ffh` (ff_in, ff_out, head). Grid: quant in
{ternary, int4, int8} (M = 0.05 / 0.35 / 6.35) x wd in {0.3, 1, 3} x p_end in
{0.3, 0.1} x seeds 100-102 (100-105 if time). Polish arms per checkpoint
(`polish_sweep.py`): natural code unguarded on the training generator;
p_end=0.1 scoring of the p=0.3 arm; `--hbits sgd` (sum k^2). Audit with
`audit_counters.py --max_n 300`; tabulate with `compare_levels.py`.

| # | prediction | how it is read off | outcome |
|---|---|---|---|
| P1 | L0 of the polished counter falls with precision: ternary >> int4 > int8. Ternary does not count past ~150 under ~3,700 weights. int4 (M=0.35, levels <= 7) is the middle case: replication returns (channel gains 3-10), polished L0 of a few hundred. | `compare_levels.py` block "trained L0 -> natural-polish L0"; anatomize2 channel gains on the smallest int4 net | |
| P2 | With biases the int8 counter drops below 16 weights: the <bos>-share channel (ch0, gain 4-6) plus one or two emb weights are replaced by one or two bias weights. | anatomize2 program readout of the smallest int8 polish: a non-zero `ff_in.bias` or `ff_out.bias` term and no gain-4+ phase channel | |
| P3 | With biases the coin is exactly flat in k (no 1/t drift): `coin_flat_ood` ~ 0 and `pb1 = pb20 = pb300` on int8 and int4 polished counters. | audit columns pb1/pb20/pb100/pb300, coin_flat_ood | |
| P4 | SGD under L2 keeps int4/int8 levels near +-1 (`sum(k^2)/L0` of the trained nets ~1-2); the code's optimum uses large levels; the trained-L0 / polished-L0 gap grows with precision (ternary ~1x, int4 ~5-10x, int8 >50x). | `level_hist.py` on model.pt vs polish_natural; compare_levels ratio | |
| P5 | The sum(k^2) polish (`--hbits sgd`) cannot afford magnitude: on int8 it either does not count or stays large (hundreds of weights), on ternary it reproduces the natural-code result (same objective up to 14.0 vs 14.62 bits). | polish_sgd rows: L0, greedy_n vs polish_natural rows | |
| P6 | Generalization at p=0.3 is a coin flip across seeds in every class (data does not demand the rule). At p=0.1 trained nets generalize 3/3 on int8, fewer on int4, 0/3 on ternary. | compare_levels block "seeds that generalize", gen = greedy_n >= 100 and coin_dev_ood <= 0.05 | |
| P7 | At p=0.1 the natural-code optimum is a counter in every int8 restart and most int4 restarts; on ternary it stays above 20,000 bits and does not count. | polish_natural rows at p=0.1: greedy_n, mdl_total | |
| P8 | Weight decay dose: wd 3 gives the smallest trained L0 on every grid; whether it also gives the most counters is open (the 81-weight LN counter came from wd 3; without LN nothing below wd 1 has run). No prediction on direction, recorded as such. | compare_levels rows by wd | |

Things that would falsify the chapter sentence of handoff sec. 10:
- an int8 polished counter with biases that is NOT cheaper than the 16-28 weight
  bias-free ones (biases do not buy anything);
- a ternary counter with biases under ~500 weights (then the margin argument was
  about the missing constant, not about magnitude);
- the sum(k^2) polish on int8 finding the same 16-30 weight counters as the
  natural code (then SGD's objective is not what keeps SGD from them).
