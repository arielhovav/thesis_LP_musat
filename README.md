# Testing Musat's fixed-precision predictions on formal languages

This codebase tests the empirical predictions of Musat (2026), *"Neural Weight
Norm = Kolmogorov Complexity"*, using the task setup and evaluation currency of
Abudy, Well, Chemla, Katzir & Lan (2025), *"A Minimum Description Length
Approach to Regularization in Neural Networks"*. The first target language is
a^n b^n; the grammar module is written so other languages from the Abudy paper
can be added later as new files with the same interface.

## What is being tested

Musat's theorem says that in any **fixed-precision** regime, every Lp weight
norm collapses to the non-zero parameter count (his Eq. 1), so the smallest
weight norm of a looped network equals the Kolmogorov complexity of what it
computes, up to a log factor — and therefore weight decay acts as an
approximately Solomonoff-optimal simplicity prior. His Section 5.3 predictions
that this experiment addresses directly:

1. **Simple data benefits more from weight decay** — a^n b^n has very low
   Kolmogorov complexity, so weight decay should push the model toward the
   exact counting solution (test Δ% → 0), not just a smooth approximation.
2. **Quantisation strengthens the bias** — the coarser the grid (ternary <
   int4 < int8 < bf16 < fp32), the more directly ||θ||² tracks the non-zero
   count, so coarser regimes should show more exact sparsity and (per the
   theory) a stronger pull toward the generalizing solution. This is the
   quantization comparison, Experiment A.
3. **Effective complexity scales as ||θ||² log ||θ||²** — we log this quantity
   at every evaluation and plot it against generalization across all runs.
4. **Sparsity-inducing pressure converges to the same prior in fixed
   precision** — measured by the hard-zero fraction and surviving L0 at the
   end of training; if the survivor is small enough, `structure.txt` dumps
   every non-zero quantized weight so the mechanism can be read by hand
   (compare the golden-network diagrams in Abudy et al.).

On top of these, the training-length experiment probes whether generalization
improves smoothly or via a memorization phase followed by a sudden jump
(grokking), and the size experiment probes how improvement scales with width.

## How the transformer is built, and why it is faithful to Musat

Musat's formal model (his Definition 1 / Appendix A) is a single transformer
block applied iteratively to a state from x₀ = 0, with halt/emit/bit channels
that stream out one string. That formalism exists to prove a theorem about a
network that *outputs a fixed string from nothing*; it is not directly a
supervised learner. Musat says so himself (Discussion): supervised settings
"require adaptations we do not pursue here, such as loop unrolling for
feedforward and designated state coordinates for inputs and labels." This code
makes exactly those adaptations, keeping every property his theory relies on:

- **One block, looped.** `LoopedTransformer` contains a single transformer
  block whose weights are tied across `n_loops` iterations (the
  Universal-Transformer form). This preserves the structural point of the
  looped model: the entire description of the machine lives in one block's
  weights, and compute depth is decoupled from parameter count — raising
  `--n_loops` adds **zero** parameters. His "looped depth helps low-K targets"
  prediction can be probed by sweeping `--n_loops`.
- **Block structure as in his Appendix A:** causal multi-head self-attention
  and a two-layer **ReLU** feed-forward, with residual connections around both
  sub-layers. Pre-LayerNorm is added for trainability; LayerNorm's 6·d scalars
  are *not* quantized and are counted separately, so they never contaminate
  the sparsity statistics.
- **Input/output instead of emit channels.** The model reads `<bos>` + prefix
  and produces a next-token distribution at every position ("designated state
  coordinates for inputs and labels"). Halting is the emission of `#`. This is
  also precisely the framing Abudy et al. use, which is what makes their
  analytical-optimum evaluation applicable.
- **Fixed-precision weights via QAT.** Every weight *matrix* — embedding, Wq,
  Wk, Wv, Wo, both FFN matrices, and the output head — is projected onto
  Musat's grid Θ_{δ,M} = δ·Z ∩ [−M, M] on every forward pass, with a
  straight-through estimator carrying the gradient to the latent
  full-precision weight. Linear layers carry no biases, so the accounting is
  clean. Regimes (his Appendix F): `ternary` (δ = M), `int4`, `int8`,
  `bf16` (the dyadic/hardware case), and `none` (full precision, the control
  in which his theorem is provably vacuous).
- **Grid scale.** Default `--scale per_tensor` recomputes δ per tensor per
  step (ternary: δ = mean|W|, the BitNet-1.58 rule; int-b: δ = max|W|/M_b) —
  this is standard, stable QAT. For strict fidelity to a single global fixed
  (δ, M) use `--scale fixed --fixed_delta 0.05` (expect touchier optimization).
- **Weight decay is the experimental treatment.** AdamW applies decoupled
  decay only to the quantizable matrices. On a grid, a latent weight decayed
  below δ/2 quantizes to *exactly* zero — this is the mechanism by which,
  under the Lp collapse, weight decay becomes an L0 penalty. In `none` mode
  the same decay only shrinks weights, which is why we report both **hard
  zeros** (exactly 0 on the grid) and **soft zeros** (|w| < 1e-3) so the
  full-precision control is compared honestly.
- **Positional encoding is sinusoidal and parameter-free**, so it is defined
  at any test length and adds nothing to the parameter budget.

### Parameter accounting

With d = `d_model`, f = `d_ff`, V = 5 (`<pad> <bos> a b #`):

| component | count | quantizable? |
|---|---|---|
| embedding | V·d | yes |
| attention Wq,Wk,Wv,Wo | 4·d² | yes |
| feed-forward | 2·d·f | yes |
| output head | d·V | yes |
| LayerNorm ×3 | 6·d | no |

Default (d=32, h=4, f=128, loops=6): **12,800 total, 12,608 quantizable, 192
LayerNorm**. Every run prints this and stores a per-module breakdown in
`results.json → param_report`; the sparsity report then states, per module and
in total, how many of the quantizable parameters actually sit on the zero grid
point at the end ("hard_zeros"), the surviving count L0, ||θ_q||², and the
effective complexity ||θ||²log||θ||².

## Evaluation (identical currency to Abudy et al.)

Accuracy is never used. The score is |D:H| — cross-entropy in bits, with the
paper's 1e-10 probability smoothing — measured on an **exhaustive test set**
of all n = 1 … n_max_train + `--test_extra`, weighted by true probability, and
compared to the **analytically computed optimum**:
Δ% = (|D:H| − Optimal)/Optimal × 100. For Geometric(0.3) the optimum is
2.938 bits/string — the code reproduces the "2.94" in the paper's tables
exactly (`python grammar.py` prints it). We additionally report the OOD slice
(n > n_max_train only), the purest length-generalization / grokking signal.
A perfect network must both implement the 0.7/0.3 continue-vs-stop coin in the
a-run and count exactly through the deterministic b-run; failures on long
strings surface as Δ% > 0.

## Files and how to run

```
grammar.py       data + analytical optimum (run it standalone as a sanity check)
quantization.py  the Θ_{δ,M} grids + STE + sparsity counting
model.py         the looped transformer + parameter/sparsity/structure reports
evaluate.py      |D:H| vs optimum; also a CLI to score a saved checkpoint
train.py         one run (one regime, one size, one seed) → results.json
sweep.py         experiment grids: quant | grokking | size | wd | all
analyze.py       aggregates runs/ → summary.csv + plots
```

Runs use CUDA automatically when available; nothing else is needed for GPU.

**Frozen config & protocol.** The standard configuration is batch 256, lr 1e-3
(now the defaults). Runs from before the batch standardization (batch 64) are
pilot data: batch size changes optimization dynamics, so don't mix the two in
one analysis — rerun anything headed for the write-up under the frozen config.
The OOD slice is split in two: **near-OOD** (n_max+1 … n_max+10) is the
validation slice, the only out-of-range number you may use for checkpoint or
hyperparameter choices; **far-OOD** (beyond that) is report-only. Selecting on
far-OOD is test-set snooping — the peak-then-decay shape of length
generalization makes it very tempting.

```bash
pip install -r requirements.txt

# Experiment A — quantization comparison (5 regimes, fixed size)
python sweep.py quant --steps 30000 --seeds 100 101 102

# Experiment B — training length / grokking. One LONG run per regime:
# the model at step t IS the model trained for t steps, so the log-spaced
# eval history of a single run is the whole training-length curve.
python sweep.py grokking --steps 100000

# Experiment C — size sweep (d_model 8→128 × regimes)
python sweep.py size --steps 30000

# Weight decay under PER-TENSOR scaling (wd acts as plain L2)
python sweep.py wd

# THE CRUX — weight decay under a FIXED grid (wd acts as an L0/MDL penalty):
# does sparsity now show a wd dose-response, and does the late OOD blow-up
# seen at wd>0 under per-tensor scaling disappear or invert?
python sweep.py fixed --seeds 100 101 102

# Power up the pilot surprises ('wd=0 wins', 'ternary retains OOD') — 5 seeds
python sweep.py confirm --seeds 100 101 102 103 104

# Grokking hunt: small memorizable train sets (50/100/200 strings)
python sweep.py memorize --seeds 100 101

# Aggregate everything
python analyze.py            # → analysis/summary.csv + *.png

# Inspect one checkpoint later / with a harsher length probe / dump structure
python evaluate.py --ckpt runs/fixed/ternary_wd1.0_seed100/model.pt \
    --test_extra 200 --dump_structure
```

Single custom run:

```bash
python train.py --quant ternary --d_model 32 --n_loops 6 \
    --steps 30000 --weight_decay 1.0 --outdir runs/my_run
```

## Reading the results

- `analysis/grokking_*.png` — left panel: train (dashed) vs test (solid) Δ% on
  a log-x axis. A memorization phase is train Δ% going negative (beating the
  true distribution on the finite sample = overfit) while test/OOD Δ% stays
  high; a grokking jump is the later abrupt drop of the solid curve. A smooth
  co-descent of both curves is the "linear" alternative. Right panel: the
  zero-fraction trajectory — Musat's account predicts the sparsity climb and
  the generalization jump should be coupled.
- `analysis/size_sweep.png` — final test Δ% vs quantizable parameter count per
  regime: read off whether improvement with size is gradual or step-like.
- `analysis/sparsity_final.png` — final hard-zero fraction and surviving L0
  per regime: the direct test of "quantisation strengthens the bias" and of
  whether the net really becomes small.
- `analysis/effective_complexity.png` — test Δ% vs ||θ||²log||θ||² pooled over
  all runs.
- `runs/*/structure.txt` — written whenever the surviving L0 ≤ 2000: every
  non-zero quantized weight with its indices, for reading the mechanism.

## Performance notes

**Running sweeps in parallel.** At these model sizes one training process uses
~1% of a modern GPU and well under 1 GB of VRAM, so the fastest way to finish
a sweep is not a faster run but more runs at once: `--parallel N` keeps up to
N training processes alive (3–4 is a good number on a gaming laptop; beyond
that the CPU cores driving kernel launches saturate). Each run's live output
goes to `<outdir>/train.log` and the sweep prints compact `[start]`/`[done]`
lines; a failed run is reported with its exit code and log path without
stopping the others; Ctrl-C terminates all children. `--resume` skips runs
whose `results.json` already exists, so an interrupted sweep can be relaunched
safely — it's off by default so that re-running after a config change
regenerates results instead of silently keeping stale ones. Example:

```bash
python sweep.py fixed --seeds 100 101 102 --parallel 3 --resume
```

For long sweeps on a laptop: plug in and use performance mode (this workload
loads the CPU more than the GPU, so thermals are rarely the issue, but
battery-mode clocks are).

The training path is optimized for the many-tiny-steps regime this model
lives in (12.6k params means the GPU is idle most of the time; overhead, not
arithmetic, sets the pace):

- The whole training set is padded and moved to the device **once** at
  startup; each step indexes rows and trims to the batch's own max length
  (pads sit after the real tokens under causal attention, so this is exact).
  No per-step Python collation, no per-step host-to-device copy.
- Each weight matrix is quantized **once per forward** and the cached STE
  tensor is reused across all `n_loops` iterations (previously it was
  recomputed on every layer call — ~38 quantize bundles per step instead
  of 8). Numerically equivalent; a 300-step CPU check reproduces the old
  trajectory exactly.
- `--compile` wraps the model in `torch.compile` (kernel fusion). Optional:
  it needs Triton, which on Windows requires the `triton-windows` package
  and may still fail — the code falls back to eager with a warning.
- Positional encodings are cached per (length, device).

Even so, at d=32 a GPU cannot be saturated; expect the GPU to pull far ahead
of CPU only at larger `--d_model` and `--batch_size` (e.g. d=128, batch 256+,
which is nearly free on GPU). Check `nvidia-smi` during a run: low utilization
means overhead-bound, which is expected at small sizes.

New flags: `--structure_max_l0 N` controls when `structure.txt` is written at
the end of training (default 2000), and `evaluate.py --ckpt ... --dump_structure`
writes the non-zero weight listing for **any existing checkpoint**, regardless
of its L0 — useful for inspecting already-trained sparse runs.

## Practical notes and caveats

- Grokking is hyperparameter-sensitive; it typically needs a *finite* train
  set (default 500 strings, as in Abudy et al.), substantial weight decay
  (default 1.0), and patience. If nothing jumps, sweep `--weight_decay`
  (0.1–3.0) and `--train_size` (200–1000) — smaller train sets make
  memorization easier and the transition sharper.
- Δ% < 0 on *test* means overfitting the truncated test set (paper, §2.6):
  treat near-zero |Δ%| as success, clearly positive Δ% as failure.
- The per-tensor scale is the one deliberate departure from a literally fixed
  (δ, M); it keeps a finite alphabet at every instant but lets it drift.
  `--scale fixed` removes the departure at some cost in trainability — worth
  reporting both in any write-up.
- Seeds matter for phase-transition claims: run ≥3 seeds (`--seeds 100 101 102`)
  before calling something a jump.
- Extending to a^n b^n c^n, Dyck-1, etc.: add a generator + analytical optimum
  with the same `TrainSet`/`TestSet` interface as `grammar.py`; everything
  downstream is language-agnostic.
