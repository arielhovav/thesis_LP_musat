"""
quantization.py
===============
Fixed-precision weight regimes from Musat (2026),
"Neural Weight Norm = Kolmogorov Complexity".

Musat's parameter space (his Definition 2) is
    Theta_{delta,M} = { theta in (delta*Z intersect [-M, M])^d },
i.e. every weight lives on a fixed grid of step delta, bounded by M.
His three "principal cases" (Appendix F):

    ternary : delta = M              -> weights in {-M, 0, +M}
    int-b   : signed b-bit integers  -> 2^(b-1)-1 levels per sign
    dyadic  : b-bit fixed point      -> what fp16/bf16 hardware approximates

We train with quantization-aware training (QAT): each forward pass quantizes
the latent full-precision weight onto the grid, and the straight-through
estimator (STE) passes the gradient to the latent weight unchanged:

    w_q = w + (Q(w) - w).detach()

Weight decay is applied (by the optimizer) to the *latent* weights. A latent
weight that decays below delta/2 quantizes to exactly 0 -- this is the
mechanism by which, per Musat's Lp-collapse (his Eq. 1), weight decay in fixed
precision becomes an L0 (parameter-count) penalty and should produce genuinely
sparse networks. In full precision ('none') weight decay merely shrinks
weights; they essentially never hit exact zero. Comparing the sparsity that
emerges across regimes is exactly his "quantisation strengthens the bias"
prediction (Section 5.3).

Scale (the grid step delta):
  * scale='fixed'      strictly Musat: one global delta you choose, shared by
                       all layers, constant through training.
  * scale='per_tensor' practical QAT: delta is recomputed per weight tensor
                       each step. Ternary uses the BitNet-1.58 rule
                       delta = mean(|W|); int-b uses delta = max(|W|) / M_b.
                       This keeps training stable at any width; it makes the
                       grid adaptive, which slightly departs from the theorem's
                       fixed (delta, M) but preserves the finite-alphabet
                       structure the theory needs at any single point in time.
Default is 'per_tensor' (stable). Use --scale fixed --delta ... to be strict.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MODES = ("none", "bf16", "int16", "int12", "int8", "int6", "int4", "ternary")
INT_LEVELS = {"ternary": 1, "int4": 7, "int6": 31, "int8": 127, "int12": 2047, "int16": 32767}


@dataclass
class QuantConfig:
    mode: str = "none"            # one of MODES
    scale: str = "per_tensor"     # 'per_tensor' | 'fixed'
    fixed_delta: float = 0.05     # grid step used when scale == 'fixed'

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")

    @property
    def n_levels_per_sign(self) -> int | None:
        """M/delta: number of positive grid levels. None for none/bf16."""
        return INT_LEVELS.get(self.mode)


def _grid_quantize(w: torch.Tensor, cfg: QuantConfig) -> torch.Tensor:
    """Project w onto the grid delta*Z intersect [-M, M] (no gradient logic)."""
    m = cfg.n_levels_per_sign  # M / delta
    if cfg.scale == "fixed":
        delta = cfg.fixed_delta
    elif cfg.mode == "ternary":
        # BitNet b1.58 rule: delta = mean(|W|). Zero iff |w| < delta/2.
        delta = w.abs().mean().clamp(min=1e-8)
    else:
        # int-b: fit the full dynamic range of the tensor onto the grid.
        delta = (w.abs().max() / m).clamp(min=1e-8)
    q = torch.clamp(torch.round(w / delta), -m, m)
    return q * delta


def quantize(w: torch.Tensor, cfg: QuantConfig) -> torch.Tensor:
    """Quantized value of w under cfg (used for inspection / sparsity counts)."""
    if cfg.mode == "none":
        return w
    if cfg.mode == "bf16":
        return w.to(torch.bfloat16).to(w.dtype)
    return _grid_quantize(w, cfg)


def quantize_ste(w: torch.Tensor, cfg: QuantConfig) -> torch.Tensor:
    """Quantize with a straight-through estimator for training."""
    if cfg.mode == "none":
        return w
    q = quantize(w, cfg)
    return w + (q - w).detach()


@torch.no_grad()
def tensor_sparsity(w: torch.Tensor, cfg: QuantConfig, fp_zero_tol: float = 1e-3):
    """(n_zero, n_total) of the *quantized* tensor.

    For grid modes, a zero is exact (the weight sits on the 0 grid point).
    For 'none'/'bf16' exact zeros essentially never occur, so we also count
    |w| < fp_zero_tol as "effectively zero" -- reported separately as soft
    zeros so the comparison with grid modes stays honest.
    """
    q = quantize(w, cfg)
    hard = int((q == 0).sum().item())
    soft = int((q.abs() < fp_zero_tol).sum().item())
    return hard, soft, q.numel()
