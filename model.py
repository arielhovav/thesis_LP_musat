"""
model.py
========
A looped transformer in the spirit of Musat (2026), Appendix A, adapted to
supervised next-token prediction on formal languages.

Correspondence with Musat's model, and where we adapt
-----------------------------------------------------
Musat's Definition 1 / Appendix A model is: ONE transformer block applied
iteratively to a state, with designated halt/emit/bit channels, run from the
zero state, emitting a string. That formalism is built for his *theorem*
(a network that outputs one fixed string from nothing). He states explicitly
in his Discussion that "standard supervised-learning settings ... require
adaptations we do not pursue here, such as loop unrolling for feedforward and
designated state coordinates for inputs and labels."

We take the standard such adaptation (the Universal-Transformer form):

  * ONE transformer block with weight tying, applied n_loops times to the
    token sequence. This preserves the property Musat's theory actually uses:
    the entire description length of the machine lives in a single block's
    weights, and compute depth (number of loop iterations) is decoupled from
    parameter count. Increasing n_loops adds ZERO parameters.
  * Instead of halt/emit/bit channels streaming one string, the model reads
    <bos> + prefix and emits a next-token distribution at every position
    (this is the "designated state coordinates for inputs and labels"
    adaptation). Halting is trivial in this framing: the string ends when the
    model emits '#'.
  * ReLU feed-forward, residual connections around both sub-layers, and
    causal multi-head self-attention -- exactly the block structure of his
    Appendix A (pre-LayerNorm added for trainability; LN params are NOT
    quantized and are counted separately).
  * All weight MATRICES (embedding, Wq, Wk, Wv, Wo, FFN in/out, output head)
    pass through the fixed-precision quantizer of quantization.py on every
    forward pass. These are the "quantizable" parameters -- the ones weight
    decay can drive to exact zero on the grid. Linear layers carry no biases,
    so the parameter accounting is clean: everything except LayerNorm's
    gain/bias vectors is quantizable.
  * Sinusoidal (parameter-free) positional encoding, so positions add no
    parameters and are defined for any test length.

Parameter count (d = d_model, f = d_ff, V = vocab size, h = heads):
    embedding      V*d
    attention      4*d*d          (Wq, Wk, Wv, Wo)
    feed-forward   2*d*f
    output head    d*V            (independent of the embedding by default)
    LayerNorm      3 * 2d = 6d    (2 in the block + 1 final; NOT quantizable)
    total          2Vd + 4d^2 + 2df + 6d,   quantizable = 2Vd + 4d^2 + 2df
n_loops does not appear: looping is free in parameters.

Gridded, counted biases (--bias; default 'none' keeps every existing run intact)
------------------------------------------------------------------------------
Without biases the only constant available to a NoPE softmax network is the
<bos> share 1/t of the prefix mean, which (a) drifts with position, so the coin
P(b | a^k) cannot be stationary, and (b) has to be amplified by replicating
residual dimensions -- the largest gain in every circuit anatomized so far.
Musat's Appendix A block has biases, Abudy's RNNs have biases, and a bias that
lives on the same (delta, M) grid, passes through the same STE, is decayed by
the same weight decay and is priced by the same natural code is inside both
theories. --bias ffh adds a quantized bias vector to ff_in (d_ff), ff_out (d)
and head (V); --bias ffh_vo also to Wv and Wo (2d). (Wq/Wk get none: a key
bias cancels in the softmax and a query bias is a constant score offset.)
Biases are yielded by quantizable_named_weights() as "<module>.bias", so L0,
|H|, sparsity, the GA genome and every census count them automatically; the
quantizable slot count N grows by d_ff + d + V (+ 2d), i.e. log2 N by ~0.02.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantization import QuantConfig, quantize, quantize_ste, tensor_sparsity


# -----------------------------------------------------------------------------
# Quantized primitives
# -----------------------------------------------------------------------------
class QuantLinear(nn.Linear):
    """Linear layer whose weight is projected onto the fixed-precision grid.

    Performance note: the looped model calls each layer n_loops times per
    forward, but the latent weight only changes between optimizer steps, so
    quantizing on every call wastes ~n_loops x the kernel launches. The model
    fills `_qw` once at the start of each top-level forward (see
    LoopedTransformer.forward) and clears it at the end; gradients through the
    cached STE tensor accumulate across the loop iterations exactly as before.
    """

    def __init__(self, in_f: int, out_f: int, qcfg: QuantConfig, bias: bool = False):
        super().__init__(in_f, out_f, bias=bias)
        self.qcfg = qcfg
        self._qw = None            # per-forward cache (not a parameter/buffer)
        self._qb = None            # same, for the (optional) gridded bias
        if bias:
            with torch.no_grad():
                self.bias.zero_()  # start exactly on the grid's zero point

    def effective_weight(self):
        return self._qw if self._qw is not None else quantize_ste(self.weight, self.qcfg)

    def effective_bias(self):
        if self.bias is None:
            return None
        return self._qb if self._qb is not None else quantize_ste(self.bias, self.qcfg)

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.effective_bias())


class QuantEmbedding(nn.Embedding):
    def __init__(self, n: int, d: int, qcfg: QuantConfig):
        super().__init__(n, d)
        self.qcfg = qcfg
        self._qw = None

    def effective_weight(self):
        return self._qw if self._qw is not None else quantize_ste(self.weight, self.qcfg)

    def forward(self, x):
        return F.embedding(x, self.effective_weight())


_PE_CACHE: dict = {}


def sinusoidal_pe(L: int, d: int, device, dtype) -> torch.Tensor:
    """Parameter-free positional encoding, cached per (L, d, device, dtype)."""
    key = (L, d, str(device), dtype)
    pe = _PE_CACHE.get(key)
    if pe is not None:
        return pe
    pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(0, d, 2, device=device, dtype=dtype)
    div = torch.exp(-math.log(10000.0) * i / d)
    pe = torch.zeros(L, d, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: (d + 1) // 2])
    _PE_CACHE[key] = pe
    return pe


# -----------------------------------------------------------------------------
# The single (looped) block
# -----------------------------------------------------------------------------
ATTN_MODES = ("softmax", "linear", "sum")
NORM_MODES = ("ln", "none", "ln_noaffine")
BIAS_MODES = ("none", "ffh", "ffh_vo")   # see module docstring


def bias_slots(bias: str) -> set:
    """Which QuantLinear modules carry a gridded bias under --bias."""
    assert bias in BIAS_MODES, f"bias must be one of {BIAS_MODES}, got {bias!r}"
    if bias == "none":
        return set()
    s = {"ff_in", "ff_out", "head"}
    if bias == "ffh_vo":
        s |= {"wv", "wo"}
    return s


def make_norm(d: int, norm: str):
    """--norm ln   pre-LayerNorm (our trainability addition; 6d scalars, never counted)
       --norm none no normalization anywhere: Musat's Appendix A block as written.
    With --attn sum, LayerNorm is what turns the accumulated count back into a
    fraction (it divides by ||x||, which grows with position); without it the
    FFN reads the raw integer count and a ReLU is a threshold at integer
    resolution. The <bos> token, present once in every prefix, accumulates to a
    constant that serves as the bias the model otherwise lacks.
       --norm ln_noaffine  LayerNorm WITHOUT gain/bias: normalization (trainability)
                   with zero parameters, so no unquantized, uncounted magnitude
                   enters the network. The honest Musat-compatible LN.
    """
    assert norm in NORM_MODES, f"norm must be one of {NORM_MODES}, got {norm!r}"
    if norm == "ln":
        return nn.LayerNorm(d)
    if norm == "ln_noaffine":
        return nn.LayerNorm(d, elementwise_affine=False)
    return nn.Identity()


class CausalSelfAttention(nn.Module):
    """Causal self-attention in one of three forms (--attn):

    softmax  standard: out_t = sum_i softmax_i(q_t.k_i/sqrt(dh)) v_i. With
             Wq = Wk = 0 this is the MEAN of the prefix values, so a count is
             only available as a fraction (n_a - n_b)/t that shrinks with t.
    linear   unnormalized dot-product: out_t = sum_{i<=t} (q_t.k_i/sqrt(dh)) v_i,
             no softmax and no denominator (a linear RNN, Katharopoulos et al.
             2020). Can learn to become 'sum'.
    sum      the pure causal cumulative sum: out_t = sum_{i<=t} v_i. Wq and Wk
             are not used (they receive no gradient, decay to zero and leave
             |H|). At loop 0 every v_i depends only on token i, so one
             residual dimension holds n_a - n_b EXACTLY at every position: the
             transformer becomes an integer accumulator, Abudy's counting
             unit, and an exact counter costs O(1) ternary weights.
    Note: 'linear' and 'sum' need activations that grow with n; with
    --act_bits > 0 the count saturates at act_max.
    """

    def __init__(self, d: int, n_heads: int, qcfg: QuantConfig, attn: str = "softmax",
                 bias: str = "none"):
        super().__init__()
        assert d % n_heads == 0, "d_model must be divisible by n_heads"
        assert attn in ATTN_MODES, f"attn must be one of {ATTN_MODES}, got {attn!r}"
        self.h = n_heads
        self.attn = attn
        bs = bias_slots(bias)
        self.wq = QuantLinear(d, d, qcfg)
        self.wk = QuantLinear(d, d, qcfg)
        self.wv = QuantLinear(d, d, qcfg, bias="wv" in bs)
        self.wo = QuantLinear(d, d, qcfg, bias="wo" in bs)

    def forward(self, x):
        B, L, D = x.shape
        dh = D // self.h

        def split(t):
            return t.view(B, L, self.h, dh).transpose(1, 2)  # B,h,L,dh

        v = split(self.wv(x))
        if self.attn == "sum":
            out = torch.cumsum(v, dim=2)                       # causal prefix sum
        else:
            q, k = split(self.wq(x)), split(self.wk(x))
            if self.attn == "softmax":
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:                                              # linear, unnormalized
                scores = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
                causal = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
                scores = scores.masked_fill(~causal, 0.0)
                out = scores @ v
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.wo(out)


def normalize_act_max(v) -> tuple:
    """v is a float (broadcast to all 6 points) or a sequence of 6 floats,
    ordered (emb+pe, ln1_out, post_attn_resid, ln2_out, post_ffn_resid,
    final_ln) -- matching measure_act_range.py's print order. Returns a
    6-tuple. Different quantization points can need very different ranges
    (a network's gain may live in the residual stream OR in the final LN,
    depending on the solution) -- one shared range/step for all 6 points
    confounds bit-depth with clipping at whichever point needs the other
    point's range. Always calibrate per-point via measure_act_range.py."""
    if isinstance(v, (int, float)):
        return (float(v),) * 6
    v = tuple(float(x) for x in v)
    assert len(v) == 6, f"act_max must be a float or 6 values, got {v}"
    return v


def fake_quant_act(x, bits: int, max_abs: float):
    """Musat-style activation quantization: round-and-clip module-boundary
    values to a fixed grid of 2^bits levels on [-max_abs, max_abs] (step
    delta = 2*max_abs / 2^bits), straight-through for training. bits <= 0
    disables (fp32 activations, the default; the original regime). Scope:
    applied at emb(+PE) output, LN1/LN2 outputs, both residual sums, and the
    final LN output; attention-internal softmax arithmetic stays fp32.
    NOTE for the thesis: activations are runtime state, not hypothesis
    description - |H| accounting is unchanged by this flag; what changes is
    fidelity to Musat's finite-alphabet machine class, in which the analog
    1/t prefix-mean signal has a hard resolution ceiling."""
    if bits <= 0:
        return x
    delta = (2.0 * max_abs) / (2 ** bits)
    xq = torch.clamp(torch.round(x / delta) * delta, -max_abs, max_abs)
    return x + (xq - x).detach()


class Block(nn.Module):
    """Pre-LN transformer block: x + Attn(LN(x)); x + FFN(LN(x)), ReLU FFN."""

    def __init__(self, d: int, n_heads: int, d_ff: int, qcfg: QuantConfig,
                 attn: str = "softmax", norm: str = "ln", bias: str = "none"):
        super().__init__()
        self.act_bits = 0          # set by LoopedTransformer
        self.act_max = 8.0
        bs = bias_slots(bias)
        self.ln1 = make_norm(d, norm)
        self.attn = CausalSelfAttention(d, n_heads, qcfg, attn=attn, bias=bias)
        self.ln2 = make_norm(d, norm)
        self.ff_in = QuantLinear(d, d_ff, qcfg, bias="ff_in" in bs)
        self.ff_out = QuantLinear(d_ff, d, qcfg, bias="ff_out" in bs)

    def forward(self, x, use_attn: bool = True):
        m_ln1, m_attn_resid, m_ln2, m_ffn_resid = self.act_max
        if use_attn:
            x = fake_quant_act(
                x + self.attn(fake_quant_act(self.ln1(x), self.act_bits, m_ln1)),
                self.act_bits, m_attn_resid)
        x = fake_quant_act(
            x + self.ff_out(F.relu(self.ff_in(
                fake_quant_act(self.ln2(x), self.act_bits, m_ln2)))),
            self.act_bits, m_ffn_resid)
        return x


class LoopedTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        n_heads: int = 4,
        d_ff: int = 128,
        n_loops: int = 6,
        qcfg: QuantConfig | None = None,
        pos: str = "sin",
        act_bits: int = 0,
        act_max=8.0,   # float (broadcast) or 6-tuple, see normalize_act_max
        attn: str = "softmax",   # see CausalSelfAttention
        norm: str = "ln",        # see make_norm
        attn_loops: int | None = None,   # attention only on the first k loops
        bias: str = "none",              # gridded, counted biases; see docstring
    ):
        super().__init__()
        assert pos in ("sin", "nope"), f"pos must be 'sin' or 'nope', got {pos!r}"
        self.attn_mode = attn
        self.norm_mode = norm
        self.bias_mode = bias
        self.qcfg = qcfg or QuantConfig()
        self.d = d_model
        self.n_loops = n_loops
        # attn_loops: run the attention sub-layer only on the first k loop
        # iterations; later iterations are FFN-only (same tied weights, so
        # still zero extra parameters). With --attn sum this means ONE
        # accumulation followed by k..n_loops-1 rounds of FFN amplification,
        # instead of iterated cumulative sums (t^k growth). None = every loop.
        self.attn_loops = n_loops if attn_loops is None else attn_loops
        self.pos = pos   # 'sin': parameter-free sinusoidal PE (Musat-faithful
                         # default); 'nope': no positional encoding (relies on
                         # causal masking; length-generalization ablation).
                         # Both add ZERO parameters, so |H| accounting is
                         # unchanged either way.
        self.act_bits = act_bits
        self.act_max = normalize_act_max(act_max)   # 6-tuple, see above
        self.emb = QuantEmbedding(vocab_size, d_model, self.qcfg)
        self.block = Block(d_model, n_heads, d_ff, self.qcfg, attn=attn, norm=norm,
                           bias=bias)
        self.block.act_bits = act_bits
        self.block.act_max = self.act_max[1:5]      # (ln1, attn_resid, ln2, ffn_resid)
        self.ln_f = make_norm(d_model, norm)
        self.head = QuantLinear(d_model, vocab_size, self.qcfg,
                                bias="head" in bias_slots(bias))
        self._qmods = [m for m in self.modules()
                       if isinstance(m, (QuantLinear, QuantEmbedding))]

    @classmethod
    def from_config(cls, cfg: dict, vocab_size: int, qcfg: QuantConfig, **overrides):
        """Build the architecture a checkpoint/results config describes. Every
        loader in the repo goes through this so a new architecture flag (like
        --bias) needs changing in exactly one place. Keys absent from old
        configs resolve to the value those runs were trained with."""
        kw = dict(d_model=cfg["d_model"], n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
                  n_loops=cfg["n_loops"], qcfg=qcfg, pos=cfg.get("pos", "sin"),
                  act_bits=cfg.get("act_bits", 0), act_max=cfg.get("act_max", 8.0),
                  attn=cfg.get("attn", "softmax"), norm=cfg.get("norm", "ln"),
                  attn_loops=cfg.get("attn_loops"), bias=cfg.get("bias", "none"))
        kw.update(overrides)
        return cls(vocab_size, **kw)

    def forward(self, tokens):  # tokens: (B, L) long
        B, L = tokens.shape
        # Quantize every weight ONCE per forward; the loop below reuses the
        # cached STE tensors (numerically identical, ~n_loops x fewer kernels).
        for m in self._qmods:
            m._qw = quantize_ste(m.weight, m.qcfg)
            if getattr(m, "bias", None) is not None:
                m._qb = quantize_ste(m.bias, m.qcfg)
        try:
            x = self.emb(tokens) * math.sqrt(self.d)
            if self.pos == "sin":
                x = x + sinusoidal_pe(L, self.d, x.device, x.dtype)
            x = fake_quant_act(x, self.act_bits, self.act_max[0])   # emb+pe
            for i in range(self.n_loops):   # weight-tied loop: zero extra params
                x = self.block(x, use_attn=i < self.attn_loops)
            return self.head(fake_quant_act(self.ln_f(x),
                                            self.act_bits, self.act_max[5]))  # final_ln
        finally:
            for m in self._qmods:           # never reuse across optimizer steps
                m._qw = None
                m._qb = None

    # ---------------- parameter accounting ----------------
    def quantizable_named_weights(self):
        """The weight matrices that live on the fixed-precision grid, i.e. the
        parameters that weight decay can drive to exact zero."""
        for name, mod in self.named_modules():
            if isinstance(mod, (QuantLinear, QuantEmbedding)):
                yield name, mod.weight
                if getattr(mod, "bias", None) is not None:
                    yield name + ".bias", mod.bias

    def n_bias_slots(self) -> int:
        return sum(w.numel() for n, w in self.quantizable_named_weights()
                   if n.endswith(".bias"))

    def param_report(self) -> dict:
        quant = {n: w.numel() for n, w in self.quantizable_named_weights()}
        total = sum(p.numel() for p in self.parameters())
        n_quant = sum(quant.values())
        return {
            "total_params": total,
            "quantizable_params": n_quant,
            "non_quantizable_params": total - n_quant,   # LayerNorm gains/biases
            "quantizable_bias_slots": self.n_bias_slots(),  # part of quantizable_params
            "bias": self.bias_mode,
            "per_module_quantizable": quant,
            "n_loops (adds 0 params)": self.n_loops,
            "attn_loops": self.attn_loops,
        }

    @torch.no_grad()
    def sparsity_report(self, fp_zero_tol: float = 1e-3) -> dict:
        per_mod, hard_z = {}, 0
        soft_z, tot = 0, 0
        l2sq_q = 0.0
        for name, w in self.quantizable_named_weights():
            h, s, n = tensor_sparsity(w, self.qcfg, fp_zero_tol)
            per_mod[name] = {"zeros": h, "numel": n, "frac": h / n}
            hard_z += h
            soft_z += s
            tot += n
            l2sq_q += float((quantize(w, self.qcfg) ** 2).sum().item())
        l0 = tot - hard_z
        eff = l2sq_q * math.log(max(l2sq_q, 1.0 + 1e-9))  # ||th||^2 log ||th||^2
        return {
            "quantizable_total": tot,
            "hard_zeros": hard_z,
            "hard_zero_frac": hard_z / tot,
            "soft_zeros(|w|<tol)": soft_z,
            "soft_zero_frac": soft_z / tot,
            "L0_nonzero_count": l0,
            "L2sq_of_quantized": l2sq_q,
            "effective_complexity_L2sq_logL2sq": eff,
            "per_module": per_mod,
        }

    @torch.no_grad()
    def dump_nonzero_structure(self, max_entries_per_tensor: int = 400) -> str:
        """Human-readable listing of the surviving (non-zero) quantized weights.
        Useful once the net is sparse enough to read (Musat's interpretability
        remark; cf. the golden-network diagrams in Abudy et al.)."""
        lines = []
        for name, w in self.quantizable_named_weights():
            q = quantize(w, self.qcfg)
            nz = q.nonzero(as_tuple=False)
            lines.append(f"== {name}: {nz.shape[0]}/{q.numel()} non-zero ==")
            for idx in nz[:max_entries_per_tensor]:
                ij = tuple(int(i) for i in idx)
                lines.append(f"   {ij}: {q[ij].item():+.4f}")
            if nz.shape[0] > max_entries_per_tensor:
                lines.append(f"   ... ({nz.shape[0] - max_entries_per_tensor} more)")
        return "\n".join(lines)