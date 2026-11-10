"""
minsky_transformer.py
=====================
A universal computer in Musat's Appendix-A class, built by hand, so that the
constant c_U of his Theorem 4 (N(s) <= K(s) + c_U) has a number.

Substrate. Instead of a Turing machine (whose tape needs relative-position
reads this architecture does not have), a Minsky register machine, which is
Turing-complete and whose primitives are the ones every a^n b^n counter on the
explainer page already uses:

  register value   = prefix count over the tokens emitted so far (causal SUM attention)
  zero test        = the clipped comparator  clip(8 * value, 8) in {0, 8}
  program          = a ReLU lookup table (own-token one-hot, zero flag) -> next token
  tape             = the context: the machine emits its own execution trace,
                     one token per executed instruction, exactly Musat's
                     "the context length is the tape" streaming model.

Tokens. For an increment instruction q: one token (q,'P'). For a test-and-
decrement instruction q: two tokens, (q,'M') "decremented and continued" and
(q,'Z') "was zero, jumped". HALT repeats itself. Load tokens L_r put the input
into the registers as a unary prefix (Musat's program bits deposited into the
input region: |p| weights, counted separately from c_U).

Grid. Ternary, delta = M = 1, activations rounded to step 1 and clipped at +-8,
no LayerNorm, no positional encoding, no biases. Every non-zero weight is +-1,
so ||theta||_2^2 = ||theta||_0 = c'_U exactly.

Weights per piece (all +-1):
  register r      : W_v 1 + W_o 1, plus the embedding effects below
  token (q,'P')   : own-dim 1 + register effect 1
  token (q,'M')   : own-dim 1 + register effect 1
  token (q,'Z')   : own-dim 1
  load L_r        : own-dim 1 + register effect 1
  BOS             : own-dim 1 + constant 1 (+ W_v 1 + W_o 1 for the constant)
  lookup j -> k   : successor increment/HALT: 2 (one unit);
                    successor ZM: 7 (two units: ReLU(own + s_r - const), ReLU(own - s_r))
  head            : 1 per emittable token

Usage:
  python minsky_transformer.py            # builds U22, verifies against an interpreter, prints c_U
"""
from __future__ import annotations

import math
import random
import sys

import torch
import torch.nn.functional as F

from model import LoopedTransformer
from quantization import QuantConfig

# Korec's strongly universal U22 (as listed by Ivanov, Pelz & Verlan 2013, Fig. 1):
# label -> ('P', register, next) | ('ZM', register, next_if_decremented, next_if_zero) | ('STOP',)
U22 = {
    4: ('ZM', 5, 6, 7),   1: ('ZM', 1, 3, 6),    3: ('P', 7, 1),       6: ('P', 6, 4),
    7: ('ZM', 6, 9, 4),   9: ('P', 5, 10),       10: ('ZM', 7, 12, 13), 13: ('ZM', 6, 33, 1),
    12: ('P', 1, 7),      16: ('ZM', 5, 18, 23), 33: ('P', 6, 14),     14: ('ZM', 4, 1, 16),
    18: ('ZM', 5, 20, 27), 20: ('ZM', 5, 22, 30), 22: ('P', 4, 16),     23: ('ZM', 2, 32, 25),
    25: ('ZM', 0, 1, 32), 27: ('ZM', 3, 32, 1),  29: ('P', 0, 1),      30: ('P', 2, 31),
    31: ('P', 3, 32),     32: ('ZM', 4, 1, 'f'), 'f': ('STOP',),
}
U22_START, U22_REGS = 1, 8


# ------------------------------------------------------------ reference ------
def run_machine(prog, start, regs, max_steps):
    """Execute the register machine; return the trace of (label, outcome) tokens."""
    regs = list(regs)
    q, trace = start, []
    for _ in range(max_steps):
        ins = prog[q]
        if ins[0] == 'STOP':
            trace.append(('HALT',)); continue
        if ins[0] == 'P':
            regs[ins[1]] += 1; trace.append((q, 'P')); q = ins[2]
        else:
            r = ins[1]
            if regs[r] > 0:
                regs[r] -= 1; trace.append((q, 'M')); q = ins[2]
            else:
                trace.append((q, 'Z')); q = ins[3]
    return trace, regs


# ------------------------------------------------------------ constructor ----
class MinskyTransformer:
    def __init__(self, prog, start, n_regs, d_ff=256):
        self.prog, self.start, self.R = prog, start, n_regs
        # ---- vocabulary
        toks = ['PAD', 'BOS'] + [f'L{r}' for r in range(n_regs)]
        for q, ins in prog.items():
            if ins[0] == 'P': toks.append((q, 'P'))
            elif ins[0] == 'ZM': toks += [(q, 'M'), (q, 'Z')]
            else: toks.append(('HALT',))
        self.toks = toks
        self.tid = {t: i for i, t in enumerate(toks)}
        emittable = [t for t in toks if t not in ('PAD', 'BOS') and not (isinstance(t, str) and t.startswith('L'))]
        own_tokens = [t for t in toks if t != 'PAD']
        # ---- residual dims
        d = 0
        self.reg = {r: d + r for r in range(n_regs)}; d += n_regs          # register effect dims
        self.s = {r: d + r for r in range(n_regs)}; d += n_regs            # register sums
        self.cdim, self.sconst = d, d + 1; d += 2                           # constant source and its sum
        self.own = {t: d + i for i, t in enumerate(own_tokens)}; d += len(own_tokens)
        self.out = {t: d + i for i, t in enumerate(emittable)}; d += len(emittable)
        # the embedding is scaled by sqrt(d) and clipped at 8: every one-hot, effect and
        # constant must saturate to the same 8, so d >= 64 (padding dims cost no weights)
        self.d = d = max(d, 64)
        self.model = LoopedTransformer(len(toks), d, 1, d_ff, 2, QuantConfig("ternary", "fixed", 1.0),
                                       pos="nope", norm="none", attn="sum", attn_loops=1, bias="none",
                                       act_bits=4, act_max=8.0)
        m = self.model
        with torch.no_grad():
            for _, w in m.quantizable_named_weights(): w.zero_()
            E, Wv, Wo = m.emb.weight, m.block.attn.wv.weight, m.block.attn.wo.weight
            Win, Wout, H = m.block.ff_in.weight, m.block.ff_out.weight, m.head.weight
            # registers: sum of the effect dim, read back into s_r
            for r in range(n_regs):
                Wv[self.reg[r], self.reg[r]] = 1; Wo[self.s[r], self.reg[r]] = 1
            # constant: BOS writes 1 on cdim; its prefix sum is 8 everywhere
            E[self.tid['BOS'], self.cdim] = 1; Wv[self.cdim, self.cdim] = 1; Wo[self.sconst, self.cdim] = 1
            # embeddings: own-token one-hot + register effect
            for t in own_tokens:
                E[self.tid[t], self.own[t]] = 1
                if isinstance(t, str):
                    if t.startswith('L'):
                        E[self.tid[t], self.reg[int(t[1:])]] = 1
                elif t[0] != 'HALT':
                    ins = prog[t[0]]
                    if t[1] == 'P': E[self.tid[t], self.reg[ins[1]]] = 1
                    if t[1] == 'M': E[self.tid[t], self.reg[ins[1]]] = -1
            # lookup: for every token with a successor instruction k, units that emit k's token
            u = 0

            def successor(t):
                if isinstance(t, str): return start                       # BOS and loads start the program
                if t[0] == 'HALT': return 'f'
                ins = prog[t[0]]
                return ins[2] if t[1] in ('P', 'M') else ins[3]

            for t in own_tokens:
                k = successor(t)
                kins = prog[k]
                if kins[0] in ('P', 'STOP'):
                    ktok = (k, 'P') if kins[0] == 'P' else ('HALT',)
                    Win[u, self.own[t]] = 1; Wout[self.out[ktok], u] = 1; u += 1
                else:
                    r = kins[1]
                    Win[u, self.own[t]] = 1; Win[u, self.s[r]] = 1; Win[u, self.sconst] = -1
                    Wout[self.out[(k, 'M')], u] = 1; u += 1
                    Win[u, self.own[t]] = 1; Win[u, self.s[r]] = -1
                    Wout[self.out[(k, 'Z')], u] = 1; u += 1
            assert u <= d_ff, f"need d_ff >= {u}"
            self.n_units = u
            for t in emittable:
                H[self.tid[t], self.out[t]] = 1
        m.eval()

    # ---- run the transformer as the machine (greedy, teacher-free after the load prefix)
    @torch.no_grad()
    def run(self, regs, max_steps):
        seq = [self.tid['BOS']] + [self.tid[f'L{r}'] for r in range(self.R) for _ in range(regs[r])]
        trace, worst = [], 0.0
        for _ in range(max_steps):
            logits = self.model(torch.tensor([seq]))[0, -1].float()
            p = F.softmax(logits, -1)
            nxt = int(logits.argmax())
            worst = max(worst, float(-torch.log2(p[nxt])))
            trace.append(self.toks[nxt]); seq.append(nxt)
        return trace, worst

    def count(self):
        n = {name: int((w != 0).sum()) for name, w in self.model.quantizable_named_weights()}
        return n, sum(n.values())


def main():
    mt = MinskyTransformer(U22, U22_START, U22_REGS)
    per, total = mt.count()
    print(f"U22 as a ternary looped transformer: d_model={mt.d}, vocab={len(mt.toks)}, lookup units={mt.n_units}")
    print("non-zero weights per tensor:", {k: v for k, v in per.items() if v})
    print(f"c'_U = {total} non-zero +-1 weights  (||theta||_2^2 = ||theta||_0 = {total})")
    N = mt.model.param_report()["quantizable_params"]
    print(f"natural-code |H| of the machine = {total} x (log2 {N} + 1) = {total * (math.log2(N) + 1):.0f} bits")
    # verification: random register contents, step-by-step against the interpreter
    rng = random.Random(0); checked, mismatches, worst_all = 0, 0, 0.0
    for trial in range(12):
        regs = [rng.randint(0, 6) for _ in range(U22_REGS)]
        steps = 60 if trial < 10 else 200
        ref, _ = run_machine(U22, U22_START, regs, steps)
        got, worst = mt.run(regs, steps)
        worst_all = max(worst_all, worst)
        checked += steps
        if got != ref:
            mismatches += 1
            k = next(i for i, (a, b) in enumerate(zip(got, ref)) if a != b)
            print(f"  MISMATCH trial {trial} regs={regs} at step {k}: got {got[k]} expected {ref[k]}")
    print(f"verified {checked} machine steps over 12 random register loads: {mismatches} mismatches; "
          f"worst per-token cost {worst_all:.4f} bits")
    # a tiny program for the reader: R0 <- R1 + R2
    ADD = {1: ('ZM', 1, 2, 3), 2: ('P', 0, 1), 3: ('ZM', 2, 4, 'f'), 4: ('P', 0, 3), 'f': ('STOP',)}
    small = MinskyTransformer(ADD, 1, 3)
    _, tot = small.count()
    ref, regs_out = run_machine(ADD, 1, [0, 3, 2], 20)
    got, _ = small.run([0, 3, 2], 20)
    print(f"\nADD program (R0 <- R1 + R2) as a transformer: {tot} weights; trace matches interpreter: {got == ref}; "
          f"registers after run: {regs_out}")


if __name__ == "__main__":
    main()