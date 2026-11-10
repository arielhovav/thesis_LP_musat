"""
ga_polish.py
============
Seeded evolutionary polish of a trained fixed-grid TERNARY checkpoint,
minimizing the exact two-part MDL objective. Answers two questions SGD
cannot: is the trained artifact the shortest network in its basin, and if
not, how much shorter can search go?

Objective (bits, no free parameters - the natural code):
    F(theta) = |H| + |D:H|
    |H|   = L0 * (log2(N_quantizable) + 1)   address + sign per surviving weight
            N_quantizable is read off the loaded model: 12608 for the a^n b^n
            architecture at d=32 (the value previously hard-coded here), 12672
            for a^n b^n c^n at d=32 (one more vocabulary row in emb and head).
            (LayerNorm scalars are shared architecture, excluded - as in all
             our |H| accounting; state this in the paper)
    |D:H| = sum over the run's OWN training strings of -log2 q(s)
            (same seeded train set, same 1e-10 smoothing as training)

On the fixed ternary grid every nonzero weight is +-delta, so this objective
is EXACTLY the description length the fixed-grid weight decay was a proxy
for. Pure mode therefore tests: "does the training-set MDL optimum keep the
counting rule?" Guard mode (--guard) additionally rejects mutations that
break exact counting at given probe lengths, answering instead: "what is the
smallest network that still counts?" Run both; they mean different things.

Search: (1+pop) evolution strategy from the checkpoint. Mutations per child
(1-3 of): delete a nonzero weight (biased - we are hunting downhill in |H|),
flip a sign, resurrect a dead slot to +-delta. Equal-fitness moves are
accepted with probability 0.5 (neutral drift opens pruning paths). Before the
search and after every improvement, DEAD-UNIT PRUNING removes at zero cost
every ff_in row whose hidden unit has no ff_out column (and vice versa):
those weights cannot affect the output, so deleting them is a free |H| win
the random search would otherwise have to find one weight at a time.

Price per weight (--hbits):
  <number>  bits per surviving weight
  natural   log2(N_quantizable) + 1 (default): Lemma 6's address+sign code
  sgd       the price SGD's own objective implies on this grid: with coupled
            L2 decay, mean_CE + wd*||theta||^2 = mean_CE + wd*delta^2*SUM(k^2)
            over the integer grid levels k, and in bits summed over the T
            training tokens that is  |D:H| + (wd*T*delta^2/ln2) * sum(k^2).
            On the ternary grid sum(k^2) = L0, a flat ~14.0 bits per weight at
            wd=1, delta=0.05 (vs 14.62 natural). On int4/int8 it is QUADRATIC
            in the level: a level-127 weight costs 127^2 x 14 bits. Running the
            polish at this price asks whether SGD reached the optimum of ITS
            OWN objective and whether that objective can afford the magnitude
            solutions the natural code selects (Musat eq. (1): beta in
            [delta^2, M^2]). (AdamW's decay is decoupled; 'sgd' is the
            coupled-L2 reading.) The header line says "bits x sum(k^2)".

--guard_coin EPS: reject children whose coin P(b | <bos> a^k) deviates from
the scoring generator's p by more than EPS at any audit probe k (1..300). A
polish that drifts the coin to buy counter weights is a cancellation exploit
on the weighted metric; this guard asks for the cheapest network with a
STATIONARY coin.

Biases (--bias checkpoints): the genome is built from quantizable_named_weights(),
so gridded bias vectors are mutated, priced and pruned like any other slot.
Dead-unit pruning also zeroes the ff_in bias of a dead hidden unit, and a unit
whose only input is its bias counts as alive (a constant unit).

The output checkpoint records config["polish"] = {p_end, train_size, hbits,
guard, guard_mode, guard_bits, guard_coin, seed, gens, pop} so audit_counters
can score the coin against the generator the network was actually polished on.

--polish_train_size N: score |D:H| on N fresh strings from the same grammar
(same seed stream, N > the run's train_size) instead of the run's own training
set. Larger data terms make the objective care about calibration again; this
is the test of whether the coin drift the polish introduces is a small-data
effect. --out sets the output path (default <ckpt_dir>/model_polished*.pt).

Guard currencies (--guard_mode):
  bits    reject children with more than --guard_bits counter bits at any
          guard length (the original; demands a calibrated counter).
  greedy  reject children whose ARGMAX is wrong at any counter position of
          any guard length, i.e. preserve greedy_n rather than exact_n. Use
          this for networks that count far beyond their exact_n with drift
          (a 184-weight a^n b^n counter that is greedy-correct to n=225 while
          spending 3.7 bits at n=120 cannot seed a bits-guarded search at all).
  The seed checkpoint must itself satisfy the guard; the script says so and
  stops if not.

Usage:
  python ga_polish.py --ckpt runs/rule/nope_fixed_wd1.0_seed100/model.pt \\
      --gens 2000 --pop 24
  python ga_polish.py --ckpt ... --gens 2000 --pop 24 --guard 30,60,120
  python ga_polish.py --ckpt ... --gens 3000 --pop 24 --guard 30,60,120 --guard_mode greedy

Writes <ckpt_dir>/model_polished[_guarded].pt (loadable by evaluate.py,
probe_lengths.py, audit_counters.py, anatomize.py) and prints before/after
L0, |H|, |D:H|, total, and counter bits at reference lengths.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random

import torch
import torch.nn.functional as F

from audit_counters import COIN_KS, coin_profile
from grammar import PAD, collate, make_train_set, task_from_config
from model import LoopedTransformer
from quantization import QuantConfig

LN2 = math.log(2.0)


def prune_dead_units(genome, names):
    """Zero ff_in rows of hidden units with no surviving ff_out column and
    ff_out columns of units with no surviving ff_in row. Returns #removed."""
    try:
        i_in = next(i for i, n in enumerate(names) if n.endswith("ff_in"))
        i_out = next(i for i, n in enumerate(names) if n.endswith("ff_out"))
    except StopIteration:
        return 0
    fin, fout = genome.tensors[i_in], genome.tensors[i_out]      # (d_ff, d), (d, d_ff)
    bin_ = next((genome.tensors[i] for i, n in enumerate(names)
                 if n.endswith("ff_in.bias")), None)              # (d_ff,) or None
    has_in = (fin != 0).any(dim=1)
    if bin_ is not None:
        has_in = has_in | (bin_ != 0)        # bias-only unit = constant unit, alive
    has_out = (fout != 0).any(dim=0)
    dead = ~(has_in & has_out)
    removed = int((fin[dead] != 0).sum()) + int((fout[:, dead] != 0).sum())
    if bin_ is not None:
        removed += int((bin_[dead] != 0).sum())
    if removed:
        fin[dead] = 0
        fout[:, dead] = 0
        if bin_ is not None:
            bin_[dead] = 0
    return removed


def bits_per_weight(model) -> float:
    """log2(address space) + log2(number of non-zero grid values) per surviving
    weight: +1 (sign) on the ternary grid, +log2(2*127) = 8 bits on int8, +log2(14)
    on int4. Equals log2(12608)+1 (the old hard-coded constant) for the ternary
    a^n b^n d=32 model. Musat's natural code, Appendix F."""
    m = model.qcfg.n_levels_per_sign or 1
    return math.log2(model.param_report()["quantizable_params"]) + math.log2(2 * m)


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    task = task_from_config(cfg)
    from quantization import INT_LEVELS
    if cfg["quant"] not in INT_LEVELS or cfg["scale"] != "fixed":
        raise SystemExit("ga_polish targets fixed-grid integer checkpoints "
                         f"({'/'.join(INT_LEVELS)}); this one: {cfg['quant']}/{cfg['scale']}.")
    qcfg = QuantConfig(mode=cfg["quant"], scale=cfg["scale"],
                       fixed_delta=cfg["fixed_delta"])
    model = LoopedTransformer.from_config(cfg, task.vocab_size, qcfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg, ck, task


class Genome:
    """Integer grid levels in [-m, m] per quantizable matrix (m = 1 on the
    ternary grid, 127 on int8); writes level*delta into the live model tensors
    (the forward quantizer is then the identity). On int grids an extra
    mutation moves a weight by one level, so the search can trade magnitude
    against replication at the natural code's price (which is per weight,
    not per unit of magnitude)."""

    def __init__(self, model, delta, m: int = 1):
        self.delta, self.m = delta, m
        self.names, self.tensors = [], []
        for name, w in model.quantizable_named_weights():
            self.names.append(name)
            self.tensors.append(torch.round(w / delta).clamp(-m, m).to(torch.int8 if m <= 127 else torch.int32))
        self._model_refs = [w for _, w in model.quantizable_named_weights()]

    def install(self):
        with torch.no_grad():
            for sgn, w in zip(self.tensors, self._model_refs):
                w.copy_(sgn.to(w.dtype) * self.delta)

    def l0(self):
        return int(sum(int((t != 0).sum()) for t in self.tensors))

    def sumsq(self):
        """sum of squared grid levels k^2 = ||theta||^2 / delta^2 (= L0 on ternary)."""
        return int(sum(int((t.to(torch.int64) ** 2).sum()) for t in self.tensors))

    def clone(self):
        g = object.__new__(Genome)
        g.delta, g.m, g.names = self.delta, self.m, self.names
        g.tensors = [t.clone() for t in self.tensors]
        g._model_refs = self._model_refs
        return g

    def mutate(self, rng):
        for _ in range(rng.choice((1, 1, 2, 3))):
            t = self.tensors[rng.randrange(len(self.tensors))]
            flat = t.view(-1)
            r = rng.random()
            nz = (flat != 0).nonzero().flatten()
            if self.m > 1 and r < 0.30 and len(nz):       # int grid: +-1 level
                i = nz[rng.randrange(len(nz))]
                flat[i] = max(-self.m, min(self.m, int(flat[i]) + rng.choice((-1, 1))))
            elif r < 0.60 and len(nz):                     # delete
                flat[nz[rng.randrange(len(nz))]] = 0
            elif r < 0.85 and len(nz):                     # flip sign
                i = nz[rng.randrange(len(nz))]
                flat[i] = -flat[i]
            else:                                          # resurrect
                zz = (flat == 0).nonzero().flatten()
                if len(zz):
                    flat[zz[rng.randrange(len(zz))]] = rng.choice((-1, 1))


@torch.no_grad()
def data_bits(model, inp, tgt):
    """Total -log2 q over the training set (sum, not mean), 1e-10 smoothing."""
    lp = F.log_softmax(model(inp).float(), dim=-1)
    p = lp.exp().clamp_min(1e-10)
    tb = -(p.log() / LN2).gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return float((tb * (tgt != PAD)).sum())


@torch.no_grad()
def counter_bits_at(model, ns, device, task):
    out = {}
    for n in ns:
        inp, tgt = collate([task.string_for_n(n)], device)
        lp = F.log_softmax(model(inp).float(), dim=-1)
        p = lp.exp().clamp_min(1e-10)
        tb = -(p.log() / LN2).gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        tb = (tb * (tgt != PAD))[0]
        lo, hi = task.counter_span(n)
        out[n] = float(tb[lo:hi].sum())
    return out


@torch.no_grad()
def greedy_ok_at(model, ns, device, task):
    """{n: True iff argmax == target at every counter position} (one batch)."""
    inp, tgt = collate([task.string_for_n(n) for n in ns], device)
    am = model(inp).argmax(-1)
    out = {}
    for j, n in enumerate(ns):
        lo, hi = task.counter_span(n)
        out[n] = bool((am[j, lo:hi] == tgt[j, lo:hi]).all())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gens", type=int, default=2000)
    ap.add_argument("--pop", type=int, default=24, help="children per generation")
    ap.add_argument("--guard", default="",
                    help="comma-separated lengths; reject children whose "
                         "counter bits exceed --guard_bits at any of them")
    ap.add_argument("--guard_bits", type=float, default=0.5)
    ap.add_argument("--guard_mode", default="bits", choices=["bits", "greedy"],
                    help="bits: counter bits <= --guard_bits at guard lengths; "
                         "greedy: argmax correct at every counter position")
    ap.add_argument("--hbits", default="natural",
                    help="bits per surviving weight: a number, 'natural' "
                         "(log2 N_quantizable + log2(2m), default), 'sgd' (the price "
                         "implied by this run's wd, delta and training tokens: sum k^2), "
                         "'l1' (coupled-L1 price, wd*T*delta/ln2 bits per unit level, see --sgd_wd), "
                         "'fraction' (Abudy's magnitude-aware value code), 'elias' (sign + "
                         "universal integer code of |k|: the log-magnitude price) or "
                         "'codebook' (distinct |levels| paid once + log2 K index per weight)")
    ap.add_argument("--sgd_wd", type=float, default=None,
                    help="with --hbits sgd: the weight decay to price at (default: the "
                         "checkpoint's own); on a ternary delta=1 grid wd = 14.64*ln2/T "
                         "makes the sgd price equal the natural code")
    ap.add_argument("--polish_train_size", type=int, default=None,
                    help="score |D:H| on this many strings from the grammar "
                         "(default: the run's own training set)")
    ap.add_argument("--polish_pend", type=float, default=None,
                    help="score |D:H| on strings sampled with this geometric end-"
                         "probability instead of the run's p_geom (a longer tail "
                         "makes the data demand counting beyond the training support)")
    ap.add_argument("--guard_coin", type=float, default=None,
                    help="reject children whose coin P(b|a^k) deviates from the "
                         "scoring p by more than this at any probe k in "
                         f"{list(COIN_KS)} (stationary-coin guard)")
    ap.add_argument("--out", default=None, help="output checkpoint path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg, ck, task = load(args.ckpt, device)
    train = make_train_set(cfg["train_size"], seed=cfg["seed"], p=cfg["p_geom"],
                           task=task)
    n_tokens = sum(len(s) for s in train.seqs)
    hbits_label = args.hbits
    price_sumsq = False                      # False: |H| = L0 x hbits; True: sum(k^2) x hbits
    price_sumabs = False                     # True: |H| = sum|k| x hbits (coupled-L1 price)
    if args.hbits == "natural":
        args.hbits = bits_per_weight(model)
        print(f"task={task.name}  |H| = L0 x {args.hbits:.2f} bits (natural code; "
              f"N = {model.param_report()['quantizable_params']} slots incl. "
              f"{model.param_report()['quantizable_bias_slots']} bias slots)")
    elif args.hbits == "l1":
        price_sumabs = True
        wd_price = args.sgd_wd if args.sgd_wd is not None else cfg["weight_decay"]
        if wd_price <= 0:
            raise SystemExit("--hbits l1 needs a positive weight decay: pass --sgd_wd (natural-code-equivalent: "
                             f"{bits_per_weight(model) * LN2 / (n_tokens * cfg['fixed_delta']):.5f})")
        args.hbits = wd_price * n_tokens * cfg["fixed_delta"] / LN2
        print(f"task={task.name}  |H| = {args.hbits:.2f} bits x sum|k|  (coupled-L1 price: wd={wd_price} x T={n_tokens} "
              f"x delta={cfg['fixed_delta']} / ln2 per level; natural would be {bits_per_weight(model):.2f} bits/weight)")
    elif args.hbits == "sgd":
        price_sumsq = True
        wd_price = args.sgd_wd if args.sgd_wd is not None else cfg["weight_decay"]
        if wd_price <= 0:
            raise SystemExit("--hbits sgd needs a positive weight decay: this checkpoint has "
                             f"wd={cfg['weight_decay']}; pass --sgd_wd (natural-code-equivalent on a "
                             f"ternary delta=1 grid: {bits_per_weight(model) * LN2 / n_tokens:.5f})")
        args.hbits = wd_price * n_tokens * cfg["fixed_delta"] ** 2 / LN2
        print(f"task={task.name}  |H| = {args.hbits:.2f} bits x sum(k^2)  (SGD-implied: "
              f"wd={wd_price} x T={n_tokens} tokens x delta^2="
              f"{cfg['fixed_delta']**2:g} / ln2 per squared level; on ternary "
              f"sum(k^2) = L0, so this equals a flat {args.hbits:.2f} bits/weight; "
              f"natural would be {bits_per_weight(model):.2f} bits/weight)")
    elif args.hbits in ("fraction", "codebook", "elias"):
        args.hbits = bits_per_weight(model)      # only used for reporting
    else:
        args.hbits = float(args.hbits)
        print(f"task={task.name}  |H| = L0 x {args.hbits:.2f} bits")

    # ---- magnitude-aware codes (same address cost as the natural code: log2 N
    # + 1 sign bit per weight; only the VALUE part changes) -------------------
    addr_bits = math.log2(model.param_report()["quantizable_params"]) + 1.0
    m_lv = getattr(model.qcfg, "n_levels_per_sign", None) or 1
    delta = float(cfg["fixed_delta"])
    den0 = int(round(1.0 / delta))

    def elias(n):                              # prefix-free code for a positive integer
        n = max(int(n), 1)
        return math.log2(n) + 2 * math.log2(math.log2(n) + 1) + 1

    frac_table = [0.0] * (m_lv + 1)            # Abudy value code for level k: k/den0 reduced
    for k in range(1, m_lv + 1):
        g = math.gcd(k, den0)
        frac_table[k] = elias(k // g) + elias(den0 // g)

    def h_fraction(genome):
        bits = 0.0
        for t in genome.tensors:
            ks = t.abs().to(torch.int64).flatten()
            ks = ks[ks != 0]
            if ks.numel():
                bits += float(sum(frac_table[int(k)] for k in ks.tolist()))
        return genome.l0() * addr_bits + bits

    def h_l1(genome):
        return float(args.hbits) * sum(float(t.abs().sum()) for t in genome.tensors)

    def h_elias(genome):
        """Universal integer code for the level: sign bit + Elias-gamma(|k|), i.e.
        ~2 log2 k + 2 bits. The Kolmogorov-honest price of a magnitude: constant
        for +-1, logarithmic for large levels (127 -> 15 bits), never quadratic."""
        bits = 0.0
        for t in genome.tensors:
            ks = t.abs().to(torch.int64).flatten()
            ks = ks[ks != 0].to(torch.float64)
            if ks.numel():
                lg = torch.log2(ks)
                bits += float((lg + 2 * torch.log2(lg + 1) + 2).sum())
        return genome.l0() * addr_bits + bits

    def h_codebook(genome):
        """Two-part value code: a codebook of the K distinct |levels| (log2(2m)
        bits each) plus a log2 K index per weight. Equal magnitudes are paid
        once: the exact symmetric comparator (+127, -127) costs one entry, a
        tilted one (+116, -120) costs two."""
        levels = set()
        for t in genome.tensors:
            ks = t.abs().to(torch.int64).flatten()
            levels.update(ks[ks != 0].tolist())
        K = max(len(levels), 1)
        return genome.l0() * (addr_bits + math.log2(K)) + K * math.log2(2 * m_lv)

    if hbits_label == "l1":
        price_fn = h_l1
    elif hbits_label == "elias":
        price_fn = h_elias
        print(f"task={task.name}  |H| = L0 x {addr_bits:.2f} address bits + sign + Elias-gamma(|k|) "
              f"(k=1: 2.0 bits, k=2: 5.0, k=7: 9.8, k=127: 16.0; log-magnitude price)")
    elif hbits_label == "fraction":
        price_fn = h_fraction
        print(f"task={task.name}  |H| = L0 x {addr_bits:.2f} address bits + Abudy fraction value "
              f"code (level k -> k/{den0} reduced, Elias-coded numerator and denominator; "
              f"k=1: {frac_table[1]:.1f} bits, k=127: {frac_table[m_lv]:.1f} bits, "
              f"k=120: {frac_table[min(120, m_lv)]:.1f} bits)")
    elif hbits_label == "codebook":
        price_fn = h_codebook
        print(f"task={task.name}  |H| = L0 x ({addr_bits:.2f} + log2 K) + K x {math.log2(2*m_lv):.2f} bits, "
              f"K = number of distinct |levels| (codebook code: equal magnitudes are paid once)")
    else:
        price_fn = None

    def h_bits(genome):
        if price_fn is not None:
            return price_fn(genome)
        return (genome.sumsq() if price_sumsq else genome.l0()) * args.hbits
    n_str, p_use = cfg["train_size"], cfg["p_geom"]
    if args.polish_train_size or args.polish_pend:
        n_str = args.polish_train_size or cfg["train_size"]
        p_use = args.polish_pend if args.polish_pend else cfg["p_geom"]
        train = make_train_set(n_str, seed=cfg["seed"], p=p_use, task=task)
        print(f"scoring |D:H| on {n_str} strings with p_end={p_use:g} "
              f"({sum(len(s) for s in train.seqs)} tokens, longest n={train.n_max}) "
              f"instead of the run's {cfg['train_size']} at p={cfg['p_geom']:g}")
    inp, tgt = collate(train.seqs, device)
    guard_ns = [int(x) for x in args.guard.split(",") if x.strip()]

    g = Genome(model, cfg["fixed_delta"], model.qcfg.n_levels_per_sign or 1)
    removed0 = prune_dead_units(g, g.names)
    g.install()
    if removed0:
        print(f"dead-unit pruning removed {removed0} weights from the seed at zero cost")

    def guard_violations(m):
        """{n: value} for guard lengths the live model fails, {} if it passes."""
        bad = {}
        if guard_ns:
            if args.guard_mode == "greedy":
                ok = greedy_ok_at(m, guard_ns, device, task)
                bad.update({n: "argmax wrong" for n, v in ok.items() if not v})
            else:
                cb = counter_bits_at(m, guard_ns, device, task)
                bad.update({n: round(v, 2) for n, v in cb.items() if v > args.guard_bits})
        if args.guard_coin is not None:
            coin = coin_profile(m, task, device)
            bad.update({f"coin@k={k}": round(v, 3) for k, v in coin.items()
                        if abs(v - p_use) > args.guard_coin})
        return bad

    any_guard = bool(guard_ns) or args.guard_coin is not None

    def fitness(genome, use_guard=True):
        genome.install()
        if use_guard and any_guard and guard_violations(model):
            return None, None, None
        d = data_bits(model, inp, tgt)
        h = h_bits(genome)
        return h + d, h, d

    if args.guard_coin is not None:
        print(f"guard: coin |P(b|a^k) - {p_use:g}| <= {args.guard_coin} at k in {list(COIN_KS)}")
    if any_guard:
        bad = guard_violations(model)
        if bad:
            hint = ("try --guard_mode greedy (preserves argmax correctness, "
                    "tolerates calibration drift)" if args.guard_mode == "bits"
                    else "choose --guard lengths within this artifact's greedy_n")
            raise SystemExit(
                f"the seed checkpoint itself violates the {args.guard_mode} "
                f"guard at {bad}; a guarded search preserves an existing "
                f"capability - {hint}.")
        if guard_ns:
            print(f"guard: {args.guard_mode} at n={guard_ns}"
                  + (f" (<= {args.guard_bits} bits)" if args.guard_mode == "bits" else ""))
    f0, h0, d0 = fitness(g, use_guard=False)
    probe_ns = [25, 50, 100, 150]
    cb0 = counter_bits_at(model, probe_ns, device, task)
    print(f"start: L0={g.l0()}  sum(k^2)={g.sumsq()}  |H|={h0:.0f} bits  |D:H|={d0:.0f} bits  "
          f"total={f0:.0f} bits")
    print(f"       counter bits at {probe_ns}: "
          + "  ".join(f"{n}:{cb0[n]:.2f}" for n in probe_ns))

    best, fbest, hb, db = g, f0, h0, d0
    since_report = 0
    for gen in range(1, args.gens + 1):
        improved = False
        for _ in range(args.pop):
            child = best.clone()
            child.mutate(rng)
            f, h, d = fitness(child)
            if f is None:
                continue
            if f < fbest or (f == fbest and rng.random() < 0.5):
                if f < fbest:
                    improved = True
                if prune_dead_units(child, child.names):
                    f, h, d = fitness(child)          # re-score after free deletions
                    if f is None:
                        continue
                best, fbest, hb, db = child, f, h, d
        since_report += 1
        if improved or since_report >= max(50, args.gens // 20):
            print(f"gen {gen:>5}: L0={best.l0()}  sum(k^2)={best.sumsq()}  |H|={hb:.0f}  "
                  f"|D:H|={db:.0f}  total={fbest:.0f}")
            since_report = 0

    best.install()
    cb1 = counter_bits_at(model, probe_ns, device, task)
    gk = greedy_ok_at(model, probe_ns, device, task)
    print("       greedy-correct at " + "  ".join(f"{n}:{'yes' if gk[n] else 'NO'}"
                                                  for n in probe_ns))
    print(f"\nfinal: L0={best.l0()}  sum(k^2)={best.sumsq()}  |H|={hb:.0f}  |D:H|={db:.0f}  "
          f"total={fbest:.0f} bits   (start total {f0:.0f}; "
          f"saved {f0 - fbest:.0f} bits)")
    if price_sumsq:
        print(f"       natural-code price of this network: "
              f"{best.l0() * bits_per_weight(model):.0f} + {db:.0f} = "
              f"{best.l0() * bits_per_weight(model) + db:.0f} bits")
    print("       counter bits at " + "  ".join(f"{n}:{cb1[n]:.2f}"
                                                for n in probe_ns))
    suffix = ("_guarded" if guard_ns else "") + \
             ("_greedy" if guard_ns and args.guard_mode == "greedy" else "")
    out = args.out or os.path.join(os.path.dirname(args.ckpt), f"model_polished{suffix}.pt")
    ck2 = copy.deepcopy(ck)
    ck2["state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ck2["config"]["polish"] = {
        "p_end": p_use, "train_size": n_str, "hbits": hbits_label,
        "hbits_value": args.hbits,
        "price": hbits_label if hbits_label in ("fraction", "codebook", "elias", "l1") else ("sumsq" if price_sumsq else "L0"),
        "sgd_wd": (args.sgd_wd if args.sgd_wd is not None else cfg["weight_decay"]) if (price_sumsq or price_sumabs) else None,
        "guard": guard_ns, "guard_mode": args.guard_mode if guard_ns else None,
        "guard_bits": args.guard_bits if guard_ns else None,
        "guard_coin": args.guard_coin, "seed": args.seed, "gens": args.gens,
        "pop": args.pop, "final_total_bits": fbest, "final_L0": best.l0(),
        "final_sumsq": best.sumsq(), "from_ckpt": args.ckpt}
    torch.save(ck2, out)
    print(f"polished checkpoint -> {out}")
    print("next: probe_lengths.py / anatomize.py / audit_counters.py on it.")


if __name__ == "__main__":
    main()