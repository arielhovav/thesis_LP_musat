"""
anatomize2.py
=============
Mechanistic account of a compressed counting network, for BOTH languages
(a^n b^n and a^n b^n c^n) and for any checkpoint (model.pt, model_best_near.pt,
model_polished*.pt). Designed for the small nets (L0 ~ 80-1500) that the
fixed-grid runs and ga_polish.py produce.

Premise (verified in step 2, not assumed): under NoPE, a network whose attention
is uniform over the causal prefix computes, at every position, a function of
(own token, counts of each letter class in the prefix). For strings of the form
a^i b^j [c^k] every prefix is again of that form, so the entire reachable
behaviour is a finite table indexed by (i, j[, k]). The steps:

  1. Structural census + Abudy-style |H|.
     Per-matrix non-zeros; live residual dimensions (written by at least one
     non-zero); live FFN hidden units (non-zero row in ff_in AND column in
     ff_out); an Abudy-style structural description length that prices each
     connection by log2 of the LIVE unit counts (plus sign and value bits) and
     each unit by a prefix code, next to Musat's address-space |H|
     (log2 N_quantizable + 1 per weight). Both are written to the report.

  2. Prefix-mean reduction check.
     Zero Wq and Wk (attention becomes exactly uniform) and measure the max
     change in next-token log-probability over all on-grammar prefixes. Small
     => the table below is a complete description of the network.

  3. Complete I/O table over reachable prefixes, i up to --max_i:
        a^i                     (coin: expect P(a)=1-p, P(b)=p)
        a^i b^j, 0<j<i          (expect b)      a^i b^i           (expect c / '#')
        a^i b^i c^k, 0<k<i      (expect c)      a^i b^i c^i       (expect '#')
     plus --off_grammar extra prefixes past each boundary (j>i, k>i) to see what
     the net does on invalid input. Writes io_table.csv (one row per prefix with
     the full distribution, expected symbol, argmax, bits on the expected symbol)
     and prints, per phase, the first i at which the argmax is wrong, and the
     DECISION BOUNDARY: for each i, the first j (k) at which the network's
     argmax switches from 'continue' to 'close' - a perfect counter switches at
     exactly j=i (k=i); a drifting one switches early or late, and the table of
     (i, switch point) is the network's counting law in one column.

  4. Neuron readout (the algorithm).
     ... and CHANNELS: live residual dimensions whose activations are identical
     up to sign (|r| > 0.995) carry the same scalar. On a ternary grid a gain of
     g*delta can only be built by writing the scalar into g dimensions, so the
     channel's replication count IS the gain the network paid for. The report
     lists each channel (members, gain, which quantity it tracks), then the
     PROGRAM: every live hidden unit as a signed sum of channels (ff_in), what
     it writes back (ff_out), and every output logit as a signed sum of
     channels (head). <stem>_circuit.png draws it: channel curves along one
     full string on the left, the channel -> unit -> logit graph on the right.
     For every live FFN hidden unit and live residual dimension, the activation
     at the LAST loop, last position, over all on-grammar prefixes, correlated
     with the candidate quantities a counter needs: i, j, k, i-j, i-k, and the
     current phase. Units whose activation is a near-linear function of i-j (in
     the b-run) or i-k (in the c-run) are the comparator; units that are
     constant within a phase are phase detectors. Prints the top units by |r|
     for each quantity, and for --dump_units the activation vs (j or k) at a few
     fixed i so the threshold can be read off by eye. Writes neurons.csv.

Usage:
  python anatomize2.py --ckpt runs/rule_settled/nope_fixed_wd1.0_seed104/model_polished_guarded_greedy.pt --max_i 60
  python anatomize2.py --ckpt runs/anbncn/rule_settled/nope_fixed_wd1_seed100/model.pt --max_i 40 --dump_units 6
Outputs go next to the checkpoint: <stem>_io_table.csv, <stem>_neurons.csv,
<stem>_anatomy.txt (the printed report).
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from grammar import BOS, PAD, task_from_config
from model import LoopedTransformer
from quantization import QuantConfig, quantize

LN2 = math.log(2.0)


# ----------------------------------------------------------------- loading ---
def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    task = task_from_config(cfg)
    qcfg = QuantConfig(mode=cfg["quant"], scale=cfg["scale"],
                       fixed_delta=cfg["fixed_delta"])
    model = LoopedTransformer.from_config(cfg, task.vocab_size, qcfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg, task


# ------------------------------------------------------------- 1. census -----
@torch.no_grad()
def structural_census(model, task):
    q = {name: quantize(w, model.qcfg) for name, w in model.quantizable_named_weights()}
    nz = {n: int((w != 0).sum()) for n, w in q.items()}
    d, f = model.d, model.block.ff_in.weight.shape[0]
    L0 = sum(nz.values())

    has_bias = any(n.endswith(".bias") for n in q)
    # live residual dims: written into by emb columns, wo rows, ff_out rows
    # (and, with --bias, the wo / ff_out bias vectors)
    written = torch.zeros(d, dtype=torch.bool)
    written |= (q["emb"] != 0).any(dim=0).cpu()
    written |= (q["block.attn.wo"] != 0).any(dim=1).cpu()
    written |= (q["block.ff_out"] != 0).any(dim=1).cpu()
    for bn in ("block.attn.wo.bias", "block.ff_out.bias"):
        if bn in q:
            written |= (q[bn] != 0).cpu()
    # read from by wq/wk/wv columns, ff_in columns, head columns
    read = torch.zeros(d, dtype=torch.bool)
    for n in ("block.attn.wq", "block.attn.wk", "block.attn.wv", "block.ff_in", "head"):
        read |= (q[n] != 0).any(dim=0).cpu()
    live_resid = int((written & read).sum())
    # live hidden units: some input weight and some output weight
    hid_in = (q["block.ff_in"] != 0).any(dim=1).cpu()
    if "block.ff_in.bias" in q:                  # bias-only unit = constant, alive
        hid_in |= (q["block.ff_in.bias"] != 0).cpu()
    hid_out = (q["block.ff_out"] != 0).any(dim=0).cpu()
    live_hidden = hid_in & hid_out
    n_live_hidden = int(live_hidden.sum())
    # dead weights: connections into/out of units that are not live
    dead_ffin = int(((q["block.ff_in"] != 0).cpu() & ~live_hidden[:, None]).sum())
    dead_ffout = int(((q["block.ff_out"] != 0).cpu() & ~live_hidden[None, :]).sum())

    # Abudy-style structural |H|: units cost a prefix code, each connection
    # log2(#live source units) + log2(#live target units) + value bits.
    # Value bits: ternary sign = 1 bit plus 1 bit for "this weight exists on
    # the grid" (the fraction 1/1 in their scheme is the cheapest value).
    def elias(n):
        n = max(int(n), 1)
        return math.log2(n) + 2 * math.log2(math.log2(n) + 1) + 1

    V = task.vocab_size
    units = {"emb": V, "resid": live_resid, "hidden": n_live_hidden, "head": V}
    h_units = sum(elias(u) for u in units.values())
    per_conn = {}
    src_dst = {}
    conn_cost = 0.0
    for name, w in q.items():
        n_nz = nz[name]
        if name == "emb":
            src, dst = V, live_resid
        elif name == "head":
            src, dst = live_resid, V
        elif name == "block.ff_in":
            src, dst = live_resid, n_live_hidden
        elif name == "block.ff_out":
            src, dst = n_live_hidden, live_resid
        elif name.endswith(".bias"):            # a bias names only its target unit
            src = 1
            dst = {"block.ff_in.bias": n_live_hidden, "head.bias": V}.get(name, live_resid)
        else:                                   # attention matrices
            src, dst = live_resid, live_resid
        c = n_nz * (math.log2(max(src, 2)) + math.log2(max(dst, 2)) + 2)
        src_dst[name] = (src, dst)
        per_conn[name] = c
        conn_cost += c
    h_abudy = h_units + conn_cost
    m_lv = getattr(model.qcfg, "n_levels_per_sign", None) or 1
    h_musat = L0 * (math.log2(sum(w.numel() for w in q.values())) + math.log2(2 * m_lv))
    # Abudy's own value code (Lan et al. 2022; Abudy et al. 2025 sec. 2.5): each
    # weight is a signed fraction, sign bit + prefix-free code of numerator and
    # denominator. On the grid w = k*delta the fraction is k / round(1/delta),
    # reduced. This is magnitude-aware: k = 127 costs more than k = 1.
    delta = float(model.qcfg.fixed_delta)
    den0 = int(round(1.0 / delta))
    val_bits = 0.0
    for w in q.values():
        ks = torch.round(w / delta).abs().to(torch.int64).flatten()
        for k in ks[ks != 0].tolist():
            g = math.gcd(int(k), den0)
            val_bits += 1 + elias(int(k) // g) + elias(den0 // g)
    h_abudy_frac = h_units + sum(
        nz[name] * (math.log2(max(src_dst[name][0], 2)) + math.log2(max(src_dst[name][1], 2)))
        for name in q) + val_bits
    return {
        "nonzeros": nz, "L0": L0, "d_model": d, "d_ff": f,
        "live_resid": live_resid, "live_hidden": n_live_hidden,
        "dead_ffin_weights": dead_ffin, "dead_ffout_weights": dead_ffout,
        "H_musat_bits": h_musat, "H_abudy_bits": h_abudy, "H_abudy_frac_bits": h_abudy_frac,
        "H_abudy_value_bits": val_bits,
        "has_bias": has_bias,
        "L0_bias": sum(v for n, v in nz.items() if n.endswith(".bias")),
        "H_abudy_units_bits": h_units, "H_abudy_conn_bits": per_conn,
        "live_hidden_idx": live_hidden.nonzero().flatten().tolist(),
        "live_resid_idx": (written & read).nonzero().flatten().tolist(),
    }


# --------------------------------------------------------- prefix enumeration
def prefixes(task, max_i, off):
    """Yield (counts tuple, expected next symbol name). counts = (i, j[, k])."""
    p_rule = None
    for i in range(1, max_i + 1):
        yield (i,) + (0,) * (task.k - 1), "coin"
        # b-run
        for j in range(1, i + off + 1):
            if j < i:
                exp = "b"
            elif j == i:
                exp = "c" if task.k == 3 else "#"
            else:
                exp = "off"
            yield (i, j) + (0,) * (task.k - 2), exp
            if j > i:
                continue
        if task.k == 3:
            for k in range(1, i + off + 1):
                exp = "c" if k < i else ("#" if k == i else "off")
                yield (i, i, k), exp


def tokens_of(task, counts):
    return [BOS] + task.prefix(counts)


@torch.no_grad()
def next_token_dist(model, task, device, count_list, batch=256):
    """Softmax next-token distributions at the last position, batched by length."""
    out = [None] * len(count_list)
    order = sorted(range(len(count_list)), key=lambda t: sum(count_list[t]))
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        seqs = [tokens_of(task, count_list[t]) for t in idx]
        L = max(len(x) for x in seqs)
        inp = torch.full((len(seqs), L), PAD, dtype=torch.long)
        lens = []
        for r, x in enumerate(seqs):
            inp[r, :len(x)] = torch.tensor(x)
            lens.append(len(x))
        logits = model(inp.to(device)).float()
        for r, t in enumerate(idx):
            out[t] = F.softmax(logits[r, lens[r] - 1], dim=-1).cpu().numpy()
    return out


# ------------------------------------------------------------ 4. neuron hooks
class Recorder:
    """Captures the FFN hidden activations (post-ReLU) and the residual stream
    entering the head, at the last loop, last real position of each sequence."""

    def __init__(self, model):
        self.model = model
        self.hidden_calls, self.resid = [], None
        self.h1 = model.block.ff_in.register_forward_hook(
            lambda m, i, o: self.hidden_calls.append(F.relu(o).detach()))
        self.h2 = model.ln_f.register_forward_hook(
            lambda m, i, o: setattr(self, "resid", i[0].detach()))

    def reset(self):
        self.hidden_calls, self.resid = [], None

    def close(self):
        self.h1.remove(); self.h2.remove()


@torch.no_grad()
def record_activations(model, task, device, count_list, batch=256):
    rec = Recorder(model)
    H = [None] * len(count_list)
    R = [None] * len(count_list)
    order = sorted(range(len(count_list)), key=lambda t: sum(count_list[t]))
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        seqs = [tokens_of(task, count_list[t]) for t in idx]
        L = max(len(x) for x in seqs)
        inp = torch.full((len(seqs), L), PAD, dtype=torch.long)
        lens = [len(x) for x in seqs]
        for r, x in enumerate(seqs):
            inp[r, :len(x)] = torch.tensor(x)
        rec.reset()
        model(inp.to(device))
        hid = rec.hidden_calls[-1]                 # last loop: (B, L, d_ff)
        for r, t in enumerate(idx):
            H[t] = hid[r, lens[r] - 1].cpu().numpy()
            R[t] = rec.resid[r, lens[r] - 1].cpu().numpy()
    rec.close()
    return np.stack(H), np.stack(R)


def find_channels(R, idx, thr=0.995):
    """Group residual dims (columns of R restricted to idx) that are identical up
    to sign. Returns list of dicts {rep, members:[(dim, sign)], gain}."""
    if not idx:
        return []
    X = R[:, idx]
    sd = X.std(0)
    Xn = (X - X.mean(0)) / np.where(sd < 1e-9, 1.0, sd)
    C = (Xn.T @ Xn) / max(len(X) - 1, 1)
    parent = list(range(len(idx)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if sd[a] > 1e-9 and sd[b] > 1e-9 and abs(C[a, b]) > thr:
                parent[find(b)] = find(a)
    groups = {}
    for a in range(len(idx)):
        groups.setdefault(find(a), []).append(a)
    chans = []
    for root, mem in groups.items():
        rep = mem[0]
        members = [(idx[m], 1 if (C[rep, m] >= 0 or sd[m] < 1e-9) else -1) for m in mem]
        chans.append({"rep": idx[rep], "members": members, "gain": len(mem),
                      "rep_local": rep})
    chans.sort(key=lambda c: -c["gain"])
    return chans


def channel_weight_sums(W, chans, axis):
    """Sum a quantized matrix over each channel's member dims with member signs.
    axis=1: W is (out, d_model) and we sum columns -> (out, n_chan)
    axis=0: W is (d_model, out) and we sum rows    -> (n_chan, out)"""
    W = W.detach().cpu().numpy()
    cols = []
    for ch in chans:
        if axis == 1:
            v = sum(sgn * W[:, dim] for dim, sgn in ch["members"])
        else:
            v = sum(sgn * W[dim, :] for dim, sgn in ch["members"])
        cols.append(v)
    return np.stack(cols, axis=1 if axis == 1 else 0)


def draw_circuit(path, task, chans, chan_labels, curves, hid_idx, Win, Wout, Whead,
                 delta, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, max(6, 0.32 * max(len(chans), len(hid_idx), 4))),
                                   gridspec_kw={"width_ratios": [1.1, 1.4]})
    # ---- left: channel values along one string
    xs, phase_marks, series = curves
    for c, (lab, ys) in enumerate(series):
        axL.plot(xs, ys, lw=1.6, label=f"ch{c}: {lab}")
    for x0, name in phase_marks:
        axL.axvline(x0, color="k", lw=0.6, ls=":")
        axL.text(x0, axL.get_ylim()[1], name, fontsize=8, va="top", ha="left")
    axL.axhline(0, color="k", lw=0.5)
    axL.set_xlabel("position in the string (last position of each prefix)")
    axL.set_ylabel("channel value (representative dim, final residual)")
    axL.set_title("what the channels compute along a^i b^i" + (" c^i" if task.k == 3 else ""))
    axL.legend(fontsize=7, loc="best", ncol=2)

    # ---- right: circuit graph, three columns
    nC, nH = len(chans), len(hid_idx)
    outs = [task.id2tok[t] for t in range(2, task.vocab_size)]
    nO = len(outs)
    ys_c = np.linspace(0.95, 0.05, max(nC, 2))[:nC]
    ys_h = np.linspace(0.95, 0.05, max(nH, 2))[:nH]
    ys_o = np.linspace(0.8, 0.2, max(nO, 2))[:nO]
    xC, xH, xO = 0.08, 0.5, 0.92
    axR.set_xlim(0, 1); axR.set_ylim(0, 1); axR.axis("off")
    axR.set_title(title, fontsize=9)
    wmax = max(1e-9, np.abs(Win).max(), np.abs(Wout).max(), np.abs(Whead).max())
    for c in range(nC):
        for h in range(nH):
            w = Win[h, c]
            if abs(w) > 1e-9:
                axR.plot([xC, xH], [ys_c[c], ys_h[h]], color="tab:blue" if w > 0 else "tab:red",
                         lw=0.4 + 2.5 * abs(w) / wmax, alpha=0.7)
            w2 = Wout[c, h]
            if abs(w2) > 1e-9:
                axR.plot([xH, xC], [ys_h[h], ys_c[c]], color="tab:blue" if w2 > 0 else "tab:red",
                         lw=0.4 + 2.5 * abs(w2) / wmax, alpha=0.35, ls="--")
        for o in range(nO):
            w = Whead[o, c]
            if abs(w) > 1e-9:
                axR.plot([xC, xO], [ys_c[c], ys_o[o]], color="tab:blue" if w > 0 else "tab:red",
                         lw=0.4 + 2.5 * abs(w) / wmax, alpha=0.6)
    for c in range(nC):
        axR.scatter([xC], [ys_c[c]], s=60 + 25 * chans[c]["gain"], color="w", edgecolor="k", zorder=3)
        axR.text(xC - 0.02, ys_c[c], f"ch{c} x{chans[c]['gain']}\n{chan_labels[c]}",
                 fontsize=6.5, ha="right", va="center")
    for h in range(nH):
        axR.scatter([xH], [ys_h[h]], s=80, marker="s", color="w", edgecolor="k", zorder=3)
        axR.text(xH, ys_h[h] + 0.018, f"h{hid_idx[h]}", fontsize=6.5, ha="center", va="bottom")
    for o in range(nO):
        axR.scatter([xO], [ys_o[o]], s=90, marker="D", color="w", edgecolor="k", zorder=3)
        axR.text(xO + 0.02, ys_o[o], f"logit {outs[o]}", fontsize=8, ha="left", va="center")
    axR.text(xC, 1.0, "channels (residual)\nsize = gain", fontsize=8, ha="center", va="bottom")
    axR.text(xH, 1.0, "FFN units (ReLU)", fontsize=8, ha="center", va="bottom")
    axR.text(xO, 1.0, "head", fontsize=8, ha="center", va="bottom")
    axR.legend(handles=[Line2D([], [], color="tab:blue", label="+ (sum of member weights, in delta units)"),
                        Line2D([], [], color="tab:red", label="-"),
                        Line2D([], [], color="k", ls="--", label="unit writes back to channel")],
               fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_i", type=int, default=40)
    ap.add_argument("--off_grammar", type=int, default=2,
                    help="extra prefixes past each boundary (j>i, k>i)")
    ap.add_argument("--dump_units", type=int, default=4,
                    help="print activation-vs-count curves for this many top units")
    ap.add_argument("--skip_neurons", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, task = load(args.ckpt, device)
    stem = os.path.splitext(args.ckpt)[0]
    lines = []

    def say(s=""):
        print(s); lines.append(s)

    say(f"anatomize2: {args.ckpt}")
    say(f"task={task.name} attn={cfg.get('attn','softmax')} quant={cfg['quant']}/{cfg['scale']} pos={cfg.get('pos','sin')} "
        f"d={cfg['d_model']} d_ff={cfg['d_ff']} loops={cfg['n_loops']} wd={cfg['weight_decay']}")
    if cfg.get("pos", "sin") != "nope":
        say("WARNING: sinusoidal PE; the prefix-count reduction does not hold. "
            "The I/O table is still exact, the neuron readout is not.")

    # ---- 1. census
    c = structural_census(model, task)
    say("\n=== 1. structural census ===")
    for n, v in c["nonzeros"].items():
        say(f"  {n:>16}: {v:5d} non-zero")
    say(f"  L0 = {c['L0']}" + (f" (of which {c['L0_bias']} bias)" if c["has_bias"] else "")
        + f"   live residual dims = {c['live_resid']}/{c['d_model']}   "
        f"live FFN hidden units = {c['live_hidden']}/{c['d_ff']}")
    say(f"  dead weights (into/out of dead hidden units): ff_in {c['dead_ffin_weights']}, "
        f"ff_out {c['dead_ffout_weights']}  (removable at zero cost)")
    say(f"  |H| Musat natural code (log2 N + log2(2m) per weight) = {c['H_musat_bits']:.0f} bits")
    say(f"  |H| Abudy-style structural (live-unit addressing, 2 value bits/weight) = {c['H_abudy_bits']:.0f} bits "
        f"({c['H_abudy_units_bits']:.0f} for units + connections)")
    say(f"  |H| Abudy fraction code (sign + prefix-free numerator/denominator of k/{int(round(1/model.qcfg.fixed_delta))}) "
        f"= {c['H_abudy_frac_bits']:.0f} bits (value bits {c['H_abudy_value_bits']:.0f})")
    say(f"  live hidden units: {c['live_hidden_idx']}")

    # ---- 3. (enumeration first, reused by 2 and 4)
    items = list(prefixes(task, args.max_i, args.off_grammar))
    counts = [x for x, _ in items]
    expected = [e for _, e in items]
    say(f"\nenumerated {len(items)} prefixes up to i={args.max_i} "
        f"(+{args.off_grammar} off-grammar per boundary)")

    # ---- 2. reduction check
    say("\n=== 2. prefix-mean reduction check (zero Wq, Wk) ===")
    dist = next_token_dist(model, task, device, counts)
    saved = {n: w.detach().clone() for n, w in model.quantizable_named_weights()
             if n.endswith("wq") or n.endswith("wk")}
    with torch.no_grad():
        for n, w in model.quantizable_named_weights():
            if n in saved:
                w.zero_()
    dist0 = next_token_dist(model, task, device, counts)
    with torch.no_grad():
        for n, w in model.quantizable_named_weights():
            if n in saved:
                w.copy_(saved[n])
    D = np.stack(dist); D0 = np.stack(dist0)
    dlog = np.abs(np.log2(D.clip(1e-10)) - np.log2(D0.clip(1e-10)))
    argmax_same = float((D.argmax(1) == D0.argmax(1)).mean())
    say(f"  max |delta log2 p| over all prefixes & symbols = {dlog.max():.3f} bits; "
        f"mean = {dlog.mean():.4f}; argmax unchanged on {100*argmax_same:.1f}% of prefixes")
    say("  " + ("=> attention is (effectively) uniform: the network IS a function of "
                "(token, prefix counts); the table below is complete."
                if dlog.max() < 0.5 else
                "=> attention is content-selective; the table is exact for these "
                "prefixes but the count-reduction is only approximate."))

    # ---- 3. I/O table
    say("\n=== 3. I/O table ===")
    names = task.id2tok
    sym_ids = {names[t]: t for t in range(task.vocab_size)}
    exp_id = {"b": sym_ids["b"], "#": sym_ids["#"]}
    if task.k == 3:
        exp_id["c"] = sym_ids["c"]
    rows = []
    first_fail = {}
    boundary = {}                       # i -> (phase, switch point)
    p = cfg["p_geom"]
    for cnt, e, pr in zip(counts, expected, dist):
        am = names[int(pr.argmax())]
        if e == "coin":
            ok = am == "a" or am == "b"    # both legal; judged by calibration
            bits = float(-math.log2(max(pr[sym_ids["b"]], 1e-10)))
            note = f"P(b)={pr[sym_ids['b']]:.3f} (true {p:.2f})"
        elif e == "off":
            ok = True; bits = float("nan"); note = "off-grammar"
        else:
            ok = am == e
            bits = float(-math.log2(max(pr[exp_id[e]], 1e-10)))
            note = ""
            if not ok:
                i = cnt[0]
                first_fail.setdefault(e, i)
        rows.append({"i": cnt[0], "j": cnt[1], "k": cnt[2] if task.k == 3 else "",
                     "expected": e, "argmax": am, "bits_expected": round(bits, 4),
                     **{f"P({names[t]})": round(float(pr[t]), 5) for t in range(2, task.vocab_size)},
                     "note": note})
    # decision boundaries
    by_i = {}
    for r in rows:
        by_i.setdefault(r["i"], []).append(r)
    say(f"  {'i':>4} {'b->close at j':>14} {'(exact: i)':>10}"
        + (f" {'c->end at k':>12} {'(exact: i)':>10}" if task.k == 3 else "")
        + f" {'P(b|a^i)':>9}")
    for i in sorted(by_i):
        rs = by_i[i]
        close_sym = "c" if task.k == 3 else "#"
        jb = next((r["j"] for r in rs if r["k"] in ("", 0) and r["j"] >= 1
                   and r["argmax"] == close_sym), None)
        line = f"  {i:>4} {str(jb):>14} {'ok' if jb == i else 'X':>10}"
        if task.k == 3:
            kb = next((r["k"] for r in rs if r["j"] == i and r["k"] != "" and r["k"] >= 1
                       and r["argmax"] == "#"), None)
            line += f" {str(kb):>12} {'ok' if kb == i else 'X':>10}"
        coin = next((r for r in rs if r["expected"] == "coin"), None)
        line += f" {coin['P(b)']:>9.3f}" if coin else ""
        boundary[i] = (jb, kb if task.k == 3 else None)
        if i <= 12 or i % 5 == 0 or (jb != i) or (task.k == 3 and kb != i):
            say(line)
    say("  first argmax failure per phase: "
        + ", ".join(f"{k}-run at i={v}" for k, v in sorted(first_fail.items()))
        if first_fail else "  argmax correct on every on-grammar prefix up to i="
        f"{args.max_i}")
    path_io = f"{stem}_io_table.csv"
    with open(path_io, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    say(f"  -> {path_io}")

    # ---- 4. neuron readout
    if not args.skip_neurons:
        say("\n=== 4. neuron readout (last loop, last position, on-grammar prefixes) ===")
        on = [t for t, e in enumerate(expected) if e != "off"]
        H, R = record_activations(model, task, device, [counts[t] for t in on])
        cnt = np.array([counts[t] for t in on], dtype=float)
        i_, j_ = cnt[:, 0], cnt[:, 1]
        k_ = cnt[:, 2] if task.k == 3 else np.zeros_like(i_)
        phase = np.array([expected[t] for t in on])
        feats = {"i": i_, "j": j_, "i-j": i_ - j_}
        if task.k == 3:
            feats.update({"k": k_, "i-k": i_ - k_})
        masks = {"all": np.ones(len(on), bool),
                 "a-run": phase == "coin",
                 "b-run": np.isin(phase, ["b", "c" if task.k == 3 else "#"]) & (k_ == 0)}
        if task.k == 3:
            masks["c-run"] = np.isin(phase, ["c", "#"]) & (j_ == i_) & (k_ >= 1)
        units = [("hid", u, H[:, u]) for u in c["live_hidden_idx"]] + \
                [("res", u, R[:, u]) for u in c["live_resid_idx"]]
        nrows = []
        for kind, u, act in units:
            row = {"unit": f"{kind}{u}", "mean": float(act.mean()), "std": float(act.std())}
            for mname, m in masks.items():
                if m.sum() < 5:
                    continue
                row[f"mean_{mname}"] = float(act[m].mean())
                for fname, fv in feats.items():
                    row[f"r_{fname}_{mname}"] = corr(act[m], fv[m])
            nrows.append(row)
        # report the comparators
        for mname, fname in [("b-run", "i-j")] + ([("c-run", "i-k")] if task.k == 3 else []):
            key = f"r_{fname}_{mname}"
            top = sorted((r for r in nrows if key in r), key=lambda r: -abs(r[key]))[:8]
            say(f"  units tracking {fname} in the {mname} (|r|):  "
                + "  ".join(f"{r['unit']}:{r[key]:+.2f}" for r in top))
        for fname in ("i",):
            key = f"r_{fname}_a-run"
            top = sorted((r for r in nrows if key in r), key=lambda r: -abs(r[key]))[:6]
            say(f"  units tracking i in the a-run (the <bos>-dilution / coin carrier):  "
                + "  ".join(f"{r['unit']}:{r[key]:+.2f}" for r in top))
        # phase detectors: large mean difference between runs, low within-run std
        pd = []
        for r in nrows:
            ms = [r.get(f"mean_{m}") for m in ("a-run", "b-run", "c-run") if f"mean_{m}" in r]
            if len(ms) >= 2 and r["std"] > 1e-6:
                pd.append((max(ms) - min(ms), r["unit"], ms))
        pd.sort(reverse=True)
        say("  phase detectors (largest mean gap across runs):  "
            + "  ".join(f"{u}:[" + ",".join(f"{m:.2f}" for m in ms) + "]" for _, u, ms in pd[:6]))
        # dump curves
        if args.dump_units:
            key = f"r_i-j_b-run"
            top = sorted((r for r in nrows if key in r), key=lambda r: -abs(r[key]))[:args.dump_units]
            for r in top:
                kind, u = r["unit"][:3], int(r["unit"][3:])
                act = H[:, u] if kind == "hid" else R[:, u]
                say(f"  {r['unit']} activation vs j at fixed i (b-run):")
                for i0 in (10, 20, min(30, args.max_i)):
                    sel = (i_ == i0) & masks["b-run"]
                    js = j_[sel].astype(int); a = act[sel]
                    o = np.argsort(js)
                    say(f"    i={i0:>2}: " + " ".join(f"{a[t]:+.2f}" for t in o[:16])
                        + (" ..." if len(o) > 16 else ""))
        # ---------------- channels + program + figure
        say("\n=== 5. channels and the program ===")
        chans = find_channels(R, c["live_resid_idx"])
        say(f"  {len(c['live_resid_idx'])} live residual dims form {len(chans)} channels "
            f"(dims identical up to sign, |r|>0.995). gain = replication count.")
        chan_labels = []
        for ci, ch in enumerate(chans):
            act = R[:, ch["rep"]]
            best = []
            for mname, m in masks.items():
                if mname == "all" or m.sum() < 5:
                    continue
                for fname, fv in feats.items():
                    best.append((abs(corr(act[m], fv[m])), f"{fname}@{mname}", corr(act[m], fv[m])))
            best.sort(reverse=True)
            lab = f"{best[0][1]} r={best[0][2]:+.2f}" if best and best[0][0] > 0.5 else "phase/const"
            chan_labels.append(lab)
            mem = ", ".join(f"{'+' if s > 0 else '-'}{d}" for d, s in ch["members"])
            say(f"  ch{ci:<2} gain {ch['gain']:<2} tracks {lab:<22} dims [{mem}]")
        with torch.no_grad():
            qd = {n: (quantize(w, model.qcfg) / cfg["fixed_delta"] if cfg["scale"] == "fixed"
                      else quantize(w, model.qcfg)).detach()
                  for n, w in model.quantizable_named_weights()}
        hid_idx = c["live_hidden_idx"]
        Win = channel_weight_sums(qd["block.ff_in"][hid_idx, :], chans, axis=1)      # (nH, nC)
        Wout = channel_weight_sums(qd["block.ff_out"][:, hid_idx], chans, axis=0)    # (nC, nH)
        Whead = channel_weight_sums(qd["head"][2:, :], chans, axis=1)               # (nO, nC)
        outs = [task.id2tok[t] for t in range(2, task.vocab_size)]
        # gridded biases (in units of delta), if the checkpoint has them
        b_in = qd["block.ff_in.bias"][hid_idx].cpu().numpy() if "block.ff_in.bias" in qd else None
        b_out = (channel_weight_sums(qd["block.ff_out.bias"].view(-1, 1), chans, axis=0)[:, 0]
                 if "block.ff_out.bias" in qd else None)                           # (nC,)
        b_head = qd["head.bias"][2:].cpu().numpy() if "head.bias" in qd else None

        def lin(vec, names, b=None):
            terms = [f"{v:+.0f}*{n}" for v, n in zip(vec, names) if abs(v) > 1e-9]
            if b is not None and abs(b) > 1e-9:
                terms.append(f"{b:+.0f}")
            return " ".join(terms) if terms else "0"
        cn = [f"ch{i}" for i in range(len(chans))]
        say("  hidden units (pre-activation = sum of channel weights x channel value"
            + (" + bias" if b_in is not None else "") + ", in units of delta; then ReLU):")
        for h, u in enumerate(hid_idx):
            r = next((r for r in nrows if r["unit"] == f"hid{u}"), {})
            role = ""
            for key, tag in ((f"r_i-j_b-run", "i-j"), (f"r_i-k_c-run", "i-k"), (f"r_i_a-run", "i")):
                if key in r and abs(r[key]) > 0.6:
                    role += f" [{tag}: r={r[key]:+.2f}]"
            say(f"    h{u:<4} = ReLU( {lin(Win[h], cn, None if b_in is None else b_in[h])} ){role}")
            wr = Wout[:, h]
            if np.abs(wr).sum() > 0:
                say(f"           writes {lin(wr, cn)}")
        if b_out is not None and np.abs(b_out).sum() > 0:
            say(f"  ff_out bias writes {lin(b_out, cn)} into the residual every loop "
                f"(a position-independent constant: the anchor the <bos> share used to supply)")
        say("  output logits (from channels, via the head" + (" + head bias" if b_head is not None else "")
            + "; LN_f gain omitted):")
        for o, name in enumerate(outs):
            say(f"    logit({name}) = {lin(Whead[o], cn, None if b_head is None else b_head[o])}")
        # curves along one full string, i0 = min(20, max_i)
        i0 = min(20, args.max_i)
        seq_pts = [((i0, j) + (0,) * (task.k - 2)) for j in range(0, i0 + 1)]
        if task.k == 3:
            seq_pts += [(i0, i0, k) for k in range(1, i0 + 1)]
        pos = {tuple(counts[t]): n for n, t in enumerate(on)}
        xs = list(range(len(seq_pts)))
        series = []
        for ci, ch in enumerate(chans):
            ys = [R[pos[p], ch["rep"]] if p in pos else np.nan for p in seq_pts]
            series.append((chan_labels[ci], ys))
        marks = [(0, "a-run end"), (i0 + 0.5 if i0 else 0, "b-run")]
        if task.k == 3:
            marks.append((2 * i0 + 0.5, "c-run"))
        fig_path = f"{stem}_circuit.png"
        try:
            draw_circuit(fig_path, task, chans, chan_labels, (xs, marks, series), hid_idx,
                         Win, Wout, Whead, cfg["fixed_delta"],
                         f"{os.path.basename(stem)}: L0={c['L0']}, {len(chans)} channels, "
                         f"{len(hid_idx)} live units")
            say(f"  -> {fig_path}")
        except Exception as e:
            say(f"  (figure failed: {e})")
        path_n = f"{stem}_neurons.csv"
        keys = sorted({k for r in nrows for k in r}, key=lambda s: (s != "unit", s))
        with open(path_n, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader(); w.writerows(nrows)
        say(f"  -> {path_n}")

    with open(f"{stem}_anatomy.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nreport -> {stem}_anatomy.txt")


if __name__ == "__main__":
    main()
