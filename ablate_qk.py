"""
ablate_qk.py
============
Is a small counter's attention really a prefix mean? Zero Wq and Wk (attention
becomes exactly uniform over the causal prefix), drop the weights they held,
and re-score with the audit's own metrics. If greedy_n / exact_n survive, the
Wq/Wk weights are incidental and the circuit belongs to the uniform-attention
family (and is cheaper by those weights). If counting dies, attention is doing
content-selective work and the circuit is a different mechanism.

Usage:
  python ablate_qk.py --ckpt runs/int8_settled/softmax_noln_int8_wd1.0_seed100/polish_guarded_bits15_2.pt
"""
from __future__ import annotations

import argparse
import copy
import math

import torch

from audit_counters import coin_profile, counter_profile, probe_ns, train_bits
from grammar import make_train_set, task_from_config
from model import LoopedTransformer
from quantization import QuantConfig, quantize


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    task = task_from_config(cfg)
    qcfg = QuantConfig(mode=cfg["quant"], scale=cfg["scale"], fixed_delta=cfg["fixed_delta"])
    model = LoopedTransformer.from_config(cfg, task.vocab_size, qcfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg, task


def score(model, cfg, task, device, max_n, fail_bits):
    ns = probe_ns(max_n)
    prof = counter_profile(model, ns, device, cfg["p_geom"], task)

    def rng(get):
        e = 0
        for n in ns:
            if get(n) > fail_bits:
                break
            e = n
        return e

    train = make_train_set(cfg["train_size"], seed=cfg["seed"], p=cfg["p_geom"], task=task)
    coin = coin_profile(model, task, device)
    ood = [k for k in coin if k > train.n_max]
    return {
        "greedy_n": rng(lambda n: 0.0 if prof[n][3] else 1.0),
        "exact_n": rng(lambda n: prof[n][1]),
        "c50": round(prof[50][1], 2) if 50 in prof else float("nan"),
        "coin_dev": max(abs(coin[k] - cfg["p_geom"]) for k in ood) if ood else float("nan"),
        "DH_bits": round(train_bits(model, train.seqs, device)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_n", type=int, default=300)
    ap.add_argument("--fail_bits", type=float, default=0.5)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, task = load(args.ckpt, device)

    q = {n: quantize(w, model.qcfg) for n, w in model.quantizable_named_weights()}
    L0 = sum(int((w != 0).sum()) for w in q.values())
    n_qk = sum(int((q[n] != 0).sum()) for n in ("block.attn.wq", "block.attn.wk"))
    m_lv = getattr(model.qcfg, "n_levels_per_sign", None) or 1
    bpw = math.log2(model.param_report()["quantizable_params"]) + math.log2(2 * m_lv)
    print(f"{args.ckpt}\n  L0 = {L0}, of which Wq+Wk = {n_qk}  "
          f"(|H| {L0*bpw:.0f} -> {(L0-n_qk)*bpw:.0f} bits if they go)")

    abl = copy.deepcopy(model)
    with torch.no_grad():
        for n, w in abl.quantizable_named_weights():
            base = n[:-len(".bias")] if n.endswith(".bias") else n
            if base.endswith("wq") or base.endswith("wk"):
                w.zero_()

    print(f"\n  {'variant':22s} {'greedy_n':>8} {'exact_n':>7} {'c50':>6} {'coin_dev':>8} {'|D:H|':>7}")
    for name, m in (("original", model), ("Wq = Wk = 0", abl)):
        s = score(m, cfg, task, device, args.max_n, args.fail_bits)
        print(f"  {name:22s} {s['greedy_n']:>8} {s['exact_n']:>7} {s['c50']:>6} "
              f"{s['coin_dev']:>8.3f} {s['DH_bits']:>7}")


if __name__ == "__main__":
    main()
