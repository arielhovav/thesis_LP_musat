"""
evaluate.py
===========
Evaluation exactly in the currency of Abudy et al.: |D:H| in bits, compared
to the analytical optimum, with Delta% = (|D:H| - Optimal)/Optimal * 100.

Reported quantities per evaluation:
  train_bits        avg -log2 q(s) per training string
  train_delta_pct   vs the empirical training optimum (negative => overfit)
  test_bits         probability-weighted -log2 q(s) over the exhaustive set
  test_delta_pct    vs the analytical optimum (the headline number)
  ood_bits/delta    combined n > n_train_max (kept for older runs), split into:
  near_ood_*        n_train_max < n <= n_train_max+10 -- the VALIDATION slice,
                    the only OOD number you may use to pick checkpoints/HPs
  far_ood_*         n > n_train_max+10 -- REPORT-ONLY length generalization

Also usable as a script to evaluate a saved checkpoint:
    python evaluate.py --ckpt runs/xyz/model.pt
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from grammar import PAD, TestSet, TrainSet, collate

LN2 = math.log(2.0)


@torch.no_grad()
def bits_per_sequence(model, seqs, device, batch_size: int = 128,
                      smooth: float = 1e-10) -> np.ndarray:
    """Total -log2 q(s) for each sequence s (sum over its tokens).

    Following the paper (Section 2.6), output probabilities are smoothed by
    adding 1e-10 before the log so that a zero probability on the correct
    symbol yields a large finite score instead of infinity.
    """
    model.eval()
    out = np.zeros(len(seqs), dtype=np.float64)
    order = np.argsort([len(s) for s in seqs])  # bucket by length
    for start in range(0, len(seqs), batch_size):
        idx = order[start : start + batch_size]
        inp, tgt = collate([seqs[i] for i in idx], device)
        logits = model(inp)
        logp = F.log_softmax(logits.float(), dim=-1)
        p = logp.exp().clamp_min(smooth)                    # smoothing
        nll2 = -(p.log() / LN2)                             # bits
        tok_bits = nll2.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        tok_bits = tok_bits * (tgt != PAD)                  # mask padding
        seq_bits = tok_bits.sum(dim=1).cpu().numpy()
        out[idx] = seq_bits
    return out


@torch.no_grad()
def full_eval(model, train: TrainSet, test: TestSet, device,
              batch_size: int = 128) -> dict:
    tr = bits_per_sequence(model, train.seqs, device, batch_size)
    train_bits = float(tr.mean())
    tr_opt = train.optimal_bits_per_string
    te = bits_per_sequence(model, test.seqs, device, batch_size)
    test_bits = float(np.sum(test.weights * te))
    in_mask, near_mask, far_mask = test.split_masks()

    def _weighted(mask):
        w = test.weights[mask]
        if w.sum() <= 0:
            return float("nan"), float("nan")
        w = w / w.sum()
        got = float(np.sum(w * te[mask]))
        ns = np.asarray(test.ns)[mask]
        opt = float(np.sum(w * np.array(
            [-(math.log2(test.p) + (n - 1) * math.log2(1 - test.p)) for n in ns])))
        return got, (got - opt) / opt * 100.0

    in_bits, in_delta = _weighted(in_mask)
    near_bits, near_delta = _weighted(near_mask)
    far_bits, far_delta = _weighted(far_mask)
    ood_bits, ood_delta = _weighted(near_mask | far_mask)   # legacy combined
    return {
        "train_bits": train_bits,
        "train_delta_pct": (train_bits - tr_opt) / tr_opt * 100.0,
        "test_bits": test_bits,
        "test_delta_pct": (test_bits - test.optimal_bits) / test.optimal_bits * 100.0,
        "test_optimal_bits": test.optimal_bits,
        "in_range_bits": in_bits,
        "in_range_delta_pct": in_delta,
        # near-OOD = validation slice (usable for selection); far-OOD = held out
        "near_ood_bits": near_bits,
        "near_ood_delta_pct": near_delta,
        "far_ood_bits": far_bits,
        "far_ood_delta_pct": far_delta,
        # combined n > n_train_max, kept for compatibility with older runs
        "ood_bits": ood_bits,
        "ood_delta_pct": ood_delta,
    }


# ------------------------------- CLI -----------------------------------------
if __name__ == "__main__":
    import argparse
    import json

    from grammar import make_exhaustive_test_set, make_train_set, task_from_config
    from model import LoopedTransformer
    from quantization import QuantConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_extra", type=int, default=None)
    ap.add_argument("--dump_structure", action="store_true",
                    help="also write the non-zero quantized weights to "
                         "structure.txt next to the checkpoint (any L0)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]
    task = task_from_config(cfg)          # 'anbn' for pre---task checkpoints
    qcfg = QuantConfig(mode=cfg["quant"], scale=cfg["scale"],
                       fixed_delta=cfg["fixed_delta"])
    model = LoopedTransformer.from_config(cfg, task.vocab_size, qcfg).to(device)
    model.load_state_dict(ck["state_dict"])

    train = make_train_set(cfg["train_size"], seed=cfg["seed"], p=cfg["p_geom"],
                           task=task)
    extra = args.test_extra if args.test_extra is not None else cfg["test_extra"]
    test = make_exhaustive_test_set(train.n_max, extra=extra, p=cfg["p_geom"],
                                    task=task)
    print(f"task = {task.name}")

    print(json.dumps(full_eval(model, train, test, device), indent=2))
    print(json.dumps(model.sparsity_report(), indent=2, default=str))

    if args.dump_structure:
        import os
        path = os.path.join(os.path.dirname(args.ckpt) or ".", "structure.txt")
        with open(path, "w") as f:
            f.write(model.dump_nonzero_structure())
        print(f"non-zero structure written to {path}")
