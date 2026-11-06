"""
measure_act_range.py
=====================
Diagnostic for choosing --act_max: measures the ACTUAL max-abs activation at
every point fake_quant_act() would clip, on a real (normally fp32) checkpoint,
across a range of string lengths including long ones. Run this BEFORE an
--act_bits sweep on any new architecture/regime -- act_max must cover the
natural dynamic range, or clipping (not bit depth) will dominate the result,
as happened in the first --act_bits sweep on the dense nope-wd0 regime.

Usage:
  python measure_act_range.py --ckpt runs/rule/nope_fixed_wd1.0_seed100/model.pt --max_n 300
  python measure_act_range.py --ckpt runs/pos/ternary_nope_seed102/model.pt --max_n 300
"""
import argparse
import torch
import torch.nn.functional as F

from grammar import collate, task_from_config
from model import LoopedTransformer, sinusoidal_pe
from quantization import QuantConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_n", type=int, default=300)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]
    task = task_from_config(cfg)
    qcfg = QuantConfig(mode=cfg["quant"], scale=cfg["scale"], fixed_delta=cfg["fixed_delta"])
    model = LoopedTransformer.from_config(cfg, task.vocab_size, qcfg, act_bits=0).to(device)   # force fp32 activations: measure the UNCLIPPED range
    model.load_state_dict(ck["state_dict"])
    model.eval()

    points = {}
    def rec(name):
        def hook(x):
            points[name] = max(points.get(name, 0.0), float(x.abs().max()))
            return x
        return hook

    ns = sorted(set([1, 5, 10, 20] + list(range(25, args.max_n + 1, 25)) + [args.max_n]))
    with torch.no_grad():
        for n in ns:
            inp, _ = collate([task.string_for_n(n)], device)
            x = model.emb(inp) * (model.d ** 0.5)
            if model.pos == "sin":
                x = x + sinusoidal_pe(inp.shape[1], model.d, x.device, x.dtype)
            rec("post_emb+pe")(x)
            blk = model.block
            for i in range(model.n_loops):
                if i < model.attn_loops:
                    h1 = blk.ln1(x); rec("ln1_out")(h1)
                    x = x + blk.attn(h1); rec("post_attn_resid")(x)
                h2 = blk.ln2(x); rec("ln2_out")(h2)
                x = x + blk.ff_out(F.relu(blk.ff_in(h2))); rec("post_ffn_resid")(x)
            xf = model.ln_f(x); rec("final_ln")(xf)

    print(f"checkpoint: {args.ckpt}  (task={task.name} quant={cfg['quant']} "
          f"scale={cfg['scale']} pos={cfg.get('pos','sin')} wd={cfg['weight_decay']})")
    print(f"max |activation| at each fake_quant_act() insertion point, n=1..{args.max_n}:")
    overall = 0.0
    for name, v in points.items():
        print(f"  {name:>18}: {v:8.3f}")
        overall = max(overall, v)
    headroom = overall * 1.25
    print(f"\noverall max: {overall:.3f}")
    print(f"suggested --act_max: {headroom:.1f}  (25% headroom over the observed max; "
          f"use the same value across the bit-depth sweep so only bits varies)")


if __name__ == "__main__":
    main()