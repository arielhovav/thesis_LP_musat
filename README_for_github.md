# a^n b^n under fixed-precision MDL

Experiments on whether weight decay behaves like Kolmogorov complexity / MDL
(Musat 2026) in a looped transformer trained on a^n b^n and a^n b^n c^n,
evaluated with the protocol of Abudy et al. (2025).

- `SUMMARY.md` — current state: golden networks, all results, remaining runs.
- `anbn_counters.html` — the nine counter circuits, drawn with worked traces.
- `handoff_v2_bias_patch.md`, `handoff_v3_digital.md`, `predictions_*.md` — run plans and pre-registrations.
- Code: `train.py` (SGD/AdamW on fixed-precision grids), `ga_polish.py` (MDL search with five value codes),
  `audit_counters.py`, `anatomize2.py`, `precision_mdl.py`, `golden.py`, `golden_digital.py`, `sweep.py`,
  `polish_sweep.py`, `compare_levels.py`, `level_hist.py`; `lmexp/` is the small-LM weight-decay experiment.
- `golden*.pt` — the hand-built and MDL-found counters (tracked; everything under `runs/` is not).

Setup: Python 3.12, `pip install torch numpy matplotlib`. Every command in the handoffs is PowerShell from this directory.
