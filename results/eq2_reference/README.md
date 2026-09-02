# Eq. (2) reference results (whole-vocabulary entropy)

Produced by the 2026-09-01 full pipeline run, when `run_eval.py --uq_signal`
defaulted to `entropy` -- the proposal's Eq. (2) exactly as written.

Kept because the default changed to `decision_entropy` on 2026-09-01 after the
n=6,456 external measurement (AUROC 0.525 [0.511, 0.539] for Eq. (2) against
0.687 [0.675, 0.700] restricted to the decision tokens; see
`results/bigbench_uq.md`). These files are the "before", so the respecification
of section 4.7 can be shown rather than asserted.

Reproduce any of them with `--uq_signal entropy`.
