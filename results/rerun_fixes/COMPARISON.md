# B1 / B2 / B3 fixes — baseline vs. re-run

Baseline: `results` (2026-09-01 full run + 2026-09-02 join pass). Re-run: `results/rerun_fixes`.

Every number below comes from the JSON artifacts, not from a log. A defect that changed no conclusion is still worth recording as such, so unchanged rows are reported rather than omitted.

## B1 — SAE knock-out (`sae.py::causal_knockout`)

The dictionary was fitted on `hidden_states[20]` but the hook fired on `layers[20]`, whose output is `hidden_states[21]`. The hook now fires on `layers[19]`. `S_semantic` is untouched by this defect and must reproduce exactly; `S_causal` is the quantity at risk.

| feature | concept | S_sem base | S_sem rerun | S_causal base | S_causal rerun | excess base | excess rerun |
|---|---|---|---|---|---|---|---|
| #14294 | qt_interval | 0.703 | 0.703 | 0.0016 | 0.0000 | 0.0016 | 0.0000 |
| #7626 | pregnancy | 0.615 | 0.615 | 0.0000 | 0.0000 | -0.0016 | -0.0031 |
| #14058 | creatinine | 0.603 | 0.603 | 0.0158 | 0.0250 | 0.0158 | 0.0250 |
| #11855 | age | 0.592 | 0.592 | 0.0146 | 0.0191 | 0.0146 | 0.0191 |
| #5438 | heart_rate | 0.586 | 0.586 | 0.0098 | 0.0000 | 0.0098 | -0.0164 |
| #2883 | drug | 0.541 | 0.541 | 0.0162 | 0.0205 | 0.0162 | 0.0205 |
| #7679 | asthma | 0.538 | 0.538 | 0.0000 | 0.0000 | -0.0016 | -0.0016 |
| #16231 | age | 0.520 | 0.520 | 0.0156 | 0.0139 | 0.0156 | 0.0139 |
| #7696 | age | 0.519 | 0.519 | — | — | — | — |
| #3446 | drug | 0.511 | 0.511 | — | — | — | — |
| #4754 | age | 0.501 | 0.501 | — | — | — | — |
| #13369 | age | 0.494 | 0.494 | — | — | — | — |
| #6550 | age | 0.490 | 0.490 | — | — | — | — |
| #11615 | qt_interval | 0.480 | 0.480 | — | — | — | — |
| #11193 | drug | 0.477 | 0.477 | — | — | — | — |
| #6673 | heart_rate | 0.476 | 0.476 | — | — | — | — |
| #15148 | drug | 0.445 | 0.445 | — | — | — | — |
| #9285 | drug | 0.444 | 0.444 | — | — | — | — |
| #7910 | age | 0.441 | 0.441 | — | — | — | — |
| #64 | age | 0.438 | 0.438 | — | — | — | — |
| #13629 | creatinine | 0.435 | 0.435 | — | — | — | — |
| #14695 | inr | 0.418 | 0.418 | — | — | — | — |
| #11033 | creatinine | 0.418 | 0.418 | — | — | — | — |
| #1713 | age | 0.414 | 0.414 | — | — | — | — |
| #1052 | age | 0.413 | 0.413 | — | — | — | — |

`S_semantic` reproduced exactly on every feature, as it must — the fix was confined to the causal stage.

Largest |excess| over a matched random control: baseline **0.0162**, rerun **0.0250** logits. A decision flip needs ~1–5 logits.

## B2 — Steering sign and layer alignment (`steering.py`)

The direction points toward UNSAFE and was applied with a negated coefficient over a non-negative sweep, so every steer pushed toward SAFE on a model already answering SAFE on all items. The sweep is now two-sided and the hook fires on the module that produced the representation the direction was fitted on.

> **The two `alpha` columns do not mean the same thing.** The baseline recorded the alpha it was *asked* for while applying its negation, so a baseline row logged at `+4.00×` was physically a steer of `-4.00×` — toward SAFE. Every baseline alpha below should be read with its sign flipped. That is the defect, not a reporting choice, and it is why the baseline's 'positive' arm is not comparable to the rerun's.

| run | cells swept | alphas (as recorded) | best cell | CC | random CC | gain |
|---|---|---|---|---|---|---|
| baseline | 24 | +0.00 … +8.00 | L15 @ +4.00× | 0.106 | 0.000 | +0.106 |
| rerun | 52 | -8.00 … +8.00 | L15 @ -4.00× | 0.094 | 0.012 | +0.082 |

**The arm that did not exist before.** Positive alpha steers toward UNSAFE; that arm is the one capable of moving a model stuck on SAFE.

| arm | best gain over matched random control | at |
|---|---|---|
| toward UNSAFE (alpha > 0) — **never tested before** | +0.024 | L5 @ +8.00× |
| toward SAFE (alpha < 0) — the only arm the baseline actually applied | +0.082 | L15 @ -4.00× |

Item accuracies observed across the rerun sweep: [0.482, 0.488, 0.5, 0.506, 0.512, 0.518, 0.524, 0.535] …. In the baseline every cell sat at 0.500 (the model answering SAFE on everything); accuracy moving off 0.500 is the direct signature that steering now reaches the decision.

## B3 — Post-final-norm donor at the last layer (`patching.py`)

`hidden_states[n_layers]` is emitted after `model.norm`, so the deepest ACE row wrote a normed tensor into a pre-norm residual stream. `Patcher.run` now captures the last decoder layer's true output and substitutes it. **Only the final row was affected** — every other row is a regression check.

### MedCalc test (renal / creatinine)

> ⚠️ **Not a clean A/B.** The baseline used 80 pairs and this re-run used 86. The patching baselines predate the `data/medcalc` re-partition (FIXES.md A4), so these two runs differ by the dataset as well as by the B3 fix. **The re-run numbers are the valid ones**; the row-by-row regression check below cannot be read as evidence either way and is reported for completeness only. To get a clean A/B, re-run the baseline command on today's data with the fix reverted.

- pairs used: 80 → 86
- identity control max |err|: 0.0000 → 0.0000 (must be ~0 in both; it is the hook-placement check)
- **last layer (31) ACE: 0.0000 → 0.0000**, excess 0.0000 → 0.0000
- 31 non-final row(s) differ, which is expected here: the item sets are not the same (see the warning above), so this is not a regression signal.
- largest |excess| over control, any layer: 0.0086 → 0.0064 logits

### MedCalc held-out (QT / ondansetron)

> ⚠️ **Not a clean A/B.** The baseline used 80 pairs and this re-run used 85. The patching baselines predate the `data/medcalc` re-partition (FIXES.md A4), so these two runs differ by the dataset as well as by the B3 fix. **The re-run numbers are the valid ones**; the row-by-row regression check below cannot be read as evidence either way and is reported for completeness only. To get a clean A/B, re-run the baseline command on today's data with the fix reverted.

- pairs used: 80 → 85
- identity control max |err|: 0.0000 → 0.0000 (must be ~0 in both; it is the hook-placement check)
- **last layer (31) ACE: 0.0000 → 0.0000**, excess -0.0102 → 0.0000
- 31 non-final row(s) differ, which is expected here: the item sets are not the same (see the warning above), so this is not a regression signal.
- largest |excess| over control, any layer: 0.0211 → 0.0221 logits

## B3 (cont.) — Sufficiency / injection sweeps

`sufficiency` shares `Patcher.run`, so it inherits the fix. Its default layer list skips the last layer, so these are expected to reproduce exactly; a change here means `--layers` reached the affected row.

- **MedCalc test**: largest |excess| 0.0156 → 0.0134; monotone layers 0 → 0; cells changed: 0/32; alpha=0 control 0.0000 → 0.0000
- **MedCalc held-out**: largest |excess| 0.0604 → 0.0833; monotone layers 3 → 3; cells changed: 0/32; alpha=0 control 0.0000 → 0.0000

