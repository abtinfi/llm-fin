## Ablation matrix -- split=`test`, model=`BioMistral/BioMistral-7B`, seeds=[0]

UQ engine defers on **`decision_entropy`**  (Eq. (2) restricted to the decision tokens, the respecified 4.7 term: AUROC 0.687 [0.675, 0.700] over 6,456 items. Rank-equivalent to `logit_margin` for K=2.)

Single seed. Decoding is greedy and therefore deterministic: another seed reproduces this run exactly, so no seed-to-seed spread is reported. The uncertainty that does exist is over items, and is given as a bootstrap CI in the next table.

### Primary: the section 4.6 ablation ladder

| Variant | Base | RAG | Sym | UQ | CL | Accuracy (strict) | Δ Acc vs base | Model's own answer acc. | Causal Consistency | Δ CC vs base | Violation Rate | Coverage | Gate fired |
|---|:-:|:-:|:-:|:-:|:-:|---|---|---|---|---|---|---|---|
| (1) Base LLM | YES | - | - | - | - | 0.520 | +0.000 | 0.520 | 0.026 | +0.000 | 0.268 | 1.000 | 0.000 |
| (2) + RAG | YES | YES | - | - | - | 0.467 | -0.054 | 0.467 | 0.000 | -0.026 | 0.010 | 1.000 | 0.000 |
| (3) + Symbolic Gate (NS-AI) | YES | YES | YES | - | - | 0.942 † | +0.422 | 0.467 | 0.861 † | +0.835 | 0.010 | 1.000 | 0.884 |
| (4) + UQ Engine (NS-AI+UQ) | YES | YES | YES | YES | - | 0.891 † | +0.371 | 0.467 | 0.861 † | +0.835 | 0.005 | 0.902 | 0.884 |

**Model's own answer acc.** is the decision word parsed from the model's generation, before the gate overrides it and before UQ defers it: what the LLM itself concluded under that row's prompt. † Accuracy and CC on gate-fired items are an identity check, not a measurement: the gate applies the rule and threshold the labels were generated from. With the gate firing on every item, that row's accuracy is 1.000 by construction; read the model's own column, and the adherence table below, for what the model knows.


### Supplementary ablations: one contribution added to the base model

| Variant | Base | RAG | Sym | UQ | CL | Accuracy (strict) | Δ Acc vs base | Model's own answer acc. | Causal Consistency | Δ CC vs base | Violation Rate | Coverage | Gate fired |
|---|:-:|:-:|:-:|:-:|:-:|---|---|---|---|---|---|---|---|
| (5) Base + Symbolic Gate only | YES | - | YES | - | - | 0.942 † | +0.422 | 0.520 | 0.870 † | +0.843 | 0.029 | 1.000 | 0.884 |
| (6) Base + UQ only | YES | - | - | YES | - | 0.000 | -0.520 | 0.520 | 0.000 | -0.026 | 0.000 | 0.000 | 0.000 |
| (7) Base + Constraint Layer only | YES | - | - | - | YES | 0.487 | -0.033 | 0.487 | 0.026 | +0.000 | 0.117 | 1.000 | 0.000 |
| (8) All four contributions | YES | YES | YES | YES | YES | 0.895 † | +0.375 | 0.460 | 0.861 † | +0.835 | 0.005 | 0.915 | 0.884 |

### 95% bootstrap CI over items (seed 0)

| Variant | Accuracy | 95% CI | Causal Consistency | 95% CI |
|---|---|---|---|---|
| (1) Base LLM | 0.520 | [0.473, 0.567] | 0.026 | [0.000, 0.061] |
| (2) + RAG | 0.467 | [0.420, 0.513] | 0.000 | [0.000, 0.000] |
| (3) + Symbolic Gate (NS-AI) | 0.942 | [0.920, 0.962] | 0.861 | [0.791, 0.922] |
| (4) + UQ Engine (NS-AI+UQ) | 0.891 | [0.859, 0.920] | 0.861 | [0.791, 0.922] |
| (5) Base + Symbolic Gate only | 0.942 | [0.920, 0.962] | 0.870 | [0.809, 0.930] |
| (6) Base + UQ only | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] |
| (7) Base + Constraint Layer only | 0.487 | [0.440, 0.533] | 0.026 | [0.000, 0.061] |
| (8) All four contributions | 0.895 | [0.866, 0.922] | 0.861 | [0.791, 0.922] |

### Paired McNemar of each contribution **against the baseline** (seed 0, item-level correctness)

This is the supervisor's question: what does adding this contribution to the base model do?

| Variant vs base | B01 (base wrong→right) | B10 (base right→wrong) | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 57 | 81 | 0.04985 | 0.0997 |
| base→nsai | 198 | 9 | 1.641e-47 | 8.207e-47 |
| base→nsai_uq | 190 | 24 | 3.118e-33 | 9.355e-33 |
| base→sym | 190 | 1 | 1.223e-55 | 7.341e-55 |
| base→uq | 0 | 233 | 1.449e-70 | 1.014e-69 |
| base→cl | 54 | 69 | 0.2066 | 0.2066 |
| base→nsai_uq_cl | 192 | 24 | 9.858e-34 | 3.943e-33 |

### Paired McNemar along the cumulative ladder (seed 0)

| Comparison | B01 | B10 | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 57 | 81 | 0.04985 | 0.04985 |
| rag→nsai | 214 | 1 | 8.204e-63 | 2.461e-62 |
| nsai→nsai_uq | 0 | 23 | 2.384e-07 | 4.768e-07 |

### Where the gate's contribution comes from (seed 0)

The gate's accuracy on items it fires on is partly circular: it applies the same rule and threshold the labels were generated from, and on the MedCalc benchmark the same extractor that filters the data runs inside the gate. Splitting the split by whether the gate fired separates the circular part from the part that is not: on gate-declined items the row IS the neural pathway, so any difference there is real.

On gate-fired items the three accuracy columns separate the two things the question conflates. *Final* is the rule checking itself (identity). *Model's own answer* is whether the LLM, given that row's prompt, reached the guideline's conclusion without the rule's help: that is guideline adherence. *Agrees with gate* is how often the gate merely confirmed the model rather than overruled it.

| Variant | subset | n | Final accuracy | Model's own answer acc. | Base accuracy, same items | Model agrees with gate |
|---|---|---|---|---|---|---|
| (3) + Symbolic Gate (NS-AI) | gate fired | 396 | 0.997 (identity) | 0.460 | 0.520 | 0.457 |
| (3) + Symbolic Gate (NS-AI) | gate declined | 52 | 0.519 | 0.519 | 0.519 | — |
| (4) + UQ Engine (NS-AI+UQ) | gate fired | 396 | 0.997 (identity) | 0.460 | 0.520 | 0.457 |
| (4) + UQ Engine (NS-AI+UQ) | gate declined | 52 | 0.077 | 0.519 | 0.519 | — |
| (5) Base + Symbolic Gate only | gate fired | 396 | 0.997 (identity) | 0.520 | 0.520 | 0.518 |
| (5) Base + Symbolic Gate only | gate declined | 52 | 0.519 | 0.519 | 0.519 | — |
| (8) All four contributions | gate fired | 396 | 0.997 (identity) | 0.455 | 0.520 | 0.452 |
| (8) All four contributions | gate declined | 52 | 0.115 | 0.500 | 0.519 | — |

### Why a UQ row can read 0.000 coverage

Split conformal picks the largest uncertainty threshold whose error rate on the calibration split is at most alpha = 0.10. On this calibration split of 124 items no threshold reaches that target (the risk-coverage table below shows how far off it is on these items). The method then falls back to its most conservative threshold, which retains 2.4% of the calibration items, and on the `test` split retains none. **That is the method behaving correctly, not a failure to run**: a 10% error target is unreachable for a model at this accuracy, so the only way to honour it is to answer nothing. It is also the exact situation Adaptive Conformal Inference exists for -- see `results/uq_coverage_*.md`, where the threshold is allowed to move.


### Risk-coverage of the UQ signal, (4) + UQ Engine (NS-AI+UQ) (seed 0)

Every threshold a deferral rule could pick, on the 52 items the UQ engine governs here (gate-decided items are never deferred and are excluded). Error is of the model's own answer on the items kept. The signal (`decision_entropy`) takes 17 distinct values, so 17 operating points exist; the rows below are those nearest each coverage level. Full curve: `table_medcalc2_test_riskcov_nsai_uq.csv`.

| Coverage target | Threshold | Coverage | Error | Items kept |
|---|---|---|---|---|
| 100% | inf | 1.000 | 0.481 | 52 |
| 90% | 0.6628 | 0.923 | 0.500 | 48 |
| 75% | 0.6275 | 0.750 | 0.487 | 39 |
| 50% | 0.5303 | 0.500 | 0.385 | 26 |
| 25% | 0.4332 | 0.269 | 0.571 | 14 |
| 10% | 0.2577 | 0.096 | 0.200 | 5 |
| 5% | 0.1825 | 0.058 | 0.000 | 3 |
| 1% | 0.1825 | 0.058 | 0.000 | 3 |

- AURC (area under the risk-coverage curve, lower is better): **0.441**; a signal that ranks at random scores the full-coverage error, 0.481.
- Lowest error at ≥1% coverage: **0.000** (coverage 0.058).
- Target error α = 0.10: reachable on these items up to coverage **0.058**.
- Deployed threshold τ = 0.3523, set on the calibration split (coverage there 0.083); here it keeps 0.902 of all items.

### Risk-coverage of the UQ signal, (6) Base + UQ only (seed 0)

Every threshold a deferral rule could pick, on the 448 items the UQ engine governs here (gate-decided items are never deferred and are excluded). Error is of the model's own answer on the items kept. The signal (`decision_entropy`) takes 21 distinct values, so 21 operating points exist; the rows below are those nearest each coverage level. Full curve: `table_medcalc2_test_riskcov_uq.csv`.

| Coverage target | Threshold | Coverage | Error | Items kept |
|---|---|---|---|---|
| 100% | 0.6931 | 1.000 | 0.480 | 448 |
| 90% | 0.6927 | 0.953 | 0.482 | 427 |
| 75% | 0.6888 | 0.723 | 0.488 | 324 |
| 50% | 0.6759 | 0.504 | 0.504 | 226 |
| 25% | 0.6466 | 0.257 | 0.548 | 115 |
| 10% | 0.6059 | 0.116 | 0.577 | 52 |
| 5% | 0.5697 | 0.045 | 0.450 | 20 |
| 1% | 0.5437 | 0.011 | 0.400 | 5 |

- AURC (area under the risk-coverage curve, lower is better): **0.504**; a signal that ranks at random scores the full-coverage error, 0.480.
- Lowest error at ≥1% coverage: **0.357** (coverage 0.031).
- Target error α = 0.10: **not reachable at any threshold** on these items -- so a rule that must honour α can only abstain.
- Deployed threshold τ = 0.5029, set on the calibration split (coverage there 0.024); here it keeps 0.000 of all items.

### Risk-coverage of the UQ signal, (8) All four contributions (seed 0)

Every threshold a deferral rule could pick, on the 52 items the UQ engine governs here (gate-decided items are never deferred and are excluded). Error is of the model's own answer on the items kept. The signal (`decision_entropy`) takes 19 distinct values, so 19 operating points exist; the rows below are those nearest each coverage level. Full curve: `table_medcalc2_test_riskcov_nsai_uq_cl.csv`.

| Coverage target | Threshold | Coverage | Error | Items kept |
|---|---|---|---|---|
| 100% | inf | 1.000 | 0.500 | 52 |
| 90% | 0.6466 | 0.904 | 0.511 | 47 |
| 75% | 0.617 | 0.712 | 0.486 | 37 |
| 50% | 0.5822 | 0.558 | 0.552 | 29 |
| 25% | 0.4611 | 0.231 | 0.500 | 12 |
| 10% | 0.3395 | 0.096 | 0.400 | 5 |
| 5% | 0.1668 | 0.077 | 0.250 | 4 |
| 1% | 0.1668 | 0.077 | 0.250 | 4 |

- AURC (area under the risk-coverage curve, lower is better): **0.491**; a signal that ranks at random scores the full-coverage error, 0.500.
- Lowest error at ≥1% coverage: **0.250** (coverage 0.077).
- Target error α = 0.10: **not reachable at any threshold** on these items -- so a rule that must honour α can only abstain.
- Deployed threshold τ = 0.4751, set on the calibration split (coverage there 0.167); here it keeps 0.915 of all items.

### Notes

- Causal Consistency is pair-level: both counterfactual arms must be correct. A constant-answer model scores 0.
- Abstentions count as failures for accuracy and causal consistency, and as non-violations for violation rate. Coverage is reported so this trade-off is visible.
- **Accuracy (strict)** counts an abstention as wrong and is the row-comparable number. Selective accuracy is in `summary_*.json`; it rises trivially as coverage falls and must never be compared across rows without coverage.
- Rows (5)-(7) add ONE contribution to the base model. Rows (1)-(4) are the cumulative ladder of the proposal's section 4.6. Row (8) is everything at once.
- **The gate's accuracy is an upper bound.** It evaluates the same constraint the ground truth was generated from, so on items it fires on it cannot be wrong unless extraction is. The figure to quote next to it is its firing rate here, and its coverage on unfiltered real notes (42.7%, `results/medcalc_ablation.md`).
- The constraint layer is scored through the identical readout as every other row (generate, then parse the first decision word), not by argmax over answer logits, so the numbers are comparable down the column.
