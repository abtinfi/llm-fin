## Ablation matrix -- split=`test`, model=`aaditya/Llama3-OpenBioLLM-8B`, seeds=[0]

UQ engine defers on **`decision_entropy`**  (Eq. (2) restricted to the decision tokens, the respecified 4.7 term: AUROC 0.687 [0.675, 0.700] over 6,456 items. Rank-equivalent to `logit_margin` for K=2.)

Single seed. Decoding is greedy and therefore deterministic: another seed reproduces this run exactly, so no seed-to-seed spread is reported. The uncertainty that does exist is over items, and is given as a bootstrap CI in the next table.

### Primary: the section 4.6 ablation ladder

| Variant | Base | RAG | Sym | UQ | CL | Accuracy (strict) | Δ Acc vs base | Model's own answer acc. | Causal Consistency | Δ CC vs base | Violation Rate | Coverage | Gate fired |
|---|:-:|:-:|:-:|:-:|:-:|---|---|---|---|---|---|---|---|
| (1) Base LLM | YES | - | - | - | - | 0.668 | +0.000 | 0.668 | 0.004 | +0.000 | 0.963 | 1.000 | 0.000 |
| (2) + RAG | YES | YES | - | - | - | 0.635 | -0.032 | 0.635 | 0.041 | +0.037 | 0.816 | 1.000 | 0.000 |
| (3) + Symbolic Gate (NS-AI) | YES | YES | YES | - | - | 0.752 † | +0.085 | 0.635 | 0.363 † | +0.359 | 0.609 | 1.000 | 0.423 |
| (4) + UQ Engine (NS-AI+UQ) | YES | YES | YES | YES | - | 0.395 † | -0.272 | 0.635 | 0.348 † | +0.344 | 0.088 | 0.423 | 0.423 |

**Model's own answer acc.** is the decision word parsed from the model's generation, before the gate overrides it and before UQ defers it: what the LLM itself concluded under that row's prompt. † Accuracy and CC on gate-fired items are an identity check, not a measurement: the gate applies the rule and threshold the labels were generated from. With the gate firing on every item, that row's accuracy is 1.000 by construction; read the model's own column, and the adherence table below, for what the model knows.


### Supplementary ablations: one contribution added to the base model

| Variant | Base | RAG | Sym | UQ | CL | Accuracy (strict) | Δ Acc vs base | Model's own answer acc. | Causal Consistency | Δ CC vs base | Violation Rate | Coverage | Gate fired |
|---|:-:|:-:|:-:|:-:|:-:|---|---|---|---|---|---|---|---|
| (5) Base + Symbolic Gate only | YES | - | YES | - | - | 0.768 † | +0.101 | 0.668 | 0.351 † | +0.347 | 0.673 | 1.000 | 0.423 |
| (6) Base + UQ only | YES | - | - | YES | - | 0.000 | -0.668 | 0.668 | 0.000 | -0.004 | 0.000 | 0.000 | 0.000 |

### 95% bootstrap CI over items (seed 0)

| Variant | Accuracy | 95% CI | Causal Consistency | 95% CI |
|---|---|---|---|---|
| (1) Base LLM | 0.668 | [0.653, 0.682] | 0.004 | [0.001, 0.008] |
| (2) + RAG | 0.635 | [0.620, 0.650] | 0.041 | [0.029, 0.054] |
| (3) + Symbolic Gate (NS-AI) | 0.752 | [0.739, 0.766] | 0.363 | [0.334, 0.393] |
| (4) + UQ Engine (NS-AI+UQ) | 0.395 | [0.380, 0.411] | 0.348 | [0.319, 0.378] |
| (5) Base + Symbolic Gate only | 0.768 | [0.755, 0.781] | 0.351 | [0.322, 0.381] |
| (6) Base + UQ only | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] |

### Paired McNemar of each contribution **against the baseline** (seed 0, item-level correctness)

This is the supervisor's question: what does adding this contribution to the base model do?

| Variant vs base | B01 (base wrong→right) | B10 (base right→wrong) | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 199 | 326 | 3.292e-08 | 3.292e-08 |
| base→nsai | 494 | 163 | 1.277e-39 | 2.554e-39 |
| base→nsai_uq | 396 | 1462 | 3.348e-143 | 1.339e-142 |
| base→sym | 396 | 1 | 2.466e-117 | 7.398e-117 |
| base→uq | 0 | 2614 | 0 | 0 |

### Paired McNemar along the cumulative ladder (seed 0)

| Comparison | B01 | B10 | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 199 | 326 | 3.292e-08 | 3.292e-08 |
| rag→nsai | 466 | 8 | 2.484e-126 | 4.969e-126 |
| nsai→nsai_uq | 0 | 1397 | 0 | 0 |

### Where the gate's contribution comes from (seed 0)

The gate's accuracy on items it fires on is partly circular: it applies the same rule and threshold the labels were generated from, and on the MedCalc benchmark the same extractor that filters the data runs inside the gate. Splitting the split by whether the gate fired separates the circular part from the part that is not: on gate-declined items the row IS the neural pathway, so any difference there is real.

On gate-fired items the three accuracy columns separate the two things the question conflates. *Final* is the rule checking itself (identity). *Model's own answer* is whether the LLM, given that row's prompt, reached the guideline's conclusion without the rule's help: that is guideline adherence. *Agrees with gate* is how often the gate merely confirmed the model rather than overruled it.

| Variant | subset | n | Final accuracy | Model's own answer acc. | Base accuracy, same items | Model agrees with gate |
|---|---|---|---|---|---|---|
| (3) + Symbolic Gate (NS-AI) | gate fired | 1658 | 0.934 (identity) | 0.657 | 0.695 | 0.711 |
| (3) + Symbolic Gate (NS-AI) | gate declined | 2258 | 0.619 | 0.619 | 0.647 | — |
| (4) + UQ Engine (NS-AI+UQ) | gate fired | 1658 | 0.934 (identity) | 0.657 | 0.695 | 0.711 |
| (4) + UQ Engine (NS-AI+UQ) | gate declined | 2258 | 0.000 | 0.619 | 0.647 | — |
| (5) Base + Symbolic Gate only | gate fired | 1658 | 0.934 (identity) | 0.695 | 0.695 | 0.761 |
| (5) Base + Symbolic Gate only | gate declined | 2258 | 0.647 | 0.647 | 0.647 | — |

### Why a UQ row can read 0.000 coverage

Split conformal picks the largest uncertainty threshold whose error rate on the calibration split is at most alpha = 0.10. On this calibration split of 1156 items no threshold reaches that target (the risk-coverage table below shows how far off it is on these items). The method then falls back to its most conservative threshold, which retains 0.0% of the calibration items, and on the `test` split retains none. **That is the method behaving correctly, not a failure to run**: a 10% error target is unreachable for a model at this accuracy, so the only way to honour it is to answer nothing. It is also the exact situation Adaptive Conformal Inference exists for -- see `results/uq_coverage_*.md`, where the threshold is allowed to move.


### Risk-coverage of the UQ signal, (4) + UQ Engine (NS-AI+UQ) (seed 0)

Every threshold a deferral rule could pick, on the 2258 items the UQ engine governs here (gate-decided items are never deferred and are excluded). Error is of the model's own answer on the items kept. The signal (`decision_entropy`) takes 95 distinct values, so 95 operating points exist; the rows below are those nearest each coverage level. Full curve: `table_mimic3bnote_test_riskcov_nsai_uq.csv`.

| Coverage target | Threshold | Coverage | Error | Items kept |
|---|---|---|---|---|
| 100% | inf | 1.000 | 0.381 | 2258 |
| 90% | 0.6466 | 0.895 | 0.338 | 2021 |
| 75% | 0.1825 | 0.752 | 0.320 | 1699 |
| 50% | 0.09461 | 0.488 | 0.299 | 1103 |
| 25% | 0.0575 | 0.248 | 0.283 | 559 |
| 10% | 0.03815 | 0.098 | 0.258 | 221 |
| 5% | 0.0294 | 0.051 | 0.259 | 116 |
| 1% | 0.01126 | 0.010 | 0.348 | 23 |

- AURC (area under the risk-coverage curve, lower is better): **0.306**; a signal that ranks at random scores the full-coverage error, 0.381.
- Lowest error at ≥1% coverage: **0.239** (coverage 0.020).
- Target error α = 0.10: reachable on these items up to coverage **0.000**.
- Deployed threshold τ = -inf, set on the calibration split (coverage there 0.000; certifiable: False); here it keeps 0.423 of all items.

### Risk-coverage of the UQ signal, (6) Base + UQ only (seed 0)

Every threshold a deferral rule could pick, on the 3916 items the UQ engine governs here (gate-decided items are never deferred and are excluded). Error is of the model's own answer on the items kept. The signal (`decision_entropy`) takes 158 distinct values, so 158 operating points exist; the rows below are those nearest each coverage level. Full curve: `table_mimic3bnote_test_riskcov_uq.csv`.

| Coverage target | Threshold | Coverage | Error | Items kept |
|---|---|---|---|---|
| 100% | inf | 1.000 | 0.332 | 3916 |
| 90% | 0.07033 | 0.902 | 0.319 | 3532 |
| 75% | 0.04018 | 0.749 | 0.325 | 2933 |
| 50% | 0.02031 | 0.507 | 0.334 | 1986 |
| 25% | 0.009571 | 0.252 | 0.318 | 985 |
| 10% | 0.004447 | 0.101 | 0.278 | 395 |
| 5% | 0.001545 | 0.051 | 0.240 | 200 |
| 1% | 0.000238 | 0.010 | 0.250 | 40 |

- AURC (area under the risk-coverage curve, lower is better): **0.315**; a signal that ranks at random scores the full-coverage error, 0.332.
- Lowest error at ≥1% coverage: **0.206** (coverage 0.041).
- Target error α = 0.10: **not reachable at any threshold** on these items -- so a rule that must honour α can only abstain.
- Deployed threshold τ = -inf, set on the calibration split (coverage there 0.000; certifiable: False); here it keeps 0.000 of all items.

### Notes

- Causal Consistency is pair-level: both counterfactual arms must be correct. A constant-answer model scores 0.
- Abstentions count as failures for accuracy and causal consistency, and as non-violations for violation rate. Coverage is reported so this trade-off is visible.
- **Accuracy (strict)** counts an abstention as wrong and is the row-comparable number. Selective accuracy is in `summary_*.json`; it rises trivially as coverage falls and must never be compared across rows without coverage.
- Rows (5)-(7) add ONE contribution to the base model. Rows (1)-(4) are the cumulative ladder of the proposal's section 4.6. Row (8) is everything at once.
- **The gate's accuracy is an upper bound.** It evaluates the same constraint the ground truth was generated from, so on items it fires on it cannot be wrong unless extraction is. The figure to quote next to it is its firing rate here, and its coverage on unfiltered real notes (42.7%, `results/medcalc_ablation.md`).
- The constraint layer is scored through the identical readout as every other row (generate, then parse the first decision word), not by argmax over answer logits, so the numbers are comparable down the column.
