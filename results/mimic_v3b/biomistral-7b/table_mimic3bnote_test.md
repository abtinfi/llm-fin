## Ablation matrix -- split=`test`, model=`BioMistral/BioMistral-7B`, seeds=[0]

UQ engine defers on **`decision_entropy`**  (Eq. (2) restricted to the decision tokens, the respecified 4.7 term: AUROC 0.687 [0.675, 0.700] over 6,456 items. Rank-equivalent to `logit_margin` for K=2.)

Single seed. Decoding is greedy and therefore deterministic: another seed reproduces this run exactly, so no seed-to-seed spread is reported. The uncertainty that does exist is over items, and is given as a bootstrap CI in the next table.

### Primary: the section 4.6 ablation ladder

| Variant | Base | RAG | Sym | UQ | CL | Accuracy (strict) | Δ Acc vs base | Model's own answer acc. | Causal Consistency | Δ CC vs base | Violation Rate | Coverage | Gate fired |
|---|:-:|:-:|:-:|:-:|:-:|---|---|---|---|---|---|---|---|
| (1) Base LLM | YES | - | - | - | - | 0.519 | +0.000 | 0.519 | 0.050 | +0.000 | 0.574 | 1.000 | 0.000 |
| (2) + RAG | YES | YES | - | - | - | 0.331 | -0.188 | 0.331 | 0.011 | -0.039 | 0.003 | 1.000 | 0.000 |
| (3) + Symbolic Gate (NS-AI) | YES | YES | YES | - | - | 0.594 † | +0.075 | 0.331 | 0.350 † | +0.300 | 0.091 | 1.000 | 0.423 |
| (4) + UQ Engine (NS-AI+UQ) | YES | YES | YES | YES | - | 0.395 † | -0.123 | 0.331 | 0.348 † | +0.298 | 0.088 | 0.423 | 0.423 |

**Model's own answer acc.** is the decision word parsed from the model's generation, before the gate overrides it and before UQ defers it: what the LLM itself concluded under that row's prompt. † Accuracy and CC on gate-fired items are an identity check, not a measurement: the gate applies the rule and threshold the labels were generated from. With the gate firing on every item, that row's accuracy is 1.000 by construction; read the model's own column, and the adherence table below, for what the model knows.


### Supplementary ablations: one contribution added to the base model

| Variant | Base | RAG | Sym | UQ | CL | Accuracy (strict) | Δ Acc vs base | Model's own answer acc. | Causal Consistency | Δ CC vs base | Violation Rate | Coverage | Gate fired |
|---|:-:|:-:|:-:|:-:|:-:|---|---|---|---|---|---|---|---|
| (5) Base + Symbolic Gate only | YES | - | YES | - | - | 0.753 † | +0.234 | 0.519 | 0.375 † | +0.325 | 0.543 | 1.000 | 0.423 |
| (6) Base + UQ only | YES | - | - | YES | - | 0.000 | -0.519 | 0.519 | 0.000 | -0.050 | 0.000 | 0.000 | 0.000 |

### 95% bootstrap CI over items (seed 0)

| Variant | Accuracy | 95% CI | Causal Consistency | 95% CI |
|---|---|---|---|---|
| (1) Base LLM | 0.519 | [0.503, 0.534] | 0.050 | [0.037, 0.064] |
| (2) + RAG | 0.331 | [0.317, 0.346] | 0.011 | [0.005, 0.018] |
| (3) + Symbolic Gate (NS-AI) | 0.594 | [0.578, 0.609] | 0.350 | [0.321, 0.380] |
| (4) + UQ Engine (NS-AI+UQ) | 0.395 | [0.380, 0.411] | 0.348 | [0.319, 0.378] |
| (5) Base + Symbolic Gate only | 0.753 | [0.739, 0.766] | 0.375 | [0.345, 0.406] |
| (6) Base + UQ only | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] |

### Paired McNemar of each contribution **against the baseline** (seed 0, item-level correctness)

This is the supervisor's question: what does adding this contribution to the base model do?

| Variant vs base | B01 (base wrong→right) | B10 (base right→wrong) | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 742 | 1477 | 9.545e-56 | 2.864e-55 |
| base→nsai | 1509 | 1215 | 1.919e-08 | 1.919e-08 |
| base→nsai_uq | 943 | 1426 | 2.866e-23 | 5.731e-23 |
| base→sym | 943 | 27 | 5.778e-240 | 2.311e-239 |
| base→uq | 0 | 2031 | 0 | 0 |

### Paired McNemar along the cumulative ladder (seed 0)

| Comparison | B01 | B10 | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 742 | 1477 | 9.545e-56 | 9.545e-56 |
| rag→nsai | 1139 | 110 | 4.278e-216 | 8.556e-216 |
| nsai→nsai_uq | 0 | 777 | 2.516e-234 | 7.548e-234 |

### Where the gate's contribution comes from (seed 0)

The gate's accuracy on items it fires on is partly circular: it applies the same rule and threshold the labels were generated from, and on the MedCalc benchmark the same extractor that filters the data runs inside the gate. Splitting the split by whether the gate fired separates the circular part from the part that is not: on gate-declined items the row IS the neural pathway, so any difference there is real.

On gate-fired items the three accuracy columns separate the two things the question conflates. *Final* is the rule checking itself (identity). *Model's own answer* is whether the LLM, given that row's prompt, reached the guideline's conclusion without the rule's help: that is guideline adherence. *Agrees with gate* is how often the gate merely confirmed the model rather than overruled it.

| Variant | subset | n | Final accuracy | Model's own answer acc. | Base accuracy, same items | Model agrees with gate |
|---|---|---|---|---|---|---|
| (3) + Symbolic Gate (NS-AI) | gate fired | 1658 | 0.934 (identity) | 0.313 | 0.381 | 0.247 |
| (3) + Symbolic Gate (NS-AI) | gate declined | 2258 | 0.344 | 0.344 | 0.620 | — |
| (4) + UQ Engine (NS-AI+UQ) | gate fired | 1658 | 0.934 (identity) | 0.313 | 0.381 | 0.247 |
| (4) + UQ Engine (NS-AI+UQ) | gate declined | 2258 | 0.000 | 0.344 | 0.620 | — |
| (5) Base + Symbolic Gate only | gate fired | 1658 | 0.934 (identity) | 0.381 | 0.381 | 0.415 |
| (5) Base + Symbolic Gate only | gate declined | 2258 | 0.620 | 0.620 | 0.620 | — |

### Why a UQ row can read 0.000 coverage

Split conformal picks the largest uncertainty threshold whose error rate on the calibration split is at most alpha = 0.10. On this calibration split of 1174 items no threshold reaches that target (the risk-coverage table below shows how far off it is on these items). The method then falls back to its most conservative threshold, which retains 0.0% of the calibration items, and on the `test` split retains none. **That is the method behaving correctly, not a failure to run**: a 10% error target is unreachable for a model at this accuracy, so the only way to honour it is to answer nothing. It is also the exact situation Adaptive Conformal Inference exists for -- see `results/uq_coverage_*.md`, where the threshold is allowed to move.


### Risk-coverage of the UQ signal, (4) + UQ Engine (NS-AI+UQ) (seed 0)

Every threshold a deferral rule could pick, on the 2258 items the UQ engine governs here (gate-decided items are never deferred and are excluded). Error is of the model's own answer on the items kept. The signal (`decision_entropy`) takes 36 distinct values, so 36 operating points exist; the rows below are those nearest each coverage level. Full curve: `table_mimic3bnote_test_riskcov_nsai_uq.csv`.

| Coverage target | Threshold | Coverage | Error | Items kept |
|---|---|---|---|---|
| 100% | 0.6931 | 1.000 | 0.656 | 2258 |
| 90% | 0.6698 | 0.897 | 0.644 | 2025 |
| 75% | 0.6466 | 0.738 | 0.624 | 1666 |
| 50% | 0.6059 | 0.491 | 0.589 | 1108 |
| 25% | 0.5569 | 0.246 | 0.517 | 555 |
| 10% | 0.5029 | 0.089 | 0.488 | 201 |
| 5% | 0.4751 | 0.052 | 0.432 | 118 |
| 1% | 0.4056 | 0.012 | 0.308 | 26 |

- AURC (area under the risk-coverage curve, lower is better): **0.573**; a signal that ranks at random scores the full-coverage error, 0.656.
- Lowest error at ≥1% coverage: **0.308** (coverage 0.012).
- Target error α = 0.10: reachable on these items up to coverage **0.004**.
- Deployed threshold τ = -inf, set on the calibration split (coverage there 0.000; certifiable: False); here it keeps 0.423 of all items.

### Risk-coverage of the UQ signal, (6) Base + UQ only (seed 0)

Every threshold a deferral rule could pick, on the 3916 items the UQ engine governs here (gate-decided items are never deferred and are excluded). Error is of the model's own answer on the items kept. The signal (`decision_entropy`) takes 52 distinct values, so 52 operating points exist; the rows below are those nearest each coverage level. Full curve: `table_mimic3bnote_test_riskcov_uq.csv`.

| Coverage target | Threshold | Coverage | Error | Items kept |
|---|---|---|---|---|
| 100% | 0.6931 | 1.000 | 0.481 | 3916 |
| 90% | 0.6912 | 0.864 | 0.464 | 3385 |
| 75% | 0.6888 | 0.780 | 0.457 | 3053 |
| 50% | 0.6698 | 0.474 | 0.400 | 1856 |
| 25% | 0.6374 | 0.243 | 0.351 | 950 |
| 10% | 0.5943 | 0.100 | 0.338 | 393 |
| 5% | 0.5569 | 0.043 | 0.305 | 167 |
| 1% | 0.1522 | 0.010 | 0.268 | 41 |

- AURC (area under the risk-coverage curve, lower is better): **0.409**; a signal that ranks at random scores the full-coverage error, 0.481.
- Lowest error at ≥1% coverage: **0.268** (coverage 0.010).
- Target error α = 0.10: reachable on these items up to coverage **0.001**.
- Deployed threshold τ = -inf, set on the calibration split (coverage there 0.000; certifiable: False); here it keeps 0.000 of all items.

### Notes

- Causal Consistency is pair-level: both counterfactual arms must be correct. A constant-answer model scores 0.
- Abstentions count as failures for accuracy and causal consistency, and as non-violations for violation rate. Coverage is reported so this trade-off is visible.
- **Accuracy (strict)** counts an abstention as wrong and is the row-comparable number. Selective accuracy is in `summary_*.json`; it rises trivially as coverage falls and must never be compared across rows without coverage.
- Rows (5)-(7) add ONE contribution to the base model. Rows (1)-(4) are the cumulative ladder of the proposal's section 4.6. Row (8) is everything at once.
- **The gate's accuracy is an upper bound.** It evaluates the same constraint the ground truth was generated from, so on items it fires on it cannot be wrong unless extraction is. The figure to quote next to it is its firing rate here, and its coverage on unfiltered real notes (42.7%, `results/medcalc_ablation.md`).
- The constraint layer is scored through the identical readout as every other row (generate, then parse the first decision word), not by argmax over answer logits, so the numbers are comparable down the column.
