## Ablation matrix -- split=`test`, model=`mistralai/Mistral-7B-Instruct-v0.2`, seeds=[0]

UQ engine defers on **`decision_entropy`**  (Eq. (2) restricted to the decision tokens, the respecified 4.7 term: AUROC 0.687 [0.675, 0.700] over 6,456 items. Rank-equivalent to `logit_margin` for K=2.)

Single seed. Decoding is greedy and therefore deterministic: another seed reproduces this run exactly, so no seed-to-seed spread is reported. The uncertainty that does exist is over items, and is given as a bootstrap CI in the next table.

### Primary: the section 4.6 ablation ladder

| Variant | Base | RAG | Sym | UQ | CL | Accuracy (strict) | Δ Acc vs base | Model's own answer acc. | Causal Consistency | Δ CC vs base | Violation Rate | Coverage | Gate fired |
|---|:-:|:-:|:-:|:-:|:-:|---|---|---|---|---|---|---|---|
| (1) Base LLM | YES | - | - | - | - | 0.647 | +0.000 | 0.647 | 0.019 | +0.000 | 0.959 | 1.000 | 0.000 |
| (2) + RAG | YES | YES | - | - | - | 0.741 | +0.094 | 0.741 | 0.383 | +0.364 | 0.233 | 1.000 | 0.000 |
| (3) + Symbolic Gate (NS-AI) | YES | YES | YES | - | - | 0.795 † | +0.148 | 0.741 | 0.522 † | +0.504 | 0.228 | 1.000 | 0.390 |
| (4) + UQ Engine (NS-AI+UQ) | YES | YES | YES | YES | - | 0.555 † | -0.091 | 0.741 | 0.399 † | +0.380 | 0.000 | 0.571 | 0.390 |

**Model's own answer acc.** is the decision word parsed from the model's generation, before the gate overrides it and before UQ defers it: what the LLM itself concluded under that row's prompt. † Accuracy and CC on gate-fired items are an identity check, not a measurement: the gate applies the rule and threshold the labels were generated from. With the gate firing on every item, that row's accuracy is 1.000 by construction; read the model's own column, and the adherence table below, for what the model knows.


### Supplementary ablations: one contribution added to the base model

| Variant | Base | RAG | Sym | UQ | CL | Accuracy (strict) | Δ Acc vs base | Model's own answer acc. | Causal Consistency | Δ CC vs base | Violation Rate | Coverage | Gate fired |
|---|:-:|:-:|:-:|:-:|:-:|---|---|---|---|---|---|---|---|
| (5) Base + Symbolic Gate only | YES | - | YES | - | - | 0.794 † | +0.148 | 0.647 | 0.396 † | +0.377 | 0.640 | 1.000 | 0.390 |
| (6) Base + UQ only | YES | - | - | YES | - | 0.089 | -0.557 | 0.647 | 0.000 | -0.019 | 0.029 | 0.099 | 0.000 |

### 95% bootstrap CI over items (seed 0)

| Variant | Accuracy | 95% CI | Causal Consistency | 95% CI |
|---|---|---|---|---|
| (1) Base LLM | 0.647 | [0.645, 0.649] | 0.019 | [0.018, 0.020] |
| (2) + RAG | 0.741 | [0.739, 0.743] | 0.383 | [0.378, 0.388] |
| (3) + Symbolic Gate (NS-AI) | 0.795 | [0.793, 0.797] | 0.522 | [0.518, 0.527] |
| (4) + UQ Engine (NS-AI+UQ) | 0.555 | [0.553, 0.558] | 0.399 | [0.394, 0.403] |
| (5) Base + Symbolic Gate only | 0.794 | [0.793, 0.796] | 0.396 | [0.391, 0.400] |
| (6) Base + UQ only | 0.089 | [0.088, 0.091] | 0.000 | [0.000, 0.000] |

### Paired McNemar of each contribution **against the baseline** (seed 0, item-level correctness)

This is the supervisor's question: what does adding this contribution to the base model do?

| Variant vs base | B01 (base wrong→right) | B10 (base right→wrong) | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 46080 | 30406 | 0 | 0 |
| base→nsai | 46517 | 21950 | 0 | 0 |
| base→nsai_uq | 32446 | 47639 | 0 | 0 |
| base→sym | 24544 | 0 | 0 | 0 |
| base→uq | 0 | 92677 | 0 | 0 |

### Paired McNemar along the cumulative ladder (seed 0)

| Comparison | B01 | B10 | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 46080 | 30406 | 0 | 0 |
| rag→nsai | 8893 | 0 | 0 | 0 |
| nsai→nsai_uq | 0 | 39760 | 0 | 0 |

### Where the gate's contribution comes from (seed 0)

The gate's accuracy on items it fires on is partly circular: it applies the same rule and threshold the labels were generated from, and on the MedCalc benchmark the same extractor that filters the data runs inside the gate. Splitting the split by whether the gate fired separates the circular part from the part that is not: on gate-declined items the row IS the neural pathway, so any difference there is real.

On gate-fired items the three accuracy columns separate the two things the question conflates. *Final* is the rule checking itself (identity). *Model's own answer* is whether the LLM, given that row's prompt, reached the guideline's conclusion without the rule's help: that is guideline adherence. *Agrees with gate* is how often the gate merely confirmed the model rather than overruled it.

| Variant | subset | n | Final accuracy | Model's own answer acc. | Base accuracy, same items | Model agrees with gate |
|---|---|---|---|---|---|---|
| (3) + Symbolic Gate (NS-AI) | gate fired | 64826 | 1.000 (identity) | 0.863 | 0.621 | 0.863 |
| (3) + Symbolic Gate (NS-AI) | gate declined | 101438 | 0.663 | 0.663 | 0.663 | — |
| (4) + UQ Engine (NS-AI+UQ) | gate fired | 64826 | 1.000 (identity) | 0.863 | 0.621 | 0.863 |
| (4) + UQ Engine (NS-AI+UQ) | gate declined | 101438 | 0.271 | 0.663 | 0.663 | — |
| (5) Base + Symbolic Gate only | gate fired | 64826 | 1.000 (identity) | 0.621 | 0.621 | 0.621 |
| (5) Base + Symbolic Gate only | gate declined | 101438 | 0.663 | 0.663 | 0.663 | — |

### Risk-coverage of the UQ signal, (4) + UQ Engine (NS-AI+UQ) (seed 0)

Every threshold a deferral rule could pick, on the 101438 items the UQ engine governs here (gate-decided items are never deferred and are excluded). Error is of the model's own answer on the items kept. The signal (`decision_entropy`) takes 262 distinct values, so 262 operating points exist; the rows below are those nearest each coverage level. Full curve: `table_mimic3b_test_riskcov_nsai_uq.csv`.

| Coverage target | Threshold | Coverage | Error | Items kept |
|---|---|---|---|---|
| 100% | 0.6931 | 1.000 | 0.337 | 101438 |
| 90% | 0.6059 | 0.908 | 0.302 | 92144 |
| 75% | 0.291 | 0.748 | 0.268 | 75844 |
| 50% | 0.07393 | 0.503 | 0.161 | 51008 |
| 25% | 0.006901 | 0.245 | 0.060 | 24882 |
| 10% | 0.0001687 | 0.100 | 0.000 | 10145 |
| 5% | 1.875e-05 | 0.051 | 0.000 | 5127 |
| 1% | 6.559e-06 | 0.010 | 0.000 | 1006 |

- AURC (area under the risk-coverage curve, lower is better): **0.164**; a signal that ranks at random scores the full-coverage error, 0.337.
- Lowest error at ≥1% coverage: **0.000** (coverage 0.011).
- Target error α = 0.10: reachable on these items up to coverage **0.298**.
- Deployed threshold τ = 0.01067, set on the calibration split (coverage there 0.295; certifiable: True); here it keeps 0.571 of all items.

### Risk-coverage of the UQ signal, (6) Base + UQ only (seed 0)

Every threshold a deferral rule could pick, on the 166264 items the UQ engine governs here (gate-decided items are never deferred and are excluded). Error is of the model's own answer on the items kept. The signal (`decision_entropy`) takes 470 distinct values, so 470 operating points exist; the rows below are those nearest each coverage level. Full curve: `table_mimic3b_test_riskcov_uq.csv`.

| Coverage target | Threshold | Coverage | Error | Items kept |
|---|---|---|---|---|
| 100% | 0.6931 | 1.000 | 0.353 | 166264 |
| 90% | 0.4056 | 0.898 | 0.345 | 149258 |
| 75% | 0.1204 | 0.752 | 0.359 | 125048 |
| 50% | 0.001545 | 0.499 | 0.313 | 82962 |
| 25% | 1.11e-05 | 0.250 | 0.181 | 41595 |
| 10% | 7.675e-07 | 0.100 | 0.097 | 16552 |
| 5% | 1.648e-07 | 0.050 | 0.000 | 8308 |
| 1% | 1.354e-08 | 0.010 | 0.000 | 1640 |

- AURC (area under the risk-coverage curve, lower is better): **0.260**; a signal that ranks at random scores the full-coverage error, 0.353.
- Lowest error at ≥1% coverage: **0.000** (coverage 0.010).
- Target error α = 0.10: reachable on these items up to coverage **0.106**.
- Deployed threshold τ = 7.452e-07, set on the calibration split (coverage there 0.099; certifiable: True); here it keeps 0.099 of all items.

### Notes

- Causal Consistency is pair-level: both counterfactual arms must be correct. A constant-answer model scores 0.
- Abstentions count as failures for accuracy and causal consistency, and as non-violations for violation rate. Coverage is reported so this trade-off is visible.
- **Accuracy (strict)** counts an abstention as wrong and is the row-comparable number. Selective accuracy is in `summary_*.json`; it rises trivially as coverage falls and must never be compared across rows without coverage.
- Rows (5)-(7) add ONE contribution to the base model. Rows (1)-(4) are the cumulative ladder of the proposal's section 4.6. Row (8) is everything at once.
- **The gate's accuracy is an upper bound.** It evaluates the same constraint the ground truth was generated from, so on items it fires on it cannot be wrong unless extraction is. The figure to quote next to it is its firing rate here, and its coverage on unfiltered real notes (42.7%, `results/medcalc_ablation.md`).
- The constraint layer is scored through the identical readout as every other row (generate, then parse the first decision word), not by argmax over answer logits, so the numbers are comparable down the column.
