## Ablation matrix -- split=`test`, model=`aaditya/Llama3-OpenBioLLM-8B`, seeds=[0]

UQ engine defers on **`decision_entropy`**  (Eq. (2) restricted to the decision tokens, the respecified 4.7 term: AUROC 0.687 [0.675, 0.700] over 6,456 items. Rank-equivalent to `logit_margin` for K=2.)

Single seed. Decoding is greedy and therefore deterministic: another seed reproduces this run exactly, so no seed-to-seed spread is reported. The uncertainty that does exist is over items, and is given as a bootstrap CI in the next table.

### Primary: the section 4.6 ablation ladder

| Variant | Base | RAG | Sym | UQ | CL | Accuracy (strict) | Δ Acc vs base | Model's own answer acc. | Causal Consistency | Δ CC vs base | Violation Rate | Coverage | Gate fired |
|---|:-:|:-:|:-:|:-:|:-:|---|---|---|---|---|---|---|---|
| (1) Base LLM | YES | - | - | - | - | 0.679 | +0.000 | 0.679 | 0.000 | +0.000 | 1.000 | 1.000 | 0.000 |
| (2) + RAG | YES | YES | - | - | - | 0.221 | -0.457 | 0.221 | 0.000 | +0.000 | 0.058 | 1.000 | 0.000 |
| (3) + Symbolic Gate (NS-AI) | YES | YES | YES | - | - | 0.431 † | -0.247 | 0.221 | 0.394 † | +0.394 | 0.051 | 1.000 | 0.390 |
| (4) + UQ Engine (NS-AI+UQ) | YES | YES | YES | YES | - | 0.396 † | -0.283 | 0.221 | 0.394 † | +0.394 | 0.000 | 0.396 | 0.390 |

**Model's own answer acc.** is the decision word parsed from the model's generation, before the gate overrides it and before UQ defers it: what the LLM itself concluded under that row's prompt. † Accuracy and CC on gate-fired items are an identity check, not a measurement: the gate applies the rule and threshold the labels were generated from. With the gate firing on every item, that row's accuracy is 1.000 by construction; read the model's own column, and the adherence table below, for what the model knows.


### Supplementary ablations: one contribution added to the base model

| Variant | Base | RAG | Sym | UQ | CL | Accuracy (strict) | Δ Acc vs base | Model's own answer acc. | Causal Consistency | Δ CC vs base | Violation Rate | Coverage | Gate fired |
|---|:-:|:-:|:-:|:-:|:-:|---|---|---|---|---|---|---|---|
| (5) Base + Symbolic Gate only | YES | - | YES | - | - | 0.787 † | +0.109 | 0.679 | 0.394 † | +0.394 | 0.662 | 1.000 | 0.390 |
| (6) Base + UQ only | YES | - | - | YES | - | 0.076 | -0.602 | 0.679 | 0.000 | +0.000 | 0.016 | 0.081 | 0.000 |
| (7) Base + Constraint Layer only | YES | - | - | - | YES | 0.900 | +0.221 | 0.900 | 0.769 | +0.769 | 0.060 | 1.000 | 0.000 |
| (8) All four contributions | YES | YES | YES | YES | YES | 0.441 † | -0.238 | 0.397 | 0.394 † | +0.394 | 0.000 | 0.447 | 0.390 |

### 95% bootstrap CI over items (seed 0)

| Variant | Accuracy | 95% CI | Causal Consistency | 95% CI |
|---|---|---|---|---|
| (1) Base LLM | 0.679 | [0.676, 0.681] | 0.000 | [0.000, 0.000] |
| (2) + RAG | 0.221 | [0.219, 0.223] | 0.000 | [0.000, 0.000] |
| (3) + Symbolic Gate (NS-AI) | 0.431 | [0.429, 0.434] | 0.394 | [0.390, 0.399] |
| (4) + UQ Engine (NS-AI+UQ) | 0.396 | [0.394, 0.398] | 0.394 | [0.390, 0.399] |
| (5) Base + Symbolic Gate only | 0.787 | [0.786, 0.789] | 0.394 | [0.390, 0.399] |
| (6) Base + UQ only | 0.076 | [0.075, 0.078] | 0.000 | [0.000, 0.000] |
| (7) Base + Constraint Layer only | 0.900 | [0.898, 0.901] | 0.769 | [0.765, 0.773] |
| (8) All four contributions | 0.441 | [0.438, 0.443] | 0.394 | [0.390, 0.399] |

### Paired McNemar of each contribution **against the baseline** (seed 0, item-level correctness)

This is the supervisor's question: what does adding this contribution to the base model do?

| Variant vs base | B01 (base wrong→right) | B10 (base right→wrong) | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 0 | 76065 | 0 | 0 |
| base→nsai | 18080 | 59202 | 0 | 0 |
| base→nsai_uq | 18080 | 65096 | 0 | 0 |
| base→sym | 18080 | 0 | 0 | 0 |
| base→uq | 0 | 100133 | 0 | 0 |
| base→cl | 49442 | 12705 | 0 | 0 |
| base→nsai_uq_cl | 26554 | 66098 | 0 | 0 |

### Paired McNemar along the cumulative ladder (seed 0)

| Comparison | B01 | B10 | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 0 | 76065 | 0 | 0 |
| rag→nsai | 34943 | 0 | 0 | 0 |
| nsai→nsai_uq | 0 | 5894 | 0 | 0 |

### Where the gate's contribution comes from (seed 0)

The gate's accuracy on items it fires on is partly circular: it applies the same rule and threshold the labels were generated from, and on the MedCalc benchmark the same extractor that filters the data runs inside the gate. Splitting the split by whether the gate fired separates the circular part from the part that is not: on gate-declined items the row IS the neural pathway, so any difference there is real.

On gate-fired items the three accuracy columns separate the two things the question conflates. *Final* is the rule checking itself (identity). *Model's own answer* is whether the LLM, given that row's prompt, reached the guideline's conclusion without the rule's help: that is guideline adherence. *Agrees with gate* is how often the gate merely confirmed the model rather than overruled it.

| Variant | subset | n | Final accuracy | Model's own answer acc. | Base accuracy, same items | Model agrees with gate |
|---|---|---|---|---|---|---|
| (3) + Symbolic Gate (NS-AI) | gate fired | 64826 | 1.000 (identity) | 0.461 | 0.721 | 0.461 |
| (3) + Symbolic Gate (NS-AI) | gate declined | 101438 | 0.068 | 0.068 | 0.652 | — |
| (4) + UQ Engine (NS-AI+UQ) | gate fired | 64826 | 1.000 (identity) | 0.461 | 0.721 | 0.461 |
| (4) + UQ Engine (NS-AI+UQ) | gate declined | 101438 | 0.010 | 0.068 | 0.652 | — |
| (5) Base + Symbolic Gate only | gate fired | 64826 | 1.000 (identity) | 0.721 | 0.721 | 0.721 |
| (5) Base + Symbolic Gate only | gate declined | 101438 | 0.652 | 0.652 | 0.652 | — |
| (8) All four contributions | gate fired | 64826 | 1.000 (identity) | 0.473 | 0.721 | 0.473 |
| (8) All four contributions | gate declined | 101438 | 0.084 | 0.348 | 0.652 | — |

### Risk-coverage of the UQ signal, (4) + UQ Engine (NS-AI+UQ) (seed 0)

Every threshold a deferral rule could pick, on the 101438 items the UQ engine governs here (gate-decided items are never deferred and are excluded). Error is of the model's own answer on the items kept. The signal (`decision_entropy`) takes 40 distinct values, so 40 operating points exist; the rows below are those nearest each coverage level. Full curve: `table_mimic3b_test_riskcov_nsai_uq.csv`.

| Coverage target | Threshold | Coverage | Error | Items kept |
|---|---|---|---|---|
| 100% | inf | 1.000 | 0.932 | 101438 |
| 90% | inf | 1.000 | 0.932 | 101438 |
| 75% | inf | 1.000 | 0.932 | 101438 |
| 50% | 0.3523 | 0.095 | 0.283 | 9624 |
| 25% | 0.3523 | 0.095 | 0.283 | 9624 |
| 10% | 0.3523 | 0.095 | 0.283 | 9624 |
| 5% | 0.1594 | 0.050 | 0.293 | 5028 |
| 1% | 0.09933 | 0.010 | 0.000 | 1002 |

- AURC (area under the risk-coverage curve, lower is better): **0.869**; a signal that ranks at random scores the full-coverage error, 0.932.
- Lowest error at ≥1% coverage: **0.209** (coverage 0.024).
- Target error α = 0.10: reachable on these items up to coverage **0.010**.
- Deployed threshold τ = 0.09933, set on the calibration split (coverage there 0.106; certifiable: True); here it keeps 0.396 of all items.

### Risk-coverage of the UQ signal, (6) Base + UQ only (seed 0)

Every threshold a deferral rule could pick, on the 166264 items the UQ engine governs here (gate-decided items are never deferred and are excluded). Error is of the model's own answer on the items kept. The signal (`decision_entropy`) takes 42 distinct values, so 42 operating points exist; the rows below are those nearest each coverage level. Full curve: `table_mimic3b_test_riskcov_uq.csv`.

| Coverage target | Threshold | Coverage | Error | Items kept |
|---|---|---|---|---|
| 100% | 0.1323 | 1.000 | 0.321 | 166264 |
| 90% | 0.08165 | 0.904 | 0.318 | 150278 |
| 75% | 0.07033 | 0.753 | 0.329 | 125188 |
| 50% | 0.04231 | 0.511 | 0.317 | 84879 |
| 25% | 0.03264 | 0.241 | 0.186 | 40073 |
| 10% | 0.02789 | 0.112 | 0.105 | 18583 |
| 5% | 0.02511 | 0.053 | 0.024 | 8827 |
| 1% | 0.02142 | 0.006 | 0.002 | 1070 |

- AURC (area under the risk-coverage curve, lower is better): **0.261**; a signal that ranks at random scores the full-coverage error, 0.321.
- Lowest error at ≥1% coverage: **0.001** (coverage 0.014).
- Target error α = 0.10: reachable on these items up to coverage **0.081**.
- Deployed threshold τ = 0.02646, set on the calibration split (coverage there 0.082; certifiable: True); here it keeps 0.081 of all items.

### Risk-coverage of the UQ signal, (8) All four contributions (seed 0)

Every threshold a deferral rule could pick, on the 101438 items the UQ engine governs here (gate-decided items are never deferred and are excluded). Error is of the model's own answer on the items kept. The signal (`decision_entropy`) takes 74 distinct values, so 74 operating points exist; the rows below are those nearest each coverage level. Full curve: `table_mimic3b_test_riskcov_nsai_uq_cl.csv`.

| Coverage target | Threshold | Coverage | Error | Items kept |
|---|---|---|---|---|
| 100% | 0.6759 | 1.000 | 0.652 | 101438 |
| 90% | 0.5029 | 0.901 | 0.613 | 91425 |
| 75% | 0.4193 | 0.738 | 0.544 | 74865 |
| 50% | 0.3395 | 0.509 | 0.459 | 51619 |
| 25% | 0.2371 | 0.231 | 0.262 | 23397 |
| 10% | 0.1668 | 0.093 | 0.106 | 9479 |
| 5% | 0.1262 | 0.051 | 0.081 | 5133 |
| 1% | 0.0575 | 0.011 | 0.000 | 1068 |

- AURC (area under the risk-coverage curve, lower is better): **0.423**; a signal that ranks at random scores the full-coverage error, 0.652.
- Lowest error at ≥1% coverage: **0.000** (coverage 0.011).
- Target error α = 0.10: reachable on these items up to coverage **0.075**.
- Deployed threshold τ = 0.1668, set on the calibration split (coverage there 0.094; certifiable: True); here it keeps 0.447 of all items.

### Notes

- Causal Consistency is pair-level: both counterfactual arms must be correct. A constant-answer model scores 0.
- Abstentions count as failures for accuracy and causal consistency, and as non-violations for violation rate. Coverage is reported so this trade-off is visible.
- **Accuracy (strict)** counts an abstention as wrong and is the row-comparable number. Selective accuracy is in `summary_*.json`; it rises trivially as coverage falls and must never be compared across rows without coverage.
- Rows (5)-(7) add ONE contribution to the base model. Rows (1)-(4) are the cumulative ladder of the proposal's section 4.6. Row (8) is everything at once.
- **The gate's accuracy is an upper bound.** It evaluates the same constraint the ground truth was generated from, so on items it fires on it cannot be wrong unless extraction is. The figure to quote next to it is its firing rate here, and its coverage on unfiltered real notes (42.7%, `results/medcalc_ablation.md`).
- The constraint layer is scored through the identical readout as every other row (generate, then parse the first decision word), not by argmax over answer logits, so the numbers are comparable down the column.
