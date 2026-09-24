## Ablation matrix -- split=`heldout`, model=`mistralai/Mistral-7B-Instruct-v0.2`, seeds=[0]

UQ engine defers on **`decision_entropy`**  (Eq. (2) restricted to the decision tokens, the respecified 4.7 term: AUROC 0.687 [0.675, 0.700] over 6,456 items. Rank-equivalent to `logit_margin` for K=2.)

Single seed. Decoding is greedy and therefore deterministic: another seed reproduces this run exactly, so no seed-to-seed spread is reported. The uncertainty that does exist is over items, and is given as a bootstrap CI in the next table.

### Primary: the section 4.6 ablation ladder

| Variant | Base | RAG | Sym | UQ | CL | Accuracy (strict) | Δ Acc vs base | Model's own answer acc. | Causal Consistency | Δ CC vs base | Violation Rate | Coverage | Gate fired |
|---|:-:|:-:|:-:|:-:|:-:|---|---|---|---|---|---|---|---|
| (1) Base LLM | YES | - | - | - | - | 0.704 | +0.000 | 0.704 | 0.032 | +0.000 | 0.889 | 1.000 | 0.000 |
| (2) + RAG | YES | YES | - | - | - | 0.844 | +0.140 | 0.844 | 0.550 | +0.518 | 0.274 | 1.000 | 0.000 |
| (3) + Symbolic Gate (NS-AI) | YES | YES | YES | - | - | 1.000 † | +0.296 | 0.844 | 1.000 † | +0.968 | 0.000 | 1.000 | 1.000 |
| (4) + UQ Engine (NS-AI+UQ) | YES | YES | YES | YES | - | 1.000 † | +0.296 | 0.844 | 1.000 † | +0.968 | 0.000 | 1.000 | 1.000 |

**Model's own answer acc.** is the decision word parsed from the model's generation, before the gate overrides it and before UQ defers it: what the LLM itself concluded under that row's prompt. † Accuracy and CC on gate-fired items are an identity check, not a measurement: the gate applies the rule and threshold the labels were generated from. With the gate firing on every item, that row's accuracy is 1.000 by construction; read the model's own column, and the adherence table below, for what the model knows.


### Supplementary ablations: one contribution added to the base model

| Variant | Base | RAG | Sym | UQ | CL | Accuracy (strict) | Δ Acc vs base | Model's own answer acc. | Causal Consistency | Δ CC vs base | Violation Rate | Coverage | Gate fired |
|---|:-:|:-:|:-:|:-:|:-:|---|---|---|---|---|---|---|---|
| (5) Base + Symbolic Gate only | YES | - | YES | - | - | 1.000 † | +0.296 | 0.704 | 1.000 † | +0.968 | 0.000 | 1.000 | 1.000 |
| (6) Base + UQ only | YES | - | - | YES | - | 0.003 | -0.701 | 0.704 | 0.000 | -0.032 | 0.000 | 0.003 | 0.000 |

### 95% bootstrap CI over items (seed 0)

| Variant | Accuracy | 95% CI | Causal Consistency | 95% CI |
|---|---|---|---|---|
| (1) Base LLM | 0.704 | [0.699, 0.709] | 0.032 | [0.029, 0.036] |
| (2) + RAG | 0.844 | [0.840, 0.848] | 0.550 | [0.540, 0.560] |
| (3) + Symbolic Gate (NS-AI) | 1.000 | [1.000, 1.000] | 1.000 | [1.000, 1.000] |
| (4) + UQ Engine (NS-AI+UQ) | 1.000 | [1.000, 1.000] | 1.000 | [1.000, 1.000] |
| (5) Base + Symbolic Gate only | 1.000 | [1.000, 1.000] | 1.000 | [1.000, 1.000] |
| (6) Base + UQ only | 0.003 | [0.003, 0.004] | 0.000 | [0.000, 0.000] |

### Paired McNemar of each contribution **against the baseline** (seed 0, item-level correctness)

This is the supervisor's question: what does adding this contribution to the base model do?

| Variant vs base | B01 (base wrong→right) | B10 (base right→wrong) | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 7481 | 2362 | 0 | 0 |
| base→nsai | 10816 | 0 | 0 | 0 |
| base→nsai_uq | 10816 | 0 | 0 | 0 |
| base→sym | 10816 | 0 | 0 | 0 |
| base→uq | 0 | 25606 | 0 | 0 |

### Paired McNemar along the cumulative ladder (seed 0)

| Comparison | B01 | B10 | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 7481 | 2362 | 0 | 0 |
| rag→nsai | 5697 | 0 | 0 | 0 |
| nsai→nsai_uq | 0 | 0 | 1 | 1 |

### Where the gate's contribution comes from (seed 0)

The gate's accuracy on items it fires on is partly circular: it applies the same rule and threshold the labels were generated from, and on the MedCalc benchmark the same extractor that filters the data runs inside the gate. Splitting the split by whether the gate fired separates the circular part from the part that is not: on gate-declined items the row IS the neural pathway, so any difference there is real.

On gate-fired items the three accuracy columns separate the two things the question conflates. *Final* is the rule checking itself (identity). *Model's own answer* is whether the LLM, given that row's prompt, reached the guideline's conclusion without the rule's help: that is guideline adherence. *Agrees with gate* is how often the gate merely confirmed the model rather than overruled it.

| Variant | subset | n | Final accuracy | Model's own answer acc. | Base accuracy, same items | Model agrees with gate |
|---|---|---|---|---|---|---|
| (3) + Symbolic Gate (NS-AI) | gate fired | 36542 | 1.000 (identity) | 0.844 | 0.704 | 0.844 |
| (4) + UQ Engine (NS-AI+UQ) | gate fired | 36542 | 1.000 (identity) | 0.844 | 0.704 | 0.844 |
| (5) Base + Symbolic Gate only | gate fired | 36542 | 1.000 (identity) | 0.704 | 0.704 | 0.704 |

**No non-circular evidence on this split for (3) + Symbolic Gate (NS-AI), (4) + UQ Engine (NS-AI+UQ), (5) Base + Symbolic Gate only:** the gate fired on every item, so there is no gate-declined subset on which the row's accuracy measures anything but the rule. The gate's contribution here must be reported as its coverage, and the model's own answer as the accuracy.


### Risk-coverage of the UQ signal, (6) Base + UQ only (seed 0)

Every threshold a deferral rule could pick, on the 36542 items the UQ engine governs here (gate-decided items are never deferred and are excluded). Error is of the model's own answer on the items kept. The signal (`decision_entropy`) takes 337 distinct values, so 337 operating points exist; the rows below are those nearest each coverage level. Full curve: `table_mimic3b_heldout_riskcov_uq.csv`.

| Coverage target | Threshold | Coverage | Error | Items kept |
|---|---|---|---|---|
| 100% | 0.6931 | 1.000 | 0.296 | 36542 |
| 90% | 0.003018 | 0.899 | 0.277 | 32843 |
| 75% | 0.0006632 | 0.747 | 0.274 | 27302 |
| 50% | 0.0001129 | 0.503 | 0.248 | 18374 |
| 25% | 2.986e-05 | 0.255 | 0.221 | 9318 |
| 10% | 1.11e-05 | 0.103 | 0.195 | 3757 |
| 5% | 6.186e-06 | 0.051 | 0.160 | 1864 |
| 1% | 2.152e-06 | 0.010 | 0.089 | 370 |

- AURC (area under the risk-coverage curve, lower is better): **0.239**; a signal that ranks at random scores the full-coverage error, 0.296.
- Lowest error at ≥1% coverage: **0.089** (coverage 0.010).
- Target error α = 0.10: reachable on these items up to coverage **0.018**.
- Deployed threshold τ = 7.452e-07, set on the calibration split (coverage there 0.099; certifiable: True); here it keeps 0.003 of all items. The calibration split holds the training families, so on the held-out family τ is transferred, not fitted.

### Notes

- Causal Consistency is pair-level: both counterfactual arms must be correct. A constant-answer model scores 0.
- Abstentions count as failures for accuracy and causal consistency, and as non-violations for violation rate. Coverage is reported so this trade-off is visible.
- **Accuracy (strict)** counts an abstention as wrong and is the row-comparable number. Selective accuracy is in `summary_*.json`; it rises trivially as coverage falls and must never be compared across rows without coverage.
- Rows (5)-(7) add ONE contribution to the base model. Rows (1)-(4) are the cumulative ladder of the proposal's section 4.6. Row (8) is everything at once.
- **The gate's accuracy is an upper bound.** It evaluates the same constraint the ground truth was generated from, so on items it fires on it cannot be wrong unless extraction is. The figure to quote next to it is its firing rate here, and its coverage on unfiltered real notes (42.7%, `results/medcalc_ablation.md`).
- The constraint layer is scored through the identical readout as every other row (generate, then parse the first decision word), not by argmax over answer logits, so the numbers are comparable down the column.
