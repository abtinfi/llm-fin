## Ablation matrix -- split=`heldout`, model=`BioMistral/BioMistral-7B`, seeds=[0]

UQ engine defers on **`decision_entropy`**  (Eq. (2) restricted to the decision tokens, the respecified 4.7 term: AUROC 0.687 [0.675, 0.700] over 6,456 items. Rank-equivalent to `logit_margin` for K=2.)

Single seed. Decoding is greedy and therefore deterministic: another seed reproduces this run exactly, so no seed-to-seed spread is reported. The uncertainty that does exist is over items, and is given as a bootstrap CI in the next table.

| Variant | Base | RAG | Sym | UQ | CL | Accuracy (strict) | Δ Acc vs base | Causal Consistency | Δ CC vs base | Violation Rate | Coverage | Gate fired |
|---|:-:|:-:|:-:|:-:|:-:|---|---|---|---|---|---|---|
| (1) Base LLM | YES | - | - | - | - | 0.556 | +0.000 | 0.000 | -0.278 | 1.000 | 1.000 | 0.000 |
| (2) + RAG | YES | YES | - | - | - | 0.444 | -0.111 | 0.000 | -0.278 | 0.000 | 1.000 | 0.000 |
| (3) + Symbolic Gate (NS-AI) | YES | YES | YES | - | - | 1.000 | +0.444 | 1.000 | +0.722 | 0.000 | 1.000 | 1.000 |
| (4) + UQ Engine (NS-AI+UQ) | YES | YES | YES | YES | - | 1.000 | +0.444 | 1.000 | +0.722 | 0.000 | 1.000 | 1.000 |
| (5) Base + Symbolic Gate only | YES | - | YES | - | - | 1.000 | +0.444 | 1.000 | +0.722 | 0.000 | 1.000 | 1.000 |
| (6) Base + UQ only | YES | - | - | YES | - | 0.000 | -0.556 | 0.000 | -0.278 | 0.000 | 0.000 | 0.000 |
| (7) Base + Constraint Layer only | YES | - | - | - | YES | 0.889 | +0.333 | 0.767 | +0.489 | 0.104 | 1.000 | 0.000 |
| (8) All four contributions | YES | YES | YES | YES | YES | 1.000 | +0.444 | 1.000 | +0.722 | 0.000 | 1.000 | 1.000 |

### 95% bootstrap CI over items (seed 0)

| Variant | Accuracy | 95% CI | Causal Consistency | 95% CI |
|---|---|---|---|---|
| (1) Base LLM | 0.556 | [0.463, 0.648] | 0.278 | [0.167, 0.407] |
| (2) + RAG | 0.444 | [0.352, 0.537] | 0.167 | [0.074, 0.259] |
| (3) + Symbolic Gate (NS-AI) | 1.000 | [1.000, 1.000] | 1.000 | [1.000, 1.000] |
| (4) + UQ Engine (NS-AI+UQ) | 1.000 | [1.000, 1.000] | 1.000 | [1.000, 1.000] |
| (5) Base + Symbolic Gate only | 1.000 | [1.000, 1.000] | 1.000 | [1.000, 1.000] |
| (6) Base + UQ only | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] |
| (7) Base + Constraint Layer only | 0.889 | [0.824, 0.944] | 0.796 | [0.685, 0.889] |
| (8) All four contributions | 1.000 | [1.000, 1.000] | 1.000 | [1.000, 1.000] |

### Paired McNemar of each contribution **against the baseline** (seed 0, item-level correctness)

This is the supervisor's question: what does adding this contribution to the base model do?

| Variant vs base | B01 (base wrong→right) | B10 (base right→wrong) | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 48 | 60 | 0.2898 | 0.2898 |
| base→nsai | 48 | 0 | 7.105e-15 | 4.263e-14 |
| base→nsai_uq | 48 | 0 | 7.105e-15 | 4.263e-14 |
| base→sym | 48 | 0 | 7.105e-15 | 4.263e-14 |
| base→uq | 0 | 60 | 1.735e-18 | 1.214e-17 |
| base→cl | 43 | 7 | 2.099e-07 | 4.197e-07 |
| base→nsai_uq_cl | 48 | 0 | 7.105e-15 | 4.263e-14 |

### Paired McNemar along the cumulative ladder (seed 0)

| Comparison | B01 | B10 | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 48 | 60 | 0.2898 | 0.5796 |
| rag→nsai | 60 | 0 | 1.735e-18 | 5.204e-18 |
| nsai→nsai_uq | 0 | 0 | 1 | 1 |

### Where the gate's contribution comes from (seed 0)

The gate's accuracy on items it fires on is partly circular: it applies the same rule and threshold the labels were generated from, and on the MedCalc benchmark the same extractor that filters the data runs inside the gate. Splitting the split by whether the gate fired separates the circular part from the part that is not: on gate-declined items the row IS the neural pathway, so any difference there is real.

| Variant | subset | n | accuracy | base accuracy on the same subset |
|---|---|---|---|---|
| (3) + Symbolic Gate (NS-AI) | gate fired | 108 | 1.000 | 0.556 |
| (4) + UQ Engine (NS-AI+UQ) | gate fired | 108 | 1.000 | 0.556 |
| (5) Base + Symbolic Gate only | gate fired | 108 | 1.000 | 0.556 |
| (8) All four contributions | gate fired | 108 | 1.000 | 0.556 |

### Why a UQ row can read 0.000 coverage

Split conformal picks the largest uncertainty threshold whose error rate on the calibration split is at most alpha = 0.10. On this calibration split of 124 items no threshold reaches that target, because the model it is governing is near chance. The method then falls back to its most conservative threshold, which retains 2.4% of the calibration items, and on the test split retains none. **That is the method behaving correctly, not a failure to run**: a 10% error target is unreachable for a model at this accuracy, so the only way to honour it is to answer nothing. It is also the exact situation Adaptive Conformal Inference exists for -- see `results/uq_coverage_*.md`, where the threshold is allowed to move.


### Notes

- Causal Consistency is pair-level: both counterfactual arms must be correct. A constant-answer model scores 0.
- Abstentions count as failures for accuracy and causal consistency, and as non-violations for violation rate. Coverage is reported so this trade-off is visible.
- **Accuracy (strict)** counts an abstention as wrong and is the row-comparable number. Selective accuracy is in `summary_*.json`; it rises trivially as coverage falls and must never be compared across rows without coverage.
- Rows (5)-(7) add ONE contribution to the base model. Rows (1)-(4) are the cumulative ladder of the proposal's section 4.6. Row (8) is everything at once.
- **The gate's accuracy is an upper bound.** It evaluates the same constraint the ground truth was generated from, so on items it fires on it cannot be wrong unless extraction is. The figure to quote next to it is its firing rate here, and its coverage on unfiltered real notes (42.7%, `results/medcalc_ablation.md`).
- The constraint layer is scored through the identical readout as every other row (generate, then parse the first decision word), not by argmax over answer logits, so the numbers are comparable down the column.
