## Ablation matrix -- split=`test`, model=`BioMistral/BioMistral-7B`, seeds=[0]

UQ engine defers on **`decision_entropy`**  (Eq. (2) restricted to the decision tokens, the respecified 4.7 term: AUROC 0.687 [0.675, 0.700] over 6,456 items. Rank-equivalent to `logit_margin` for K=2.)

Single seed. Decoding is greedy and therefore deterministic: another seed reproduces this run exactly, so no seed-to-seed spread is reported. The uncertainty that does exist is over items, and is given as a bootstrap CI in the next table.

| Variant | Base | RAG | Sym | UQ | CL | Accuracy (strict) | Δ Acc vs base | Causal Consistency | Δ CC vs base | Violation Rate | Coverage | Gate fired |
|---|:-:|:-:|:-:|:-:|:-:|---|---|---|---|---|---|---|
| (1) Base LLM | YES | - | - | - | - | 0.534 | +0.000 | 0.069 | +0.000 | 0.862 | 1.000 | 0.000 |
| (2) + RAG | YES | YES | - | - | - | 0.500 | -0.034 | 0.000 | -0.069 | 0.000 | 1.000 | 0.000 |
| (3) + Symbolic Gate (NS-AI) | YES | YES | YES | - | - | 0.690 | +0.155 | 0.379 | +0.310 | 0.000 | 1.000 | 0.379 |
| (4) + UQ Engine (NS-AI+UQ) | YES | YES | YES | YES | - | 0.379 | -0.155 | 0.379 | +0.310 | 0.000 | 0.379 | 0.379 |
| (5) Base + Symbolic Gate only | YES | - | YES | - | - | 0.690 | +0.155 | 0.379 | +0.310 | 0.621 | 1.000 | 0.379 |
| (6) Base + UQ only | YES | - | - | YES | - | 0.034 | -0.500 | 0.000 | -0.069 | 0.034 | 0.052 | 0.000 |
| (7) Base + Constraint Layer only | YES | - | - | - | YES | 0.534 | +0.000 | 0.069 | +0.000 | 0.552 | 1.000 | 0.000 |
| (8) All four contributions | YES | YES | YES | YES | YES | 0.379 | -0.155 | 0.379 | +0.310 | 0.000 | 0.379 | 0.379 |

### 95% bootstrap CI over items (seed 0)

| Variant | Accuracy | 95% CI | Causal Consistency | 95% CI |
|---|---|---|---|---|
| (1) Base LLM | 0.534 | [0.397, 0.655] | 0.069 | [0.000, 0.172] |
| (2) + RAG | 0.500 | [0.362, 0.621] | 0.000 | [0.000, 0.000] |
| (3) + Symbolic Gate (NS-AI) | 0.690 | [0.569, 0.810] | 0.379 | [0.207, 0.552] |
| (4) + UQ Engine (NS-AI+UQ) | 0.379 | [0.259, 0.500] | 0.379 | [0.207, 0.552] |
| (5) Base + Symbolic Gate only | 0.690 | [0.569, 0.810] | 0.379 | [0.207, 0.552] |
| (6) Base + UQ only | 0.034 | [0.000, 0.086] | 0.000 | [0.000, 0.000] |
| (7) Base + Constraint Layer only | 0.534 | [0.397, 0.655] | 0.069 | [0.000, 0.172] |
| (8) All four contributions | 0.379 | [0.259, 0.500] | 0.379 | [0.207, 0.552] |

### Paired McNemar of each contribution **against the baseline** (seed 0, item-level correctness)

This is the supervisor's question: what does adding this contribution to the base model do?

| Variant vs base | B01 (base wrong→right) | B10 (base right→wrong) | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 25 | 27 | 0.8899 | 1 |
| base→nsai | 27 | 18 | 0.2327 | 0.6981 |
| base→nsai_uq | 9 | 18 | 0.1221 | 0.6104 |
| base→sym | 9 | 0 | 0.003906 | 0.02344 |
| base→uq | 0 | 29 | 3.725e-09 | 2.608e-08 |
| base→cl | 9 | 9 | 1 | 1 |
| base→nsai_uq_cl | 9 | 18 | 0.1221 | 0.6104 |

### Paired McNemar along the cumulative ladder (seed 0)

| Comparison | B01 | B10 | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 25 | 27 | 0.8899 | 0.8899 |
| rag→nsai | 11 | 0 | 0.0009766 | 0.001953 |
| nsai→nsai_uq | 0 | 18 | 7.629e-06 | 2.289e-05 |

### Where the gate's contribution comes from (seed 0)

The gate's accuracy on items it fires on is partly circular: it applies the same rule and threshold the labels were generated from, and on the MedCalc benchmark the same extractor that filters the data runs inside the gate. Splitting the split by whether the gate fired separates the circular part from the part that is not: on gate-declined items the row IS the neural pathway, so any difference there is real.

| Variant | subset | n | accuracy | base accuracy on the same subset |
|---|---|---|---|---|
| (3) + Symbolic Gate (NS-AI) | gate fired | 22 | 1.000 | 0.591 |
| (3) + Symbolic Gate (NS-AI) | gate declined | 36 | 0.500 | 0.500 |
| (4) + UQ Engine (NS-AI+UQ) | gate fired | 22 | 1.000 | 0.591 |
| (4) + UQ Engine (NS-AI+UQ) | gate declined | 36 | 0.000 | 0.500 |
| (5) Base + Symbolic Gate only | gate fired | 22 | 1.000 | 0.591 |
| (5) Base + Symbolic Gate only | gate declined | 36 | 0.500 | 0.500 |
| (8) All four contributions | gate fired | 22 | 1.000 | 0.591 |
| (8) All four contributions | gate declined | 36 | 0.000 | 0.500 |

### Notes

- Causal Consistency is pair-level: both counterfactual arms must be correct. A constant-answer model scores 0.
- Abstentions count as failures for accuracy and causal consistency, and as non-violations for violation rate. Coverage is reported so this trade-off is visible.
- **Accuracy (strict)** counts an abstention as wrong and is the row-comparable number. Selective accuracy is in `summary_*.json`; it rises trivially as coverage falls and must never be compared across rows without coverage.
- Rows (5)-(7) add ONE contribution to the base model. Rows (1)-(4) are the cumulative ladder of the proposal's section 4.6. Row (8) is everything at once.
- **The gate's accuracy is an upper bound.** It evaluates the same constraint the ground truth was generated from, so on items it fires on it cannot be wrong unless extraction is. The figure to quote next to it is its firing rate here, and its coverage on unfiltered real notes (42.7%, `results/medcalc_ablation.md`).
- The constraint layer is scored through the identical readout as every other row (generate, then parse the first decision word), not by argmax over answer logits, so the numbers are comparable down the column.
