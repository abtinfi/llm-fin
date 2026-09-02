## Ablation matrix -- split=`test`, model=`BioMistral/BioMistral-7B`, seeds=[0]

UQ engine defers on **`entropy`**  (the proposal's Eq. (2))

Single seed. Decoding is greedy and therefore deterministic: another seed reproduces this run exactly, so no seed-to-seed spread is reported. The uncertainty that does exist is over items, and is given as a bootstrap CI in the next table.

| Variant | Base | RAG | Sym | UQ | CL | Accuracy (strict) | Δ Acc vs base | Causal Consistency | Δ CC vs base | Violation Rate | Coverage | Gate fired |
|---|:-:|:-:|:-:|:-:|:-:|---|---|---|---|---|---|---|
| (1) Base LLM | YES | - | - | - | - | 0.506 | +0.000 | 0.011 | +0.000 | 0.322 | 1.000 | 0.000 |
| (2) + RAG | YES | YES | - | - | - | 0.494 | -0.011 | 0.000 | -0.011 | 0.022 | 1.000 | 0.000 |
| (3) + Symbolic Gate (NS-AI) | YES | YES | YES | - | - | 0.994 | +0.489 | 0.989 | +0.978 | 0.000 | 1.000 | 0.989 |
| (4) + UQ Engine (NS-AI+UQ) | YES | YES | YES | YES | - | 0.994 | +0.489 | 0.989 | +0.978 | 0.000 | 1.000 | 0.989 |
| (5) Base + Symbolic Gate only | YES | - | YES | - | - | 0.994 | +0.489 | 0.989 | +0.978 | 0.000 | 1.000 | 0.989 |
| (6) Base + UQ only | YES | - | - | YES | - | 0.028 | -0.478 | 0.000 | -0.011 | 0.000 | 0.050 | 0.000 |
| (7) Base + Constraint Layer only | YES | - | - | - | YES | 0.506 | +0.000 | 0.022 | +0.011 | 0.044 | 1.000 | 0.000 |
| (8) All four contributions | YES | YES | YES | YES | YES | 0.994 | +0.489 | 0.989 | +0.978 | 0.000 | 1.000 | 0.989 |

### 95% bootstrap CI over items (seed 0)

| Variant | Accuracy | 95% CI | Causal Consistency | 95% CI |
|---|---|---|---|---|
| (1) Base LLM | 0.506 | [0.433, 0.578] | 0.011 | [0.000, 0.033] |
| (2) + RAG | 0.494 | [0.422, 0.572] | 0.000 | [0.000, 0.000] |
| (3) + Symbolic Gate (NS-AI) | 0.994 | [0.983, 1.000] | 0.989 | [0.967, 1.000] |
| (4) + UQ Engine (NS-AI+UQ) | 0.994 | [0.983, 1.000] | 0.989 | [0.967, 1.000] |
| (5) Base + Symbolic Gate only | 0.994 | [0.983, 1.000] | 0.989 | [0.967, 1.000] |
| (6) Base + UQ only | 0.028 | [0.006, 0.056] | 0.000 | [0.000, 0.000] |
| (7) Base + Constraint Layer only | 0.506 | [0.433, 0.578] | 0.022 | [0.000, 0.056] |
| (8) All four contributions | 0.994 | [0.983, 1.000] | 0.989 | [0.967, 1.000] |

### Paired McNemar of each contribution **against the baseline** (seed 0, item-level correctness)

This is the supervisor's question: what does adding this contribution to the base model do?

| Variant vs base | B01 (base wrong→right) | B10 (base right→wrong) | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 28 | 30 | 0.8957 | 1 |
| base→nsai | 88 | 0 | 6.462e-27 | 4.524e-26 |
| base→nsai_uq | 88 | 0 | 6.462e-27 | 4.524e-26 |
| base→sym | 88 | 0 | 6.462e-27 | 4.524e-26 |
| base→uq | 0 | 86 | 2.585e-26 | 7.755e-26 |
| base→cl | 25 | 25 | 1 | 1 |
| base→nsai_uq_cl | 88 | 0 | 6.462e-27 | 4.524e-26 |

### Paired McNemar along the cumulative ladder (seed 0)

| Comparison | B01 | B10 | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 28 | 30 | 0.8957 | 1 |
| rag→nsai | 90 | 0 | 1.616e-27 | 4.847e-27 |
| nsai→nsai_uq | 0 | 0 | 1 | 1 |

### Where the gate's contribution comes from (seed 0)

The gate's accuracy on items it fires on is partly circular: it applies the same rule and threshold the labels were generated from, and on the MedCalc benchmark the same extractor that filters the data runs inside the gate. Splitting the split by whether the gate fired separates the circular part from the part that is not: on gate-declined items the row IS the neural pathway, so any difference there is real.

| Variant | subset | n | accuracy | base accuracy on the same subset |
|---|---|---|---|---|
| (3) + Symbolic Gate (NS-AI) | gate fired | 178 | 1.000 | 0.506 |
| (3) + Symbolic Gate (NS-AI) | gate declined | 2 | 0.500 | 0.500 |
| (4) + UQ Engine (NS-AI+UQ) | gate fired | 178 | 1.000 | 0.506 |
| (4) + UQ Engine (NS-AI+UQ) | gate declined | 2 | 0.500 | 0.500 |
| (5) Base + Symbolic Gate only | gate fired | 178 | 1.000 | 0.506 |
| (5) Base + Symbolic Gate only | gate declined | 2 | 0.500 | 0.500 |
| (8) All four contributions | gate fired | 178 | 1.000 | 0.506 |
| (8) All four contributions | gate declined | 2 | 0.500 | 0.500 |

### Notes

- Causal Consistency is pair-level: both counterfactual arms must be correct. A constant-answer model scores 0.
- Abstentions count as failures for accuracy and causal consistency, and as non-violations for violation rate. Coverage is reported so this trade-off is visible.
- **Accuracy (strict)** counts an abstention as wrong and is the row-comparable number. Selective accuracy is in `summary_*.json`; it rises trivially as coverage falls and must never be compared across rows without coverage.
- Rows (5)-(7) add ONE contribution to the base model. Rows (1)-(4) are the cumulative ladder of the proposal's section 4.6. Row (8) is everything at once.
- **The gate's accuracy is an upper bound.** It evaluates the same constraint the ground truth was generated from, so on items it fires on it cannot be wrong unless extraction is. The figure to quote next to it is its firing rate here, and its coverage on unfiltered real notes (42.7%, `results/medcalc_ablation.md`).
- The constraint layer is scored through the identical readout as every other row (generate, then parse the first decision word), not by argmax over answer logits, so the numbers are comparable down the column.
