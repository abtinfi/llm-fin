## Ablation matrix -- split=`heldout`, model=`mistralai/Mistral-7B-Instruct-v0.2`, seeds=[0]

UQ engine defers on **`decision_entropy`**  (Eq. (2) restricted to the decision tokens, the respecified 4.7 term: AUROC 0.687 [0.675, 0.700] over 6,456 items. Rank-equivalent to `logit_margin` for K=2.)

Single seed. Decoding is greedy and therefore deterministic: another seed reproduces this run exactly, so no seed-to-seed spread is reported. The uncertainty that does exist is over items, and is given as a bootstrap CI in the next table.

| Variant | Base | RAG | Sym | UQ | CL | Accuracy (strict) | Δ Acc vs base | Causal Consistency | Δ CC vs base | Violation Rate | Coverage | Gate fired |
|---|:-:|:-:|:-:|:-:|:-:|---|---|---|---|---|---|---|
| (1) Base LLM | YES | - | - | - | - | 0.500 | +0.000 | 0.000 | +0.000 | 1.000 | 1.000 | 0.000 |
| (2) + RAG | YES | YES | - | - | - | 0.562 | +0.062 | 0.125 | +0.125 | 0.000 | 1.000 | 0.000 |
| (3) + Symbolic Gate (NS-AI) | YES | YES | YES | - | - | 0.812 | +0.312 | 0.625 | +0.625 | 0.000 | 1.000 | 0.500 |
| (4) + UQ Engine (NS-AI+UQ) | YES | YES | YES | YES | - | 0.688 | +0.188 | 0.500 | +0.500 | 0.000 | 0.844 | 0.500 |
| (5) Base + Symbolic Gate only | YES | - | YES | - | - | 0.750 | +0.250 | 0.500 | +0.500 | 0.500 | 1.000 | 0.500 |
| (6) Base + UQ only | YES | - | - | YES | - | 0.062 | -0.438 | 0.000 | +0.000 | 0.000 | 0.062 | 0.000 |
| (7) Base + Constraint Layer only | YES | - | - | - | YES | 0.500 | +0.000 | 0.000 | +0.000 | 1.000 | 1.000 | 0.000 |
| (8) All four contributions | YES | YES | YES | YES | YES | 0.688 | +0.188 | 0.500 | +0.500 | 0.000 | 0.875 | 0.500 |

### 95% bootstrap CI over items (seed 0)

| Variant | Accuracy | 95% CI | Causal Consistency | 95% CI |
|---|---|---|---|---|
| (1) Base LLM | 0.500 | [0.312, 0.656] | 0.000 | [0.000, 0.000] |
| (2) + RAG | 0.562 | [0.406, 0.719] | 0.125 | [0.000, 0.312] |
| (3) + Symbolic Gate (NS-AI) | 0.812 | [0.656, 0.938] | 0.625 | [0.375, 0.875] |
| (4) + UQ Engine (NS-AI+UQ) | 0.688 | [0.531, 0.844] | 0.500 | [0.250, 0.750] |
| (5) Base + Symbolic Gate only | 0.750 | [0.594, 0.875] | 0.500 | [0.250, 0.750] |
| (6) Base + UQ only | 0.062 | [0.000, 0.156] | 0.000 | [0.000, 0.000] |
| (7) Base + Constraint Layer only | 0.500 | [0.312, 0.656] | 0.000 | [0.000, 0.000] |
| (8) All four contributions | 0.688 | [0.531, 0.844] | 0.500 | [0.250, 0.750] |

### Paired McNemar of each contribution **against the baseline** (seed 0, item-level correctness)

This is the supervisor's question: what does adding this contribution to the base model do?

| Variant vs base | B01 (base wrong→right) | B10 (base right→wrong) | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 16 | 14 | 0.8555 | 1 |
| base→nsai | 16 | 6 | 0.05248 | 0.2624 |
| base→nsai_uq | 14 | 8 | 0.2863 | 1 |
| base→sym | 8 | 0 | 0.007812 | 0.04688 |
| base→uq | 0 | 14 | 0.0001221 | 0.0008545 |
| base→cl | 0 | 0 | 1 | 1 |
| base→nsai_uq_cl | 14 | 8 | 0.2863 | 1 |

### Paired McNemar along the cumulative ladder (seed 0)

| Comparison | B01 | B10 | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 16 | 14 | 0.8555 | 0.8555 |
| rag→nsai | 8 | 0 | 0.007812 | 0.02344 |
| nsai→nsai_uq | 0 | 4 | 0.125 | 0.25 |

### Where the gate's contribution comes from (seed 0)

The gate's accuracy on items it fires on is partly circular: it applies the same rule and threshold the labels were generated from, and on the MedCalc benchmark the same extractor that filters the data runs inside the gate. Splitting the split by whether the gate fired separates the circular part from the part that is not: on gate-declined items the row IS the neural pathway, so any difference there is real.

| Variant | subset | n | accuracy | base accuracy on the same subset |
|---|---|---|---|---|
| (3) + Symbolic Gate (NS-AI) | gate fired | 16 | 1.000 | 0.500 |
| (3) + Symbolic Gate (NS-AI) | gate declined | 16 | 0.625 | 0.500 |
| (4) + UQ Engine (NS-AI+UQ) | gate fired | 16 | 1.000 | 0.500 |
| (4) + UQ Engine (NS-AI+UQ) | gate declined | 16 | 0.375 | 0.500 |
| (5) Base + Symbolic Gate only | gate fired | 16 | 1.000 | 0.500 |
| (5) Base + Symbolic Gate only | gate declined | 16 | 0.500 | 0.500 |
| (8) All four contributions | gate fired | 16 | 1.000 | 0.500 |
| (8) All four contributions | gate declined | 16 | 0.375 | 0.500 |

### Notes

- Causal Consistency is pair-level: both counterfactual arms must be correct. A constant-answer model scores 0.
- Abstentions count as failures for accuracy and causal consistency, and as non-violations for violation rate. Coverage is reported so this trade-off is visible.
- **Accuracy (strict)** counts an abstention as wrong and is the row-comparable number. Selective accuracy is in `summary_*.json`; it rises trivially as coverage falls and must never be compared across rows without coverage.
- Rows (5)-(7) add ONE contribution to the base model. Rows (1)-(4) are the cumulative ladder of the proposal's section 4.6. Row (8) is everything at once.
- **The gate's accuracy is an upper bound.** It evaluates the same constraint the ground truth was generated from, so on items it fires on it cannot be wrong unless extraction is. The figure to quote next to it is its firing rate here, and its coverage on unfiltered real notes (42.7%, `results/medcalc_ablation.md`).
- The constraint layer is scored through the identical readout as every other row (generate, then parse the first decision word), not by argmax over answer logits, so the numbers are comparable down the column.
