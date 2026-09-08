## Ablation matrix -- split=`test`, model=`mistralai/Mistral-7B-Instruct-v0.2`, seeds=[0]

UQ engine defers on **`decision_entropy`**  (Eq. (2) restricted to the decision tokens, the respecified 4.7 term: AUROC 0.687 [0.675, 0.700] over 6,456 items. Rank-equivalent to `logit_margin` for K=2.)

Single seed. Decoding is greedy and therefore deterministic: another seed reproduces this run exactly, so no seed-to-seed spread is reported. The uncertainty that does exist is over items, and is given as a bootstrap CI in the next table.

| Variant | Base | RAG | Sym | UQ | CL | Accuracy (strict) | Δ Acc vs base | Causal Consistency | Δ CC vs base | Violation Rate | Coverage | Gate fired |
|---|:-:|:-:|:-:|:-:|:-:|---|---|---|---|---|---|---|
| (1) Base LLM | YES | - | - | - | - | 0.500 | +0.000 | 0.000 | +0.000 | 1.000 | 1.000 | 0.000 |
| (2) + RAG | YES | YES | - | - | - | 0.552 | +0.052 | 0.138 | +0.138 | 0.034 | 1.000 | 0.000 |
| (3) + Symbolic Gate (NS-AI) | YES | YES | YES | - | - | 0.672 | +0.172 | 0.379 | +0.379 | 0.034 | 1.000 | 0.379 |
| (4) + UQ Engine (NS-AI+UQ) | YES | YES | YES | YES | - | 0.397 | -0.103 | 0.379 | +0.379 | 0.000 | 0.448 | 0.379 |
| (5) Base + Symbolic Gate only | YES | - | YES | - | - | 0.690 | +0.190 | 0.379 | +0.379 | 0.621 | 1.000 | 0.379 |
| (6) Base + UQ only | YES | - | - | YES | - | 0.069 | -0.431 | 0.000 | +0.000 | 0.069 | 0.103 | 0.000 |
| (7) Base + Constraint Layer only | YES | - | - | - | YES | 0.603 | +0.103 | 0.207 | +0.207 | 0.172 | 1.000 | 0.000 |
| (8) All four contributions | YES | YES | YES | YES | YES | 0.414 | -0.086 | 0.379 | +0.379 | 0.000 | 0.466 | 0.379 |

### 95% bootstrap CI over items (seed 0)

| Variant | Accuracy | 95% CI | Causal Consistency | 95% CI |
|---|---|---|---|---|
| (1) Base LLM | 0.500 | [0.379, 0.638] | 0.000 | [0.000, 0.000] |
| (2) + RAG | 0.552 | [0.414, 0.690] | 0.138 | [0.034, 0.276] |
| (3) + Symbolic Gate (NS-AI) | 0.672 | [0.552, 0.793] | 0.379 | [0.207, 0.552] |
| (4) + UQ Engine (NS-AI+UQ) | 0.397 | [0.276, 0.517] | 0.379 | [0.207, 0.552] |
| (5) Base + Symbolic Gate only | 0.690 | [0.569, 0.810] | 0.379 | [0.207, 0.552] |
| (6) Base + UQ only | 0.069 | [0.017, 0.138] | 0.000 | [0.000, 0.000] |
| (7) Base + Constraint Layer only | 0.603 | [0.483, 0.724] | 0.207 | [0.069, 0.379] |
| (8) All four contributions | 0.414 | [0.293, 0.534] | 0.379 | [0.207, 0.552] |

### Paired McNemar of each contribution **against the baseline** (seed 0, item-level correctness)

This is the supervisor's question: what does adding this contribution to the base model do?

| Variant vs base | B01 (base wrong→right) | B10 (base right→wrong) | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 28 | 25 | 0.7838 | 1 |
| base→nsai | 28 | 18 | 0.1839 | 0.9196 |
| base→nsai_uq | 12 | 18 | 0.3616 | 1 |
| base→sym | 11 | 0 | 0.0009766 | 0.005859 |
| base→uq | 0 | 25 | 5.96e-08 | 4.172e-07 |
| base→cl | 24 | 18 | 0.4408 | 1 |
| base→nsai_uq_cl | 13 | 18 | 0.4731 | 1 |

### Paired McNemar along the cumulative ladder (seed 0)

| Comparison | B01 | B10 | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 28 | 25 | 0.7838 | 0.7838 |
| rag→nsai | 7 | 0 | 0.01562 | 0.03125 |
| nsai→nsai_uq | 0 | 16 | 3.052e-05 | 9.155e-05 |

### Where the gate's contribution comes from (seed 0)

The gate's accuracy on items it fires on is partly circular: it applies the same rule and threshold the labels were generated from, and on the MedCalc benchmark the same extractor that filters the data runs inside the gate. Splitting the split by whether the gate fired separates the circular part from the part that is not: on gate-declined items the row IS the neural pathway, so any difference there is real.

| Variant | subset | n | accuracy | base accuracy on the same subset |
|---|---|---|---|---|
| (3) + Symbolic Gate (NS-AI) | gate fired | 22 | 1.000 | 0.500 |
| (3) + Symbolic Gate (NS-AI) | gate declined | 36 | 0.472 | 0.500 |
| (4) + UQ Engine (NS-AI+UQ) | gate fired | 22 | 1.000 | 0.500 |
| (4) + UQ Engine (NS-AI+UQ) | gate declined | 36 | 0.028 | 0.500 |
| (5) Base + Symbolic Gate only | gate fired | 22 | 1.000 | 0.500 |
| (5) Base + Symbolic Gate only | gate declined | 36 | 0.500 | 0.500 |
| (8) All four contributions | gate fired | 22 | 1.000 | 0.500 |
| (8) All four contributions | gate declined | 36 | 0.056 | 0.500 |

### Notes

- Causal Consistency is pair-level: both counterfactual arms must be correct. A constant-answer model scores 0.
- Abstentions count as failures for accuracy and causal consistency, and as non-violations for violation rate. Coverage is reported so this trade-off is visible.
- **Accuracy (strict)** counts an abstention as wrong and is the row-comparable number. Selective accuracy is in `summary_*.json`; it rises trivially as coverage falls and must never be compared across rows without coverage.
- Rows (5)-(7) add ONE contribution to the base model. Rows (1)-(4) are the cumulative ladder of the proposal's section 4.6. Row (8) is everything at once.
- **The gate's accuracy is an upper bound.** It evaluates the same constraint the ground truth was generated from, so on items it fires on it cannot be wrong unless extraction is. The figure to quote next to it is its firing rate here, and its coverage on unfiltered real notes (42.7%, `results/medcalc_ablation.md`).
- The constraint layer is scored through the identical readout as every other row (generate, then parse the first decision word), not by argmax over answer logits, so the numbers are comparable down the column.
