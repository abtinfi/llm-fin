## Ablation matrix -- split=`test`, model=`BioMistral/BioMistral-7B`, seeds=[0]

UQ engine defers on **`decision_entropy`**  (Eq. (2) restricted to the decision tokens, the respecified 4.7 term: AUROC 0.687 [0.675, 0.700] over 6,456 items. Rank-equivalent to `logit_margin` for K=2.)

Single seed. Decoding is greedy and therefore deterministic: another seed reproduces this run exactly, so no seed-to-seed spread is reported. The uncertainty that does exist is over items, and is given as a bootstrap CI in the next table.

| Variant | Base | RAG | Sym | UQ | CL | Accuracy (strict) | Δ Acc vs base | Causal Consistency | Δ CC vs base | Violation Rate | Coverage | Gate fired |
|---|:-:|:-:|:-:|:-:|:-:|---|---|---|---|---|---|---|
| (1) Base LLM | YES | - | - | - | - | 0.602 | +0.000 | 0.219 | +0.000 | 0.125 | 1.000 | 0.000 |
| (2) + RAG | YES | YES | - | - | - | 0.648 | +0.047 | 0.312 | +0.094 | 0.141 | 1.000 | 0.000 |
| (3) + Symbolic Gate (NS-AI) | YES | YES | YES | - | - | 0.836 | +0.234 | 0.672 | +0.453 | 0.125 | 1.000 | 0.562 |
| (4) + UQ Engine (NS-AI+UQ) | YES | YES | YES | YES | - | 0.633 | +0.031 | 0.562 | +0.344 | 0.000 | 0.633 | 0.562 |
| (5) Base + Symbolic Gate only | YES | - | YES | - | - | 0.805 | +0.203 | 0.609 | +0.391 | 0.062 | 1.000 | 0.562 |
| (6) Base + UQ only | YES | - | - | YES | - | 0.211 | -0.391 | 0.016 | -0.203 | 0.031 | 0.234 | 0.000 |
| (7) Base + Constraint Layer only | YES | - | - | - | YES | 0.727 | +0.125 | 0.453 | +0.234 | 0.375 | 1.000 | 0.000 |
| (8) All four contributions | YES | YES | YES | YES | YES | 0.578 | -0.023 | 0.562 | +0.344 | 0.000 | 0.578 | 0.562 |

### 95% bootstrap CI over items (seed 0)

| Variant | Accuracy | 95% CI | Causal Consistency | 95% CI |
|---|---|---|---|---|
| (1) Base LLM | 0.602 | [0.516, 0.688] | 0.219 | [0.125, 0.328] |
| (2) + RAG | 0.648 | [0.562, 0.727] | 0.312 | [0.203, 0.422] |
| (3) + Symbolic Gate (NS-AI) | 0.836 | [0.766, 0.898] | 0.672 | [0.562, 0.781] |
| (4) + UQ Engine (NS-AI+UQ) | 0.633 | [0.547, 0.719] | 0.562 | [0.438, 0.688] |
| (5) Base + Symbolic Gate only | 0.805 | [0.734, 0.867] | 0.609 | [0.484, 0.734] |
| (6) Base + UQ only | 0.211 | [0.141, 0.281] | 0.016 | [0.000, 0.047] |
| (7) Base + Constraint Layer only | 0.727 | [0.648, 0.805] | 0.453 | [0.328, 0.578] |
| (8) All four contributions | 0.578 | [0.492, 0.664] | 0.562 | [0.438, 0.688] |

### Paired McNemar of each contribution **against the baseline** (seed 0, item-level correctness)

This is the supervisor's question: what does adding this contribution to the base model do?

| Variant vs base | B01 (base wrong→right) | B10 (base right→wrong) | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 18 | 12 | 0.3616 | 1 |
| base→nsai | 35 | 5 | 1.383e-06 | 6.913e-06 |
| base→nsai_uq | 26 | 22 | 0.6655 | 1 |
| base→sym | 26 | 0 | 2.98e-08 | 1.788e-07 |
| base→uq | 0 | 50 | 1.776e-15 | 1.243e-14 |
| base→cl | 33 | 17 | 0.03284 | 0.1314 |
| base→nsai_uq_cl | 26 | 29 | 0.7877 | 1 |

### Paired McNemar along the cumulative ladder (seed 0)

| Comparison | B01 | B10 | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 18 | 12 | 0.3616 | 0.3616 |
| rag→nsai | 24 | 0 | 1.192e-07 | 2.384e-07 |
| nsai→nsai_uq | 0 | 26 | 2.98e-08 | 8.941e-08 |

### Where the gate's contribution comes from (seed 0)

The gate's accuracy on items it fires on is partly circular: it applies the same rule and threshold the labels were generated from, and on the MedCalc benchmark the same extractor that filters the data runs inside the gate. Splitting the split by whether the gate fired separates the circular part from the part that is not: on gate-declined items the row IS the neural pathway, so any difference there is real.

| Variant | subset | n | accuracy | base accuracy on the same subset |
|---|---|---|---|---|
| (3) + Symbolic Gate (NS-AI) | gate fired | 72 | 1.000 | 0.639 |
| (3) + Symbolic Gate (NS-AI) | gate declined | 56 | 0.625 | 0.554 |
| (4) + UQ Engine (NS-AI+UQ) | gate fired | 72 | 1.000 | 0.639 |
| (4) + UQ Engine (NS-AI+UQ) | gate declined | 56 | 0.161 | 0.554 |
| (5) Base + Symbolic Gate only | gate fired | 72 | 1.000 | 0.639 |
| (5) Base + Symbolic Gate only | gate declined | 56 | 0.554 | 0.554 |
| (8) All four contributions | gate fired | 72 | 1.000 | 0.639 |
| (8) All four contributions | gate declined | 56 | 0.036 | 0.554 |

### Notes

- Causal Consistency is pair-level: both counterfactual arms must be correct. A constant-answer model scores 0.
- Abstentions count as failures for accuracy and causal consistency, and as non-violations for violation rate. Coverage is reported so this trade-off is visible.
- **Accuracy (strict)** counts an abstention as wrong and is the row-comparable number. Selective accuracy is in `summary_*.json`; it rises trivially as coverage falls and must never be compared across rows without coverage.
- Rows (5)-(7) add ONE contribution to the base model. Rows (1)-(4) are the cumulative ladder of the proposal's section 4.6. Row (8) is everything at once.
- **The gate's accuracy is an upper bound.** It evaluates the same constraint the ground truth was generated from, so on items it fires on it cannot be wrong unless extraction is. The figure to quote next to it is its firing rate here, and its coverage on unfiltered real notes (42.7%, `results/medcalc_ablation.md`).
- The constraint layer is scored through the identical readout as every other row (generate, then parse the first decision word), not by argmax over answer logits, so the numbers are comparable down the column.
