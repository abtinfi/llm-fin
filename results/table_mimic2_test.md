## Ablation matrix -- split=`test`, model=`BioMistral/BioMistral-7B`, seeds=[0]

UQ engine defers on **`decision_entropy`**  (Eq. (2) restricted to the decision tokens, the respecified 4.7 term: AUROC 0.687 [0.675, 0.700] over 6,456 items. Rank-equivalent to `logit_margin` for K=2.)

Single seed. Decoding is greedy and therefore deterministic: another seed reproduces this run exactly, so no seed-to-seed spread is reported. The uncertainty that does exist is over items, and is given as a bootstrap CI in the next table.

| Variant | Base | RAG | Sym | UQ | CL | Accuracy (strict) | Δ Acc vs base | Causal Consistency | Δ CC vs base | Violation Rate | Coverage | Gate fired |
|---|:-:|:-:|:-:|:-:|:-:|---|---|---|---|---|---|---|
| (1) Base LLM | YES | - | - | - | - | 0.621 | +0.000 | 0.069 | +0.000 | 0.867 | 1.000 | 0.000 |
| (2) + RAG | YES | YES | - | - | - | 0.440 | -0.181 | 0.000 | -0.069 | 0.000 | 1.000 | 0.000 |
| (3) + Symbolic Gate (NS-AI) | YES | YES | YES | - | - | 0.681 | +0.060 | 0.379 | +0.310 | 0.000 | 1.000 | 0.379 |
| (4) + UQ Engine (NS-AI+UQ) | YES | YES | YES | YES | - | 0.414 | -0.207 | 0.379 | +0.310 | 0.000 | 0.414 | 0.379 |
| (5) Base + Symbolic Gate only | YES | - | YES | - | - | 0.741 | +0.121 | 0.379 | +0.310 | 0.667 | 1.000 | 0.379 |
| (6) Base + UQ only | YES | - | - | YES | - | 0.052 | -0.569 | 0.000 | -0.069 | 0.044 | 0.069 | 0.000 |

### 95% bootstrap CI over items (seed 0)

| Variant | Accuracy | 95% CI | Causal Consistency | 95% CI |
|---|---|---|---|---|
| (1) Base LLM | 0.621 | [0.534, 0.707] | 0.069 | [0.000, 0.172] |
| (2) + RAG | 0.440 | [0.353, 0.534] | 0.000 | [0.000, 0.000] |
| (3) + Symbolic Gate (NS-AI) | 0.681 | [0.595, 0.767] | 0.379 | [0.207, 0.552] |
| (4) + UQ Engine (NS-AI+UQ) | 0.414 | [0.328, 0.509] | 0.379 | [0.207, 0.552] |
| (5) Base + Symbolic Gate only | 0.741 | [0.664, 0.819] | 0.379 | [0.207, 0.552] |
| (6) Base + UQ only | 0.052 | [0.017, 0.095] | 0.000 | [0.000, 0.000] |

### Paired McNemar of each contribution **against the baseline** (seed 0, item-level correctness)

This is the supervisor's question: what does adding this contribution to the base model do?

| Variant vs base | B01 (base wrong→right) | B10 (base right→wrong) | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 39 | 60 | 0.04388 | 0.08775 |
| base→nsai | 44 | 37 | 0.5052 | 0.5052 |
| base→nsai_uq | 18 | 42 | 0.00267 | 0.008011 |
| base→sym | 14 | 0 | 0.0001221 | 0.0004883 |
| base→uq | 0 | 66 | 2.711e-20 | 1.355e-19 |

### Paired McNemar along the cumulative ladder (seed 0)

| Comparison | B01 | B10 | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 39 | 60 | 0.04388 | 0.04388 |
| rag→nsai | 28 | 0 | 7.451e-09 | 1.49e-08 |
| nsai→nsai_uq | 0 | 31 | 9.313e-10 | 2.794e-09 |

### Where the gate's contribution comes from (seed 0)

The gate's accuracy on items it fires on is partly circular: it applies the same rule and threshold the labels were generated from, and on the MedCalc benchmark the same extractor that filters the data runs inside the gate. Splitting the split by whether the gate fired separates the circular part from the part that is not: on gate-declined items the row IS the neural pathway, so any difference there is real.

| Variant | subset | n | accuracy | base accuracy on the same subset |
|---|---|---|---|---|
| (3) + Symbolic Gate (NS-AI) | gate fired | 44 | 1.000 | 0.682 |
| (3) + Symbolic Gate (NS-AI) | gate declined | 72 | 0.486 | 0.583 |
| (4) + UQ Engine (NS-AI+UQ) | gate fired | 44 | 1.000 | 0.682 |
| (4) + UQ Engine (NS-AI+UQ) | gate declined | 72 | 0.056 | 0.583 |
| (5) Base + Symbolic Gate only | gate fired | 44 | 1.000 | 0.682 |
| (5) Base + Symbolic Gate only | gate declined | 72 | 0.583 | 0.583 |

### Notes

- Causal Consistency is pair-level: both counterfactual arms must be correct. A constant-answer model scores 0.
- Abstentions count as failures for accuracy and causal consistency, and as non-violations for violation rate. Coverage is reported so this trade-off is visible.
- **Accuracy (strict)** counts an abstention as wrong and is the row-comparable number. Selective accuracy is in `summary_*.json`; it rises trivially as coverage falls and must never be compared across rows without coverage.
- Rows (5)-(7) add ONE contribution to the base model. Rows (1)-(4) are the cumulative ladder of the proposal's section 4.6. Row (8) is everything at once.
- **The gate's accuracy is an upper bound.** It evaluates the same constraint the ground truth was generated from, so on items it fires on it cannot be wrong unless extraction is. The figure to quote next to it is its firing rate here, and its coverage on unfiltered real notes (42.7%, `results/medcalc_ablation.md`).
- The constraint layer is scored through the identical readout as every other row (generate, then parse the first decision word), not by argmax over answer logits, so the numbers are comparable down the column.
