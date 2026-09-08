## Ablation matrix -- split=`test`, model=`mistralai/Mistral-7B-Instruct-v0.2`, seeds=[0]

UQ engine defers on **`decision_entropy`**  (Eq. (2) restricted to the decision tokens, the respecified 4.7 term: AUROC 0.687 [0.675, 0.700] over 6,456 items. Rank-equivalent to `logit_margin` for K=2.)

Single seed. Decoding is greedy and therefore deterministic: another seed reproduces this run exactly, so no seed-to-seed spread is reported. The uncertainty that does exist is over items, and is given as a bootstrap CI in the next table.

| Variant | Base | RAG | Sym | UQ | CL | Accuracy (strict) | Δ Acc vs base | Causal Consistency | Δ CC vs base | Violation Rate | Coverage | Gate fired |
|---|:-:|:-:|:-:|:-:|:-:|---|---|---|---|---|---|---|
| (1) Base LLM | YES | - | - | - | - | 0.570 | +0.000 | 0.141 | +0.000 | 0.734 | 1.000 | 0.000 |
| (2) + RAG | YES | YES | - | - | - | 0.703 | +0.133 | 0.422 | +0.281 | 0.266 | 1.000 | 0.000 |
| (3) + Symbolic Gate (NS-AI) | YES | YES | YES | - | - | 0.875 | +0.305 | 0.750 | +0.609 | 0.109 | 1.000 | 0.562 |
| (4) + UQ Engine (NS-AI+UQ) | YES | YES | YES | YES | - | 0.734 | +0.164 | 0.562 | +0.422 | 0.000 | 0.766 | 0.562 |
| (5) Base + Symbolic Gate only | YES | - | YES | - | - | 0.789 | +0.219 | 0.578 | +0.438 | 0.422 | 1.000 | 0.562 |
| (6) Base + UQ only | YES | - | - | YES | - | 0.094 | -0.477 | 0.000 | -0.141 | 0.000 | 0.094 | 0.000 |
| (7) Base + Constraint Layer only | YES | - | - | - | YES | 0.734 | +0.164 | 0.484 | +0.344 | 0.203 | 1.000 | 0.000 |
| (8) All four contributions | YES | YES | YES | YES | YES | 0.750 | +0.180 | 0.594 | +0.453 | 0.000 | 0.773 | 0.562 |

### 95% bootstrap CI over items (seed 0)

| Variant | Accuracy | 95% CI | Causal Consistency | 95% CI |
|---|---|---|---|---|
| (1) Base LLM | 0.570 | [0.484, 0.656] | 0.141 | [0.062, 0.234] |
| (2) + RAG | 0.703 | [0.625, 0.781] | 0.422 | [0.297, 0.531] |
| (3) + Symbolic Gate (NS-AI) | 0.875 | [0.812, 0.930] | 0.750 | [0.641, 0.844] |
| (4) + UQ Engine (NS-AI+UQ) | 0.734 | [0.656, 0.805] | 0.562 | [0.438, 0.688] |
| (5) Base + Symbolic Gate only | 0.789 | [0.719, 0.859] | 0.578 | [0.453, 0.703] |
| (6) Base + UQ only | 0.094 | [0.047, 0.148] | 0.000 | [0.000, 0.000] |
| (7) Base + Constraint Layer only | 0.734 | [0.656, 0.805] | 0.484 | [0.359, 0.609] |
| (8) All four contributions | 0.750 | [0.672, 0.820] | 0.594 | [0.469, 0.719] |

### Paired McNemar of each contribution **against the baseline** (seed 0, item-level correctness)

This is the supervisor's question: what does adding this contribution to the base model do?

| Variant vs base | B01 (base wrong→right) | B10 (base right→wrong) | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 47 | 30 | 0.06755 | 0.06755 |
| base→nsai | 48 | 9 | 1.52e-07 | 7.601e-07 |
| base→nsai_uq | 40 | 19 | 0.008641 | 0.02592 |
| base→sym | 28 | 0 | 7.451e-09 | 4.47e-08 |
| base→uq | 0 | 61 | 8.674e-19 | 6.072e-18 |
| base→cl | 43 | 22 | 0.0125 | 0.02592 |
| base→nsai_uq_cl | 43 | 20 | 0.005152 | 0.02061 |

### Paired McNemar along the cumulative ladder (seed 0)

| Comparison | B01 | B10 | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 47 | 30 | 0.06755 | 0.06755 |
| rag→nsai | 22 | 0 | 4.768e-07 | 1.431e-06 |
| nsai→nsai_uq | 0 | 18 | 7.629e-06 | 1.526e-05 |

### Where the gate's contribution comes from (seed 0)

The gate's accuracy on items it fires on is partly circular: it applies the same rule and threshold the labels were generated from, and on the MedCalc benchmark the same extractor that filters the data runs inside the gate. Splitting the split by whether the gate fired separates the circular part from the part that is not: on gate-declined items the row IS the neural pathway, so any difference there is real.

| Variant | subset | n | accuracy | base accuracy on the same subset |
|---|---|---|---|---|
| (3) + Symbolic Gate (NS-AI) | gate fired | 72 | 1.000 | 0.611 |
| (3) + Symbolic Gate (NS-AI) | gate declined | 56 | 0.714 | 0.518 |
| (4) + UQ Engine (NS-AI+UQ) | gate fired | 72 | 1.000 | 0.611 |
| (4) + UQ Engine (NS-AI+UQ) | gate declined | 56 | 0.393 | 0.518 |
| (5) Base + Symbolic Gate only | gate fired | 72 | 1.000 | 0.611 |
| (5) Base + Symbolic Gate only | gate declined | 56 | 0.518 | 0.518 |
| (8) All four contributions | gate fired | 72 | 1.000 | 0.611 |
| (8) All four contributions | gate declined | 56 | 0.429 | 0.518 |

### Notes

- Causal Consistency is pair-level: both counterfactual arms must be correct. A constant-answer model scores 0.
- Abstentions count as failures for accuracy and causal consistency, and as non-violations for violation rate. Coverage is reported so this trade-off is visible.
- **Accuracy (strict)** counts an abstention as wrong and is the row-comparable number. Selective accuracy is in `summary_*.json`; it rises trivially as coverage falls and must never be compared across rows without coverage.
- Rows (5)-(7) add ONE contribution to the base model. Rows (1)-(4) are the cumulative ladder of the proposal's section 4.6. Row (8) is everything at once.
- **The gate's accuracy is an upper bound.** It evaluates the same constraint the ground truth was generated from, so on items it fires on it cannot be wrong unless extraction is. The figure to quote next to it is its firing rate here, and its coverage on unfiltered real notes (42.7%, `results/medcalc_ablation.md`).
- The constraint layer is scored through the identical readout as every other row (generate, then parse the first decision word), not by argmax over answer logits, so the numbers are comparable down the column.
