## Ablation matrix -- split=`heldout`, model=`BioMistral/BioMistral-7B`, seeds=[0]

UQ engine defers on **`entropy`**  (the proposal's Eq. (2))

| Variant | Base | RAG | Sym | UQ | Accuracy (strict) | Sel. Acc. | Causal Consistency | Violation Rate | Coverage |
|---|:-:|:-:|:-:|:-:|---|---|---|---|---|
| (1) Base LLM | YES | - | - | - | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.125 ± 0.000 | 0.125 ± 0.000 | 1.000 ± 0.000 |
| (2) + RAG | YES | YES | - | - | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| (3) + Symbolic Gate (NS-AI) | YES | YES | YES | - | 0.750 ± 0.000 | 0.750 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| (4) + UQ Engine (NS-AI+UQ) | YES | YES | YES | YES | 0.500 ± 0.000 | 1.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 |

### Causal Consistency, 95% bootstrap CI (seed 0, pair-level)

| Variant | CC | 95% CI |
|---|---|---|
| (1) Base LLM | 0.125 | [0.000, 0.312] |
| (2) + RAG | 0.000 | [0.000, 0.000] |
| (3) + Symbolic Gate (NS-AI) | 0.500 | [0.250, 0.750] |
| (4) + UQ Engine (NS-AI+UQ) | 0.500 | [0.250, 0.750] |

### Paired McNemar vs. previous variant (seed 0, item-level correctness)

| Comparison | B01 (prev wrong→right) | B10 (prev right→wrong) | p (exact) | p (Holm) |
|---|---|---|---|---|
| base→rag | 2 | 2 | 1 | 1 |
| rag→nsai | 8 | 0 | 0.007812 | 0.02344 |
| nsai→nsai_uq | 0 | 8 | 0.007812 | 0.02344 |

### Notes

- Causal Consistency is pair-level: both counterfactual arms must be correct. A constant-answer model scores 0.
- Abstentions count as failures for accuracy and causal consistency, and as non-violations for violation rate. Coverage is reported so this trade-off is visible.
- **Accuracy (strict)** counts an abstention as wrong and is the row-comparable number. **Sel. Acc.** is computed over answered items only; it rises trivially as coverage falls, so it must never be compared across rows without coverage.
- The symbolic gate extracts facts from the vignette by regex and declines to fire when the causal factor is stated indirectly. Measured firing rate on this split is reported below; cases where it declines fall through to the neural pathway.
- Vignettes are hardened: near-threshold values, indirect presentation of the causal factor in half the templates, and three distractor labs per case. Run `build_dataset.py --no_hardening` to reproduce the earlier, easier version.