# Aim 3: the Constraint-Aware Layer, with the proposal's full objective

**A rank-32 adapter at layer 30, trained on 55 counterfactual pairs with all
four terms of the proposal's objective, takes Causal Consistency on unseen
pairs of the same family from 0.000 to 0.767 at a language cost of −1.2%
perplexity — and produces nothing at all on the family where the probes said
the information was missing.**

Both halves were predicted in advance from mechanistic measurements taken
before any adapter existed. Both held.

---

## 1. The objective, now complete

The proposal specifies

    L = L_LM + l1*L_ontology + l2*L_causal + l3*L_uncertainty

The earlier version of this experiment implemented two of the four terms and
said so. All four are now implemented:

| term | what it is here | weight |
|---|---|---|
| `L_causal` | cross-entropy on the SAFE/UNSAFE decision token; the label comes from the clinical rule, not a judge | 1.0 |
| `L_LM` | KL(base ‖ adapted) over the full vocabulary at the decision position, against the frozen base model | 0.01 |
| `L_ontology` | **pairwise**: the ontology says the decision is *monotone* in the decisive quantity, so the SAFE arm's decision margin must sit above the UNSAFE arm's by a margin. A hinge on that ordering is a statement about the mechanism, and a model can satisfy every per-item label while violating it | 0.1 |
| `L_uncertainty` | for each training item, an **ablated copy** with the decisive number redacted; on those the decision distribution is pushed toward maximum entropy | 0.1 |

Optimisation is per pair rather than per item, because `L_ontology` is defined
on the two arms jointly and no single-item loss can express it.

## 2. The prediction, made before training

`results/aim123_internals.md` measured whether the decisive clinical quantity
is linearly readable from the residual stream:

| Family | Probe pair-CC | Reading |
|---|---|---|
| QT / ondansetron (QTc > 500) | **0.976** (null 0.200) | represented |
| Renal / metformin (eGFR < 30) | **0.042** (null 0.033) | not represented |

> A layer that edits hidden states can only amplify what the hidden states
> already carry. It should work on QT and fail on renal.

## 3. Result

Base model and adapted model are scored under the identical decision rule.

| Trained on | pairs | QT held-out CC | QT held-out acc | Renal CC |
|---|---|---|---|---|
| *(base model)* | — | **0.000** | 0.500 | 0.033 |
| **QT, real labels** | 55 | **0.767** | **0.883** | 0.000 |
| QT, shuffled labels *(control)* | 55 | 0.200 | 0.583 | 0.000 |
| Renal, real labels | 140 | 0.000 | 0.500 | 0.011 |
| Synthetic benchmark, from calib | 16 | 0.438 | 0.719 | — |

The QT evaluation split is 30 pairs the adapter never saw, in a family the
renal training never touches.

### The control, read correctly

A per-arm-random predictor scores **0.25** pair-CC, not 0. The shuffled-label
control lands at **0.200 — at chance** — against 0.767 with real labels, and it
could not fit even its own training set:

| | real labels | shuffled labels |
|---|---|---|
| final `L_causal` | **0.245** | 0.689 (≈ ln 2 = 0.693) |
| final train decision accuracy | **0.909** | 0.564 |
| `L_ontology` over training | 0.985 → **0.015** | (disabled — see below) |
| QT held-out CC | **0.767** | 0.200 |

`l1` is forced to zero in the shuffled run. The ontology term is defined on the
true arm ordering, so leaving it on would feed the real labels back into a run
whose entire purpose is to have none. The code does this itself and prints why.

**The ontology term was doing work.** Its hinge falls from 0.985 to 0.015 over
training, meaning the adapter did not merely get the individual labels right —
it made the two arms' decision margins respect the direction the threshold
implies.

### The negative half

Trained on 140 renal pairs with real labels, the adapter never fitted its own
training set: decision accuracy 0.464 → 0.564 over eight epochs, `L_causal`
flat at 0.689, and the ontology hinge stuck at 0.93. Renal CC moved 0.033 →
0.011. **Nothing was learned, exactly as the probe predicted: the layer has
nothing to amplify, because creatinine is not in the residual stream in any
linearly readable form.**

> A constraint layer on hidden states is not a general fix. It works precisely
> where the model already represents the decisive quantity and merely fails to
> read it out, and does nothing where the quantity was never encoded. Which
> case you are in is measurable in advance, with a probe, for far less compute
> than training the layer.

## 4. `L_uncertainty` did not do what it was supposed to

This is the clearest negative result of the run and it is not buried.

The term pushes the decision toward maximum entropy on notes whose decisive
number has been redacted. Mean decision entropy on redacted notes, in nats,
where **ln 2 = 0.693 is maximal**:

| split | base model | with constraint layer | n | direction |
|---|---|---|---|---|
| renal test | 0.626 | **0.457** | 180 | **wrong way** |
| QT held-out | 0.670 | **0.553** | 60 | **wrong way** |

The adapter became *more* confident on notes that no longer state the fact.

And yet the term was satisfied **during training** — `L_unc` stayed between
0.003 and 0.16 throughout, i.e. entropy on the training set's redacted copies
stayed near ln 2. So the precise statement is:

> The uncertainty term is fitted on the notes it is trained on and does not
> generalise to unseen notes. Confidence learned from `L_causal` transfers;
> calibrated ignorance does not.

That is worth saying because it is the opposite of the assumption behind an
uncertainty-aware safety layer. Three things would test it properly and none
were run: a much larger `l3`, redaction applied to held-out notes during
training, and a check of whether the effect is specific to redaction or is just
the adapter becoming globally more confident.

## 5. RQ3: does it cost general language ability?

`src/perplexity.py` on WikiText-2 (200 documents), adapter attached exactly as
evaluated above:

| | WikiText-2 perplexity |
|---|---|
| Base model | 7.1622 |
| **+ QT constraint layer** | **7.0775** (−1.18%) |
| + synthetic-benchmark constraint layer | 7.1266 (−0.50%) |

Unchanged for practical purposes, and slightly *lower*, which at this magnitude
is noise rather than improvement. The two numbers are not identical, which
confirms the adapter was actually active.

**RQ3 is answered affirmatively for this constraint and this layer:** Causal
Consistency 0.000 → 0.767 at a language cost of −1.2%. Quote the pair, never
the first number alone — a layer that shouted UNSAFE at everything would also
raise Causal Consistency while destroying the model.

## 6. It is now a row of the ablation table

The previous version of this experiment scored the adapter by argmax over the
two answer logits, while every row of the ablation table scores by generating
text and parsing the first decision word. Two readouts that need not agree, so
its number could not honestly be placed in that table.

The adapter is now attached inside the model wrapper and scored through the
identical path (`run_eval.py --variants cl --adapter ...`). The two readouts
agree to three decimals on the QT split (0.883 / 0.767 either way), which is
itself worth knowing.

From `results/table_medcalc_heldout.md`, QT family, 30 unseen pairs:

| Row | Accuracy | Δ vs base | CC | Δ vs base | Violation | McNemar vs base |
|---|---|---|---|---|---|---|
| (1) Base LLM | 0.500 | — | 0.000 | — | 1.000 | — |
| (2) + RAG | 0.500 | +0.000 | 0.000 | +0.000 | 0.000 | p = 1 |
| (5) Base + Gate only | 1.000 | +0.500 | 1.000 | +1.000 | 0.000 | p = 1.9e−09 |
| **(7) Base + Constraint Layer** | **0.883** | **+0.383** | **0.767** | **+0.767** | 0.133 | **p = 1.5e−05** |

The symbolic gate still wins where it applies. The constraint layer's claim is
different and complementary: it needs no extractable fact, no regex and no
threshold written into code. On the 57% of real notes where the gate cannot
fire at all, a learned layer is the only one of the two that can be attempted.

**The cost is in the violation column.** The base model answers SAFE to all 60
items, so its violation rate is 1.000 and its CC is 0. The constraint layer
takes CC to 0.767 but still calls 13.3% of genuinely unsafe cases safe, where
the gate calls none. On the synthetic benchmark the same trade is sharper: CC
0.219 → 0.422, violation 0.125 → 0.344. A learned layer buys consistency and
sells some safety margin; the gate does not.

## 7. Limitations

- **One family, 55 training pairs, one seed.** The 0.767 is on 30 held-out
  pairs; the binomial 95% CI is [0.600, 0.900].
- **Within-family generalisation.** The QT eval pairs are unseen notes of a
  family the adapter trained on. No cross-family transfer was expected on the
  MedCalc benchmark and none is claimed; the synthetic run does show some
  (CC 0.234 → 0.422 on trained families, 0.312 → 0.438 on two unseen ones) but
  from only 16 training pairs.
- **`L_uncertainty` does not generalise** (§4).
- **The violation-rate cost is real** (§6) and no threshold tuning was done to
  trade it back.
- **WikiText-2 perplexity is a weak proxy** for clinical language ability.
- **The layer index was chosen from the probe curve**, and changed once from 5
  to 30 after a convergence failure, on training behaviour, before any test
  number was looked at.
- **Renal remains unsolved by any neural method here.** "Not linearly readable
  at these two pooling choices" is not the same as "absent".
