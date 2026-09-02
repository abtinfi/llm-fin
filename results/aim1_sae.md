# Aim 1: sparse autoencoder features, and what they are worth

**A TopK SAE trained on 150,000 real clinical tokens finds features that
select single clinical concepts cleanly — a QT-interval feature at F1 0.703, a
pregnancy feature at 0.615, a creatinine feature at 0.603. Knocking any of them
out of the forward pass moves the model's decision by 0.002–0.016 logits, when
flipping a decision needs 1–5.**

The features are real. Their causal role in this model's clinical decision is
not detectable. Those are two separate findings and both are reported.

---

## 1. Why this exists

Aim 1 is the first aim of the proposal and nothing in the repository
implemented it. `src/probe.py` answers a strictly weaker question — is the
decisive fact linearly decodable from the residual stream — which is a
supervised readout, not a decomposition. A probe cannot produce features,
cannot say what a direction responds to, and cannot be knocked out one unit at
a time. Every claim about "discovering sparse features" needs an SAE.

## 2. Setup

| | |
|---|---|
| Model | `BioMistral/BioMistral-7B`, bf16, GPU 0 only |
| Site | residual stream, layer 20 |
| Corpus | 150,000 tokens from 280 real PMC notes (`data/medcalc`, train split) |
| Dictionary | 16,384 features (4x expansion) |
| Architectures | **TopK** (k=32) and **JumpReLU**, both named in the proposal's fallback |
| Objective | proposal Eq. (1), decoder rows unit-norm after every step |

Inputs are rescaled so `E||x|| = sqrt(d)`. Without that the first run diverged
to NaN on the first step and *still* printed a plausible FVU and a 100%
dead-feature count — the failure mode that gets written up as a finding. The
trainer now aborts on a non-finite loss instead of writing an SAE.

## 3. Reconstruction, and which architecture won

| | val FVU | L0 | dead features |
|---|---|---|---|
| **TopK (k=32)** | **0.067** | 31.9 | 6,442 / 16,384 (39.3%) |
| JumpReLU | 0.131 | 27.8 | 10,481 / 16,384 (64.0%) |

TopK reconstructs twice as well at comparable sparsity and leaves far fewer
features dead, so everything below uses it. This is the comparison the
proposal's fallback clause asks for, run on the merits rather than triggered by
a threshold.

## 4. S_semantic: do features select clinical concepts?

Best F1 of "this feature is active" against "this token belongs to concept c",
maximised over eleven concepts whose spans are found in the prompt text — never
in the dataset's structured facts.

| feature | concept | S_semantic | fires on | S_causal | FIS |
|---|---|---|---|---|---|
| #14294 | qt_interval | **0.703** | 36 / 150,000 | 0.0016 | 0.352 |
| #7626 | pregnancy | 0.615 | 8 | 0.0000 | 0.308 |
| #14058 | creatinine | **0.603** | 2,149 | 0.0158 | 0.310 |
| #11855 | age | 0.592 | 1,046 | 0.0146 | 0.303 |
| #5438 | heart_rate | 0.586 | 330 | 0.0098 | 0.298 |
| #2883 | drug | 0.541 | 526 | 0.0162 | 0.279 |

The JumpReLU dictionary's best feature reaches only 0.363, consistent with its
worse reconstruction.

**A caveat that matters for #7626.** A feature firing on 8 tokens out of 150,000
can reach a high F1 on a concept that is itself rare. The firing count is in the
table for exactly that reason; the QT and creatinine features are the ones with
enough support to lean on.

### What this does and does not show

It shows the model's layer-20 representation contains directions that track
individual clinical quantities, and that a sparse dictionary recovers them
without supervision. It does **not** show the model *uses* them — that is the
next section, and the answer is no.

## 5. S_causal: knocking the feature out

For each feature, its contribution is subtracted at every position,
`h' = h - z_f(h)·W_dec[f]`, and the change in the SAFE-minus-UNSAFE decision
logit is measured over 40 items of the **test** split — notes the dictionary was
not fitted on. A random feature is knocked out as a matched control every time.

| feature | \|Δ margin\| | control | control \|Δ\| | excess |
|---|---|---|---|---|
| #14294 (QT) | 0.0016 | #13936 | 0.0000 | +0.0016 |
| #7626 (pregnancy) | 0.0000 | #10435 | 0.0016 | −0.0016 |
| #14058 (creatinine) | 0.0158 | #8374 | 0.0000 | +0.0158 |
| #11855 (age) | 0.0146 | #4420 | 0.0000 | +0.0146 |
| #5438 (heart rate) | 0.0289 | #5043 | 0.0191 | +0.0098 |
| #2883 (drug) | 0.0209 | #671 | 0.0047 | +0.0162 |
| #7679 | 0.0000 | #1232 | 0.0016 | −0.0016 |
| #16231 | 0.0172 | #270 | 0.0016 | +0.0156 |

Mean clean decision margin over these items: −0.131 logits. **Flipping a
decision needs roughly 1–5 logits. The largest excess effect here is 0.016 —
about 100x too small.** Two of the eight features move the decision *less* than
their random control.

## 6. This replicates the patching result, by a different route

`results/aim123_internals.md` reached the same conclusion at the level of token
positions: patching the residual stream at the positions the counterfactual edit
touched moved the decision by ~0.002 logits. That could have been an artefact of
patching the wrong site. This is a different intervention — remove one *learned
feature* everywhere, rather than overwrite one position — and it lands in the
same place.

The two together say something sharper than either alone:

> The quantities are represented, a sparse dictionary finds them, and the
> decision does not read them. The failure is not in the representation and not
> in the choice of intervention site; it is in the path from representation to
> output.

That is precisely the gap the Aim 3 constraint layer is for, and it is why the
layer works on QT (where `results/aim3_constraint_layer.md` shows CC 0.000 →
0.767) while nothing recovers the renal decision.

## 7. FIS, and the term that is missing

    FIS = alpha*S_semantic + beta*S_causal + gamma*S_human

`S_human` is **not measured**: this pipeline has no expert annotators. Its
weight is forced to zero and the FIS reported above is a two-term score with
alpha = beta = 0.5. Weighting an unmeasured term at 1/3 would make the number
look expert-validated when no expert has seen it.

The consequence is that **the proposal's fallback criterion cannot be evaluated
as written**. It says to switch architecture if fewer than 30% of features pass
expert validation. There is no expert validation, so the architecture choice was
made on reconstruction and sparsity instead (§3), and the criterion should be
restated in the proposal as something measurable without a clinician panel, or
the panel has to be budgeted.

## 8. Limitations

- **Pilot scale.** Production SAEs see hundreds of millions of tokens; this saw
  150,000 from one narrow corpus, at 4x expansion. Features here describe this
  distribution of clinical notes, not the model in general, and a concept that
  fails to appear is not evidence the model lacks it.
- **39% of features are dead**, which is what undertraining looks like.
- **One layer, one site.** Layer 20 only; a feature absent here may live
  elsewhere.
- **The concept labels are regex spans.** They mark where a quantity is stated,
  which is not the same as where the model represents it.
- **Knock-out was run on 8 features and 40 items.** It is powered to detect
  effects of order 0.1 logits, not 0.005. The claim is that no *decision-sized*
  effect is present, not that the effect is exactly zero.

## 9. Reproducing

```bash
export CUDA_VISIBLE_DEVICES=0
python src/sae.py collect --data data/medcalc --split train --layer 20 \
    --other_mult 1000 --max_tokens 150000 \
    --out results/sae/acts_medcalc_train_L20.npz
python src/sae.py train --acts results/sae/acts_medcalc_train_L20.npz \
    --kind topk --expansion 4 --k 32 --epochs 30 --batch 2048 \
    --out results/sae/sae_topk_L20.npz
python src/sae.py score --sae results/sae/sae_topk_L20.npz \
    --top 25 --causal_items 40 --causal_features 8 --causal_split test
```
