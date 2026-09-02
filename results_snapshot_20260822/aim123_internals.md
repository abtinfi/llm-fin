# Inside the model: probes, patching, and a steering prototype

**First work in this project that opens the model up rather than measuring its
behaviour.** Corresponds to the proposal's Aim 1 (discovery), Aim 2 (causal
necessity/sufficiency) and a minimal prototype of Aim 3 (control).

**The headline: BioMistral-7B represents one of the two clinical quantities
strongly and the other not at all — and its answer is driven by neither.**

---

## 1. Why probes and patching, and in that order

The behavioural result is that the model ignores the decisive number. That is
compatible with two very different mechanisms, which call for opposite fixes:

- the number is **represented** internally and not read out → Aim 3 has
  something to act on, and a constraint layer is the right instrument;
- the number is **never represented** → no intervention on hidden states can
  recover it, and the symbolic route is a necessity rather than a shortcut.

Activation patching alone cannot separate these. Its normalised effect is
(patched − clean) / (counterfactual − clean), and that denominator is the gap
between the two arms' outputs. Where the model's arms are output-identical the
metric divides by noise. So the probe answers the prior question — *is it
there?* — and patching answers *does it drive the answer?*

## 2. Setup

| | |
|---|---|
| Model | `BioMistral/BioMistral-7B`, bf16, no quantisation, GPU 0 only |
| Representation | residual stream, all 33 layers (`hidden_states`) |
| Pooling | **final** = last prompt position (where the answer is predicted from); **mean** = averaged over non-pad positions |
| Probe | StandardScaler → PCA(128, fitted per training fold) → logistic, C=0.05 |
| Splits | 5-fold, **grouped by `pair_id`** so both arms never straddle the boundary |
| Metric | **pair-level consistency** — both arms correct — identical to Causal Consistency, so probe and model numbers are directly comparable |

### The null control, and the version of it that was wrong

Every real pair holds exactly one SAFE and one UNSAFE arm, so a constant
predictor scores **0** pair-consistency and a per-arm-random one scores
**0.25**.

The first control shuffled labels *globally*. That breaks the one-of-each
structure: roughly half the pairs end up with two identical labels, which a
constant predictor gets right, and the floor climbs to ~0.23–0.28 for reasons
having nothing to do with the activations. It made the code print "LEAK, not
readable" on a perfectly clean run.

The control used here permutes labels **within each pair** — randomly swapping
which arm is called SAFE. That destroys the association with the activations
while preserving the structure, so null and signal are on the same scale.

Two properties of this null are worth stating:

- It is **conservative when real signal is strong**: half the pairs keep their
  true labels, so a probe trained on the permuted labels picks up a fraction of
  the real direction. The QT null (0.164) sits above the renal null (0.013) for
  exactly this reason.
- The reported floor is the **maximum across all 33 layers**, which also
  absorbs the multiple-comparison cost of reporting the best layer. Observed
  maxima sit ~2.5 standard errors above their means, as expected from sampling
  noise at these sample sizes.

## 3. Probe results

| Benchmark | Pooling | Best layer | Item acc | **Pair CC** | Null (max) | **Margin** |
|---|---|---|---|---|---|---|
| Real notes, renal | final | 1 | 0.519 | **0.042** | 0.033 | **+0.009** |
| Real notes, renal | mean | 5 | 0.544 | **0.093** | 0.042 | **+0.051** |
| **Real notes, QT** | final | 5 | 0.924 | **0.859** | 0.235 | **+0.624** |
| **Real notes, QT** | mean | 30 | 0.988 | **0.976** | 0.200 | **+0.776** |
| Synthetic vignettes | final | 15 | 0.820 | **0.641** | 0.359 | **+0.281** |
| Synthetic vignettes | mean | 14 | 0.781 | **0.656** | 0.438 | **+0.219** |

The QT curve is smooth and interpretable rather than a lucky layer: 0.000 at
layer 0 (the final token's embedding cannot know about a number stated
elsewhere — a perfect sanity check), 0.482 at layer 1, 0.694 at layer 2,
peaking at **0.859** by layer 5 and holding 0.65–0.80 through layer 32.

### Two controls that stopped stronger claims

**Could the QT probe just be reading the edited number?** No — but it also
cannot be said to compute QTc.

| Predictor | Pair CC |
|---|---|
| Best single threshold on raw QT | 0.706 |
| Best single threshold on true QTc | 1.000 |
| Linear probe on (QT, heart rate) | **1.000** |
| **BioMistral residual stream** | **0.976** / 0.859 |

Heart rate ranges 47–178, so raw QT alone caps out at 0.706 and the model's
0.976 is well past it. But a linear probe on the two *raw* inputs also reaches
1.000, so **this experiment cannot distinguish "the model computes the
rate-corrected QTc" from "the model encodes QT and heart rate separately and
the probe combines them."** The defensible claim is the weaker one: the
representation supports a linear readout of the decision that raw QT alone
does not.

**Is the renal null meaningful, or is that decision simply not linearly
decodable?** It is meaningful.

| Predictor | Pair CC |
|---|---|
| Creatinine alone | 0.907 |
| **Creatinine + age + sex** | **1.000** |
| True eGFR | 1.000 |
| **BioMistral residual stream** | **0.042** / 0.093 |

The renal decision is trivially linearly decodable from the raw facts. If the
model encoded creatinine in any linearly readable form, the probe would find
it. It does not. **The residual stream carries no linearly readable
representation of creatinine — the single most important number in the note.**

Both decisions are equally decodable in principle. The model represents one and
not the other. *Why* is not established here; the notes differ in that QT and
heart rate are stated together in stereotyped cardiology phrasing while
creatinine is one of a dozen labs in a list. That is a hypothesis, not a
finding.

## 4. Patching results (Aim 2)

Patching the residual stream at **only the token positions the edit touched**,
one layer at a time, taking those rows from the counterfactual run.

Patching every position instead would be close to useless: writing all of layer
*l* from run B makes the forward pass *become* run B from there on, so the
curve would measure how early you overwrote, not what the number does.

| Benchmark | Pairs | Clean gap between arms | Identity control | Max \|excess over control\| |
|---|---|---|---|---|
| Real notes, renal | 80 | **+0.0017** logits | **0.0e+00** | 0.0086 @ L9 |
| Real notes, QT | 80 | **+0.0008** logits | **0.0e+00** | 0.0211 @ L19 |
| Synthetic vignettes | — | — | — | **not runnable** (see §6) |

For scale, flipping a decision needs roughly 1–5 logits. The largest effect
found anywhere is **~100–500× too small to matter**.

Three things make this null trustworthy rather than a broken hook:

1. **The identity control is exactly zero.** Patching the clean run from its
   own activations changes nothing, to machine precision, so the hook is
   writing where it claims to.
2. **A position control** — patching the same number of positions chosen at
   random from positions the edit did *not* touch — is subtracted at every
   layer.
3. **Internal consistency at layer 0.** Nothing has propagated to other
   positions yet there, so patching should recover close to the *full*
   counterfactual effect. It does (76% of it on renal) — and that full effect
   is itself ~0.001 logits. The null is real, not an artefact of patching the
   wrong site.

Changing creatinine from 2.1 to 2.9 mg/dL takes eGFR from 33.6 to 22.8 — from
permitted to contraindicated — and moves the model's decision logit by
**0.0017**.

## 5. Steering prototype (Aim 3, smallest honest version)

The probe direction is exactly the `P_causal` of the proposal's
`h'_l = h_l + α·P_causal(h_l)`. So the cheapest possible test of Aim 3 is to
add it back and see whether the represented fact can be *made* causal.

Guards: the direction is fitted on training folds only and applied to held-out
items fold by fold; a matched-norm **random direction** is run at every α; the
whole α sweep is reported rather than a selected point; the sign is read from
the probe's own classes.

**α must be scaled per layer.** The residual norm in this model grows ~240×
with depth (0.13 at layer 0, 1.43 at layer 5, 17.1 at layer 25, 347 at layer
32). A fixed absolute α means something completely different at each depth; the
first sweep used absolute values and was uninterpretable. It is now expressed
in multiples of that layer's mean norm.

### Result: no effect

24 cells (4 layers × 6 α). Unsteered CC on this split is 0.000.

| Layer | norm | α=0 | 0.5× | 1× | 2× | 4× | 8× |
|---|---|---|---|---|---|---|---|
| 5 | 1.43 | +0.000 | +0.000 | +0.000 | −0.059 | −0.024 | +0.047 |
| 15 | 5.20 | +0.000 | +0.035 | +0.024 | +0.024 | **+0.106** | +0.000 |
| 25 | 17.08 | +0.000 | −0.024 | +0.059 | +0.000 | −0.012 | +0.000 |
| 30 | 29.99 | +0.000 | −0.035 | +0.035 | −0.012 | +0.012 | +0.000 |

(margin over the matched-norm random direction)

The single cell above 0.10 is layer 15 at α=4×. Its neighbours at 2× and 8×
give +0.024 and +0.000 — **there is no dose-response, just an isolated peak
between two nulls.** For scale, the random-direction control reaches ±0.071 on
its own somewhere in the sweep, and the best probe margin is only 1.5× that.
With 85 pairs the standard error of a pair-CC estimate is ~0.04, so the maximum
of 24 noise draws is expected around 0.12. +0.106 is exactly that.

**Verdict: steering along a single probe direction does not make the fact
causal.**

An earlier version of this script called it a positive result, because it
compared the best cell against a fixed 0.10 threshold — which ignores that the
best of 24 cells is elevated by construction. The check now requires the best
margin to stand clear of the control's own largest excursion *and* to show a
dose-response across neighbouring α.

The reading: **the fact is decodable but not consumable as a single fixed
direction.** A linear readout can find it; the downstream computation cannot be
made to use it by a rank-one nudge. Aim 3 therefore needs a *trained* layer
with the proposal's full objective, not a vector — which is what the proposal
specifies anyway, and this now says so with evidence rather than assumption.

This is a prototype and not Aim 3: one direction, one layer, one position, no
`L_ontology` / `L_causal` / `L_uncertainty` term, nothing trained.

## 6. What this says about the two benchmarks

Patching could not run on `data/` at all: 60 of 64 pairs do not tokenise to the
same length, because **the two arms differ in their distractor labs as well as
the causal value** (bug 5 in `HANDOFF.md` §5, found through this attempt). All
64 test pairs are affected.

The probe null gave an independent signature of the same bug. On a clean
minimal-pair benchmark an uninformative probe predicts the same label for both
arms and scores ~0; on a contaminated one it can tell the arms apart and drifts
to the 0.25 random baseline:

| Benchmark | Pairs | Mean null across layers |
|---|---|---|
| Real notes, renal (clean) | 215 | **0.013** |
| Real notes, QT (clean) | 85 | 0.164 |
| Synthetic (contaminated) | 64 | **0.268** |

`build_dataset.py` is fixed and verified in a scratch build (64/64 → 0), but
`data/` has not been regenerated and no synthetic result has been re-run.

## 7. Limitations

- **Linear probes only.** A negative probe result rules out a linearly readable
  representation, not any representation at all. A non-linear probe or an SAE
  (Aim 1 proper) could still find structure in the renal case.
- **Two pooling choices, one position.** "final" and "mean" do not exhaust the
  question; probing at the token that *states* the creatinine was not done.
- **One model, one seed, greedy decoding.**
- **85 QT pairs and 215 renal pairs.** The QT margin (+0.624) is far outside
  sampling noise; the synthetic margin (+0.281) is more modest at 64 pairs.
- **Steering explores a single direction at a single position.** A null here
  does not rule out that a trained low-rank map would work.

## 8. Reproducing

```bash
export CUDA_VISIBLE_DEVICES=0

python src/probe.py --data data/medcalc/counterfactual_heldout.jsonl \
    --cache results/acts/medcalc_heldout.npz \
    --out results/probe_medcalc_heldout.json

python src/patching.py --data data/medcalc/counterfactual_test.jsonl \
    --limit 80 --out results/patching_medcalc_test.json

python src/steering.py --data data/medcalc/counterfactual_heldout.jsonl \
    --cache results/acts/medcalc_heldout.npz \
    --layer 5 15 25 30 --alphas 0 0.5 1 2 4 8 \
    --out results/steering_qt.json
```

**Only one 7B process fits on the card.** Activation caches live in
`results/acts/*.npz` (~100 MB each); probe re-fits reuse them and need no GPU.
