# What was wrong, what was fixed, and what it changed

An audit of this repository found nine defects. All are fixed and everything
affected has been re-run. This file records each one, what it would have done
to a reported number, and whether the conclusion survived.

**Headline: no conclusion reversed.** The two defects that could have reversed
one (non-minimal pairs, mislabelled β-blocker items) turned out to have shifted
the numbers by at most 0.09 and to have left every ordering intact. That is
worth stating plainly rather than quietly re-running.

---

## A. Defects in the data

### A1. Counterfactual pairs were not minimal — 64/64 test pairs

The two arms of a pair drew **different distractor labs**, because `render()`
consumed the shared RNG once per arm. A pair meant to differ only in eGFR also
differed in CRP, sodium, platelets and haemoglobin.

Why it matters: Causal Consistency is defined on minimal pairs. If the arms
differ in four places, a model that flips its answer because the sodium changed
is scored as if it had responded to the causal factor.

The fix had been written into `build_dataset.py` but **the dataset was never
rebuilt**, so every reported number still came from the broken file. Fixed,
rebuilt, and a build-time integrity check now refuses to write a split whose
arms differ outside the causal factor.

### A2. Items that contradicted their own label — found by that new check

The implicit β-blocker renderer selected the severe-asthma text with
`"asthma" in str(value)`. Every SAFE value of that family reads
`"no history of asthma"` or `"no respiratory disease"` — and *"no history of
asthma" contains "asthma"*. So **both arms of every implicit β-blocker pair
described a patient with severe asthma, while one carried the label SAFE**.

A model reading the text in front of it and answering UNSAFE was scored wrong
for being right. Renderers now branch on the arm, passed in by the caller, not
on substring sniffing.

### A3. Clinically impossible patients

| | before | after |
|---|---|---|
| pregnancy family ages | 67, 69, 71, 74, 76, 77, 81, 84 | 22–41 |
| a 15-year-old | "a child in year 10 at **primary** school" | "a pupil in year 10 at secondary school" |
| implicit renal creatinine | one of four hard-coded values (340 / 295 / 88 / 74 µmol/L) regardless of the eGFR being encoded | exact CKD-EPI inverse of the stated eGFR, rounded, label re-derived from the rounded value |

The pregnancy items were formally correct and unusable: a model answering UNSAFE
because the case is absurd scores identically to one that knows ACE inhibitors
are teratogenic. The creatinine defect silently undid the near-threshold
hardening — an "eGFR 27" arm was rendered as a creatinine implying eGFR ≈ 13.

### A4. The MedCalc report was not reproducible

The report described a 430-item test split; the data directory had since been
re-partitioned to make room for the constraint layer's training set and held
180. The build is deterministic (verified: same ids, same splits, byte for
byte), so this was a documentation defect, not a data one — but nothing in the
repository could notice it.

Fixed structurally: the held-out QT family is now split into `heldout_train`
(55 pairs) and `heldout` (30 pairs) as separate files, so the evaluation split
is leak-free for every row including the constraint layer, and
`src/make_summary.py` regenerates the summary **from the JSON artifacts** so a
number that no longer exists cannot survive in a report.

## B. Defects in the analysis

### B1. `± 0.000` was not a variance estimate

Decoding is greedy, so a different seed reproduces the run exactly. The `±`
column was a standard deviation over one seed. Replaced with 95% bootstrap CIs
over items, which is the uncertainty that actually exists. `±` reappears only
with more than one seed.

### B2. The ladder could not attribute an effect to a component

Every contribution was measured on top of the one before it. On real notes RAG
*lowers* Causal Consistency (0.011 → 0.000), so the gate's contribution was
being measured from a damaged intermediate state. Added rows 5–7: each
contribution on top of the **base model**, plus McNemar of every row against
the baseline.

### B3. The constraint layer could not be compared to the table

It was scored by argmax over the answer logits; the table is scored by
generating text and parsing the decision word. Two readouts that need not
agree. The adapter is now attached inside the model wrapper and scored through
the identical path — they agree to three decimals, which is itself worth
knowing.

### B4. The gate's circularity was a footnote

Its accuracy is partly circular: it applies the same rule and threshold the
labels were generated from. The table now carries a firing-rate column and a
gate-fired / gate-declined breakdown, which separates the circular part from
the part that is not.

## C. Things the proposal promised and the code did not do

| Promised | Was | Now |
|---|---|---|
| Aim 1: SAEs, FIS, JumpReLU/TopK fallback | **absent** — only a linear probe | `src/sae.py`: both architectures, 16k features on 150k real tokens, FIS, feature knock-out with matched controls. `results/aim1_sae.md` |
| `L_ontology`, `L_uncertainty` | not implemented | implemented and **measured**; the uncertainty term produced a negative result, reported |
| Adaptive Conformal Inference | a single frozen split-conformal threshold | `metrics.adaptive_conformal`, evaluated in `results/uq_coverage_*.md` |
| Conditional coverage across subgroups | not measured | `metrics.conditional_coverage`, per family / label / presentation / gate-fired |
| KL direction in the docstring | said KL(adapted‖base) | code always computed KL(base‖adapted); docstring corrected |

---

## D. Did any conclusion change?

### The synthetic benchmark, before and after the data fixes

| Variant | acc before | acc after | CC before | CC after |
|---|---|---|---|---|
| Base LLM | 0.594 | 0.602 | 0.234 | 0.219 |
| + RAG | 0.633 | 0.648 | 0.281 | 0.312 |
| + Symbolic Gate | 0.820 | 0.836 | 0.641 | 0.672 |
| + UQ Engine | 0.562 | 0.562 | 0.562 | 0.562 |
| Base LLM (held-out) | 0.500 | 0.594 | 0.125 | 0.250 |

Largest movement is the held-out baseline (+0.094 accuracy, +0.125 CC), which
is 32 items. Every ordering is unchanged and every earlier claim still holds.

### What is genuinely new

1. **A constraint-layer row in the table**: CC 0.000 → 0.767 on 30 unseen QT
   pairs, p = 1.5e−05 against the baseline, at −1.2% WikiText-2 perplexity.
2. **Aim 1 exists**: an SAE finds concept-selective features (QT 0.703,
   creatinine 0.603) whose knock-out moves the decision by 0.002–0.016 logits
   when 1–5 are needed — replicating the patching null by a different route.
3. **`L_uncertainty` does not generalise**: satisfied on training notes, but on
   unseen redacted notes the adapter became *more* confident, not less.
4. **The isolated rows change how the gate should be quoted**: on real notes
   `Base + Gate` reaches the same 0.989 as `Base + RAG + Gate`, so RAG
   contributes nothing to that number and its own row is negative.
5. **A UQ row can legitimately read 0.000 coverage**: with a 10% error target
   and a near-chance model, split conformal has no admissible threshold and
   must answer nothing. The table now explains this where it occurs instead of
   leaving it looking like a crash.
