# The full ablation on real clinical text

> **SUPERSEDED IN PART — read this first.**
>
> The numbers below were produced on a 430-item test split. That split was
> later re-partitioned to create the training set the Aim 3 constraint layer
> needs, and the QT family was split again so the held-out evaluation is
> leak-free for every row. The current splits are 180 renal test items and 60
> QT evaluation items, and the current numbers are in
> `results/table_medcalc_test.md` and `results/table_medcalc_heldout.md`.
>
> **Every finding in this report replicates on the new splits**: base CC 0.023
> -> 0.011, RAG 0.000 -> 0.000, gate 0.986 -> 0.989, and on the QT family the
> base model still answers SAFE to every single item. What is superseded is the
> arithmetic, not the conclusions. Sections 1, 2 and 4 (construction, guards,
> and the coverage analysis) are unaffected and remain the reference for how
> this benchmark is built.
>
> The reproduce commands in section 7 build the CURRENT splits. See
> `results/FIXES.md` section A4.

**All four stages, run on real PMC case-report notes instead of synthetic
vignettes.** The headline: on real clinical text the base model's Causal
Consistency is **0.023**, retrieval-augmentation takes it to **0.000**, and the
symbolic gate takes it to **0.986** — but the gate can only be *applied* to
**42.7%** of real notes in the first place, and that ceiling is the finding
that matters most.

---

## 1. Why this benchmark exists

The benchmark in `data/` is 128 items of templated text. `src/components.py`
already flagged the limitation in its own header:

> "Because the vignettes are templated, extraction is near-perfect, so the
> gate's measured contribution is an UPPER BOUND. On free-text notes the
> extractor will degrade and the gap will shrink."

That prediction had never been tested. This tests it, and quantifies the gap.

## 2. Construction

**Source.** `ncbi/MedCalc-Bench-v1.2` — 11,643 rows of real patient notes drawn
from PMC case reports, each with structured `Relevant Entities` and a verified
`Ground Truth Answer` for a named clinical calculator.

**Task.** Same as the original: is a proposed prescription safe for this
patient? A published threshold on a computed clinical quantity decides.

| Family | Quantity | Threshold | Drug | Split |
|---|---|---|---|---|
| `metformin_renal` | eGFR (CKD-EPI 2021) | < 30 mL/min/1.73m² → unsafe | metformin | test |
| `ondansetron_qt` | QTc (Bazett) | > 500 ms → unsafe | ondansetron | **held out** |

Both thresholds are quoted from FDA labelling, not written from memory. The
metformin one is the sentence `fetch_openfda.py` already pulled down:
*"Severe renal impairment: (eGFR below 30 mL/min/1.73 m2)"*. This is part of
P4 being avoided rather than repeated.

**Counterfactual pairs.** One arm is the **real, unedited note**. The other is
the same note with exactly one number changed — the creatinine (or the QT
interval) — so the computed value crosses the threshold and the label flips.
Everything else is byte-identical.

| Split | Items | Pairs | Labels |
|---|---|---|---|
| test | 430 | 215 | 215 SAFE / 215 UNSAFE |
| heldout | 170 | 85 | 85 / 85 |
| calib | 80 | 40 | 40 / 40 |

### Formula validation

Nothing is taken on trust: the labels depend on the formulas twice over, once
to label and once to solve for the counterfactual creatinine. Every
implementation is checked against MedCalc-Bench's own answers before any item
is emitted.

| Formula | Rows | Exact | Within their band | Mismatch |
|---|---|---|---|---|
| MDRD | 295 | 265 | 26 | **0** |
| CKD-EPI 2021 | 503 | 427 | 41 | 27 |
| QTc Bazett | 100 | 100 | — | **0** |
| Cockcroft-Gault | 151 | 56 | 15 | **77** → **dropped** |

Cockcroft-Gault was dropped rather than patched: MedCalc applies an adjusted
body weight rule to it, and its output is in mL/min, which does not match the
FDA threshold's units anyway. Inverse solve round-trips to a maximum error of
1.0e-9.

Labels use **CKD-EPI 2021 for every renal note**, not whichever calculator the
source row happened to use. A benchmark whose ground truth depends on which row
of a CSV an item came from measures bookkeeping, not a clinical rule.

### Seven guards, three of which caught something real

| Guard | What it prevents | Dropped |
|---|---|---|
| **Creatinine stated once, or repeatedly but consistently** | **items with no unambiguous ground truth** | **418** |
| Margin from the threshold on both arms | testing arithmetic precision, not reasoning | 37 |
| Formula reproduces MedCalc's answer | mislabelled items from a wrong formula | 27 |
| Note's creatinine matches MedCalc's entity | gate reading a different value from the label | 20 |
| Deduplicate by note | same note counted twice | 14 |
| Note does not already state eGFR/CrCl | text contradicting its own label (bug #2 class) | 12 |
| **Physiological plausibility** | **impossible source data** | **12** |
| Round first, then re-derive the label | label not following from the visible number | 1 |

558 of the 764 candidate notes are dropped. The largest single loss is
**330 notes that state creatinine more than once with differing values**
(a further 88 never state it with a unit at all) — "0.7 on admission … 3.3 on day 5". Such a note has no
single eGFR; MedCalc-Bench resolves it by picking one, but nothing in the text
tells a reader which, so the item would have no ground truth. Same failure
class as bug #2 in `HANDOFF.md` §5.

The plausibility guard caught real corruption in the source: `pmc-6997309-1`
states creatinine as "1.1 µmol/L", roughly 1/60th of the lower limit of normal
and almost certainly 1.1 mg/dL mislabelled. MedCalc-Bench propagates it and
reports an eGFR of **12,129** mL/min/1.73m² against a physiological maximum
near 120. A formula check passes on such a row because the formula is being
applied correctly to nonsense.

### Integrity checks on the emitted data

| Check | Result |
|---|---|
| Pairs differing by more than one word | **0 / 215** |
| Pairs whose labels fail to flip | **0 / 215** |
| Pairs lacking one SAFE and one UNSAFE arm | **0 / 215** |
| Creatinine present in the vignette text | **430 / 430** |

**Answer leakage is structurally impossible at pair level.** Both arms share
the same note body, so any cue in the text appears in both. 104 test notes
mention dialysis — 52 SAFE, 52 UNSAFE. 76 mention renal failure — 38 / 38.
Guessing from any such phrase yields exactly chance.

## 3. Results — test split (430 items, 215 pairs)

| Variant | Accuracy (strict) | Causal Consistency | 95% CI | Violation | Coverage |
|---|---|---|---|---|---|
| (1) Base LLM | 0.507 | **0.023** | [0.005, 0.047] | 0.358 | 1.000 |
| (2) + RAG | 0.498 | **0.000** | [0.000, 0.000] | 0.042 | 1.000 |
| (3) + Symbolic Gate | 0.993 | **0.986** | [0.967, 1.000] | 0.000 | 1.000 |
| (4) + UQ Engine | 0.993 | 0.986 | [0.967, 1.000] | 0.000 | 0.998 |

Paired McNemar on identical items:

| Comparison | B01 | B10 | p |
|---|---|---|---|
| base → rag | 81 | 85 | **0.816 — no net change** |
| rag → nsai | 213 | **0** | **1.5e-64** |

### Held-out split (170 items, unseen rule family)

| Variant | Accuracy | CC | Violation | What it actually answered |
|---|---|---|---|---|
| (1) Base LLM | 0.500 | **0.000** | **1.000** | **SAFE on all 170** |
| (2) + RAG | 0.500 | **0.000** | 0.000 | **UNSAFE on all 170** |
| (3) + Symbolic Gate | **1.000** | **1.000** | 0.000 | 85 / 85 |

This is the cleanest possible demonstration of why Causal Consistency is
scored pair-level. Both neural variants score a respectable-looking 0.500
accuracy on a balanced set while emitting a **constant answer** and scoring
0.000 CC. Accuracy alone would have hidden it completely.

## 4. What each stage actually did

### Stage 1 — the base model is at chance on real notes

Accuracy 0.507 on a balanced set. It answers UNSAFE on 63.5% of items.
Arm-wise: 0.372 on truly-SAFE cases, 0.642 on truly-UNSAFE. On the held-out
family it degenerates completely to a single answer.

The synthetic benchmark gave 0.594 / CC 0.234. Real clinical prose removes even
that. The task there is genuinely harder: the note states a creatinine, and
deciding safety requires computing an eGFR from it.

### Stage 2 — RAG does not fail to retrieve; it fails *because* it retrieves

This is a sharper result than "RAG does not help", and it is not a plumbing
failure. Retrieval works essentially perfectly:

| Split | Top-3 contains a document stating the threshold |
|---|---|
| test | **99.5%** |
| heldout | **100%** |

The real FDA contraindications text — containing the literal string *"eGFR
below 30 mL/min/1.73 m2"* — was in the prompt for practically every case. What
happened:

| | says UNSAFE | acc on truly-SAFE | acc on truly-UNSAFE |
|---|---|---|---|
| base | 63.5% | 0.372 | 0.642 |
| **+ RAG** | **96.0%** | **0.037** | **0.958** |

Retrieving contraindication text makes the model answer "contraindicated"
almost regardless of the patient's numbers. Net accuracy is unchanged
(p = 0.816) because the gain on unsafe cases exactly offsets the collapse on
safe ones — but Causal Consistency goes to **0.000**, because it can no longer
get both arms of any pair right.

**78 truly-safe cases were flipped from a correct answer to a wrong one.**
Inspecting them shows the mechanism:

- *Misapplied distractor.* One case: "the patient has chronic kidney disease
  stage 3b, and metformin is contraindicated in this stage." Stage 3b was read
  off the retrieved CKD-staging distractor, and it means eGFR 30–44 — **above**
  the metformin threshold. The retrieved text supplied a fact the model then
  reasoned from incorrectly.
- *Bare assertion.* After retrieving the boxed warning, the model emits
  "UNSAFE" with no justification at all.
- *Instruction echo.* In some cases it repeats the prompt's formatting
  instruction back instead of answering.

So the defensible claim is stronger than before: **retrieval of authoritative
warning text induces a systematic bias toward the warning, overriding the
patient data.** For a clinical safety system that is a specific, actionable
failure mode, not a null result.

### Stage 3 — the gate works, and its real limit is coverage, not accuracy

On the items it is given: fires on 424/430 (98.6%), **0 wrong decisions**,
extraction of creatinine / age / sex accurate on 100%. Held-out: 170/170,
100% correct on a rule family never seen in the test split.

**This 0.986 is inflated by construction and must not be quoted alone.** The
same function that locates the creatinine (`creatinine_mentions`) is used both
to filter the dataset and inside the gate. The benchmark therefore contains
exactly the notes this gate can read. That is a deliberate choice — items
without an unambiguous creatinine have no valid ground truth and cannot be in a
benchmark at all — but it makes the accuracy figure circular.

**The uncircular number is coverage.** Applying the gate to all 764 distinct
real renal notes in the source, regardless of whether they yield a valid
benchmark item:

| Outcome | n | Share |
|---|---|---|
| **Gate can decide** | **326** | **42.7%** |
| Creatinine stated with conflicting values | 327 | 42.8% |
| No creatinine stated with a unit | 98 | 12.8% |
| Implausible after extraction | 10 | 1.3% |
| Age not extractable | 3 | 0.4% |

**The symbolic gate is applicable to 42.7% of real clinical notes**, against
56.2% on the original templated vignettes. `components.py` predicted the
degradation; this is its size.

More important is *why* it fails. The dominant blocker is not weak regexes —
it is that clinical narratives report the same lab repeatedly as it changes
over the admission. No improvement to extraction fixes that. Deciding which
creatinine is "the" creatinine requires temporal reasoning about the patient's
course, which is a different and much harder problem than fact extraction.
**Any constraint layer built on "read the value, check the threshold" inherits
this ~43% ceiling on real notes.** That is a direct constraint on Aim 3.

### Stage 4 — nothing left to do

Coverage 0.998; it abstains on essentially nothing. With the gate deciding
98.6% of items at zero error, the residual neural population is 6 items. There
is no uncertainty signal to measure on 6 items, and no deferral decision worth
making. This reproduces the structural point from `results/bigbench_uq.md`: the
gate and the UQ engine compete for the same cases, and a strong gate leaves the
UQ engine nothing to govern.

## 5. Comparison with the original benchmark

| | Original (`data/`) | This (`data/medcalc/`) |
|---|---|---|
| Text | templated synthetic vignettes | real PMC case-report notes |
| Test items | 128 | **430** |
| Held-out items | 32 | **170** |
| Decisive quantity | stated outright ("eGFR of 27") | must be **computed** from a stated lab |
| Thresholds | hand-written from memory | **quoted from FDA labelling** |
| Base LLM CC | 0.234 | **0.023** |
| RAG CC | 0.281 | **0.000** |
| Gate CC | 0.641 | 0.986 (circular — see §4) |
| Gate coverage | 56.2% | **42.7%** of real notes |
| Retrieval hit rate | 64.8% test | **99.5%** test |

## 6. Limitations

- **The gate's accuracy is circular** (§4). Its coverage figure is not.
- **Single seed, deterministic decoding.** P2 is untouched by this run.
- **Two rule families, one drug each.** The original has ten. Breadth was
  traded for realism of text.
- **The counterfactual arm is synthetic.** One arm of every pair is a real
  note; the other is a real note with one number altered. The altered value is
  physiologically plausible and consistent with the rest of the note, but it is
  not an observed patient.
- **The edited number is the only cue that changes**, which is what makes the
  pair valid, but also means the model never has to weigh conflicting
  evidence — a real reviewer often does.
- **Notes come from published case reports**, which are selected for being
  interesting and are not representative of routine records. MIMIC-IV would be
  the next step; the Demo v2.2 is open access with no credentialing.

## 7. Reproducing

```bash
export CUDA_VISIBLE_DEVICES=0

# fetch source (55 MB) into data/external/, then build
python src/build_medcalc.py --out data/medcalc

# gate extraction diagnostic -- extraction accuracy against ground-truth facts
python src/medcalc_gate.py --data data/medcalc/counterfactual_test.jsonl \
                                  data/medcalc/counterfactual_heldout.jsonl

# the four-stage ablation
python src/run_eval.py --backend hf --model_id BioMistral/BioMistral-7B \
    --seeds 0 --batch_size 4 --data data/medcalc --gate medcalc --tag _medcalc
python src/run_eval.py --backend hf --model_id BioMistral/BioMistral-7B \
    --seeds 0 --batch_size 4 --data data/medcalc --gate medcalc --tag _medcalc \
    --split heldout

python src/make_table.py --split test    --tag _medcalc \
    --out results/table_medcalc_test.md
python src/make_table.py --split heldout --tag _medcalc \
    --out results/table_medcalc_heldout.md
```
