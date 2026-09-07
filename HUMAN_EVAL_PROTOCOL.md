# Human evaluation protocol

Everything here is ready to run. The one thing it cannot supply is the raters.

`src/human_eval.py` draws the sample, blinds it, writes a packet a clinician
can open in a browser, reads the responses back, computes inter- and
intra-rater agreement, derives `S_human`, and evaluates the proposal's
fallback trigger. Until real responses exist, `ingest` **refuses to emit a
zero-valued `S_human`** — "not measured" and "measured at zero" are different
claims and only one of them is true.

---

## 1. What this unblocks, and where the proposal asks for it

| Requirement | Proposal location | Status without raters |
|---|---|---|
| `S_human`, the third FIS term | Aim 1 Approach; §4.4 | weight forced to 0; FIS is a two-term score, stamped as such in every artifact |
| "<30% passing expert validation → fall back to JumpReLU/TopK" | §4.4 | **not evaluable.** The architecture choice was made on reconstruction error (FVU 0.067 vs 0.131) instead, which is a different criterion and must be reported as one |
| Blind expert rating of explanation correctness and usefulness | §4.5 | not implemented |
| "human-in-the-loop" | abstract, four-aims line | see the note below |

**A discrepancy to carry into the write-up.** Human evaluation appears in Aim
1's Approach, §4.4 and §4.5. It does **not** appear in §4.7, the Aim 4
methodology section, which is entirely conformal UQ, abstention and
conditional coverage. "Human-in-the-loop" survives only in the abstract's
four-aims line, and §2's Aim 4 explicitly places hospital deployment "beyond
the current project scope". Attributing the blind expert rating to Aim 4 is a
misreading of the proposal; it belongs to Aims 1 and 2.

---

## 2. The three instruments

Drawn together so one rater session covers all three.

**A — Feature interpretability (Aim 1, feeds `S_human`).**
Each item is one SAE feature, shown as its highest-activating spans in
context. The rater is **not** shown the concept the pipeline assigned, its
`S_semantic`, or its FIS. Showing them would ask "do you agree with this
regex", which is a weaker question than the one the FIS is defined on. The
rater names the concept themselves, rates coherence 1–5, and answers the
accept/reject question the trigger is computed from.

**B — Explanation quality (Aim 2, §4.5).**
The case, and the system's own justification sentence, with the variant that
produced it hidden. Rated for correctness 1–5, usefulness 1–5, and a
safety-critical-error flag.

Note what sampling this exposed: of 900 responses on the real-notes test
split, **102 carried no justification at all** — the model answered with the
bare decision word despite the prompt asking for "one short sentence of
justification". Those are excluded from the instrument rather than shown to a
clinician as an explanation to rate, and the exclusion count is reported,
because "there was no explanation to evaluate" is a finding about Aim 2's
blind rating rather than a preprocessing detail.

**C — Clinical safety.**
Stratified over the two case types that matter: an UNSAFE prescription the
system called SAFE (a violation), and the cases the UQ engine declined. The
rater says whether they agree with the decision, how much harm could follow,
and whether the system should have deferred.

## 3. Blinding and randomisation

- Item keys are `sha256` digests of the source identifiers, so nothing in a
  key reveals the variant, the model or the label.
- The answer key is written to a **separate file** (`answer_key.json`) that no
  packet ever includes.
- Every rater receives a different item order, so an order effect cannot be
  mistaken for agreement.
- 10% of items are repeated within each packet to measure intra-rater
  reliability. A rater who scores the same item differently twenty items
  apart is telling you the rubric is ambiguous, and there is no other way to
  learn that.
- Verified on the generated packet: no variant name, adapter path or gold
  label appears. The only matches for those strings are `NSAIDs` and
  `FDA label` occurring inside the clinical text itself.

## 4. Raters

Three independent raters, none of whom has worked on this system.
Inclusion: a clinician who prescribes the drug classes involved — metformin,
ondansetron, nitrofurantoin, apixaban — routinely. No conferring during
rating.

**Sample size.** 115 items × 3 raters = 1,035 response cells, of which 28
items carry the feature instrument. That is enough to estimate the pass rate
against the 30% trigger with a 95% interval of roughly ±0.17, which separates
"clearly above 30%" from "clearly below" but not much finer. Increasing
`--n_features` widens the feature instrument at the cost of rater time; the
trigger is the only quantity whose precision depends on it.

## 5. Analysis

- **Ordinal fields** (coherence, correctness, usefulness, harm): Krippendorff's
  α with the ordinal difference function, which tolerates missing cells and
  any number of raters per item without imputation.
- **Binary fields** (accept/reject, safety error, agree/defer): Fleiss' κ.
- Both are hand-rolled in `human_eval.py`. `statsmodels` is not installed in
  this environment and two coefficients do not justify adding a dependency to
  a pinned research image.
- **`S_human` per feature** = the fraction of raters who accepted it.
- **Pass rate** = the fraction of features accepted by at least half the
  raters. The §4.4 trigger fires when that is below 0.30.

Report α and κ **before** any S_human number. A pass rate computed from raters
who do not agree with each other is not a measurement, and the dry run
demonstrates the failure mode directly: fabricated ratings give a
plausible-looking pass rate of 0.89 alongside α = −0.31 and κ = 0.13, which is
the agreement statistics correctly reporting that the "raters" were noise.

## 6. Running it

```bash
python src/human_eval.py sample --n_features 25 --n_explanations 60 --n_safety 40
python src/human_eval.py packet --raters 3
# ... clinicians fill results/human_eval/packets/responses_rater*.csv ...
python src/human_eval.py ingest
```

To exercise the plumbing without raters:

```bash
python src/human_eval.py dry_run --raters 3
```

Everything the dry run writes lands under `results/human_eval_SIMULATED/` and
carries `SIMULATED — NOT EXPERT DATA` in its JSON and in every response cell.
It must never be reported.

## 7. The second thing waiting on a human

`data/curation_worksheet.csv` — 145 rows, `constraint_value` empty on all of
them. Five of the ten rule families have thresholds that are `absent` or
`construct_mismatch` against their FDA labels, including `ondansetron_qt`,
the family carrying the Aim 3 result. Two of them are the dangerous kind: a
number **is** present in the label and encodes something else (`aspirin_reye`
is an OTC dosing instruction, `spironolactone_hyperkalaemia` a heart-failure
initiation criterion). This needs a curator, not a better parser, and it
belongs in the same packet as the ratings above.
