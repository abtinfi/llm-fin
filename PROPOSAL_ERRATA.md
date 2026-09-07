# Errata and respecifications for the proposal

`AbstractOfProposal (1).pdf` is a flattened, watermarked PDF and cannot be
edited in place. This file is the change list: each entry gives the location,
the sentence as written, the sentence it should become, and the measurement
that forces the change. Nothing here is a matter of taste — every item is
either a claim the implementation contradicts or a term the implementation
does not use.

---

## 1. SNOMED CT is named twice and used nowhere

**§1, page 2, "Long-Term Vision"**

> …because it provides rigorous structured biomedical knowledge constraints
> (via established ontologies like **SNOMED CT [30]** and UMLS [31]), and the
> high cost of error necessitates mechanistic transparency.

→ replace with

> …because it provides rigorous structured biomedical knowledge constraints
> (via established ontologies — here **UMLS** for concept identity and
> **MED-RT** for drug–condition relations), and the high cost of error
> necessitates mechanistic transparency.

**§3.2, page 3, "Neuro-Symbolic Integration and Medical Ontologies"**

> Integrating symbolic knowledge bases (e.g., **SNOMED CT**) with neural
> networks provides a structured safety boundary [7].

→ replace with

> Integrating symbolic knowledge bases (e.g., **UMLS/MED-RT**) with neural
> networks provides a structured safety boundary [7].

**Why.** SNOMED CT appears in **zero** Python files and zero shell scripts in
the implementation. `grep -rn SNOMED` finds it only in prose. It reaches the
repository once, incidentally, as `rootSource: "SNOMEDCT_US"` inside cached
UMLS API responses — SNOMED atoms arrive through UTS because UMLS aggregates
them, not because anything targets SNOMED as a source vocabulary.

What is actually used is UMLS for concept identity (CUIs, atoms, synonyms) and
**MED-RT** for the drug–disease relations that carry the contraindications.
`src/umls_grounding.py` sets `CLINICAL_SABS = "MED-RT"` and says why: the plain
`/relations` endpoint is dominated by RxNorm formulation links, which are not
clinical relations. RxNorm is filtered **out** as noise; it is not a knowledge
source here either.

This is a substitution of vocabulary, not of ambition, and it should be
described as such rather than left to look like a silent equivalence.

## 2. §4.2 — the SCM's grounding must state its measured coverage

**§4.2, page 4**

> The Structural Causal Model (SCM) is initialized from biomedical ontologies
> (e.g., UMLS) and refined using expert-curated causal relations.

→ replace with

> The Structural Causal Model is initialized from **UMLS concept identities
> and MED-RT drug–condition relations**, and refined using expert-curated
> causal relations. The realised graph has **184 nodes and 234 edges (224
> UMLS-derived, 10 curated)**, every edge provenance-tagged. Coverage is
> partial and is reported as such: **5 of the 10 rule families are
> ontology-attested** (4 exact, 1 synonymous, 5 absent).

**Why.** The sentence as written implies the ontology supplies the graph and
curation polishes it. The measurement is the other way round for half the
families. In particular `ondansetron_qt` — the family carrying the Aim 3
constraint-layer result — is **absent from MED-RT**, so the graph licenses no
contraindication path for it and that result rests on curation at both ends.
`statin_macrolide` is a drug–drug interaction and cannot be attested by a
drug–disease relation at all.

One further limit worth stating rather than discovering later:
`mechanism_annotations()` deliberately does **not** claim a
Disease→mechanism→contraindication path. MED-RT asserts
`propranolol has_physiologic_effect Bronchoconstriction` and
`propranolol contraindicated_with_disease Asthma` as two independent facts and
never connects them. Joining them is a clinician's inference, not an ontology
lookup.

## 3. §4.7 — Equation (2) measures the wrong quantity

**§4.7, page 6.** Equation (2) defines predictive entropy over the whole
vocabulary:

> H(y) ≈ −(1/N) Σᵢ Σ_{w∈V} P(tᵢ=w | t_<i, x) log P(tᵢ=w | t_<i, x)

→ respecify the uncertainty term as the same entropy **restricted to the
decision tokens**, and add:

> The whole-vocabulary form of Eq. (2) was measured over 6,456 items
> (MedMCQA 4,183 + MedQA-USMLE 1,273 + PubMedQA 1,000) at **AUROC 0.525
> [0.511, 0.539]** — indistinguishable from chance, and 0.490 on MedQA alone.
> Restricting the identical entropy to the answer tokens gives **0.687
> [0.675, 0.700]**, a paired difference of **+0.163 [+0.144, +0.181]**,
> p < 0.0001 Holm-corrected, replicated independently on all three
> benchmarks. A whole-vocabulary control (`max_entropy`) does not help: what
> matters is restricting attention to the decision, not the functional form.

**Why.** This is a measurement, not a preference. It also matters that the
implementation was never broken — the finding is that §4.7 asks for the wrong
quantity, and the deferral behaviour it wants works once the signal is right
(pooled error 0.558 → 0.282 at 20% coverage, monotone at every step).

The "before" numbers are preserved under `results/eq2_reference/` so the
respecification can be shown rather than asserted.

## 4. §4.2 — the Token-to-Concept Attribution Layer needs its scope stated

**§4.2, page 4**

> …we introduce an explicit Token-to-Concept Attribution Layer [8, 15]. This
> attribution layer will be quantitatively evaluated for faithfulness using
> standard sufficiency and comprehensiveness metrics.

Keep the sentence; add the constraint the implementation exposed:

> The layer can only name a concept for which the sparse dictionary contains
> a feature. Concept coverage is therefore a property of the dictionary and
> must be reported alongside any attribution result.

**Why.** See `results/attribution.md`. On the renal family the layer ranks the
causally edited token at the **98th percentile** and names the right concept
**90%** of the time against a 25% chance baseline. On the QT family it scores
at chance, because the scored dictionary contained **no** `qt_interval`
feature whose knock-out beat its matched random control.

Whether that is a vocabulary gap or an artifact of our own measurement is a
separate question, and the implementation settles it separately rather than
assuming: the weights are knock-out excesses measured on a split of renal
items only, where a QT feature is causally inert by construction and scores
zero whether or not it exists. See `results/attribution.md` §4.

Note also a terminology collision the proposal creates for itself: §4.2's
faithfulness **sufficiency** (an ERASER rationale metric, lower is better) and
§4.5's causal **sufficiency** (feature injection, larger effect is better) are
different quantities. The implementation keeps them in separate files with
separate key names for exactly this reason.

## 5. §4.4 — the "<30% expert validation" trigger is not evaluable as written

**§4.4, page 5**

> If standard SAEs fail (<30% passing expert validation), we trigger a
> fallback to JumpReLU or TopK SAE architectures.

Either supply raters, or restate the criterion in terms the pipeline can
evaluate. As written the trigger is defined on a quantity that does not exist
without a clinician panel, so it cannot fire and cannot fail to fire.

**What was done instead, and it must be reported as a substitution:** both
architectures were trained and the choice was made on **reconstruction error**
— TopK FVU 0.067 / L0 31.9 / 39.3% dead against JumpReLU FVU 0.131 / L0 27.8 /
64.0% dead. That is a defensible criterion. It is not the proposal's criterion.

`HUMAN_EVAL_PROTOCOL.md` and `src/human_eval.py` make the proposal's version
runnable the moment raters are available.

## 6. Aim 4 — "human-in-the-loop" is in the abstract but not in the method

The abstract's four-aims line says Aim 4 will "perform synthetic and
**human-in-the-loop** computational validation". §2's Aim 4 Approach mentions
only synthetic counterfactual benchmarks and explicitly places hospital
deployment "strictly as a future translation pathway beyond the current
project scope". §4.7 — the Aim 4 methodology — contains no human evaluation
at all.

Pick one. Either drop "human-in-the-loop" from the abstract, or add the rating
protocol to §4.7. The human evaluation that the proposal *does* specify
belongs to **Aim 1** (`S_human`, §4.4) and **Aim 2** (§4.5's blind rating), and
attributing it to Aim 4 misreads the document.

## 7. §4.6 — the ablation matrix as run is larger than as specified

§4.6's matrix has four rows: Base LLM, +RAG, +Symbolic Gate, +UQ Engine. The
implementation runs **eight**, and the four extra ones are the point.

A cumulative ladder cannot attribute an effect to a component. On real
clinical notes RAG makes Causal Consistency **worse**, so the "gate's
contribution" measured on top of RAG is measured from a damaged baseline.
Rows 5–7 therefore add exactly one contribution to the base model each, and
every McNemar test is taken against the base model rather than against the
previous rung. Row 8 is all four at once.

The four-row matrix should be replaced by the eight-row one, with the reason
stated.

## 8. §4.6 — MIMIC-IV, and what a demo can and cannot support

> MIMIC-IV [29] will be used for retrospective evaluation and scenario
> construction rather than direct causal graph discovery.

Add:

> The open-access **MIMIC-IV Clinical Database Demo v2.2** (ODbL, no
> credentialing) supplies the structured-labs half of this: 71 pairs / 142
> items across four rule families, the only arm in the project where **both**
> sides of every counterfactual pair are real measured values. It carries no
> free-text notes, so the note is rendered from structured fields, and the two
> arms are different **timepoints** in the same patient, so the clinical state
> genuinely differed. A *retrospective evaluation* claim in the full sense
> still requires credentialed MIMIC-IV.

**Why.** This should be stated positively rather than left as a gap: the
pipeline runs end to end with no credentialed source at all, and
`src/check_data.py --strict` is wired in as stage 0 and fails the run if that
ever stops being true.

---

## Numbers that changed after the audit, and where they live

Nine defects were found and fixed (`results/FIXES.md`); no conclusion
reversed. Two corrections postdate that file and are recorded here so the
proposal text is not written against stale figures:

| Quantity | Was | Is | Why |
|---|---|---|---|
| Aim 2 causal sufficiency (injection), largest excess | 0.0134 / 0.0833 logits | **0.0101 / 0.0646 logits** | the reported figures came from a superseded 40-pair run; the uncapped 86-pair artifacts were on disk but `make_summary.py` preferred the older directory |
| `patching.py --mode sufficiency` random control | not reproducible | reproducible | the matched-norm random control was drawn from torch's global generator, which nothing seeded, so `--seed` never reached it and `excess` changed between identical runs |

Neither changes a conclusion: the injection effects remain roughly two orders
of magnitude below the 1–5 logits a decision flip needs.
