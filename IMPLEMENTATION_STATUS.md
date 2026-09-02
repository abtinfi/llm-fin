# Implementation status against `AbstractOfProposal (1).pdf`

Written 2026-09-01. Supersedes the Aim-status table in `BACKLOG.md` §2, which
was written 2026-08-20 and is now stale: it lists Aims 1 and 2 as "NOT
STARTED", but both were implemented and run on 2026-08-20/22
(`results/aim123_internals.md`, `results/aim1_sae.md`,
`results/aim3_constraint_layer.md`).

Everything below is judged against the proposal text, not against the
supervisor's four requirements (those were all met and are tracked separately
in `BACKLOG.md` §1).

Legend: **DONE** = implemented and run; **PARTIAL** = implemented, but a
component the proposal names is missing; **NOT IMPLEMENTED** = no code exists.

---

## 1. One-line answer

The **experimental machinery of all four Aims exists and has been executed
end-to-end**. What is missing is not the pipeline but three specific things the
proposal names: (a) **human expert evaluation**, which nothing in this repo can
substitute for; (b) **ontology grounding via UMLS/SNOMED CT**, which the API
key just added to `.env` now unblocks; and (c) **causal sufficiency (feature
injection)**, the other half of Aim 2.

Scientifically the pipeline has mostly produced **negative results**, and they
are internally consistent: the model's decisions are not driven by the
representations the SAE finds, so the symbolic route does the work.

---

## 2. Aim by aim

### Aim 1 — Discovery (SAEs, FIS, concept probes) — **PARTIAL**

| Proposal element | Status | Where |
|---|---|---|
| SAE on middle-layer activations, Eq. (1) | **DONE** | `src/sae.py` (`collect`/`train`/`score`), layer 20, 16,384 features (4× expansion) |
| Fallback to JumpReLU **or** TopK if standard SAE fails | **DONE** — both trained, not just as a fallback | `results/sae/sae_{topk,jumprelu}_L20.npz`; TopK FVU 0.067 / L0 31.9 / 39.3% dead, JumpReLU FVU 0.131 / L0 27.8 / 64.0% dead |
| `FIS = α·S_semantic + β·S_causal + γ·S_human` | **PARTIAL — 2 of 3 terms** | `S_semantic` = max F1 of feature-fires vs concept-token, `S_causal` = knock-out effect. **`S_human` is forced to weight 0** — there are no expert annotators |
| Activation maximisation | **DONE** in effect | top-activating tokens per feature reported in `results/aim1_sae.md` |
| Sparse **concept probing classifiers** (distributed subspaces, not 1:1) | **DONE** | `src/probe.py` — StandardScaler → PCA(128) → logistic, 5-fold grouped by `pair_id`, with a within-pair label-permutation null |
| Mapping features → **biomedical concepts** | **PARTIAL / weak** | concepts are **11 hand-written regexes** in `sae.py` (`creatinine`, `qt_interval`, `egfr`, `heart_rate`, `potassium`, `inr`, `age`, `drug`, `renal_disease`, `pregnancy`, `asthma`) — *not* UMLS CUIs or SNOMED codes |
| Stability across **layers, prompts, and model seeds** | **NOT IMPLEMENTED** | one layer (20), one model, one seed. Probes do sweep all 33 layers, but the SAE does not |
| Human expert evaluation of explanation usefulness | **NOT IMPLEMENTED** | requires clinicians |
| The "<30% pass expert validation → trigger fallback" criterion | **NOT EVALUABLE** | it is defined in terms of expert validation, which does not exist here |

**Key result:** best FIS ≈ 0.35 (TopK feature #14294, `qt_interval`,
S_semantic 0.703). Semantic coherence is real; causal relevance is ~0.

### Aim 2 — Mechanism (necessity, sufficiency, ACE) — **PARTIAL**

| Proposal element | Status | Where |
|---|---|---|
| Causal **necessity** (feature knock-out) | **DONE** | `sae.py::causal_knockout` — each top feature zeroed in the forward pass |
| **Negative controls** (random features) | **DONE** | a firing-rate-matched random feature is knocked out for *every* scored feature |
| **ACE** `E[Y\|do(F+Δf)] − E[Y\|do(F)]` | **DONE**, by layer | `src/patching.py` — activation patching at only the edited token positions, plus an identity control (exactly 0.0) and a random-position control |
| Causal **sufficiency** (clamping features on counterfactual inputs) | **NOT IMPLEMENTED** | only knock-out exists; nothing clamps a feature *on* |
| "Interventions modify behaviour without degrading perplexity" | **DONE** | `src/perplexity.py`, WikiText-2: base **7.1622** → QT adapter **7.0775**, synthetic adapter **7.1266** |
| **Human evaluation** — experts blindly rate explanation correctness/usefulness | **NOT IMPLEMENTED** | requires clinicians |

**Key result (a clean null, well-controlled):** patching effects are
+0.0017 logits (renal) and +0.0008 logits (QT) against a 0.0 identity control,
where flipping a decision needs ~1–5 logits — **100–500× too small**. SAE
knock-out replicates it independently at 0.002–0.016 logits over matched
controls. H2 as stated is **not supported on this model**.

Probes show why this is not a broken hook: the QT decision *is* linearly
readable from the residual stream (pair-CC **0.976** vs null 0.200), and
creatinine is *not* readable at all (**0.042** vs null 0.033) even though it is
trivially decodable from the raw facts (creatinine+age+sex → 1.000). The model
represents one decisive quantity, not the other, and acts on neither.

### Aim 3 — Control (Constraint-Aware Layer) — **DONE** (with a scope caveat)

| Proposal element | Status | Where |
|---|---|---|
| Residual-stream adapter `h'_l = h_l + α·P_causal(h_l)` | **DONE** | `src/constraint_layer.py` — low-rank (rank 32) trainable map inserted at layer 30 |
| Joint objective `L = L_LM + λ1·L_ontology + λ2·L_causal + λ3·L_uncertainty` | **DONE — all four terms** | `L_causal` = CE on the decision token; `L_LM` = KL to the frozen model; `L_ontology` = **pairwise** hinge on the SAFE/UNSAFE margin order; `L_uncertainty` = entropy floor on notes whose decisive number is redacted. Weights used: `lam_kl=0.01, lam_ont=0.1, lam_unc=0.1` |
| Shuffled-label control | **DONE** | `--shuffled_control`; QT 0.767 real vs **0.200** shuffled |
| Fixed-direction steering prototype (the simplest `P_causal`) | **DONE — null result, kept** | `src/steering.py` |
| Ablation Matrix of §4.6 | **DONE, and extended** | 8 rows, not 4: rows 5–7 isolate each contribution against the baseline (`src/run_eval.py`, `src/make_table.py`) |
| RQ3 — no degradation of language ability | **DONE** | perplexity 7.1622 → 7.0775 / 7.1266 (no degradation; both adapters sit slightly *below* base) |
| **MIMIC-IV** for retrospective evaluation / scenario construction | **NOT IMPLEMENTED** | blocked on PhysioNet credentialing; the open-access **Demo v2.2** (100 patients, `labevents`/`prescriptions`, no notes) is the available substitute |
| Constraints seeded from **SNOMED CT / UMLS** | **NOT IMPLEMENTED** | `src/rules.py` holds 10 hand-written rule families; thresholds hand-transcribed |

**Key result:** the layer works exactly where the probe predicted it would and
fails where the probe predicted it would fail — QT held-out CC **0.000 → 0.767**,
renal **0.000 → 0.000**. That is the strongest coherence in the project: an
Aim-1/2 measurement made a falsifiable prediction about Aim 3 and it held.

**Caveat to carry into the write-up:** the *symbolic gate* is a post-hoc text
check and is **not** the proposal's intervention layer. They are separate rows
in the table (rows 3 and 7) and must not be conflated.

### Aim 4 — Validation (counterfactual benchmark + conformal UQ) — **DONE**

| Proposal element | Status | Where |
|---|---|---|
| Synthetic counterfactual clinical benchmark | **DONE** | `src/build_dataset.py` — 128 test / 32 held-out / 32 calib, 10 rule families (2 held out), with build-time integrity checks that abort on failure |
| The same on **real** clinical text | **DONE — beyond the proposal** | `src/build_medcalc.py` — 600 items from real PMC case-report notes |
| Causal Consistency = correct counterfactuals / total | **DONE**, at **pair** level | `src/metrics.py` |
| Predictive entropy, Eq. (2) | **DONE — and measured to be unusable** | AUROC **0.525** [0.511, 0.539] over 6,456 items (`results/bigbench_uq.md`) |
| **Adaptive Conformal Inference** for the abstention threshold | **DONE** | `src/coverage_report.py`, `metrics.adaptive_conformal` |
| **Conditional coverage** across subgroups | **DONE** | `results/uq_coverage_*.md` |
| Uncertainty-aware deferral | **DONE** | pooled error 0.558 → 0.282 at 20% coverage, monotone |
| Statistical rigour | **DONE — beyond the proposal** | paired McNemar + Holm correction, item-level bootstrap CIs |
| **Human-in-the-loop** validation (Aim 4's title) | **NOT IMPLEMENTED** | requires clinicians |

**Key result, and a change the proposal text needs:** Eq. (2) as written
(whole-vocabulary token entropy) is at chance. Restricting the *same* entropy
to the decision tokens gives **0.687** [0.675, 0.700]; the paired difference is
**+0.163**, p < 0.0001 Holm-corrected, replicated on MedMCQA, MedQA and
PubMedQA independently. **§4.7 needs its uncertainty term respecified** — this
is a proposal edit, not a bug fix.

---

## 3. What the UMLS key unblocks

The key in `.env` is **live** — verified 2026-09-01 against
`uts-ws.nlm.nih.gov/rest/search/current` (HTTP 200, release 2026AA). Nothing in
the codebase reads it yet: `grep -rn "UMLS\|umls" src/` returns nothing.

It is the direct unblocker for four items the proposal names explicitly:

1. **§4.2 — "the SCM is initialized from biomedical ontologies (e.g. UMLS)."**
   Currently `src/rules.py` is 10 hand-written families; there is no causal
   graph object at all.
2. **§4.4 — mapping SAE features to biomedical concepts.** The 11 regexes in
   `sae.py` become CUI-anchored concept sets (synonyms and lexical variants
   come free from UMLS), which turns `S_semantic` from "matches my regex" into
   "selects a UMLS concept" — a materially stronger claim.
3. **`L_ontology`.** The term is implemented as a pairwise margin hinge; it is
   the only term whose name currently outruns its content, because no ontology
   is consulted. UMLS relations (`may_treat`, `contraindicated_with`,
   `has_finding_site`) would make the name accurate.
4. **P4 in `BACKLOG.md` — hand-written thresholds.** UMLS will not supply
   numeric thresholds (those stay FDA-label/guideline-sourced), but it does
   supply the concept identity and drug–condition relations that the current
   string matching approximates.

---

## 4. Consolidated "still to implement" list

Ordered by how much of the proposal's claim each one blocks.

**Blocks a claim the proposal makes and cannot be worked around**

- [ ] **Human expert evaluation.** Blocks `S_human` in the FIS (Aim 1), the
      "<30% pass" fallback trigger, the blind expert rating of explanation
      correctness (Aim 2), and the "human-in-the-loop" half of Aim 4's title.
      Nothing computational substitutes for this.
- [ ] **Causal sufficiency — feature injection/clamping** (Aim 2, §4.5). Only
      knock-out exists. Half of the proposal's two explicit causal tests.
- [ ] **Token-to-Concept Attribution Layer** (§4.2), and its faithfulness
      evaluation via **sufficiency and comprehensiveness**. No code exists
      (`grep -rl "comprehensiveness\|attribution" src/` → nothing). The full
      conceptual bridge in §4.2 (Token → Activation → SAE feature → Probe →
      Concept → Causal-graph node → Explanation) is implemented up to "Probe"
      and stops there.
- [ ] **UMLS/SNOMED grounding** — the four items in §3 above. Newly unblocked.
- [ ] **A causal graph object.** §4.2's SCM and Figure 1's "Causal Knowledge
      Graph" node do not exist as a data structure anywhere.

**Blocks generalisation of results already obtained**

- [ ] **Feature stability across layers, prompts and model seeds** (§4.4).
      One layer, one model, one seed today.
- [ ] **MIMIC-IV** (§4.6). Start with the open-access Demo v2.2 — no
      credentialing, and structured `labevents` sidesteps the
      which-value-is-current problem below.
- [ ] **More than two real rule families.** `data/medcalc` has metformin/eGFR
      and ondansetron/QTc; MedCalc-Bench has 55 calculators.
- [ ] **A second model.** §4.3 names Llama-3-Med and Mistral variants; every
      number in this repo is BioMistral-7B. The Aim 2 null in particular is
      currently a statement about one model.

**Methodological gaps that weaken numbers already reported**

- [ ] **Control (non-flipping) pairs in the counterfactual benchmarks.** On
      8,000 MCQ items the model's discrimination is **−0.013**
      [−0.037, +0.013] — it flips at the same rate whether or not the truth
      changed. Until `data/` and `data/medcalc` carry wrong-vs-wrong control
      arms, every Causal Consistency number conflates causal sensitivity with
      prompt sensitivity, including the **0.986** gate result.
- [ ] **A spurious-flip rate reported beside every consistency number.**
- [ ] **Seed variance (P2).** Greedy decoding makes all seeds byte-identical,
      so `± 0.000` is not a measurement. Route the seed through *dataset
      construction* instead.
- [ ] **The gate's 0.986 is partly circular** — the dataset was filtered with
      the same extractor the gate uses. Never publish it without the
      uncircular number beside it: **coverage 42.7%** of real renal notes.
- [ ] **The which-value-is-current problem.** 42.8% of real renal notes state
      creatinine more than once with different values. This is a hard ceiling
      on any threshold-reading layer and is the strongest argument for
      structured input (MIMIC-IV `labevents`).

**Proposal text changes, not code**

- [ ] **Respecify §4.7's uncertainty term** (Eq. 2 → decision-restricted
      entropy). Measured, not guessed.
- [ ] **Decide the row-4 operating point.** `logit_margin` answers more (0.656
      vs 0.578 coverage) and is right more often (0.617 vs 0.562) but lets one
      unsafe case through (violation 0.000 → 0.016). A clinical judgement.

**Housekeeping, still open**

- [ ] Rotate the openFDA API key and the HuggingFace token (`BACKLOG.md` §5 —
      both were pasted into a public chat and there is no evidence of rotation).
- [ ] `BACKLOG.md` §2's Aim table is stale; this file is the current one.

---

## 5. Full pipeline re-run, 2026-09-01

`bash run_full_pipeline.sh` executed end-to-end on GPU 0, ~9.3 h wall clock.
Log: `logs/full_run_20260901.log`. **All 9 stages completed; zero tracebacks.**
Stage 8 (SAE scoring) had OOM-killed on the 2026-08-22 run; the streaming fix
already in the tree held, and both FIS files were written.

Previous results were snapshotted to `results_snapshot_20260822/` before the
run, so the reproduction is checkable.

**Reproduction: everything is byte-identical except one run.** Of 104 snapshotted
artifacts, only `constraint_renal.json` and the two files derived from it
(`summary_test_medcalc.json`, `table_medcalc_test.md`) changed. All 8 ablation
rows on all 4 benchmarks, all conformal/coverage reports, and both SAE FIS
tables reproduced exactly — as they must, under greedy decoding.

| Quantity | 2026-08-22 | 2026-09-01 |
|---|---|---|
| Constraint layer, **renal**, test CC | 0.033 | 0.011 |
| Constraint layer, **renal**, test acc | 0.517 | 0.500 |
| Constraint layer, **QT**, held-out CC | 0.767 | 0.767 (identical) |
| WikiText-2 ppl, QT adapter | 7.1496 | 7.0775 |
| WikiText-2 ppl, synthetic adapter | not recorded | 7.1266 |

The renal drift is adapter-training stochasticity, and it moved the run the
probe **predicted would not work** — 0.033 and 0.011 are both indistinguishable
from zero (2 pairs of 180 vs 4 of 180). The QT run, the one that carries the
Aim 3 claim, reproduced to the digit. Perplexity moved for the same reason
(adapters retrained) and stayed at or below base, so RQ3's conclusion is
unchanged.

**Nothing in §2–§4 above changes as a result of this run.**
