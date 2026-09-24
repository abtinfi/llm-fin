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

*Header updated 2026-09-02. Items (b) and (c) below were closed on
2026-09-01/02 and the tables in §2 have been corrected; the remaining list in
§4 is current.*

*Updated 2026-09-24: the full MIMIC-IV v3.1 arm (`mimic_v3b`) is complete on
all three models, with the constraint layer, conformal UQ and the matched-control
SAE re-analysis. Final numbers, figures and caveats are in §6 and PAPER.md §5.9.*

The **experimental machinery of all four Aims exists and has been executed
end-to-end**. The one thing still missing that nothing in this repository can
substitute for is **human expert evaluation** — it blocks `S_human`, the
"<30% pass" fallback trigger, the blind expert rating in Aim 2, and the
"human-in-the-loop" half of Aim 4's title.

What closed since this file was written: **causal sufficiency** (feature
injection, `patching.py --mode sufficiency`), **ontology grounding** (`sae.py`
now scores against CUI-anchored concepts; `L_ontology` reads the causal graph),
and a **real-value MIMIC-IV arm** (`data/mimic`, Demo v2.2, open access).

What those closures revealed is more useful than the closures themselves, and
belongs in the write-up:

- **Grounding is uneven and the headline features are the least grounded.**
  `drug` and `inr` are genuinely UMLS-matched; `qt_interval` matches zero UMLS
  atoms on real text and `age` has no CUI at all — and `age` is now the
  top-scoring feature.
- **Half the safety thresholds are not attested by an FDA label**, and two have
  a number in the label that encodes a *different construct*
  (`results/threshold_provenance.md`, SUMMARY.md §6).
- **`ondansetron_qt` is grounded at neither end** — absent from openFDA and
  absent from MED-RT — and it is the family carrying the Aim 3 result.

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
| Mapping features → **biomedical concepts** | **PARTIAL — now CUI-anchored, but see the caveat** | `sae.py` reads `umls_grounding.graph.concept_patterns()` since 2026-09-02 (`--concepts umls`, default). Re-scored end-to-end: best S_semantic 0.703→0.732, FIS 0.351→0.376. **The caveat is the finding:** grounding judged by *empirical* umls-only hits, not term counts — `drug` 376/376 and `inr` 18/18 are genuinely UMLS-matched, but `qt_interval` matches **0** UMLS atoms on real text and `age` has **no CUI at all** — and `age` is the new top feature and 10 of the top 25 |
| Stability across **layers, prompts, and model seeds** | **NOT IMPLEMENTED** | one layer (20), one model, one seed. Probes do sweep all 33 layers, but the SAE does not |
| Human expert evaluation of explanation usefulness | **NOT IMPLEMENTED** | requires clinicians |
| The "<30% pass expert validation → trigger fallback" criterion | **NOT EVALUABLE** | it is defined in terms of expert validation, which does not exist here |

**Key result:** best FIS ≈ 0.35 (TopK feature #14294, `qt_interval`,
S_semantic 0.703). Semantic coherence is real; causal relevance is ~0.

### Aim 2 — Mechanism (necessity, sufficiency, ACE) — **PARTIAL**

| Proposal element | Status | Where |
|---|---|---|
| Causal **necessity** (feature knock-out) | **DONE** | `sae.py::causal_knockout` — each top feature zeroed in the forward pass |
| **Negative controls** (random features) | **DONE** — matched, 5 per feature | every scored feature is compared with the mean of **5 live features matched on firing rate** (S1 fix; the first runs used one feature drawn uniformly, dead ones included). Σ S_causal over the top 25 falls **0.371 → 0.178 (−52.0%)** on test and **0.205 → 0.093 (−54.5%)** on held-out; `#10721` (drug) keeps the largest effect, 0.0573 → 0.0418. `results/sae/sae_topk_L20_fis.json`, `results/sae_heldout/sae_topk_L20_fis.json`, `results/mimic_v3b/figures/fig_sae_contraction.{pdf,png}` |
| **ACE** `E[Y\|do(F+Δf)] − E[Y\|do(F)]` | **DONE**, by layer | `src/patching.py` — activation patching at only the edited token positions, plus an identity control (exactly 0.0) and a random-position control |
| Causal **sufficiency** (clamping features on counterfactual inputs) | **DONE** (2026-09-02) | `src/patching.py --mode sufficiency`, dose sweep + alpha=0 control. Largest excess **0.0101 (test, 86 pairs) / 0.0646 (held-out, 30 pairs)** logits — replicates the knock-out null. `results/sufficiency_medcalc_*.json`. (The 0.0134/0.0833 figures previously quoted here came from the superseded 40-pair run in `results/rerun_fixes/`; `make_summary.py` preferred that directory until 2026-09-06.) |
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
| **MIMIC-IV** for retrospective evaluation / scenario construction | **DONE — full v3.1, evaluated on all three models (2026-09-24, §6)** | **Evaluated on the corrected `data/mimic_v3b`**, which supersedes `data/mimic_v3` below after the 2026-09-23 audit; results in §6 and PAPER.md §5.9. **v3.1 (credentialed):** `src/fetch_mimic_bq.py` pulls the three lab itemids from BigQuery; `src/build_mimic.py --source v3.1` builds `data/mimic_v3` — 184,677 pairs (94,797 causal, 89,880 control) from 64,601 patients, splits globally patient-disjoint, every value measured. `src/build_mimic_note.py` builds `data/mimic_v3_note`: the same pairs inside a real Brief Hospital Course excerpt from MIMIC-IV-Note v2.2 (note within 30 days), identical in both arms, so the pair still differs in one number — this closes the "no free-text notes" limit below. Both are git-ignored under the DUA; evaluated by `run_mimic_v3*.sh` under the supervisor for all three models. The *timepoints* caveat stands, and a cohort-style *retrospective evaluation* is still a different study. **Demo arm, unchanged:** `src/build_mimic.py` loads MIMIC-IV Clinical Database Demo v2.2 (ODbL, open access, **no credentialing**) and builds `data/mimic` — 46 items / 23 pairs. **The only arm where both sides of every pair are real measured values**: 23 of the 100 demo patients have two real serum creatinines (itemid 50912) straddling eGFR 30, so no number is invented. Uses the one FDA-attested threshold in the project. Two limits: the demo carries **no free-text notes**, so the note is rendered from structured fields; and the arms are different *timepoints*, so the clinical state genuinely differed. The full credentialed MIMIC-IV, and any *retrospective* claim, remain out of reach |
| Constraints seeded from **UMLS / MED-RT** (the proposal says "SNOMED CT / UMLS"; SNOMED CT is used nowhere — see `PROPOSAL_ERRATA.md`) | **PARTIAL** | The *qualitative* edge comes from MED-RT via `data/umls/causal_graph.json` (5/10 families attested). The *thresholds* are audited against FDA labels by `src/curate_thresholds.py`: **5 of 10 attested, 5 not**, including two `construct_mismatch` cases where a number is present and means something else. `ondansetron_qt` — the family carrying the Aim 3 result — is absent from BOTH openFDA and MED-RT. See `results/threshold_provenance.md` and SUMMARY.md §6 |

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
| The same on **real** clinical text | **DONE — beyond the proposal** | `src/build_medcalc.py` — 680 items from real PMC case-report notes |
| Causal Consistency = correct counterfactuals / total | **DONE**, at **pair** level | `src/metrics.py` |
| Predictive entropy, Eq. (2) | **DONE — and measured to be unusable** | AUROC **0.525** [0.511, 0.539] over 6,456 items (`results/bigbench_uq.md`) |
| **Adaptive Conformal Inference** for the abstention threshold | **DONE** | `src/coverage_report.py`, `metrics.adaptive_conformal` |
| **Conditional coverage** across subgroups | **DONE** | `results/uq_coverage_*.md` |
| Uncertainty-aware deferral | **DONE** | pooled error 0.558 → 0.282 at 20% coverage, monotone |
| Conformal threshold with a **finite-sample guarantee** | **DONE** (S3 fix) | `metrics.calibrate_threshold(rule="conformal")`: τ is the largest threshold whose one-sided Clopper-Pearson upper bound (δ = 0.10) on calibration selective error is ≤ α = 0.10, evaluated point-wise per candidate τ; −∞ (defer all) when nothing is certifiable. All 18 v3b UQ rows carry `calib_rule = conformal`; the calibration CP bound at each deployed τ is tabulated under `fig_risk_coverage` |
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
`uts-ws.nlm.nih.gov/rest/search/current` (HTTP 200, release 2026AA).

*This section described the key as unread by the codebase. That was true when
it was written on 2026-09-01 and stopped being true the next day: `src/
umls_grounding.py` reads it, `L_ontology` consults the graph it builds, and
`sae.py` scores against CUI-anchored concepts (see the closed item in §4 and
the Aim 3 row above). The four items below are kept as the record of what the
key unblocked.*

It was the direct unblocker for four items the proposal names explicitly:

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
- [x] ~~**Causal sufficiency — feature injection/clamping** (Aim 2, §4.5).~~
      Built and run 2026-09-02: `patching.py --mode sufficiency`, dose sweep
      with an alpha=0 control. Largest excess 0.0134 / 0.0833 logits — it
      replicates the knock-out null rather than overturning it.
- [x] ~~**Token-to-Concept Attribution Layer** (§4.2), and its faithfulness
      evaluation via **sufficiency and comprehensiveness**.~~ Built
      2026-09-06/08: `src/attribution.py`, `results/attribution.md`. Ranks
      the causally edited token at the 0.98 percentile (renal) and 0.85
      (QT) against ~0.50 for random, and names the decisive concept at
      0.90 vs 0.25 chance (renal) and 1.00 vs 0.67 (QT). ERASER
      faithfulness is weak for EVERY attributor including exact occlusion,
      which is the Aim 2 null at token level. Superseded text follows:
      No code existed
      (`grep -rl "comprehensiveness\|attribution" src/` → nothing). The full
      conceptual bridge in §4.2 (Token → Activation → SAE feature → Probe →
      Concept → Causal-graph node → Explanation) is implemented up to "Probe"
      and stops there.
- [x] ~~**UMLS/MED-RT grounding** — the four items in §3 above.~~ Closed
      2026-09-01/02: the graph exists, `L_ontology` reads it, and `sae.py`
      scores against CUI-anchored concepts. **Still open inside it:** the
      grounding is uneven — see §1. Do not describe `qt_interval` or `age`
      features as UMLS-matched.
- [ ] **Thresholds for the 5 unattested families.** `metformin_renal` and
      `warfarin_inr` are `attested_exact`; `nsaid_renal`,
      `nitrofurantoin_renal` and `ondansetron_qt` are `absent`;
      `aspirin_reye` and `spironolactone_hyperkalaemia` are
      `construct_mismatch` — a number IS in the label and means something
      else. Needs a curator, not a better parser.
- [ ] **A causal graph object.** §4.2's SCM and Figure 1's "Causal Knowledge
      Graph" node do not exist as a data structure anywhere.

**Blocks generalisation of results already obtained**

- [x] ~~**Feature stability across layers, prompts and model seeds**
      (§4.4).~~ Done 2026-09-08. Three seeds at layer 20 and three
      layers (16/20/24), compared by subspace and by concept selection
      rather than by feature index, which is meaningless across runs
      (`src/stability.py`, `results/stability.md`). **Concept sets are
      fairly stable** (Jaccard 0.75-1.00 across seeds); **directions
      are not** (mean max cosine ~0.55). And the layer sweep says the
      chosen layer 20 is the WORST of the three -- FVU 0.051 against
      0.026 at layer 16, and four expressible concepts against seven.
      Three models now, not one. Prompts are still unswept.
- [x] ~~**MIMIC-IV** (§4.6).~~ Built from the full credentialed v3.1
      (2026-09-23), plus a real-note arm from MIMIC-IV-Note v2.2; the Demo arm
      is kept beside it. Evaluated on all three models (2026-09-24): §6 below
      and PAPER.md §5.9. Superseded text follows:
      Start with the open-access Demo v2.2 — no
      credentialing, and structured `labevents` sidesteps the
      which-value-is-current problem below.
      **This blocks a §4.6 claim, not the pipeline.** Every Aim 1–4 stage runs
      today with no PhysioNet access; `src/check_data.py --strict` is wired in
      as stage 0 of `run_full_pipeline.sh` and fails the run if that ever
      stops being true. Note what a substitute can and cannot do: a synthetic
      note corpus cannot support a *retrospective evaluation* claim, because
      evaluating on generated notes measures the generator. Only Demo v2.2's
      real structured labs can, and only for the structured-input half.
- [ ] **More than two real rule families.** `data/medcalc` has metformin/eGFR
      and ondansetron/QTc; MedCalc-Bench has 55 calculators.
- [x] ~~**A second model.** §4.3 names Llama-3-Med and Mistral variants.~~
      Done 2026-09-08: `Llama3-OpenBioLLM-8B` and
      `Mistral-7B-Instruct-v0.2`, full pipeline each. Aim 3 transfers to
      all three (held-out CC 0.767 / 0.467 / 0.267); the Aim 2 null
      replicates. A first attempt was INVALID -- lane A never passed
      `--model_id` to `constraint_layer.py`, so every adapter was trained
      on the BioMistral default and attached to another model; all three
      are 4096-dim so it loaded cleanly. Artifacts discarded and re-run. The Aim 2 null in particular is
      currently a statement about one model.

**Methodological gaps that weaken numbers already reported**

- [x] ~~**Control (non-flipping) pairs in the counterfactual benchmarks.**~~
      Done 2026-09-07/08 in `data/medcalc_v2` (edited controls) and
      `data/mimic_v2` (controls made of two more REAL measurements, no
      invented number anywhere). `metrics.score()` reports
      `spurious_flip_rate` and `discrimination`, always with a bootstrap
      CI. Discrimination is indistinguishable from zero on every benchmark
      tried. Superseded text follows: On
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

---

## 6. MIMIC-IV v3.1 arm (`mimic_v3b`), final — 2026-09-24

All supervisor stages finished with `ALL_DONE` (`v3b_main_*`, `v3b_note_*`,
`v3b_cl_*` for BioMistral-7B, OpenBioLLM-8B and Mistral-7B-Instruct, plus
`v3b_report`). The automated audit recomputes all 336 summary metrics from the
prediction logs and matches every one (`results/mimic_v3b/AUDIT_CHECKS.md`);
splits are patient-disjoint (`AUDIT_SANITY.md` §3). The manuscript section
is generated from the same artifacts: PAPER.md §5.9.

**Scale.** 166,264 test items (17,550 distinct prompts, 27,672 patients) on
`metformin_egfr30`, `metformin_egfr45` and `spironolactone_k5_5`; 36,542
held-out items on `warfarin_inr4`, which nothing was trained or calibrated on.
Greedy decoding, one seed; CIs resample distinct prompts.

### Final metrics

From `results/mimic_v3b/CONSOLIDATED_MIMIC3B.md` (CIs for held-out and the
*unseen pairs* rows are there). † = the gate fired: strict accuracy and CC are
partly an identity with the labelling rule. **Model's own** = the answer parsed
before the gate overrides it or UQ defers it. **Coverage** = share answered;
**Violation** = share of UNSAFE items answered SAFE.

| Model | Row | Test strict acc [95% CI] | Model's own | CC [95% CI] | Coverage | Violation | Held-out strict acc | Held-out CC | Held-out violation |
|---|---|---|---|---|---|---|---|---|---|
| BioMistral-7B | (1) Base | 0.688 [0.670, 0.706] | 0.688 | 0.110 [0.103, 0.117] | 1.000 | 0.871 | 0.333 | 0.000 | 0.000 |
| BioMistral-7B | (2) + RAG | 0.342 [0.324, 0.360] | 0.342 | 0.000 [0.000, 0.000] | 1.000 | 0.000 | 0.333 | 0.000 | 0.000 |
| BioMistral-7B | (3) NS-AI | 0.606 † [0.587, 0.625] | 0.342 | 0.395 † [0.372, 0.417] | 1.000 | 0.000 | 1.000 † | 1.000 † | 0.000 |
| BioMistral-7B | (4) NS-AI + UQ | 0.414 † [0.399, 0.431] | 0.342 | 0.394 † [0.372, 0.417] | 0.415 | 0.000 | 1.000 † | 1.000 † | 0.000 |
| BioMistral-7B | `sym` Base + gate | 0.787 † [0.769, 0.805] | 0.688 | 0.394 † [0.372, 0.417] | 1.000 | 0.662 | 1.000 † | 1.000 † | 0.000 |
| BioMistral-7B | `uq` Base + UQ | 0.000 [0.000, 0.000] | 0.688 | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| BioMistral-7B | `cl` Base + constraint layer | 0.951 [0.940, 0.961] | 0.951 | 0.855 [0.831, 0.876] | 1.000 | 0.093 | 0.464 | 0.088 | 0.078 |
| BioMistral-7B | `nsai_uq_cl` NS-AI + UQ + CL | 0.657 † [0.637, 0.677] | 0.834 | 0.399 † [0.376, 0.422] | 0.684 | 0.082 | 1.000 † | 1.000 † | 0.000 |
| OpenBioLLM-8B | (1) Base | 0.679 [0.660, 0.696] | 0.679 | 0.000 [0.000, 0.000] | 1.000 | 1.000 | 0.667 | 0.000 | 1.000 |
| OpenBioLLM-8B | (2) + RAG | 0.221 [0.211, 0.233] | 0.221 | 0.000 [0.000, 0.000] | 1.000 | 0.058 | 0.667 | 0.000 | 1.000 |
| OpenBioLLM-8B | (3) NS-AI | 0.431 † [0.415, 0.449] | 0.221 | 0.394 † [0.372, 0.417] | 1.000 | 0.051 | 1.000 † | 1.000 † | 0.000 |
| OpenBioLLM-8B | (4) NS-AI + UQ | 0.396 † [0.381, 0.413] | 0.221 | 0.394 † [0.372, 0.417] | 0.396 | 0.000 | 1.000 † | 1.000 † | 0.000 |
| OpenBioLLM-8B | `sym` Base + gate | 0.787 † [0.769, 0.805] | 0.679 | 0.394 † [0.372, 0.417] | 1.000 | 0.662 | 1.000 † | 1.000 † | 0.000 |
| OpenBioLLM-8B | `uq` Base + UQ | 0.076 [0.065, 0.090] | 0.679 | 0.000 [0.000, 0.000] | 0.081 | 0.016 | 0.000 | 0.000 | 0.000 |
| OpenBioLLM-8B | `cl` Base + constraint layer | 0.900 [0.886, 0.913] | 0.900 | 0.769 [0.744, 0.793] | 1.000 | 0.060 | 0.333 | 0.000 | 0.000 |
| OpenBioLLM-8B | `nsai_uq_cl` NS-AI + UQ + CL | 0.441 † [0.425, 0.458] | 0.397 | 0.394 † [0.372, 0.417] | 0.447 | 0.000 | 1.000 † | 1.000 † | 0.000 |
| Mistral-7B-Instruct | (1) Base | 0.647 [0.628, 0.664] | 0.647 | 0.019 [0.017, 0.021] | 1.000 | 0.959 | 0.704 | 0.032 | 0.889 |
| Mistral-7B-Instruct | (2) + RAG | 0.741 [0.721, 0.760] | 0.741 | 0.383 [0.361, 0.406] | 1.000 | 0.233 | 0.844 | 0.550 | 0.274 |
| Mistral-7B-Instruct | (3) NS-AI | 0.795 † [0.775, 0.813] | 0.741 | 0.522 † [0.493, 0.552] | 1.000 | 0.228 | 1.000 † | 1.000 † | 0.000 |
| Mistral-7B-Instruct | (4) NS-AI + UQ | 0.555 † [0.536, 0.576] | 0.741 | 0.399 † [0.376, 0.422] | 0.571 | 0.000 | 1.000 † | 1.000 † | 0.000 |
| Mistral-7B-Instruct | `sym` Base + gate | 0.794 † [0.776, 0.812] | 0.647 | 0.396 † [0.373, 0.419] | 1.000 | 0.640 | 1.000 † | 1.000 † | 0.000 |
| Mistral-7B-Instruct | `uq` Base + UQ | 0.089 [0.083, 0.097] | 0.647 | 0.000 [0.000, 0.000] | 0.099 | 0.029 | 0.003 | 0.000 | 0.000 |
| Mistral-7B-Instruct | `cl` Base + constraint layer | 0.906 [0.891, 0.920] | 0.906 | 0.769 [0.740, 0.796] | 1.000 | 0.090 | 0.412 | 0.007 | 0.000 |
| Mistral-7B-Instruct | `nsai_uq_cl` NS-AI + UQ + CL | 0.466 † [0.449, 0.484] | 0.698 | 0.395 † [0.373, 0.418] | 0.474 | 0.000 | 1.000 † | 1.000 † | 0.000 |

**Aim 3, against its shuffled-label control** (the adapter's own evaluation):

| model | frozen base, test CC | rule labels | shuffled labels | frozen base, held-out CC | rule labels | shuffled labels |
|---|---|---|---|---|---|---|
| BioMistral-7B | 0.098 | **0.848** | 0.250 | 0.000 | 0.086 | 0.000 |
| OpenBioLLM-8B | 0.000 | **0.772** | 0.084 | 0.000 | 0.000 | 0.000 |
| Mistral-7B-Instruct | 0.017 | **0.772** | 0.185 | 0.026 | 0.010 | 0.384 |

**Aim 4, deployed thresholds** (test split; calibration evidence per τ):

| model | row | calibration n | calibration risk at τ | CP upper bound | test coverage (UQ-governed) | test risk at τ |
|---|---|---|---|---|---|---|
| BioMistral-7B | Base + UQ | 66,938 | — | none ≤ α (τ = −∞) | 0.0% | — |
| BioMistral-7B | NS-AI + UQ | 41,140 | 0.024 | 0.030 | 3.5% | 0.019 |
| BioMistral-7B | NS-AI + UQ + CL | 41,140 | 0.092 | 0.094 | 48.1% | 0.090 |
| OpenBioLLM-8B | Base + UQ | 66,938 | 0.062 | 0.066 | 8.1% | 0.062 |
| OpenBioLLM-8B | NS-AI + UQ | 3,827 | 0.000 | 0.006 | 0.8% | 0.000 |
| OpenBioLLM-8B | NS-AI + UQ + CL | 41,140 | 0.086 | 0.092 | 8.6% | 0.101 |
| Mistral-7B-Instruct | Base + UQ | 66,938 | 0.094 | 0.099 | 9.7% | 0.091 |
| Mistral-7B-Instruct | NS-AI + UQ | 41,140 | 0.086 | 0.090 | 29.8% | 0.088 |
| Mistral-7B-Instruct | NS-AI + UQ + CL | 41,140 | 0.095 | 0.100 | 13.8% | 0.087 |

### Figures (`results/mimic_v3b/figures/`, vector PDF + 300-dpi PNG)

| figure | shows |
|---|---|
| `fig_risk_coverage` | selective risk vs coverage of the UQ-governed items, deployed τ marked; the table under the panels states the Clopper-Pearson selection rule and gives each τ's calibration n, risk and CP upper bound. CP is evaluated point-wise during selection and deliberately not drawn as a band on the test curves (prompts repeat ~9.5×, so an item-level test bound would be far too tight) |
| `fig_sae_contraction` | top-25 L20 TopK knock-out effects, old uniform control vs 5 matched controls, test and held-out panels: −52.0% / −54.5%, `#10721` 0.0573 → 0.0418 |
| `fig_option_order` | base accuracy, CC and UNSAFE share with the options in original vs swapped order: accuracy falls to 0.50–0.52, i.e. the base models answer partly by position |

Regenerate with `/usr/bin/python3 src/generate_paper_plots.py`.

### Caveats that must travel with these numbers

1. **OpenBioLLM-8B non-answers under retrieval.** With retrieved context it
   returns no parsable SAFE/UNSAFE on 76.0% of test items (0.0% without), and
   55.2% of its NS-AI rows stay unparsable after the gate. Its RAG (0.221) and
   NS-AI (0.431) accuracies measure an answer-format failure, not clinical
   judgement. Held-out is unaffected.
2. **No transfer to the held-out warfarin family.** The constraint layer lifts
   test CC to 0.769–0.855 on every model but held-out CC stays 0.000–0.088,
   and for Mistral-7B-Instruct the shuffled-label adapter scores *higher* on
   held-out (0.384 vs 0.010). Every gated held-out row is 1.000† by identity.
   So the QT held-out result of §2 (Aim 3) transfers for that family pair
   only; it is not a general property of the method.
3. **Prompt-rendering duplicates.** The rendered note carries only age, sex,
   one lab value and the drug, so 166,264 test items are 17,550 distinct
   prompts, and 11,101 of those also appear verbatim in calibration (patients
   stay disjoint). CIs therefore resample prompts, and the calibration-to-test
   agreement of the UQ rows is optimistic for genuinely novel presentations.
   One row exceeds α on test: OpenBioLLM NS-AI + UQ + CL, 0.101 (calibration CP
   bound 0.092).
4. `spironolactone_k5_5`'s threshold is `construct_mismatch` in
   `results/threshold_provenance.md`; do not quote it as label-attested.
5. The SAE analyses are BioMistral-7B on the real-notes benchmark (180 test /
   60 held-out items), not on this arm.

### Housekeeping done with this update

- The `mimic-v3` worktree was merged (`135787b`) and removed; its local
  branch was deleted. Its gitignored per-item artifacts (prediction logs,
  adapters, the `data/mimic_v3*` builds, `data/mimic_v3b_swap`, `logs/mimic_v3*`)
  were moved into this checkout at the same paths and verified byte-identical
  (265 files) before removal. They stay gitignored under the DUA.
- The supervisor queue's v3b entries now point at the repo-root
  `run_v3b*.sh` (identical scripts; they resolve paths from their own
  location), so a re-run no longer depends on the removed worktree.
- The 2026-09-24 Teleport auto-stash was dropped after verifying every code
  change in it is in HEAD; a full patch backup is at
  `.git/stash-backups/fd378c9-teleport-auto-stash-2026-09-24.patch`.
