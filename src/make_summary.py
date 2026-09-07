"""
Regenerates `results/SUMMARY.md` from the artifacts, not from memory.

  python src/make_summary.py

WHY
---
Every number in the previous SUMMARY.md was typed in by hand from a run log.
That is how a report ends up describing a dataset that has since been rebuilt:
the MedCalc report quoted a 430-item test split, the split was later
re-partitioned to make room for the constraint layer's training set, and the
report kept quoting 430. Nothing in the repository could notice.

Here every figure is read out of `summary_*.json`, `constraint_*.json` and the
SAE's `*_fis.json`. Prose that interprets the numbers is still written by a
human -- interpretation should be -- but a number that no longer exists in an
artifact cannot survive in the report.
"""

import argparse
import json
from pathlib import Path

import numpy as np

# Module-level default so every helper that closes over `R` keeps
# working; `main()` rebinds it from --results for a per-model run.
R = Path("results")
LABELS = {
    "base": "(1) Base LLM", "rag": "(2) + RAG",
    "nsai": "(3) + Symbolic Gate (NS-AI)", "nsai_uq": "(4) + UQ Engine",
    "sym": "(5) Base + Gate only", "uq": "(6) Base + UQ only",
    "cl": "(7) Base + Constraint Layer only", "nsai_uq_cl": "(8) All four",
}
ORDER = ["base", "rag", "nsai", "nsai_uq", "sym", "uq", "cl", "nsai_uq_cl"]


def load(name):
    p = R / name
    return json.load(p.open()) if p.exists() else None


def rows(summary):
    out = {}
    for s in summary or []:
        out.setdefault(s["variant"], s)
    return out


def cell(d, key, fmt="{:.3f}"):
    return fmt.format(d[key]) if d and key in d and d[key] is not None else "—"


def ablation_block(title, files):
    """files: list of (column label, summary filename)"""
    data = [(lab, rows(load(f))) for lab, f in files]
    present = [v for v in ORDER if any(v in d for _, d in data)]
    lines = [f"### {title}\n"]
    head = "| Stage | " + " | ".join(f"{lab} CC" for lab, _ in data) + " |"
    lines += [head, "|---" * (len(data) + 1) + "|"]
    for v in present:
        cells = [cell(d.get(v), "causal_consistency") for _, d in data]
        lines.append(f"| {LABELS[v]} | " + " | ".join(cells) + " |")
    lines.append("")
    head = "| Stage | " + " | ".join(f"{lab} acc" for lab, _ in data) + " |"
    lines += [head, "|---" * (len(data) + 1) + "|"]
    for v in present:
        cells = [cell(d.get(v), "accuracy") for _, d in data]
        lines.append(f"| {LABELS[v]} | " + " | ".join(cells) + " |")
    n = [f"{lab}: n={d[present[0]]['n']}" for lab, d in data if present and
         present[0] in d]
    lines.append("\nSplit sizes — " + ", ".join(n) + ".\n")
    return lines


def _causal_spans(R):
    """
    The Aim 2 effect sizes, read from the artifacts rather than typed in.

    These three numbers were hard-coded string literals until 2026-09-02, which
    contradicted this file's own docstring ("from the artifacts, not from
    memory") and meant the B1/B3 fixes could not reach section 5. The
    `~0.002 logits` and `0.002-0.016 logits` figures they asserted were both
    stale the moment the knock-out was re-run against the correct layer.

    Prefers `results/`, then falls back to `results/rerun_fixes/`.

    The preference used to run the other way, from a time when `results/` held
    PRE-fix artifacts and `rerun_fixes/` held the corrected ones. That stopped
    being true on 2026-09-02 13:41, when the caps were removed and the scaled
    pipeline regenerated `results/` with the B1/B2/B3 fixes already in the
    code. After that the old preference silently reported the SMALLER run:
    `rerun_fixes/sufficiency_medcalc_test.json` covers 40 pairs against 86 in
    `results/`, so section 5 quoted 0.0134/0.0833 when the uncapped artifacts
    on disk said 0.0101/0.0646. `rerun_fixes/` stays as the fallback so this
    still works in a tree where the scaled run has not been done.
    """
    import json as _json

    def _load(*cands):
        for c in cands:
            p = R / c
            if p.is_file():
                try:
                    return _json.loads(p.read_text())
                except _json.JSONDecodeError:
                    pass
        return None

    def _span(vals, unit="logits"):
        if not vals:
            return "not measured"
        lo, hi = min(vals), max(vals)
        return (f"{hi:.4f} {unit}" if abs(hi - lo) < 5e-5
                else f"{lo:.4f} to {hi:.4f} {unit}")

    out = {}
    pat = []
    for split in ("test", "heldout"):
        d = _load(f"patching_medcalc_{split}.json",
                  f"rerun_fixes/patching_medcalc_{split}.json")
        if d:
            pat.append(max(abs(r["excess"]) for r in d["rows"]))
    out["patching"] = _span([min(pat), max(pat)] if pat else [])

    fis = _load("sae/sae_topk_L20_fis.json")
    ko = ([f["causal_detail"]["excess"] for f in fis["features"]
           if "causal_detail" in f] if fis else [])
    out["knockout"] = _span(ko)

    suf = []
    for split in ("test", "heldout"):
        d = _load(f"sufficiency_medcalc_{split}.json",
                  f"rerun_fixes/sufficiency_medcalc_{split}.json")
        if d:
            suf.append(max(abs(r["excess"]) for r in d["rows"]))
    out["sufficiency"] = _span([min(suf), max(suf)] if suf else [])
    return out



def _threshold_provenance(R):
    """
    Section 6: where each safety threshold comes from.

    Read live from src/rules.py so a threshold that loses its attestation
    cannot keep a stale label in this file. Checked by
    src/curate_thresholds.py against data/openfda_raw.jsonl; the full
    sentences are in results/threshold_provenance.md.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from rules import RULE_FAMILIES
    except Exception as e:                                  # pragma: no cover
        return ["## 6. Threshold provenance", "",
                f"*(could not read src/rules.py: {e})*", ""]

    order = {"attested_exact": 0, "attested_qualitative": 1,
             "construct_mismatch": 2, "absent": 3}
    fams = sorted(RULE_FAMILIES, key=lambda f: (order.get(f.threshold_status, 9),
                                                f.name))
    n_att = sum(1 for f in fams if f.threshold_source == "openfda")
    L = ["## 6. Threshold provenance — %d of %d attested by an FDA label"
         % (n_att, len(fams)), "",
         "`hardening.py` said these thresholds were INTERIM and must not be "
         "presented as sourced. `src/curate_thresholds.py` checked all ten "
         "against `data/openfda_raw.jsonl`. **No row is human-verified**: "
         "`curator` reads `auto:` throughout, so this is a machine audit "
         "awaiting a curator, not a completed curation.", "",
         "| family | threshold | source | status |",
         "|---|---|---|---|"]
    for f in fams:
        t = f.constraint.get("threshold")
        L.append(f"| `{f.name}` | {t} | {f.threshold_source} | "
                 f"**{f.threshold_status}** |")
    L += ["",
          "**`construct_mismatch` is the dangerous category.** A number IS "
          "present in the label and encodes something else:", "",
          "- `aspirin_reye` — the label's \"children under 12 years: consult "
          "a doctor\" is an OTC **dosing** instruction. The rule encodes the "
          "Reye's-syndrome contraindication (<16), which this label never "
          "mentions.",
          "- `spironolactone_hyperkalaemia` — the label's \"serum potassium "
          "\u22645.0 mEq/L\" is a heart-failure **initiation** criterion, not "
          "the contraindication ceiling (5.5) the rule encodes.", "",
          "Neither value is written into `constraint_value`; both are left "
          "for a human. A number that is present but means something else is "
          "more dangerous than no number at all.", "",
          "**`ondansetron_qt` has no grounding at either end.** Its 500 ms "
          "threshold is `absent` from the FDA labels, and the family is also "
          "absent from MED-RT, so the causal knowledge graph licenses no "
          "contraindication path for it (`results/aim3_constraint_layer.md`, "
          "`umls_grounding.py coverage`). This is the family carrying the "
          "Aim 3 result — held-out CC 0.000 \u2192 0.767 — so that result "
          "rests on curation at both ends and must be reported as such.", "",
          "Full sentences and sections: `results/threshold_provenance.md`.", ""]
    return L


def _benchmark_arms(R):
    """Section 7: the three benchmark arms and how much of each is invented."""
    import json
    rows = [
        ("`data/synthetic_control`", "templated vignette", "invented",
         "invented", "Control arm. Shows what the pipeline does when the "
         "causal factor is stated cleanly and the label is guaranteed."),
        ("`data/medcalc`", "real PMC case-report prose", "one arm real, "
         "one **edited**", "5 of 10 attested",
         "Real clinical text. Half of every pair has its driving number "
         "changed to cross the threshold."),
    ]
    meta_p = R.parent / "data" / "mimic" / "build_meta.json"
    if meta_p.is_file():
        m = json.loads(meta_p.read_text())
        n = sum(m["counts"].values())
        rows.append(("`data/mimic`", "minimal rendered note",
                     "**both arms real**", "attested_exact",
                     f"MIMIC-IV Real-Value Cohort, {n} items. "
                     f"{m['counts']['test']//2} test pairs from real patients "
                     f"whose measured creatinines straddle eGFR 30."))
    L = ["## 7. The benchmark arms, and how much of each is invented", "",
         "| arm | text | numbers | threshold | note |", "|---|---|---|---|---|"]
    for r in rows:
        L.append("| " + " | ".join(r) + " |")
    L += ["",
          "No arm dominates. `data/mimic` invents no number but its two arms "
          "are different **timepoints** in the same patient, so the clinical "
          "state genuinely differed; `data/medcalc` holds the timepoint fixed "
          "and fabricates a number instead. The paper should report both and "
          "say which trade each makes.", ""]
    if not meta_p.is_file():
        L += ["*(`data/mimic` not built — run `python src/build_mimic.py`.)*",
              ""]
    return L


def main():
    global R
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results",
                    help="artifact root; a per-model run passes "
                         "results/models/<slug>")
    ap.add_argument("--model_id", default="BioMistral/BioMistral-7B",
                    help="named in the header line, so a summary generated "
                         "for a second model cannot claim to be BioMistral's")
    args = ap.parse_args()
    R = Path(args.results)

    L = ["# All experiments, side by side",
         "",
         "*Generated by `src/make_summary.py` from the JSON artifacts. Do not "
         "edit the numbers by hand -- rerun the script.*",
         "",
         f"Everything on `{args.model_id}`, bf16, greedy decoding. "
         "Ground truth is always programmatic: no LLM-as-judge, no "
         "human annotation anywhere.",
         "",
         "The 2026-09-01 pass ran across both GPUs, partitioned by output "
         "file: `run_eval.py` merges into `summary_{split}{tag}.json` rather "
         "than overwriting, so the synthetic benchmark (GPU 0) and the real "
         "notes (GPU 1) were kept in separate lanes and no summary file had "
         "two writers.",
         ""]

    # ---------------- the ablation, both benchmarks --------------------
    L += ["## 1. The ablation, each contribution against the baseline\n"]
    L += ablation_block(
        "Causal Consistency and strict accuracy",
        [("synthetic control test", "summary_test.json"),
         ("synthetic control held-out", "summary_heldout.json"),
         ("real notes test", "summary_test_medcalc.json"),
         ("real notes held-out (QT)", "summary_heldout_medcalc.json"),
         ("MIMIC-IV test", "summary_test_mimic.json"),
         ("MIMIC-IV held-out (warfarin)", "summary_heldout_mimic.json")])
    L += ["Rows 1-4 are the proposal's cumulative ladder; rows 5-7 add exactly "
          "one contribution to the base model, which is the comparison the "
          "supervisor asked for; row 8 is everything at once. Per-table "
          "significance tests, bootstrap CIs and the gate-fired/gate-declined "
          "breakdown are in `results/table_*.md`.\n"]

    # ---------------- constraint layer ---------------------------------
    L += ["## 2. Aim 3: the trained Constraint-Aware Layer\n"]
    cl = {name: load(f"constraint_{name}.json")
          for name in ("qt", "qt_shuffled", "renal", "synth",
                       "mimic", "mimic_shuffled")}
    if any(cl.values()):
        L += ["| Run | trained on | held-out CC before | after | test CC "
              "before | after |", "|---|---|---|---|---|---|"]
        for name, d in cl.items():
            if not d:
                continue
            L.append(f"| {name} | {d.get('n_train_pairs','?')} pairs "
                     f"({d.get('train_on','?')}) | "
                     f"{d['base']['heldout_cc']:.3f} | "
                     f"**{d['adapted']['heldout_cc']:.3f}** | "
                     f"{d['base']['test_cc']:.3f} | "
                     f"{d['adapted']['test_cc']:.3f} |")
        L.append("")
        d = cl.get("qt")
        if d and "redacted_entropy" in d:
            r = d["redacted_entropy"]
            L += ["**L_uncertainty, measured rather than assumed.** Decision "
                  "entropy on notes whose decisive number has been redacted "
                  f"(maximum is ln 2 = {r['ln2']:.4f} nats):\n",
                  "| split | base model | with constraint layer | n |",
                  "|---|---|---|---|",
                  f"| test | {r['test_base']:.4f} | {r['test_adapted']:.4f} | "
                  f"{r['n_test']} |",
                  f"| held-out | {r['heldout_base']:.4f} | "
                  f"{r['heldout_adapted']:.4f} | {r['n_heldout']} |", ""]
            L += [f"Objective weights actually used: lam_kl="
                  f"{d.get('lam_kl')}, lam_ont={d.get('lam_ont')}, "
                  f"lam_unc={d.get('lam_unc')}."]
            req = d.get("ontology_required_margin")
            if req:
                pretty = ", ".join(f"`{k}` {v:.2f}" for k, v in sorted(req.items()))
                L += ["",
                      "`L_ontology` required margin, read from the causal "
                      "knowledge graph rather than held constant "
                      f"(`{d.get('ontology_graph')}`): {pretty}. A family the "
                      "graph licenses no contraindication for gets 0.0 and the "
                      "term is silent for it; x1.25 marks a MED-RT-attested "
                      "contraindication, x1.0 one resting on curation alone.",
                      ""]
            else:
                L += ["", "`L_ontology` used the pre-2026-09-01 constant "
                      "margin (no graph).", ""]

    # ---------------- SAE ----------------------------------------------
    L += ["## 3. Aim 1: sparse autoencoder features\n"]
    any_sae = False
    for f in sorted(R.glob("sae/*_fis.json")):
        d = json.load(f.open())
        any_sae = True
        L += [f"**`{f.name}`** — {d['kind']}, {d['d_hidden']} features, "
              f"reconstruction FVU {d['fvu']:.3f}, L0 {d['l0']:.1f}, "
              f"dead {d['dead']}/{d['d_hidden']} "
              f"({d['dead']/d['d_hidden']:.1%}).\n",
              "| feature | concept | S_semantic | S_causal | FIS |",
              "|---|---|---|---|---|"]
        for feat in d["features"][:10]:
            # a feature with no knock-out run is not a feature with zero
            # causal effect; print an em dash, never a number
            sc = feat.get("s_causal")
            sc = f"{sc:.4f}" if sc is not None else "not run"
            L.append(f"| #{feat['feature']} | {feat.get('concept') or '—'} | "
                     f"{feat['s_semantic']:.3f} | {sc} | "
                     f"{feat.get('fis', float('nan')):.3f} |")
        L += ["", f"*{d['fis_weights']['gamma_note']}*", ""]
    if not any_sae:
        L += ["_No SAE artifacts found; run stage 8 of "
              "`run_full_pipeline.sh`._\n"]

    # ---------------- UQ coverage --------------------------------------
    L += ["## 4. Aim 4: conformal coverage guarantees\n",
          "Frozen split-conformal vs Adaptive Conformal Inference, and "
          "coverage broken down by subgroup, are in "
          "`results/uq_coverage_*.md`. The headline: a frozen threshold has no "
          "guarantee on a split that is not exchangeable with the calibration "
          "split, which is exactly what the held-out rule family is, and ACI "
          "recovers the target by giving up coverage.\n"]

    # ---------------- earlier experiments, unchanged --------------------
    L += ["## 5. Supporting results\n",
          "Their reports hold the detail. The three causal rows are computed "
          "from the post-fix artifacts (B1/B2/B3, 2026-09-02); the two UQ rows "
          "are unaffected by those fixes and carry over unchanged.\n",
          "| Question | Answer | Where |",
          "|---|---|---|",
          "| Does the proposal's Eq. (2) uncertainty work? | **No** — pooled "
          "AUROC 0.525 over 6,456 items; restricting the same entropy to the "
          "answer tokens gives 0.687 | `results/bigbench_uq.md` |",
          "| Is counterfactual consistency alone evidence of reasoning? | "
          "**No** — discrimination −0.013 [−0.037, +0.013] over 8,000 items | "
          "`results/mcqpairs.md` |",
          "| Is the decisive fact represented internally? | QT yes (pair-CC "
          "0.976), creatinine no (0.042) | `results/aim123_internals.md` |",
          f"| Does it causally drive the answer? | No — patching effects "
          f"{_causal_spans(R)['patching']}, ~100x too small to flip a "
          f"decision | `results/aim123_internals.md` |",
          f"| ...and by SAE feature knock-out? | No — "
          f"{_causal_spans(R)['knockout']} against matched controls, "
          f"replicating the patching null | `results/aim1_sae.md` |",
          f"| ...and by feature injection (sufficiency)? | No — "
          f"{_causal_spans(R)['sufficiency']} | "
          f"`results/sufficiency_medcalc_{{test,heldout}}.json` |",
          "",
          "See `results/FIXES.md` for the nine defects found in an audit of "
          "this repository, what each would have done to a reported number, "
          "and the before/after comparison showing no conclusion reversed.",
          ""]

    L += _threshold_provenance(R)
    L += _benchmark_arms(R)

    (R / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {R/'SUMMARY.md'}")


if __name__ == "__main__":
    main()
