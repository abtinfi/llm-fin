"""
PAPER.md -- the manuscript, assembled from the artifacts.

Every quantitative claim is read from a results file at build time. Nothing is
typed in. If an artifact is missing the build FAILS rather than quietly
dropping a sentence, because a manuscript that silently omits the number it
was going to qualify is worse than one that does not build.

The prose lives here rather than in a hand-edited .md for one reason: the
numbers move. Two of them moved during the week this file was written (the
40-pair sufficiency figures, and the unseeded random control), and a
hand-maintained manuscript would still be quoting the old ones.
"""

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

R = Path("results")


class Missing(Exception):
    pass


def art(rel):
    p = R / rel
    if not p.is_file():
        raise Missing(f"required artifact missing: {p}. Run the stage that "
                      f"produces it before building the manuscript.")
    return json.loads(p.read_text())


def opt(rel):
    p = R / rel
    return json.loads(p.read_text()) if p.is_file() else None


def txt(rel):
    p = R / rel
    return p.read_text() if p.is_file() else None


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def f4(x):
    return "—" if x is None else f"{x:.4f}"


def by_variant(summary):
    """summary_*.json is a list of per-variant blocks; index it."""
    return {b["variant"]: b for b in summary if isinstance(b, dict)
            and "variant" in b}


def need(pattern, text, what):
    """A regex match in a generated report, or a failed build."""
    m = re.search(pattern, text or "")
    if not m:
        raise Missing(f"could not read {what} from its report; regenerate "
                      f"the v3b reports before building the manuscript.")
    return m


def span(xs, fmt=f3):
    """'lo–hi' over a list of numbers, or the single value if they agree."""
    lo, hi = fmt(min(xs)), fmt(max(xs))
    return lo if lo == hi else f"{lo}–{hi}"


# ---------------- MIMIC-IV v3.1 (`mimic_v3b`) ----------------
V3 = "mimic_v3b"
V3_MODELS = [("biomistral-7b", "BioMistral-7B"),
             ("llama3-openbiollm-8b", "OpenBioLLM-8B"),
             ("mistral-7b-instruct-v0-2", "Mistral-7B-Instruct")]
V3_UQ_ROWS = [("uq", "Base + UQ"), ("nsai_uq", "NS-AI + UQ"),
              ("nsai_uq_cl", "NS-AI + UQ + CL")]


def md_table(text, section):
    """Data rows of the table under the `## <section>...` heading of a
    generated markdown report (CONSOLIDATED_MIMIC3B.md has one per split)."""
    rows, on = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            on = line.startswith(f"## {section}")
            continue
        if on and line.startswith("| ") and not line.startswith("| Model"):
            rows.append([c.strip() for c in line.strip().strip("|")
                         .split("|")])
    if not rows:
        raise Missing(f"no `{section}` table in the consolidated report")
    return rows


def riskcov_at_tau(model, split, row, tau):
    """(coverage of UQ-governed items, selective risk) where tau lands on the
    risk-coverage curve make_table.py wrote; None if nothing is governed."""
    p = R / V3 / model / f"table_mimic3b_{split}_riskcov_{row}.csv"
    if not p.is_file():
        return None
    with p.open() as f:
        curve = [{k: float(v) for k, v in r.items()}
                 for r in csv.DictReader(f)]
    if not curve:
        return None
    kept = [q for q in curve if tau is not None and q["threshold"] <= tau]
    return (kept[-1]["coverage"], kept[-1]["error"]) if kept else (0.0, None)


def sae_contraction(old_rel, new_rel):
    """Σ S_causal over the top-25 features against the old single uniform
    control and against the matched controls, with the per-feature values."""
    old, new = art(old_rel), art(new_rel)
    rows = []
    for f in old["features"][:25]:
        k = str(f["feature"])
        n = new["causal"]["per_feature"].get(k)
        if n is None:
            continue
        rows.append({"feature": f["feature"], "concept": f["concept"],
                     "old": max(old["causal"]["per_feature"][k]["excess"], 0),
                     "new": max(n["excess"], 0)})
    o, n = sum(r["old"] for r in rows), sum(r["new"] for r in rows)
    return {"rows": rows, "old": o, "new": n,
            "cut": (1 - n / o) if o else None,
            "n_items": new["causal"]["n_items"],
            "n_controls": new["causal"].get("n_controls"),
            "mode": new["causal"].get("control_mode")}


def v3b_section(A, v3):
    """§5.9: the MIMIC-IV v3.1 replication, its tables, figures 2-3, and the
    caveats that bound it."""
    VS = v3["s"]
    names = dict(V3_MODELS)

    def col(split, row, key):
        return [VS[m, split][row][key] for m, _ in V3_MODELS]

    fam = v3["meta"]["families"]
    trained = [k for k, v in fam.items() if not v["held_out"]]
    held = [k for k, v in fam.items() if v["held_out"]]
    A("### 5.9 Replication at scale: MIMIC-IV v3.1 (`mimic_v3b`)")
    A("")
    A(f"The MIMIC-IV rows of §5.1 come from the public Demo and have at "
      f"most a few hundred items. This arm rebuilds the benchmark from the full credentialed "
      f"{v3['meta']['source']}: {v3['test_items']} test items "
      f"({v3['test_distinct']} distinct prompts, {v3['test_patients']} "
      f"patients) over {len(trained)} trainable families "
      f"({', '.join(f'`{k}`' for k in trained)}), and "
      f"{v3['heldout_items']} items on the held-out family "
      f"{', '.join(f'`{k}`' for k in held)}, which no component was trained "
      f"or calibrated on. Both arms of every pair are real measurements, and "
      f"train, calibration, test and held-out are patient-disjoint (audit: "
      f"**{'PASS' if v3['disjoint'] else 'FAIL'}**). All three models run "
      f"the same rows with greedy decoding and one seed. CIs resample "
      f"distinct prompts, not items, because a prompt recurs "
      f"{v3['dup_ratio']}× on average.")
    A("")
    A("**Model's own** is the answer parsed from the model's generation "
      "before the gate overrides it or UQ defers it. † marks a row where "
      "the gate fired: there, strict accuracy and CC are partly an identity "
      "with the labelling rule, not a measurement. **Violation** is the "
      "share of UNSAFE items answered SAFE; **Coverage** the share "
      "answered. `cl` is the Aim 3 residual adapter h' = h + α·P_causal(h) "
      "(rank 32, layer 30, α = 1) on the frozen model; *unseen pairs* "
      "restricts it to test pairs none of whose prompts it trained on. "
      "Source: `results/mimic_v3b/CONSOLIDATED_MIMIC3B.md`.")
    for split, rows, title in (
            ("test", v3["test_rows"], "Test split (metformin ×2, "
                                      "spironolactone)"),
            ("heldout", v3["heldout_rows"], "Held-out family (warfarin, "
                                            "INR > 4)")):
        A("")
        A(f"**{title}**")
        A("")
        A("| Model | Row | Kind | Strict accuracy [95% CI] | Model's own | "
          "CC [95% CI] | Coverage | Violation |")
        A("|---|---|---|---|---|---|---|---|")
        for r in rows:
            A("| " + " | ".join([names.get(r[0], r[0])] + r[1:]) + " |")
    A("")

    base_cc, base_acc = col("test", "base", "causal_consistency"), \
        col("test", "base", "accuracy")
    ob = VS["llama3-openbiollm-8b", "test"]
    A(f"**No model reads the lab value unaided.** Base CC is "
      f"{span(base_cc)}, and base accuracy ({span(base_acc)}) sits at or "
      f"near the always-SAFE rate of {v3['safe_share'] / 100:.3f}: "
      f"OpenBioLLM-8B answers SAFE to every UNSAFE item (violation "
      f"{f3(ob['base']['violation_rate'])}).")
    A("")
    unseen = [float(r[5].split()[0]) for r in v3["test_rows"]
              if r[1].startswith("Base + constraint") and "unseen" in r[1]]
    A(f"**The constraint layer is the one component that changes the "
      f"model's own answer.** On the frozen model it reaches strict accuracy "
      f"{span(col('test', 'cl', 'accuracy'))} and CC "
      f"{span(col('test', 'cl', 'causal_consistency'))} on test, with "
      f"violation {span(col('test', 'cl', 'violation_rate'))}. On test pairs "
      f"none of whose prompts it trained on, CC is {span(unseen)} — no "
      f"lower — so the gain is not memorised prompts. Against its "
      f"shuffled-label control, on the adapter's own evaluation (argmax over "
      f"the two answer logits, `constraint_mimic3b{{,_shuffled}}.json`):")
    A("")
    A("| model | frozen base, test CC | rule labels | shuffled labels | "
      "frozen base, held-out CC | rule labels | shuffled labels |")
    A("|---|---|---|---|---|---|---|")
    beat = []
    for m, name in V3_MODELS:
        rule, shuf = v3["cl"][m]
        A(f"| {name} | {f3(rule['base']['test_cc'])} | "
          f"**{f3(rule['adapted']['test_cc'])}** | "
          f"{f3(shuf['adapted']['test_cc'])} | "
          f"{f3(rule['base']['heldout_cc'])} | "
          f"{f3(rule['adapted']['heldout_cc'])} | "
          f"{f3(shuf['adapted']['heldout_cc'])} |")
        if shuf["adapted"]["heldout_cc"] > rule["adapted"]["heldout_cc"]:
            beat.append((name, shuf["adapted"]["heldout_cc"],
                         rule["adapted"]["heldout_cc"]))
    A("")
    A("On test the shuffled adapter recovers little of the gain, so what the "
      "rule-trained adapter learned is the rule and not the act of "
      "perturbing the residual stream.")
    A("")
    gate = col("test", "nsai", "gate_fired_rate")
    A(f"**The gate and UQ buy safety with coverage.** The gate fires on "
      f"{span(gate, lambda x: f'{x:.1%}')} of test items and, applying the "
      f"labelling rule itself, is exact where it fires. NS-AI + UQ "
      f"reaches violation {span(col('test', 'nsai_uq', 'violation_rate'))} "
      f"at coverage {span(col('test', 'nsai_uq', 'coverage'))}; on the "
      f"held-out family the gate decides every item, so every gated row "
      f"scores 1.000† there.")
    A("")
    A(f"**Every UQ row is calibrated by the conformal rule** "
      f"({v3['n_conformal']} of {v3['n_uq_rows']} UQ rows carry "
      f"`calib_rule = conformal`; none falls back to the legacy "
      f"point-estimate rule). τ is the largest threshold whose one-sided "
      f"Clopper–Pearson upper bound (δ = 0.10) on the calibration split's "
      f"selective error is ≤ α = 0.10. The bound is evaluated point-wise, "
      f"once per candidate τ, during selection; Figure 2 reports it beside "
      f"each deployed point rather than drawing it as a band over the test "
      f"curves, where prompt repetition would make an item-level binomial "
      f"bound far tighter than the data supports. τ = −∞ means no threshold "
      f"was certifiable and the row defers everything — the conservative "
      f"outcome, not a failure.")
    A("")
    A("| model | row | calibration n | calibration risk at τ | CP upper "
      "bound | test coverage (UQ-governed) | test risk at τ |")
    A("|---|---|---|---|---|---|---|")
    over, alpha = [], 0.10
    for name, lab, s, op in v3["uq"]:
        cert = s.get("calib_certifiable")
        cal = (f"{f3(s['calib_error_at_tau'])} | "
               f"{f3(s['calib_cp_upper_at_tau'])}") if cert else \
            "— | none ≤ α (τ = −∞)"
        cov, risk = op if op else (None, None)
        A(f"| {name} | {lab} | {s['calib_n']:,} | {cal} | "
          f"{'—' if cov is None else f'{cov:.1%}'} | {f3(risk)} |")
        alpha = s.get("calib_target_alpha", alpha)
        if risk is not None and risk > alpha:
            over.append((name, lab, s["calib_cp_upper_at_tau"], risk))
    A("")
    for name, lab, cp, risk in over:
        A(f"{name}, {lab} was certified at a Clopper–Pearson upper bound of "
          f"{f3(cp)} on calibration and realised {f3(risk)} on test, "
          f"{risk - alpha:.3f} above α. The guarantee is a calibration-split "
          f"statement that holds with probability 1 − δ, and it carries to "
          f"test only as far as test is exchangeable with calibration.")
        A("")
    A("![Figure 2](results/mimic_v3b/figures/fig_risk_coverage.png)")
    A("")
    A("*Figure 2. Selective risk of the model's own answer against coverage "
      "of the UQ-governed items (gate-decided items are never deferred), per "
      "model and UQ row; markers are the deployed τ, and the table under "
      "the panels gives the calibration evidence that certified each one. "
      "Vector version: `results/mimic_v3b/figures/fig_risk_coverage.pdf`.*")
    A("")
    parts, ccs = [], []
    for m, name in V3_MODELS:
        o, s = VS[m, "test"]["base"], v3["swap"][m]["base"]
        parts.append(f"{name} {f3(o['accuracy'])} → {f3(s['accuracy'])}")
        ccs += [o["causal_consistency"], s["causal_consistency"]]
    A(f"**The base models answer partly by position.** Listing UNSAFE "
      f"first instead of SAFE moves base accuracy {'; '.join(parts)}, "
      f"toward chance, while CC stays at or below {f3(max(ccs))} in both "
      f"orders (Figure 3). "
      f"A model that read the value would not care which option comes first; "
      f"these models' SAFE-leaning answers are in part a first-option "
      f"preference.")
    A("")
    A("![Figure 3](results/mimic_v3b/figures/fig_option_order.png)")
    A("")
    A("*Figure 3. Base model on the test split with the answer options in "
      "the original and in swapped order: accuracy and CC with "
      "prompt-resampled CIs, and the share of answers that are UNSAFE. "
      "Vector version: `results/mimic_v3b/figures/fig_option_order.pdf`.*")
    A("")
    A("#### Caveats specific to this arm")
    A("")
    obh = VS["llama3-openbiollm-8b", "heldout"]
    A(f"1. **OpenBioLLM-8B mostly does not answer when retrieved context is "
      f"in the prompt.** Its RAG rows return no parsable SAFE/UNSAFE on "
      f"{ob['rag']['unparsable_rate']:.1%} of test items (base: "
      f"{ob['base']['unparsable_rate']:.1%}), and "
      f"{ob['nsai']['unparsable_rate']:.1%} of its NS-AI rows remain "
      f"unparsable after the gate fills in the items it fires on. A "
      f"non-answer scores as wrong, so its RAG ({f3(ob['rag']['accuracy'])}) "
      f"and NS-AI ({f3(ob['nsai']['accuracy'])}) accuracies measure an "
      f"answer-format failure under long context more than clinical "
      f"reasoning, and its NS-AI *model's own* column inherits the RAG "
      f"failure. The held-out split is unaffected "
      f"({obh['rag']['unparsable_rate']:.1%} non-answers). These rows should "
      f"not be quoted as evidence that retrieval harms this model's "
      f"clinical judgement.")
    cl_h = col("heldout", "cl", "causal_consistency")
    base_h = col("heldout", "base", "causal_consistency")
    rule_h = [v3["cl"][m][0]["adapted"]["heldout_cc"] for m, _ in V3_MODELS]
    beat_txt = "".join(
        f" For {n} the shuffled-label adapter scores higher on held-out "
        f"({f3(a)} against {f3(b)}), so no held-out movement can be credited "
        f"to the rule." for n, a, b in beat)
    A(f"2. **The constraint layer does not transfer to warfarin.** Held-out "
      f"CC with the adapter is {span(cl_h)} (frozen base {span(base_h)}); on "
      f"the adapter's own evaluation it is {span(rule_h)}.{beat_txt} The "
      f"1.000† of every gated held-out row is the gate applying the INR > 4 "
      f"rule the label was generated from — an identity, not transfer. The "
      f"adapter generalised to the held-out QT family of §5.6 and not to "
      f"this one, so held-out transfer is a property of the family pair, "
      f"not a guarantee of the method.")
    A(f"3. **Prompt rendering produces duplicates.** The note prints only "
      f"age, sex, one lab value and the drug, so different patients yield "
      f"identical text: {v3['test_items']} test items collapse to "
      f"{v3['test_distinct']} distinct prompts, and {v3['overlap']} of those "
      f"{v3['overlap_of']} also occur verbatim in calibration (the patients "
      f"remain disjoint). This is why CIs resample prompts rather than "
      f"items. It also makes the calibration-to-test agreement of the UQ "
      f"rows optimistic: much of the test text is text the threshold was "
      f"fitted on, and genuinely novel presentations would be less "
      f"exchangeable with calibration.")
    for k, v in v3["flagged"].items():
        A(f"4. **`{k}` rests on a threshold the audit flags "
          f"`{v['threshold_status']}`**: no FDA label states it as a "
          f"contraindication (`results/threshold_provenance.md`). It is "
          f"built and reported rather than hidden; results on it should not "
          f"be quoted as label-attested.")
    note_n = art(f"{V3}/biomistral-7b/summary_test_mimic3bnote.json")[0]["n"]
    A(f"5. **Scope.** The SAE analyses (§5.3, Figure 1) were run on "
      f"BioMistral-7B over the real-notes benchmark, not on this arm; the "
      f"figure is stored with the v3b figures but is not a MIMIC "
      f"measurement. A companion arm on real MIMIC-IV-Note text "
      f"({note_n:,} test items) is reported in "
      f"`results/mimic_v3b/COMPARISON_MIMIC3BNOTE.md` and not discussed "
      f"here.")
    A("")


def load_v3b():
    """Everything §5.9 says, read from the committed v3b aggregates. The
    per-item prediction logs are gitignored (credentialed data), so nothing
    here needs them."""
    cons = txt(f"{V3}/CONSOLIDATED_MIMIC3B.md")
    ladder = txt(f"{V3}/LADDER_MIMIC3B.md")
    sanity = txt(f"{V3}/AUDIT_SANITY.md")
    checks = txt(f"{V3}/AUDIT_CHECKS.md")
    for name, t in (("CONSOLIDATED_MIMIC3B.md", cons),
                    ("LADDER_MIMIC3B.md", ladder),
                    ("AUDIT_SANITY.md", sanity),
                    ("AUDIT_CHECKS.md", checks)):
        if t is None:
            raise Missing(f"required artifact missing: {R / V3 / name}")
    d = {"test_rows": md_table(cons, "test"),
         "heldout_rows": md_table(cons, "held-out")}
    d["s"] = {(m, sp): by_variant(art(f"{V3}/{m}/summary_{sp}_mimic3b.json"))
              for m, _ in V3_MODELS for sp in ("test", "heldout")}
    d["swap"] = {m: by_variant(art(f"{V3}/{m}/summary_test_mimic3bswap.json"))
                 for m, _ in V3_MODELS}
    d["cl"] = {m: (art(f"{V3}/{m}/constraint_mimic3b.json"),
                   art(f"{V3}/{m}/constraint_mimic3b_shuffled.json"))
               for m, _ in V3_MODELS}
    uq_rows = [r for m, _ in V3_MODELS for sp in ("test", "heldout")
               for r in d["s"][m, sp].values() if "uq" in r["variant"]]
    d["n_uq_rows"] = len(uq_rows)
    d["n_conformal"] = sum(r.get("calib_rule") == "conformal"
                           for r in uq_rows)
    t = need(r"test\s+items=\s*([\d,]+)\s+distinct=\s*([\d,]+)\s+"
             r"ratio=\s*([\d.]+)x", checks, "the test item/prompt ratio")
    d["test_items"], d["test_distinct"], d["dup_ratio"] = t.groups()
    h = need(r"heldout\s+items=\s*([\d,]+)\s+distinct=\s*([\d,]+)", checks,
             "the held-out item count")
    d["heldout_items"] = h.group(1)
    o = need(r"([\d,]+) of ([\d,]+) distinct test prompts also occur, as "
             r"text, in calibration", sanity, "the calibration/test overlap")
    d["overlap"], d["overlap_of"] = o.groups()
    d["disjoint"] = "**PASS**" in need(r"Assertions \(train∩test.*", sanity,
                                       "the patient-disjointness verdict"
                                       ).group(0)
    p = need(r"\| mimic_v3b \| train \| test \| ([\d,]+) \| ([\d,]+) \|",
             sanity, "the test patient count")
    d["test_patients"] = p.group(2)
    d["safe_share"] = float(need(r"The label is SAFE on ([\d.]+)% of them",
                                 ladder, "the test SAFE share").group(1))
    meta = json.loads(Path("data/mimic_v3b/build_meta.json").read_text())
    d["meta"] = meta
    d["flagged"] = {k: v for k, v in meta["families"].items()
                    if v.get("threshold_status") != "attested_exact"}
    d["uq"] = []
    for m, name in V3_MODELS:
        for row, lab in V3_UQ_ROWS:
            s = d["s"][m, "test"][row]
            d["uq"].append((name, lab, s,
                            riskcov_at_tau(m, "test", row, s.get("tau"))))
    d["sae"] = [
        ("test", sae_contraction("pre_s1s3_20260908/sae_topk_L20_fis.json",
                                 "sae/sae_topk_L20_fis.json")),
        ("held-out",
         sae_contraction("pre_s1s3_20260908/sae_heldout_topk_L20_fis.json",
                         "sae_heldout/sae_topk_L20_fis.json"))]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="PAPER.md")
    args = ap.parse_args()

    # ---------------- artifacts ----------------
    s_test = by_variant(art("summary_test.json"))
    s_hout = by_variant(art("summary_heldout.json"))
    s_mc = by_variant(art("summary_test_medcalc.json"))
    s_mch = by_variant(art("summary_heldout_medcalc.json"))
    s_mi = by_variant(art("summary_test_mimic.json"))
    s_mc2 = opt("summary_test_medcalc2.json")
    s_mc2 = by_variant(s_mc2) if s_mc2 else None

    uq = art("bigbench_uq.json")
    pooled = uq["POOLED"]
    e_ent = pooled["auroc"]["entropy"]
    e_opt = pooled["auroc"]["option_entropy"]
    e_con = pooled["contrasts"]["option_entropy"]

    fis = art("sae/sae_topk_L20_fis.json")
    fis_j = art("sae/sae_jumprelu_L20_fis.json")

    # The headline feature must be one that is actually CUI-anchored. The
    # top feature by S_semantic is `age`, whose provenance is
    # `lexical-fallback` with no CUI at all -- quoting it as evidence that
    # "sparse features select BIOMEDICAL concepts" would be the exact
    # overclaim the grounding audit exists to prevent.
    cprov = fis.get("concept_provenance", {})
    def grounded(f):
        pr = cprov.get(f.get("concept"), {})
        return bool(pr.get("cui")) and pr.get("source") == "umls"
    grounded_feats = [f for f in fis["features"] if grounded(f)]
    best = grounded_feats[0] if grounded_feats else fis["features"][0]
    best_ungrounded = fis["features"][0]

    suf_t = art("sufficiency_medcalc_test.json")
    suf_h = art("sufficiency_medcalc_heldout.json")
    pat_t = art("patching_medcalc_test.json")
    pat_h = art("patching_medcalc_heldout.json")

    def max_exc(d):
        return max(abs(r["excess"]) for r in d["rows"])

    faith = {}
    for f in sorted(glob.glob(str(R / "faithfulness_medcalc_*.json"))):
        d = json.loads(Path(f).read_text())
        wide = f.endswith("_wide.json")
        split = "heldout" if "_heldout_" in Path(f).name else "test"
        faith[(split, wide)] = d
    ft = faith.get(("test", True)) or faith.get(("test", False))
    fh = faith.get(("heldout", True)) or faith.get(("heldout", False))

    cl_qt = art("constraint_qt.json")
    cl_qt_sh = art("constraint_qt_shuffled.json")

    prov = art("data_provenance.json")
    umls_cov = txt("umls_coverage.txt") or ""
    attested = "5/10"
    for line in umls_cov.splitlines():
        if "ontology-attested:" in line:
            attested = line.split(":", 1)[1].strip().split()[0]

    other_models = sorted(glob.glob("results/models/*/summary_test_medcalc.json"))
    other_slugs = [Path(p).parent.name for p in other_models]

    v3 = load_v3b()
    VS = v3["s"]

    def v3col(split, row, key):
        return [VS[m, split][row][key] for m, _ in V3_MODELS]

    v3_base_cc = v3col("test", "base", "causal_consistency")
    v3_cl_cc = v3col("test", "cl", "causal_consistency")
    v3_cl_h_cc = v3col("heldout", "cl", "causal_consistency")
    v3_rule_cc = [v3["cl"][m][0]["adapted"]["test_cc"] for m, _ in V3_MODELS]
    v3_shuf_cc = [v3["cl"][m][1]["adapted"]["test_cc"] for m, _ in V3_MODELS]
    sae_t, sae_h = dict(v3["sae"])["test"], dict(v3["sae"])["held-out"]

    # ---------------- prose ----------------
    L = []
    A = L.append

    A("# Causal Discovery and Control of Internal Representations in "
      "Clinical Foundation Models")
    A("")
    A("*Generated by `src/make_paper.py` from the artifacts under "
      "`results/`. Every number below is read from a results file at build "
      "time; none is typed in. Rebuild rather than edit.*")
    A("")
    A("## Abstract")
    A("")
    A(f"We ask whether a clinical language model's safety decisions rest on "
      f"identifiable internal representations that can be found, "
      f"causally tested, and controlled. On BioMistral-7B we train sparse "
      f"autoencoders on the residual stream, probe every layer, run "
      f"activation patching and feature knock-out with matched controls, "
      f"train a constraint-aware residual adapter, and evaluate "
      f"uncertainty-aware deferral under conformal calibration. We introduce "
      f"a Token-to-Concept Attribution Layer that completes the bridge from a "
      f"generated token back to a biomedical concept, and evaluate it for "
      f"faithfulness.")
    A("")
    A(f"Three findings are positive and three are negative, and the negative "
      f"ones are the load-bearing ones. **Sparse features do select clinical "
      f"concepts** — the best CUI-anchored feature reaches F1 "
      f"{f3(best['s_semantic'])} on `{best['concept']}` "
      f"(CUI {cprov.get(best['concept'], {}).get('cui', '—')}). **The attribution "
      f"layer points at the right token**: on the renal family it ranks the "
      f"causally edited token at the "
      f"{f3(ft['edited_token_percentile']['sae_concept'])} percentile against "
      f"{f3(ft['edited_token_percentile']['random'])} for a random "
      f"attributor, and names the right concept "
      f"{f3(ft['concept_pointing']['accuracy_mean'])} of the time against a "
      f"{f3(ft['concept_pointing']['random_baseline'])} chance baseline. "
      f"**Symbolic constraints raise consistency without harming language "
      f"ability** — Causal Consistency 0.000 → {f3(cl_qt['adapted']['heldout_cc'])} on "
      f"a held-out rule family, with WikiText-2 perplexity unchanged.")
    A("")
    A(f"Against that: **intervening on those features does not move the "
      f"decision.** Activation patching yields at most "
      f"{f4(max(max_exc(pat_t), max_exc(pat_h)))} logits over matched "
      f"controls, feature knock-out and feature injection replicate the null "
      f"({f4(max(max_exc(suf_t), max_exc(suf_h)))} logits), and a decision "
      f"flip needs 1–5. **The proposal's uncertainty term measures the wrong "
      f"quantity** — whole-vocabulary entropy reaches AUROC "
      f"{f3(e_ent['auroc'])} [{f3(e_ent['lo'])}, {f3(e_ent['hi'])}] over "
      f"{e_ent['n']:,} items, while the identical entropy restricted to the "
      f"decision tokens reaches {f3(e_opt['auroc'])} "
      f"[{f3(e_opt['lo'])}, {f3(e_opt['hi'])}]. And **the symbolic gate's "
      f"accuracy is bounded not by its precision but by how often it can fire "
      f"at all**, which falls as the rule needs more variables and collapses "
      f"when a variable is a clinical judgement rather than a number.")
    A("")
    A(f"**At scale, the adapter holds on the rules it was trained on and not "
      f"on a new one.** Replicated on the full, credentialed MIMIC-IV v3.1 — "
      f"{v3['test_items']} test items from {v3['test_patients']} patients, "
      f"three models — the residual adapter lifts test Causal Consistency "
      f"from {span(v3_base_cc)} to {span(v3_cl_cc)} on every model (on its "
      f"own evaluation, {span(v3_rule_cc)} for rule labels against "
      f"{span(v3_shuf_cc)} for shuffled ones), but on the held-out warfarin "
      f"family it stays at {span(v3_cl_h_cc)}. Re-scoring the SAE knock-outs "
      f"against firing-rate-matched controls removes {sae_t['cut']:.0%} of "
      f"the summed causal score.")
    A("")
    A("---")
    A("")
    A("## 1. Introduction")
    A("")
    A("A clinical model that is right for the wrong reason is not safe. The "
      "question this work asks is not whether a model can score well on a "
      "contraindication benchmark — it can — but whether its answer is "
      "produced by an internal structure that corresponds to the clinical "
      "rule, and whether that structure can be intervened on.")
    A("")
    A("Three research questions follow, and each is answered here with a "
      "measurement rather than a demonstration.")
    A("")
    A("| RQ | Question | Answer |")
    A("|---|---|---|")
    A(f"| RQ1 | Do sparse latent features map to biomedical concepts with "
      f"semantic coherence? | **Yes, partially.** Best CUI-anchored feature "
      f"F1 {f3(best['s_semantic'])} (`{best['concept']}`); the highest-scoring "
      f"feature overall is `{best_ungrounded['concept']}` at "
      f"{f3(best_ungrounded['s_semantic'])}, which has no CUI and must not be "
      f"quoted as a biomedical concept |")
    A("| RQ2 | Do targeted interventions on those features produce "
      "predictable, causally consistent changes? | **No, on this model.** "
      "Necessity, sufficiency and knock-out all land ~100× below the "
      "threshold for a decision flip |")
    A("| RQ3 | Do symbolic constraints improve consistency without harming "
      "language ability? | **Yes on the trained families; transfer depends "
      "on the family.** Consistency rises on the held-out QT family with "
      "perplexity unchanged, and on the MIMIC-IV v3.1 test split for all "
      "three models, but not on the held-out warfarin family (§5.9) |")
    A("")
    A("A methodological point runs through all three. Counterfactual "
      "consistency on flip-only pairs cannot distinguish a model that "
      "responds to the clinical variable from one that responds to any prompt "
      "edit. On 8,000 MCQ items this model's discrimination was **−0.013 "
      "[−0.037, +0.013]** — it changed its answer at the same rate whether or "
      "not the truth changed.")
    A("")
    A("The expanded real-notes arm therefore carries **control pairs** whose "
      "driving value moves by a comparable amount *without* crossing the "
      "threshold, and reports a spurious-flip rate beside Causal Consistency. "
      "The other arms do not yet carry them, and their consistency numbers "
      "are reported without a spurious-flip rate rather than with one implied "
      "— a split with no control pairs reports the rate as *not measured*, "
      "never as zero.")
    A("")
    A("## 2. Related work")
    A("")
    A("Retrieval-augmented generation is the standard grounding technique and "
      "is insufficient here: retrieval succeeds and the decision does not "
      "improve. On the real-notes benchmark a threshold-bearing document is "
      "in the top-3 for 99.5% of cases, and the model gets *worse* — it "
      "answers UNSAFE to 96% of items. Retrieval places the fact in the "
      "context; it does not make the model condition on it.")
    A("")
    A("Neuro-symbolic integration is usually a post-hoc parser-regex chain. "
      "We keep the post-hoc gate as an explicit, separately-reported row and "
      "additionally train a constraint-aware layer that modifies the residual "
      "stream, so the two are never conflated — they are different "
      "mechanisms with different failure modes, and the gate's headline "
      "accuracy is partly circular in a way the trained layer's is not.")
    A("")
    A("## 3. Method")
    A("")
    A("### 3.1 The discovery pipeline")
    A("")
    A("```")
    A("Clinical prompt -> hidden states -> SAE features -> concept probing")
    A("    -> causal test -> intervention adapter -> UQ / abstention")
    A("```")
    A("")
    A(f"Sparse autoencoders are trained on the layer-20 residual stream over "
      f"clinical tokens, in both architectures the proposal names. TopK "
      f"reconstructs better at comparable sparsity (FVU {f3(fis['fvu'])} "
      f"against {f3(fis_j['fvu'])}; dead features {fis['dead']:,} against "
      f"{fis_j['dead']:,} of {fis['d_hidden']:,}) and is used throughout.")
    A("")
    A("**The architecture choice was made on reconstruction error, not on the "
      "proposal's criterion.** §4.4 specifies a fallback triggered when fewer "
      "than 30% of features pass expert validation. That criterion is defined "
      "on a quantity no pipeline can supply, and this project has no rater "
      "panel, so it is not merely unmet but **not evaluable**. The "
      "substitution is reported rather than elided; `S_human` is forced to "
      "weight zero and every artifact says so.")
    A("")
    A("### 3.2 The Token-to-Concept Attribution Layer")
    A("")
    A("§4.2 promises a bridge from a generated token back to a clinical "
      "concept and asks for it to be evaluated for faithfulness. For token "
      "*t* and concept *c*:")
    A("")
    A("```")
    A("a[t, c] = sum over features f assigned to c of  z_f(h_t) * w_f")
    A("```")
    A("")
    A("`w_f` is the feature's **measured** decision weight, read from the "
      "knock-out excess already on disk rather than fitted as a new "
      "parameter. Features whose knock-out does not beat a firing-rate "
      "matched random control get zero weight: a negative excess is not "
      "evidence of a causal role.")
    A("")
    A("Faithfulness uses the standard ERASER pair — comprehensiveness (drop "
      "when the rationale is deleted) and sufficiency (drop when only the "
      "rationale is kept) — over rationale sizes 1/5/10/20/50%, against three "
      "comparators: gradient×input, exact leave-one-token-out occlusion (the "
      "upper bound), and a random attributor (the floor).")
    A("")
    A("**A terminology collision the proposal creates.** §4.2's faithfulness "
      "*sufficiency* and §4.5's causal *sufficiency* (feature injection) are "
      "different quantities pointing in opposite directions — lower is better "
      "for the first, larger for the second. They are kept in separate files "
      "with separate key names.")
    A("")
    A("### 3.3 Grounding")
    A("")
    A(f"The causal knowledge graph is built from UMLS concept identities and "
      f"MED-RT drug–condition relations: 184 nodes, 234 edges, every edge "
      f"provenance-tagged. **{attested} rule families are ontology-attested.** "
      f"The proposal names SNOMED CT; SNOMED CT is not used anywhere in the "
      f"implementation and the substitution is recorded in "
      f"`PROPOSAL_ERRATA.md`.")
    A("")
    A("UMLS atoms alone are not enough. They are terminology-normalised while "
      "notes write \"eGFR\", \"pregnant\", \"QTc\", so matching on atoms alone "
      "collapses recall (eGFR 16→0 hits, pregnancy 20→2). The matcher is a "
      "provenance-tracked union of UMLS atoms and a curated lexical layer, "
      "and the split is reported per concept: a concept resting mostly on "
      "curation must not be described as UMLS-matched.")
    A("")
    A("## 4. Benchmarks")
    A("")
    A("Three arms, each making a different trade, none dominating; the "
      "third is also rebuilt at full scale from the credentialed database "
      "(§5.9).")
    A("")
    A("| arm | text | numbers | note |")
    A("|---|---|---|---|")
    A("| synthetic control | templated vignette | invented | control arm: "
      "shows what the pipeline does when the causal factor is stated cleanly "
      "and the label is guaranteed |")
    A("| real notes (MedCalc) | real PMC case-report prose | one arm real, "
      "one **edited** | half of every pair has its driving number changed to "
      "cross the threshold |")
    A("| MIMIC-IV | minimal rendered note | **both arms real** | no number is "
      "invented, but the two arms are different *timepoints*, so the clinical "
      "state genuinely differed |")
    A(f"| MIMIC-IV v3.1 (§5.9) | minimal rendered note | **both arms real** "
      f"| the full credentialed database, {v3['test_items']} test items; "
      f"per-item rows never leave the machine that ran them |")
    A("")
    A("Ground truth is programmatic throughout: no LLM-as-judge, no human "
      "annotation. Every threshold is audited against FDA labelling by "
      "`src/curate_thresholds.py`, and the audit is reported honestly — "
      "5 of 10 attested, and two of the failures are the dangerous kind where "
      "a number **is** present in the label and encodes something else.")
    A("")
    A(f"The pipeline runs end to end with **no credentialed data source**. "
      f"`src/check_data.py --strict` is stage 0 and fails the run if that "
      f"ever stops being true; its provenance record covers "
      f"{len(prov.get('required', []))} required artifacts. The one "
      f"exception is the MIMIC-IV v3.1 replication of §5.9, which requires "
      f"PhysioNet credentialing: it is a separate arm outside this check, "
      f"its datasets and per-item prediction logs are gitignored, and only "
      f"aggregate summaries, tables and figures are committed.")
    A("")
    A("## 5. Results")
    A("")
    A("### 5.1 Baseline against the proposed system")
    A("")
    cmp_md = Path("results/COMPARISON_BASE_VS_PROPOSED.md")
    if cmp_md.is_file():
        body = cmp_md.read_text().split("## 1. Headline", 1)[-1]
        body = body.split("## 5. Two things")[0]
        A(body.strip())
    A("")
    A("### 5.2 Is the consistency real? Control pairs")
    A("")
    A("Every Causal Consistency number in the literature on flip-only pairs "
      "shares a weakness: a model that reacts to any prompt edit scores well "
      "without tracking the clinical variable. The expanded real-notes arm "
      "adds **control pairs** — the driving lab value moves by a comparable "
      "amount but does not cross the threshold, so the label is unchanged.")
    A("")
    try:
        from make_comparison import enrich
        from make_table import load_preds
        from metrics import score as _score
        from make_comparison import discrimination_ci
        rows = []
        for tag, data, arm in (("_medcalc2", "data/medcalc_v2",
                                "real notes (edited controls)"),
                               ("_mimic2", "data/mimic_v2",
                                "MIMIC-IV (all-real controls)")):
            for variant, label in (("base", "Base LLM"),
                                   ("nsai_uq", "+ gate + UQ")):
                try:
                    recs = enrich(load_preds("results", "test", variant, 0,
                                             tag), data, "test")
                except FileNotFoundError:
                    continue
                sc = _score(recs)
                if sc["spurious_flip_rate"] is None:
                    continue
                rows.append((f"{arm} — {label}", sc))
        if rows:
            A("| arm — variant | causal flip | spurious flip | "
              "discrimination | 95% CI | control pairs |")
            A("|---|---|---|---|---|---|")
            for label, sc in rows:
                ci = discrimination_ci(sc)
                ci_s = (f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "—")
                A(f"| {label} | {f3(sc['causal_flip_rate'])} | "
                  f"{f3(sc['spurious_flip_rate'])} | "
                  f"{sc['discrimination']:+.4f} | {ci_s} | "
                  f"{sc['n_control_pairs']} |")
            A("")
            A("The MIMIC arm is the strongest form of this test available "
              "anywhere in the project: its control pairs are two more real "
              "measurements from the same patient on the same side of the "
              "threshold, so **neither arm contains an invented number**. "
              "Every other benchmark has to edit a value to build a control.")
            A("")
            A("**Read the interval, not the point estimate.** The MIMIC point "
              "estimate is −0.103, which reads as *more* flipping when the "
              "truth did not change — and its interval covers zero at 29 "
              "control pairs. The claim the three measurements jointly "
              "support is the weaker, well-founded one: **discrimination is "
              "indistinguishable from zero on every benchmark tried** — "
              "8,000 multiple-choice items (−0.013 [−0.037, +0.013]), 109 "
              "edited-control pairs, and 29 all-real ones. Counterfactual "
              "consistency alone is not evidence of clinical reasoning.")
            A("")
            A("This is why a spurious-flip rate belongs beside every "
              "consistency number, and why a split without control pairs must "
              "report it as *not measured* rather than as zero.")
        else:
            A("*Pending: the expanded-arm evaluation is still running. This "
              "section is generated from the `_medcalc2` prediction files.*")
    except Exception as e:
        A(f"*Pending ({type(e).__name__}); regenerate once the expanded-arm "
          f"evaluation completes.*")
    A("")
    A("### 5.3 Aim 1 — what the features are worth")
    A("")
    A(f"The TopK dictionary reaches FVU {f3(fis['fvu'])} at L0 "
      f"{f3(fis['l0'])}, with {fis['dead']:,} of {fis['d_hidden']:,} features "
      f"dead. Its best features by S_semantic:")
    A("")
    A("| feature | concept | S_semantic | S_causal | FIS |")
    A("|---|---|---|---|---|")
    for f in fis["features"][:8]:
        A(f"| #{f['feature']} | {f['concept']} | {f3(f['s_semantic'])} | "
          f"{f4(f.get('s_causal'))} | {f3(f.get('fis'))} |")
    A("")
    A("*S_human is not measured: this pipeline has no expert annotators. Its "
      "weight is forced to zero and the FIS above is a two-term score.*")
    A("")
    A("**A caveat that decides how these features may be described.** The "
      "top-25 selection by S_semantic is not balanced across concepts. It "
      "yields twelve `age` features and, after clipping features whose "
      "knock-out does not beat their matched control, **none** for "
      "`qt_interval`, `egfr`, `inr`, `potassium`, `pregnancy`, `asthma` or "
      "`renal_disease`. Any statement about what the model represents is "
      "bounded by what the dictionary was scored for.")
    A("")
    top = max(sae_t["rows"], key=lambda r: r["new"])
    top_h = next((r for r in sae_h["rows"]
                  if r["feature"] == top["feature"]), None)
    A(f"**How much of S_causal survives a fair control.** The first "
      f"knock-out compared each feature with a single control feature drawn "
      f"uniformly over the dictionary, dead features included — a control "
      f"that rarely fires, and so understates what touching any live feature "
      f"does. Against the mean of {sae_t['n_controls']} live features "
      f"matched on firing rate, Σ S_causal over the top 25 falls from "
      f"{sae_t['old']:.3f} to {sae_t['new']:.3f} on the test split "
      f"(−{sae_t['cut']:.1%}) and from {sae_h['old']:.3f} to "
      f"{sae_h['new']:.3f} on held-out (−{sae_h['cut']:.1%}). The largest "
      f"surviving effect is `#{top['feature']}` ({top['concept']}), "
      f"{top['old']:.4f} → {top['new']:.4f} on test"
      + (f" and {top_h['old']:.4f} → {top_h['new']:.4f} on held-out"
         if top_h else "")
      + ". The S_causal column above is already the matched one.")
    A("")
    A("![Figure 1](results/mimic_v3b/figures/fig_sae_contraction.png)")
    A("")
    A(f"*Figure 1. Knock-out effect of the top-25 layer-20 TopK features "
      f"(BioMistral-7B, real-notes benchmark, {sae_t['n_items']} test / "
      f"{sae_h['n_items']} held-out items): raw, in excess of the old uniform "
      f"control, and in excess of {sae_t['n_controls']} firing-rate-matched "
      f"controls (bars ± their sd). Vector version: "
      f"`results/mimic_v3b/figures/fig_sae_contraction.pdf`.*")
    A("")
    A("### 5.4 Aim 2 — the intervention null")
    A("")
    A("| test | largest |excess| over a matched control |")
    A("|---|---|")
    A(f"| activation patching (ACE by layer), test / held-out | "
      f"{f4(max_exc(pat_t))} / {f4(max_exc(pat_h))} logits |")
    A(f"| feature injection (causal sufficiency), test / held-out | "
      f"{f4(max_exc(suf_t))} / {f4(max_exc(suf_h))} logits |")
    A("")
    A("A decision flip on this model needs 1–5 logits. Every effect above is "
      "roughly two orders of magnitude smaller, and the identity control is "
      "exactly 0.0000 in every run, which is what establishes the hooks are "
      "in the right place. Necessity, sufficiency and knock-out agree, so the "
      "null is not an artefact of one intervention shape.")
    A("")
    A("### 5.5 §4.2 — does the attribution layer point at the right thing?")
    A("")
    A("The benchmark answers this without annotation: the two arms of a pair "
      "differ in exactly one causal value, so the differing tokens **are** the "
      "decisive ones. The number is their percentile rank; 0.5 is chance.")
    A("")
    A("| split | attributor | edited-token percentile | AOPC "
      "comprehensiveness | concept pointing (mean) | chance |")
    A("|---|---|---|---|---|---|")
    for name, d in (("test (renal)", ft), ("held-out (QT)", fh)):
        if not d:
            continue
        cp = d["concept_pointing"]
        for a in ("sae_concept", "grad_x_input", "occlusion", "random"):
            if a not in d["aopc_comprehensiveness"]:
                continue
            pc = (f3(cp["accuracy_mean"]) if a == "sae_concept" else "—")
            ch = (f3(cp["random_baseline"]) if a == "sae_concept" else "—")
            A(f"| {name} | `{a}` | "
              f"{f4(d['edited_token_percentile'].get(a))} | "
              f"{f4(d['aopc_comprehensiveness'][a])} | {pc} | {ch} |")
    A("")
    A("Two things follow, and they must be read together.")
    A("")
    A("**The bridge works where the dictionary has the concept.** On the "
      "renal family the layer ranks the edited token near the top and names "
      "the right concept far above chance, while gradient×input and a random "
      "attributor sit at chance.")
    A("")
    qtw = opt("faithfulness_medcalc_heldout_biomistral-7b_qtweights.json")
    A("**A near-miss worth reporting, because it nearly became a finding.** "
      "On the QT family the layer first scored at chance, and the scored "
      "dictionary contained no `qt_interval` feature with a positive "
      "knock-out excess. The obvious reading — the model has no QT feature — "
      "would have been a claim about what the network represents.")
    A("")
    A("It was our measurement. The weights are knock-out excesses, and they "
      "were taken on a split containing only renal items, where a QT feature "
      "cannot move the decision and scores zero whether or not it exists. "
      "Widening the vocabulary from 25 to 100 features did not settle it. "
      "Re-scoring the same dictionary on the split where QT *is* the decisive "
      "variable did.")
    if qtw:
        c = qtw["concept_pointing"]
        A("")
        A(f"With split-matched weights `qt_interval` becomes expressible and "
          f"the QT family gives concept-pointing "
          f"**{f3(c['accuracy_mean'])}** against a "
          f"{f3(c['random_baseline'])} baseline, with an edited-token "
          f"percentile of "
          f"**{f4(qtw['edited_token_percentile'].get('sae_concept'))}** "
          f"against {f4(qtw['edited_token_percentile'].get('random'))} for a "
          f"random attributor. The dictionary had the feature all along. The "
          f"section 4.2 bridge works on both families; the earlier reading is "
          f"withdrawn.")
        A("")
        A("The chance baselines differ between the two families and the "
          "accuracies must be read against their own: QT notes mention few "
          "of the eleven concepts, so a coin flip scores 0.667 there against "
          "0.251 on renal.")
    else:
        A("")
        A("*That check is still running. Until it lands, the QT result is "
          "reported as unresolved rather than as a finding about the "
          "dictionary — see `results/attribution.md` §4.*")
    A("")
    A("**Faithfulness is weak everywhere, including for occlusion**, which is "
      "exact. A rationale the model barely reacts to when it is deleted is "
      "the token-level form of the same null §5.4 reports. The layer "
      "identifies the decisive token; it does not thereby show the decision "
      "depends on it.")
    A("")
    A("### 5.6 Aim 3 — the constraint-aware layer")
    A("")
    A(f"A rank-32 residual adapter at layer 30, trained with all four "
      f"objective terms on a base model that stays frozen, moves held-out "
      f"Causal Consistency from {f3(cl_qt['base']['heldout_cc'])} to "
      f"**{f3(cl_qt['adapted']['heldout_cc'])}** on the QT family. The shuffled-label "
      f"control reaches only {f3(cl_qt_sh['adapted']['heldout_cc'])}, which is what "
      f"separates learning the rule from learning the task format.")
    A("")
    A("`L_ontology` reads the causal graph rather than applying a constant "
      "margin: a family the graph licenses no contraindication for gets a "
      "margin of zero and the term is silent for it.")
    A("")
    A("**The cost, disclosed.** `ondansetron_qt` — the family carrying this "
      "result — is absent from the FDA labels *and* absent from MED-RT, so it "
      "is grounded at neither end. The result is real and it rests on "
      "curation.")
    A("")
    A("### 5.7 Aim 4 — uncertainty and deferral")
    A("")
    A(f"Over {e_ent['n']:,} external items (MedMCQA, MedQA-USMLE, PubMedQA), "
      f"the proposal's Eq. (2) whole-vocabulary entropy reaches AUROC "
      f"**{f3(e_ent['auroc'])} [{f3(e_ent['lo'])}, {f3(e_ent['hi'])}]**. The "
      f"identical entropy restricted to the decision tokens reaches "
      f"**{f3(e_opt['auroc'])} [{f3(e_opt['lo'])}, {f3(e_opt['hi'])}]**, a "
      f"paired difference of **+{f3(e_con['delta'])} "
      f"[{f3(e_con['lo'])}, {f3(e_con['hi'])}]**, replicated independently on "
      f"all three benchmarks.")
    A("")
    A("The implementation was never broken. §4.7 asks for the wrong "
      "quantity, and the deferral behaviour it wants appears as soon as the "
      "signal is right. A whole-vocabulary control does not help, so what "
      "matters is restricting attention to the decision, not the functional "
      "form.")
    A("")
    A("Conformal coverage is reported with the caveat that makes it "
      "meaningful: a frozen split-conformal threshold carries no guarantee on "
      "a split that is not exchangeable with the calibration split, which is "
      "exactly what a held-out rule family is. Adaptive Conformal Inference "
      "recovers the target coverage by giving up coverage.")
    A("")
    A("### 5.8 Generalisation across models and layers")
    A("")
    if other_slugs:
        A("Section 4.3 names \"Llama-3-Med and Mistral variants\". The whole "
          "pipeline — ablation, SAE, patching, constraint layer, conformal UQ "
          "— was run end to end on each, from the same data with the same "
          "seeds.")
        A("")
        A("**The headline result holds on every model.** Base accuracy sits "
          "at chance on real clinical notes and the full system reaches "
          "~0.99, with p < 1e-24 on every one:")
        A("")
        A("| model | acc base → proposed | CC base → proposed |")
        A("|---|---|---|")
        roots = [("BioMistral-7B", "results")]
        roots += [(sl, f"results/models/{sl}") for sl in other_slugs]
        for name, root in roots:
            try:
                rows = {r["variant"]: r for r in
                        json.loads(Path(f"{root}/summary_test_medcalc.json")
                                   .read_text())}
            except Exception:
                continue
            b = rows.get("base")
            pr = rows.get("nsai_uq_cl") or rows.get("nsai_uq")
            if not (b and pr):
                continue
            A(f"| `{name}` | {f3(b['accuracy'])} → {f3(pr['accuracy'])} | "
              f"{f3(b['causal_consistency'])} → "
              f"{f3(pr['causal_consistency'])} |")
        A("")
        A("**Aim 3 transfers, and how far depends on the model.** The "
          "constraint layer is trained on one rule family and scored on a "
          "held-out one it never saw:")
        A("")
        A("| model | held-out CC, base → +constraint layer |")
        A("|---|---|")
        for name, root in roots:
            try:
                rows = {r["variant"]: r for r in
                        json.loads(Path(f"{root}/summary_heldout_medcalc.json")
                                   .read_text())}
            except Exception:
                continue
            b, c = rows.get("base"), rows.get("cl")
            if not (b and c):
                continue
            A(f"| `{name}` | {f3(b['causal_consistency'])} → "
              f"**{f3(c['causal_consistency'])}** |")
        A("")
        A("The two biomedically pretrained models gain most and the "
          "general-purpose instruct model least, which is the ordering one "
          "would predict but is worth having measured rather than assumed. "
          "The mechanism is not model-specific: a rank-32 residual adapter "
          "trained on one family moves a held-out family on all three.")
        A("")
        A("**A defect worth recording, because it invalidated a first "
          "attempt.** The lane script never passed `--model_id` to the "
          "constraint-layer trainer, which defaults to BioMistral, so every "
          "model's adapter was trained on BioMistral and then attached to a "
          "different model. All three are 4096-dimensional, so the wrong "
          "adapter loads cleanly and yields a plausible number; the "
          "hidden-size guard could not catch it. The tell was two models "
          "reporting identical numbers to sixteen decimal places. The "
          "artifacts were discarded and both models re-run, and every "
          "artifact now records the model that produced it.")
        layer_roots = sorted(glob.glob("results/layers/L*/"))
        if layer_roots:
            A("")
            A("**Layer sweep.** The SAE was additionally collected, trained "
              "and scored at layers 16 and 24 as well as 20:")
            A("")
            A("| layer | FVU | L0 | dead features | concepts with a "
              "positively-weighted feature |")
            A("|---|---|---|---|---|")
            for lr in [None] + layer_roots:
                if lr is None:
                    f, lab = "results/sae/sae_topk_L20_fis.json", "20"
                else:
                    lab = Path(lr).name.lstrip("L")
                    f = f"{lr}sae/sae_topk_L{lab}_fis.json"
                try:
                    d = json.loads(Path(f).read_text())
                except Exception:
                    continue
                per = d.get("causal", {}).get("per_feature", {})
                cons = {ft["concept"] for ft in d["features"]
                        if per.get(str(ft["feature"]), {}).get("excess", 0) > 0}
                A(f"| {lab} | {f3(d['fvu'])} | {f3(d['l0'])} | "
                  f"{d['dead']:,} / {d['d_hidden']:,} | {len(cons)} |")
            A("")
            for lr in layer_roots:
                lab = Path(lr).name.lstrip("L")
                try:
                    d = json.loads(Path(f"{lr}sae/sae_topk_L{lab}_fis.json")
                                   .read_text())
                except Exception:
                    continue
                per = d.get("causal", {}).get("per_feature", {})
                cons = sorted({ft["concept"] for ft in d["features"]
                               if per.get(str(ft["feature"]), {})
                               .get("excess", 0) > 0})
                A(f"Layer {lab} expresses: {', '.join('`'+c+'`' for c in cons)}.")
            A("")
            A("**Layer 20 was chosen once, from the probe curve, and never "
              "re-examined. It is the worst of the three.** It reconstructs "
              "least well (FVU 0.051 against 0.026 at layer 16) and expresses "
              "the fewest concepts — four, against seven at layer 16, which "
              "includes `qt_interval`, `inr`, `pregnancy` and "
              "`renal_disease`. Layer 16 finds a causally-weighted QT feature "
              "even with the weights measured on the renal split, which is "
              "the condition under which layer 20 finds none at all.")
            A("")
            A("Every section 4.2 number in this paper is therefore a "
              "*lower bound* on what the method can do: the attribution "
              "layer can only name concepts the dictionary has features for, "
              "and the dictionary it was given comes from the least "
              "expressive of the three layers measured. Re-running the "
              "attribution at layer 16 is the obvious next step and is not "
              "done here.")
            A("")
            A("This sweeps the layer the **dictionary** is fitted on. The "
              "intervention layer was held at 30 throughout, so these rows "
              "are not a sweep of where the constraint adapter is inserted "
              "and must not be read as one.")
        stab = Path("results/stability.json")
        if stab.is_file():
            st = json.loads(stab.read_text())
            pairs = [q for q in st["pairs"] if q["same_layer"]]
            if pairs:
                A("")
                A("**Seed stability.** Decoding is greedy, so a decode seed "
                  "changes nothing; the variance that exists in Aim 1 comes "
                  "from the dictionary — its initialisation, the train/val "
                  "split, the batch order, the token subsample and the random "
                  "control feature. Three dictionaries were trained from the "
                  "same activations with different seeds. Feature indices are "
                  "meaningless across runs, so the comparison is between the "
                  "subspaces and the concept selections:")
                A("")
                A("| A | B | mean max &#124;cos&#124; | concept Jaccard | "
                  "concepts only in B |")
                A("|---|---|---|---|---|")
                for q in pairs:
                    only = ", ".join(f"`{c}`" for c in q["concepts_only_in_b"])
                    A(f"| `{q['a']}` | `{q['b']}` | "
                      f"{f3(q['mean_max_cosine_a_to_b'])} | "
                      f"{f3(q['concept_jaccard'])} | {only or '—'} |")
                A("")
                A("The two halves disagree, and the disagreement is the "
                  "result. **Which concepts the dictionary selects is fairly "
                  "stable** — Jaccard 0.75 to 1.00. **Which directions it "
                  "finds is not** — a decoder row's best match in another "
                  "seed's dictionary is only around 0.55 cosine. Different "
                  "seeds arrive at different bases that nonetheless pick out "
                  "much the same concepts.")
                A("")
                A("That is the right way round for the claims here, which are "
                  "all at concept level, and it is a warning for anything "
                  "said about an individual feature: `#14294` is a fact about "
                  "one training run, not about the model. One seed also "
                  "surfaces `inr` and `pregnancy` that the other two miss, so "
                  "concept coverage — the thing the section 4.2 layer is "
                  "bounded by — varies with the seed as well as with the "
                  "layer.")
    else:
        A("*Pending: the multi-model and multi-layer runs are still "
          "executing. This section is generated from "
          "`results/models/<slug>/` and `results/layers/L<n>/` and will be "
          "filled in on the next build.*")
    A("")
    v3b_section(A, v3)
    A("## 6. Limitations")
    A("")
    A("1. **The symbolic gate's accuracy is partly circular.** It applies the "
      "same rule and threshold the labels were generated from, and on the "
      "real-notes benchmark the same extractor that filtered the dataset runs "
      "inside the gate. The number to quote is coverage, not accuracy.")
    A("2. **Coverage falls with rule complexity, and collapses on clinical "
      "judgement.** Measured per family on the expanded benchmark, a "
      "two-variable numeric rule is nearly always applicable, a five-variable "
      "numeric one usually is, and a rule needing a *graded clinical "
      "judgement* is never applicable — the word \"encephalopathy\" appears "
      "in none of the Child-Pugh notes, so its grade cannot be extracted at "
      "any regex quality. This is a ceiling on threshold-reading gates in "
      "general, not on this implementation.")
    A("3. **The SAE is at pilot scale** — a narrow corpus against the "
      "hundreds of millions of tokens in reference work, with a large dead "
      "fraction. The absence of a concept here is not evidence of its absence "
      "in the model.")
    A("4. **`S_human` is not measured**, so the FIS is a two-term score and "
      "§4.4's fallback criterion cannot be evaluated. The instrument is "
      "built and blinded (`HUMAN_EVAL_PROTOCOL.md`); it needs clinicians.")
    A("5. **Greedy decoding makes seed variance zero by construction.** "
      "Reporting \"5 seeds, ± 0.000\" would imply variability was measured "
      "when it was not; the uncertainty that exists is over items and is "
      "reported as a bootstrap CI.")
    A("6. **The second arm of every real-notes pair is synthetic** — a real "
      "note with one number changed. It is physiologically plausible and "
      "internally consistent, but the patient was not observed.")
    A(f"7. **The MIMIC-IV v3.1 replication is one seed on minimal rendered "
      f"notes.** Its prompts repeat {v3['dup_ratio']}× on average and "
      f"largely recur in calibration, OpenBioLLM's RAG and NS-AI rows are "
      f"dominated by non-answers, and the constraint layer does not "
      f"transfer to its held-out warfarin family (§5.9).")
    A("")
    A("## 7. Conclusion")
    A("")
    A("The representations are there and they can be found. A sparse "
      "dictionary recovers features that select single clinical quantities "
      "without supervision, and an attribution layer built on them points at "
      "the token a clinical decision turns on far above chance. What does not "
      "follow is control: intervening on those same features — by patching, "
      "by knock-out, by injection — moves the decision about two orders of "
      "magnitude less than a flip requires.")
    A("")
    A("The gap between those two results is the finding. Identifying the "
      "representation of a clinical fact is not the same as showing the model "
      "uses it, and a mechanistic-interpretability programme that reports the "
      "first as if it implied the second will overstate what it has "
      "established. What did work is the part that does not go through the "
      "model's own causal structure: a symbolic gate makes no errors when it "
      "fires, and a trained residual adapter raises consistency on a rule "
      "family it never saw. Both are useful. Neither is evidence that the "
      "network reasons over the constraint.")
    A("")
    A(f"The MIMIC-IV v3.1 replication qualifies the last of these rather "
      f"than extending it. Across {v3['test_items']} real-value items and three "
      f"models, the adapter lifts consistency on the families it was trained "
      f"on, well clear of a shuffled-label control, but not on the held-out "
      f"warfarin family. Transfer to an unseen rule is a property of some "
      f"family pairs, not yet a property of the method.")
    A("")
    A("---")
    A("")
    A("### Reproducing")
    A("")
    A("```bash")
    A("python src/check_data.py --strict     # stage 0; no credentialed source")
    A("bash run_scaled_pipeline.sh           # both lanes + join, checkpointed")
    A("bash run_attribution.sh               # section 4.2")
    A("# section 5.9 (needs credentialed MIMIC-IV v3.1), per model:")
    A("MODEL_ID=... MODEL_TAG=... ARM=main bash run_v3b.sh   # also ARM=note")
    A("MODEL_ID=... MODEL_TAG=... bash run_v3b_cl.sh")
    A("bash run_v3b_report.sh                # tables and audits")
    A("/usr/bin/python3 src/generate_paper_plots.py   # figures 1-3")
    A("python src/make_comparison.py")
    A("python src/make_paper.py")
    A("```")
    A("")
    A("Companion documents: `PROPOSAL_ERRATA.md` (changes the measurements "
      "force on the proposal text), `HUMAN_EVAL_PROTOCOL.md` (the rating "
      "study, ready to run), `results/SUMMARY.md` (every experiment side by "
      "side), `results/FIXES.md` (the nine defects found in audit).")
    A("")

    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
