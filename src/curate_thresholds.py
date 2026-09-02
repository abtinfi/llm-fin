"""
Source every safety threshold from an FDA label, or say plainly that it cannot be.

WHY THIS FILE EXISTS
--------------------
`src/rules.py` carries 10 thresholds written from memory. `hardening.py` says so
at its own point of use: "Thresholds are UNCHANGED from src/rules.py and remain
INTERIM ... Do not present them as sourced." `data/curation_worksheet.csv` was
created to fix that and has sat with `constraint_value` empty on all 145 rows.

This fills it from `data/openfda_raw.jsonl`, searching the WHOLE label rather
than the three sections the worksheet sampled.

WHAT THIS TOOL IS NOT
---------------------
It is not a curator. It proposes a value and attaches the sentence and section
it came from; `curator` is written as `auto:curate_thresholds.py` so no row can
be mistaken for a human decision. The distinction that matters is not
"did a number appear" but "is it the SAME CONSTRUCT as the rule":

  - metformin: the label's "eGFR below 30" IS the contraindication.  -> attested
  - aspirin:   the label's "children under 12 years: consult a doctor" is a
               DOSING instruction. The rule is about Reye's syndrome (<16),
               which this label never mentions.                      -> NOT attested
  - spironolactone: the label's "serum potassium <=5.0 mEq/L" is an INITIATION
               criterion, not a contraindication ceiling.            -> NOT attested

A number that is present but means something else is more dangerous than no
number at all, so those rows are recorded as `construct_mismatch` and left for
a human, never silently written into `constraint_value`.
"""

import argparse
import csv
import json
import re
from pathlib import Path

# One entry per rule family in src/rules.py. `pattern` must match the sentence
# that states the CONTRAINDICATION, not merely one containing the number.
SPECS = {
    "metformin_renal": dict(
        drug="metformin", var="egfr", op="<", unit="mL/min/1.73m2",
        rules_value=30,
        pattern=r"(?:eGFR|estimated glomerular filtration rate)[^.]{0,60}?"
                r"(?:below|less than|<)\s*(\d+(?:\.\d+)?)",
        sections=("contraindications", "warnings_and_cautions",
                  "use_in_specific_populations"),
        kind="numeric"),
    "warfarin_inr": dict(
        drug="warfarin", var="inr", op=">", unit="",
        rules_value=4.0,
        pattern=r"\bINR\s*(?:of\s*)?(?:greater than|above|>)\s*(\d+(?:\.\d+)?)",
        sections=("warnings_and_cautions", "dosage_and_administration"),
        kind="numeric"),
    "acei_pregnancy": dict(
        drug="lisinopril", var="pregnant", op="==", unit="",
        rules_value=True,
        pattern=r"(?:discontinue|contraindicated|fetal toxicity)[^.]{0,120}?pregnan",
        sections=("boxed_warning", "contraindications",
                  "use_in_specific_populations", "warnings"),
        kind="qualitative"),
    "betablocker_asthma": dict(
        drug="propranolol", var="severe_asthma", op="==", unit="",
        rules_value=True,
        pattern=r"contraindicated[^.]{0,200}?(?:asthma|bronchospas)",
        sections=("contraindications", "warnings"),
        kind="qualitative"),
    "statin_macrolide": dict(
        drug="simvastatin", var="clarithromycin", op="==", unit="",
        rules_value=True,
        pattern=r"contraindicated[^.]{0,240}?(?:clarithromycin|macrolide)",
        sections=("contraindications", "drug_interactions"),
        kind="qualitative"),
    # ---- families whose threshold this label set does NOT attest -------------
    "nsaid_renal": dict(
        drug="ibuprofen", var="egfr", op="<", unit="mL/min/1.73m2",
        rules_value=30,
        pattern=r"(?:eGFR|glomerular filtration|creatinine clearance)[^.]{0,60}?"
                r"(?:below|less than|<)\s*(\d+(?:\.\d+)?)",
        sections=(), kind="numeric"),
    "aspirin_reye": dict(
        drug="aspirin", var="age", op="<", unit="years",
        rules_value=16,
        pattern=r"Reye[^.]{0,120}?(\d{1,2})\s*years",
        sections=(), kind="numeric",
        known_trap="The label's '<12 years' is an OTC DOSING instruction, not "
                   "the Reye's-syndrome contraindication the rule encodes."),
    "spironolactone_hyperkalaemia": dict(
        drug="spironolactone", var="potassium", op=">", unit="mEq/L",
        rules_value=5.5,
        pattern=r"contraindicated[^.]{0,160}?potassium[^.]{0,60}?"
                r"(\d+(?:\.\d+)?)\s*(?:mEq|mmol)",
        sections=(), kind="numeric",
        known_trap="The label's '<=5.0 mEq/L' is an INITIATION criterion for "
                   "heart failure dosing, not a contraindication ceiling."),
    "nitrofurantoin_renal": dict(
        drug="nitrofurantoin", var="egfr", op="<", unit="mL/min/1.73m2",
        rules_value=30,
        pattern=r"(?:creatinine clearance|CrCl|glomerular)[^.]{0,60}?"
                r"(?:below|less than|<)\s*(\d+(?:\.\d+)?)",
        sections=(), kind="numeric"),
    "ondansetron_qt": dict(
        drug="ondansetron", var="qtc", op=">", unit="ms",
        rules_value=500,
        pattern=r"QT[c]?[^.]{0,120}?(\d{3})\s*(?:msec|ms)",
        sections=(), kind="numeric"),
}


def label_sections(label):
    for sec, txt in label.items():
        if isinstance(txt, list):
            yield sec, " ".join(str(x) for x in txt)
        elif isinstance(txt, str):
            yield sec, txt


def find(label, spec):
    """Return (value, section, sentence) for the first confident match."""
    pat = re.compile(spec["pattern"], re.I)
    prefer = spec["sections"]
    order = sorted(label_sections(label),
                   key=lambda kv: (prefer.index(kv[0]) if kv[0] in prefer
                                   else len(prefer)))
    for sec, text in order:
        if prefer and sec not in prefer:
            continue
        m = pat.search(text)
        if m:
            val = m.group(1) if m.groups() else True
            s = max(0, m.start() - 120)
            return val, sec, " ".join(text[s:m.end() + 120].split())
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/openfda_raw.jsonl")
    ap.add_argument("--worksheet", default="data/curation_worksheet.csv")
    ap.add_argument("--out", default="data/curation_worksheet.csv")
    ap.add_argument("--report", default="results/threshold_provenance.md")
    args = ap.parse_args()

    labels = {json.loads(l)["query_drug"]: json.loads(l)["label"]
              for l in Path(args.labels).open()}

    findings = {}
    for fam, spec in SPECS.items():
        lab = labels.get(spec["drug"])
        if lab is None:
            findings[fam] = dict(spec=spec, status="no_label", value=None,
                                 section=None, sentence=None)
            continue
        val, sec, sent = find(lab, spec)
        if val is None:
            status = "construct_mismatch" if spec.get("known_trap") else "absent"
        elif spec["kind"] == "qualitative":
            status = "attested_qualitative"
        elif float(val) == float(spec["rules_value"]):
            status = "attested_exact"
        else:
            status = "attested_different_value"
        findings[fam] = dict(spec=spec, status=status, value=val,
                             section=sec, sentence=sent)

    # ---- write the worksheet ------------------------------------------------
    rows = list(csv.DictReader(Path(args.worksheet).open()))
    by_drug = {}
    for fam, f in findings.items():
        if f["status"].startswith("attested"):
            by_drug.setdefault(f["spec"]["drug"], []).append((fam, f))

    filled = 0
    for r in rows:
        cands = by_drug.get(r["query_drug"], [])
        for fam, f in cands:
            if f["section"] and f["section"] != r["source_section"]:
                continue
            if f["sentence"] and r["source_text"][:60] not in f["sentence"] \
                    and f["sentence"][:60] not in r["source_text"]:
                continue
            r["constraint_var"] = f["spec"]["var"]
            r["constraint_op"] = f["spec"]["op"]
            r["constraint_value"] = ("" if f["value"] is True
                                     else str(f["value"]))
            r["unit"] = f["spec"]["unit"]
            r["curator"] = "auto:curate_thresholds.py"
            r["curator_note"] = (f"{fam}: {f['status']} in [{f['section']}]. "
                                 f"NOT human-verified.")
            filled += 1
            break

    with Path(args.out).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # ---- write the provenance report ---------------------------------------
    lines = ["# Where each safety threshold actually comes from", "",
             "Generated by `src/curate_thresholds.py` from "
             "`data/openfda_raw.jsonl` (10 FDA labels). **No row here is "
             "human-verified**; `curator` reads `auto:` throughout.", "",
             "| family | rules.py | label says | status | section |",
             "|---|---|---|---|---|"]
    for fam, f in findings.items():
        v = f["value"]
        lines.append(f"| `{fam}` | {f['spec']['rules_value']} | "
                     f"{'—' if v is None else (v if v is not True else 'qualitative')} | "
                     f"**{f['status']}** | {f['section'] or '—'} |")
    lines += ["", "## The sentence each attested value rests on", ""]
    for fam, f in findings.items():
        if f["sentence"]:
            lines += [f"**`{fam}`** — [{f['section']}]", "",
                      f"> {f['sentence']}", ""]
    traps = [(fam, f) for fam, f in findings.items()
             if f["spec"].get("known_trap")]
    if traps:
        lines += ["## Numbers that are present but mean something else", "",
                  "These are the dangerous ones. A number appears in the label "
                  "and does NOT encode the rule, so it is recorded here and "
                  "never written into `constraint_value`.", ""]
        for fam, f in traps:
            lines += [f"- **`{fam}`** — {f['spec']['known_trap']}"]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n")

    print(f"worksheet rows filled: {filled}/{len(rows)}")
    for fam, f in findings.items():
        print(f"  {fam:32s} {f['status']:26s} {f['value']}")
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
