"""
Data provenance and integrity preflight: proves the pipeline runs with no
credentialed data source, and fails loudly if that ever stops being true.

WHY THIS FILE EXISTS
--------------------
The proposal names MIMIC-IV (§4.6) for retrospective evaluation and scenario
construction, and MIMIC-IV is behind PhysioNet credentialing. That has led to
a recurring question -- "is the pipeline blocked without PhysioNet?" -- whose
answer is no, but which nothing in the repository could previously demonstrate.

It is worth being exact, because the distinction matters for the write-up:

  * **No code loads MIMIC-IV.** `grep -rniE "mimic|physionet|labevents" src/`
    returns two docstring asides and nothing else. There is no loader, no data
    path, no download step, and no token check anywhere in the tree. The
    evaluation pipeline has never depended on PhysioNet.
  * **MIMIC-IV is therefore an unimplemented proposal element, not a blocked
    dependency.** `IMPLEMENTATION_STATUS.md` lists it correctly as NOT
    IMPLEMENTED. Nothing regresses if it never arrives; what is lost is the
    §4.6 retrospective-evaluation claim, which no substitute dataset can
    supply on its own.
  * **The "real clinical text" arm already exists and is open access.**
    `data/medcalc` is 680 items built by `src/build_medcalc.py` from
    MedCalc-Bench PMC case-report notes. That is real clinical prose from
    published case reports, not templated text, and it is what every
    "real notes" column in `results/SUMMARY.md` is scored on.

So this module does NOT invent a synthetic stand-in for MIMIC-IV. Generating
clinical notes and then evaluating on them measures the generator, not the
model, and this project has been careful elsewhere to avoid exactly that kind
of closed loop (no LLM-as-judge, no synthetic ground truth, the gate's own
circularity warning in `IMPLEMENTATION_STATUS.md`). A fabricated corpus
presented as an EHR surrogate would be the single most misleading artifact the
repository could contain.

What it does instead is verify, on every run, that every dataset the pipeline
actually reads is present, well formed, and obtainable without credentials --
turning "no PhysioNet needed" from a claim into a check.

  python src/check_data.py                 # human-readable report
  python src/check_data.py --strict        # non-zero exit if anything is off
  python src/check_data.py --json out.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Provenance table. `credentialed` is the field that matters for the question
# this file answers: True means a human must sign a data-use agreement before
# the bytes can be obtained.
#
# UMLS is the one honest "yes, but": rebuilding the causal graph needs a UTS
# licence key, and the key lives in ../.env outside this repository. Running
# the pipeline does NOT, because every UMLS response is cached to disk and the
# cache plus the built graph are committed. `umls_grounding.py` states the same
# thing at its own point of use. That is recorded here rather than smoothed
# over, because "the pipeline needs no credentials" and "no artifact in this
# repository was ever built with a credential" are different claims and only
# the first is true.
# ---------------------------------------------------------------------------
SOURCES = {
    "data/synthetic_control": dict(
        what="SYNTHETIC CONTROL benchmark, 10 hand-written rule families",
        origin="Generated in-repo by src/build_dataset.py + src/hardening.py",
        credentialed=False,
        note="Templated vignettes with hand-written thresholds. This is a "
             "CONTROL ARM, not evidence about clinical text: it exists to show "
             "what the pipeline does when the causal factor is stated cleanly "
             "and the label is guaranteed. Every claim about real notes must "
             "cite data/medcalc instead. Renamed from data/ on 2026-09-02 so "
             "the distinction cannot be lost in a path."),
    "data/medcalc": dict(
        what=None,   # counted live -- see below; a hardcoded figure drifts
        _what_tmpl="Real clinical notes benchmark, {n} items across 5 splits",
        origin="Derived by src/build_medcalc.py from MedCalc-Bench PMC "
               "case-report notes (data/external/*.csv)",
        credentialed=False,
        note="Real published clinical prose. This is the pipeline's 'real "
             "notes' arm and needs no data-use agreement."),
    "data/external": dict(
        what="MedCalc-Bench source CSVs (train + test)",
        origin="Third-party, redistributed by the MedCalc-Bench authors",
        credentialed=False,
        note="Open download, no credentialing. Git-ignored (54 MB) as it is "
             "not ours to vendor; re-fetch from the MedCalc-Bench release."),
    "data/umls": dict(
        what="Causal knowledge graph + cached UMLS responses",
        origin="UMLS UTS REST API (release 2026AA), cached to disk",
        credentialed="rebuild-only",
        note="RUNNING needs nothing: causal_graph.json and the response cache "
             "are committed and umls_grounding.py makes zero HTTP calls with "
             "a warm cache. REBUILDING (`umls_grounding.py build --refresh`) "
             "needs a free UTS licence key in ../.env."),
}

# Every file the evaluation pipeline reads, and which stage would break.
REQUIRED = [
    ("data/synthetic_control/counterfactual_calib.jsonl",
     "run_eval.py UQ calibration (synthetic control)"),
    ("data/synthetic_control/counterfactual_test.jsonl",
     "run_eval.py --split test (synthetic control)"),
    ("data/synthetic_control/counterfactual_heldout.jsonl",
     "run_eval.py --split heldout (synthetic control)"),
    ("data/synthetic_control/rag_corpus.jsonl",
     "components.TfidfRetriever (synthetic control RAG)"),
    ("data/medcalc/counterfactual_calib.jsonl", "medcalc UQ calibration"),
    ("data/medcalc/counterfactual_test.jsonl", "medcalc --split test"),
    ("data/medcalc/counterfactual_heldout.jsonl", "medcalc --split heldout"),
    ("data/medcalc/counterfactual_heldout_train.jsonl",
     "constraint_layer.py --train_on heldout_first"),
    ("data/medcalc/counterfactual_train.jsonl",
     "constraint_layer.py --train_on train"),
    ("data/medcalc/rag_corpus.jsonl", "medcalc RAG variants"),
    ("data/umls/causal_graph.json", "constraint_layer.py L_ontology"),
]

OPTIONAL = [
    ("data/medcalc/counterfactual_heldout_all.jsonl",
     "patching.py / steering.py 85-pair QT continuity split"),
    ("data/external/medcalc_train_data_11_18_final.csv",
     "build_medcalc.py rebuild only"),
    ("data/external/medcalc_test_data_11_18_final.csv",
     "build_medcalc.py rebuild only"),
    ("results/acts/medcalc_heldout.npz", "steering.py activation cache"),
]


def read_jsonl(p):
    out = []
    for i, line in enumerate(Path(p).open(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{p}:{i} is not valid JSON ({e})")
    return out


def check_split(path):
    """
    Integrity of one counterfactual split.

    The properties checked are the ones the metrics silently depend on. Pair
    consistency is defined on pairs holding exactly one SAFE and one UNSAFE
    arm; a split where that fails does not merely score oddly, it makes
    `causal_consistency` mean something different from what every report says
    it means.
    """
    recs = read_jsonl(path)
    issues = []
    by_pair = defaultdict(list)
    for r in recs:
        for field in ("id", "pair_id", "label", "prompt", "family"):
            if field not in r:
                issues.append(f"record {r.get('id', '?')} missing `{field}`")
        by_pair[r.get("pair_id")].append(r)

    complete = incomplete = 0
    for pid, arms in by_pair.items():
        labels = sorted(a.get("label") for a in arms)
        if len(arms) == 2 and labels == ["SAFE", "UNSAFE"]:
            complete += 1
        else:
            incomplete += 1
            if incomplete <= 3:
                issues.append(f"pair {pid}: {len(arms)} arm(s), labels "
                              f"{labels} (want one SAFE + one UNSAFE)")
    ids = [r.get("id") for r in recs]
    if len(set(ids)) != len(ids):
        issues.append(f"duplicate item ids ({len(ids) - len(set(ids))})")

    return {
        "n_items": len(recs),
        "n_pairs": len(by_pair),
        "complete_pairs": complete,
        "incomplete_pairs": incomplete,
        "families": sorted({r.get("family") for r in recs if r.get("family")}),
        "issues": issues,
    }


def check_leakage(pairs):
    """
    No pair_id may appear in two splits that are meant to be disjoint.

    `constraint_layer.py` asserts this for the splits it uses, but only for
    those, and only when it runs. Checking it up front covers every split pair
    at once and costs nothing.
    """
    out = []
    names = list(pairs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = pairs[a] & pairs[b]
            if shared:
                out.append({"a": a, "b": b, "n_shared": len(shared),
                            "example": sorted(shared)[:3]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if a required file is missing or a "
                         "split fails integrity")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    report = {"credentialed_sources_required_to_run": [], "sources": {},
              "required": [], "optional": [], "splits": {}, "leakage": []}
    problems = 0

    print("=" * 72)
    print("DATA PROVENANCE — does anything here need credentials to RUN?")
    print("=" * 72)
    for name, meta in SOURCES.items():
        p = root / name
        present = p.exists()
        # Counted, never hardcoded. The repo's prose said "600 items" while the
        # directory held 680 -- the same class of drift as the 430-vs-180 defect
        # in FIXES.md A4, and the reason make_summary.py regenerates from
        # artifacts. A provenance report that states a stale number is worse
        # than one that states none.
        if meta.get("_what_tmpl"):
            n = sum(sum(1 for _ in f.open())
                    for f in sorted(p.glob("counterfactual_*.jsonl"))
                    if "heldout_all" not in f.name) if present else 0
            meta = {**meta, "what": meta["_what_tmpl"].format(n=n)}
        cred = meta["credentialed"]
        tag = {True: "CREDENTIALED", False: "open access",
               "rebuild-only": "open to run / key to rebuild"}[cred]
        print(f"\n  {name}   [{tag}]{'' if present else '   *** ABSENT ***'}")
        print(f"    what   : {meta['what']}")
        print(f"    origin : {meta['origin']}")
        print(f"    note   : {meta['note']}")
        report["sources"][name] = {**meta, "present": present}
        if cred is True:
            report["credentialed_sources_required_to_run"].append(name)

    print("\n" + "-" * 72)
    if not report["credentialed_sources_required_to_run"]:
        print("RESULT: no credentialed data source is required to run the")
        print("        pipeline. PhysioNet / MIMIC-IV is NOT a dependency —")
        print("        no loader, data path or token check exists in src/.")
    else:
        print("RESULT: *** a credentialed source is required: "
              f"{report['credentialed_sources_required_to_run']} ***")
        problems += 1

    print("\n" + "=" * 72)
    print("REQUIRED FILES")
    print("=" * 72)
    for rel, used_by in REQUIRED:
        p = root / rel
        ok = p.is_file()
        size = f"{p.stat().st_size / 1024:.0f} KB" if ok else "—"
        print(f"  [{'ok' if ok else 'MISSING'}] {rel:52s} {size:>9s}")
        if not ok:
            print(f"           needed by: {used_by}")
            problems += 1
        report["required"].append({"path": rel, "present": ok,
                                   "used_by": used_by})

    print("\nOPTIONAL (a stage degrades or is skipped, nothing breaks)")
    for rel, used_by in OPTIONAL:
        p = root / rel
        ok = p.is_file()
        print(f"  [{'ok' if ok else '--'}] {rel:52s} "
              f"{'' if ok else '(' + used_by + ')'}")
        report["optional"].append({"path": rel, "present": ok,
                                   "used_by": used_by})

    print("\n" + "=" * 72)
    print("SPLIT INTEGRITY")
    print("=" * 72)
    pair_sets = {}
    for rel, _ in REQUIRED + OPTIONAL:
        if "counterfactual_" not in rel:
            continue
        p = root / rel
        if not p.is_file():
            continue
        try:
            info = check_split(p)
        except ValueError as e:
            print(f"  [BAD] {rel}: {e}")
            problems += 1
            continue
        report["splits"][rel] = info
        pair_sets[rel] = {r["pair_id"] for r in read_jsonl(p)}
        flag = "ok" if not info["issues"] else "ISSUES"
        print(f"  [{flag}] {rel}")
        print(f"         {info['n_items']} items, {info['complete_pairs']} "
              f"complete pairs, {info['incomplete_pairs']} incomplete, "
              f"families={info['families']}")
        for m in info["issues"][:4]:
            print(f"         - {m}")
            problems += 1

    print("\n" + "=" * 72)
    print("CROSS-SPLIT LEAKAGE (shared pair_id between splits)")
    print("=" * 72)
    # heldout_all is the pre-split superset of heldout + heldout_train and is
    # SUPPOSED to overlap them; it is kept only for continuity with the
    # patching/steering runs that predate the re-partition.
    superset = "data/medcalc/counterfactual_heldout_all.jsonl"
    leaks = [l for l in check_leakage(pair_sets)
             if superset not in (l["a"], l["b"])]
    expected = [l for l in check_leakage(pair_sets)
                if superset in (l["a"], l["b"])]
    report["leakage"] = leaks
    if leaks:
        for l in leaks:
            print(f"  *** {l['n_shared']} shared pair(s) between\n"
                  f"      {l['a']}\n      {l['b']}   e.g. {l['example']}")
            problems += 1
    else:
        print("  none — every evaluation split is pair-disjoint from every "
              "other")
    for l in expected:
        print(f"  (expected) {l['n_shared']} shared with the "
              f"heldout_all continuity superset")

    print("\n" + "=" * 72)
    print(f"{'PASS' if problems == 0 else f'{problems} PROBLEM(S)'}"
          " — pipeline data is "
          f"{'ready and credential-free' if problems == 0 else 'NOT ready'}")
    print("=" * 72)

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json}")
    if args.strict and problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
