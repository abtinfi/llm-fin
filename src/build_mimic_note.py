"""
The MIMIC-IV v3.1 arm with REAL discharge prose as context: data/mimic_v3_note.

WHAT IT ADDS, AND WHAT IT KEEPS
--------------------------------
Every other MIMIC arm renders its note from structured fields ("A 67-year-old
woman has a serum creatinine of 2.10 mg/dL"), because the Demo has no free
text. That is the limitation IMPLEMENTATION_STATUS records against the arm.

Here the prompt is a real excerpt of the patient's own discharge summary
(MIMIC-IV-Note v2.2), followed by the same rendered sentence the v3 arm uses.
The excerpt is IDENTICAL in both arms of a pair, so the two prompts still
differ in exactly one number and both numbers are still real measurements --
the minimal-pair property of data/mimic_v3 survives. What changes is that the
stated value now sits inside real clinical prose that may mention other
values, other drugs, and the same drug. That is the "which value is current"
problem the README lists as a known limitation, turned into a test.

WHICH NOTE, WHICH PART
-----------------------
* The discharge note whose charttime is nearest the pair's EARLIER
  measurement; the gap in days is recorded per item, not hidden.
* Only the "Brief Hospital Course" section (present in 88% of the 331,793
  discharge notes), cut at a sentence boundary under --max_chars. The median
  full note is ~9,800 characters, more than BioMistral's 2,048-token context.
  The section is narrative; the lab dump ("Pertinent Results") and the drug
  list ("Discharge Medications") are excluded by construction, and the build
  reports how often either header still leaks into an excerpt.

SELECTION BIAS, STATED
  Discharge notes exist only for inpatient stays, so this arm is drawn from a
  sicker sub-population than data/mimic_v3 (75% of v3 test patients have one;
  86% of held-out). Report it as a separate arm, never pooled with v3.

NO TEXT IS PRINTED
  Under the PhysioNet DUA note text must not reach an online service, which
  includes an LLM assistant reading this script's output. Everything printed
  is a count or a rate. The excerpts exist only in data/mimic_v3_note/ and in
  git-ignored prediction files.

USAGE
  python src/build_mimic_note.py --pairs data/mimic_v3 --out data/mimic_v3_note \
      --project <gcp-project> --bq <path/to/bq>
"""

import argparse
import collections
import csv
import gzip
import io
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

NOTE_TABLE = "physionet-data.mimiciv_note.discharge"

# Sizes are in CAUSAL pairs; each patient's control pair (when one exists)
# comes along, so items ~= 4 x pairs. Chosen so a full six-variant ablation of
# the long prompts fits inside one reboot window, not from any test number.
DEFAULT_PAIRS = {"test": 1000, "heldout": 500, "calib": 300, "train": 300}

BHC = re.compile(r"brief hospital course\s*:?", re.I)
# The next section header ends the excerpt: a line that is a short title
# followed by a colon, the shape every MIMIC discharge section header has.
NEXT_HEADER = re.compile(r"\n\s*[A-Z][A-Za-z /&#-]{2,60}:\s*\n")
LEAK_HEADERS = re.compile(r"(?i)(discharge medications|pertinent results|"
                          r"medications on admission)")
SENTENCE_END = re.compile(r"[.!?](?=\s)")


def bq_csv(bq, project, sql, params=()):
    cmd = [bq, f"--project_id={project}", "query", "--use_legacy_sql=false",
           "--format=csv", "--max_rows=10000000"]
    cmd += [f"--parameter={p}" for p in params]
    cmd.append(sql)
    # bq authenticates through gcloud, which must sit on PATH beside it; cron
    # and a bare `python` both lack it, and bq then fails with no stderr.
    env = dict(os.environ)
    env["PATH"] = str(Path(bq).resolve().parent) + os.pathsep + env["PATH"]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if out.returncode != 0:
        sys.exit(f"bq failed: {(out.stderr or out.stdout).strip()[:500]}")
    return list(csv.DictReader(io.StringIO(out.stdout)))


def excerpt(text, max_chars):
    """Brief Hospital Course, cut at the last sentence end under max_chars."""
    m = BHC.search(text)
    if not m:
        return None, "no_bhc_section"
    body = text[m.end():]
    nxt = NEXT_HEADER.search(body)
    if nxt:
        body = body[:nxt.start()]
    body = re.sub(r"[ \t]+", " ", body).strip()
    if len(body) < 200:
        return None, "bhc_too_short"
    if len(body) > max_chars:
        cut = body[:max_chars]
        ends = [e.end() for e in SENTENCE_END.finditer(cut)]
        body = cut[:ends[-1]] if ends else cut
    return body, None


def load_pairs(path):
    by = collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        by[r["pair_id"]].append(r)
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/mimic_v3")
    ap.add_argument("--out", default="data/mimic_v3_note")
    ap.add_argument("--project", required=True)
    ap.add_argument("--bq", default="bq")
    ap.add_argument("--max_chars", type=int, default=2000)
    ap.add_argument("--max_gap_days", type=int, default=30,
                    help="a note further than this from the pair's earlier "
                         "measurement is about a different episode, not "
                         "context for this one")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src, out = Path(args.pairs), Path(args.out)
    meta_in = json.loads((src / "build_meta.json").read_text())
    if not meta_in.get("credentialed"):
        sys.exit(f"{src} is not the credentialed v3.1 arm")

    # --- 0. note TIMES only (no text) for every discharge note ---------------
    # A first build took each patient's nearest note whatever its distance:
    # the median gap was 8 days but the 90th percentile was 761, i.e. one pair
    # in ten was given the summary of an admission two years away -- prose
    # about a different illness, which is noise, not context. Eligibility is
    # therefore decided on the gap BEFORE sampling, from timestamps alone.
    print("discharge note timestamps ...", flush=True)
    fmt = "%Y-%m-%d %H:%M:%S"
    note_times = collections.defaultdict(list)
    for r in bq_csv(args.bq, args.project,
                    f"SELECT note_id, CAST(subject_id AS STRING) AS s, "
                    f"FORMAT_DATETIME('%Y-%m-%d %H:%M:%S', charttime) AS t "
                    f"FROM `{NOTE_TABLE}`"):
        note_times[r["s"]].append((datetime.strptime(r["t"], fmt),
                                   r["note_id"]))
    print(f"  {sum(map(len, note_times.values())):,} notes, "
          f"{len(note_times):,} patients")

    def near_notes(arms):
        """This pair's notes within --max_gap_days, nearest first."""
        t0 = min(datetime.strptime(a["charttime"], fmt) for a in arms)
        c = sorted((abs((t - t0).days), nid)
                   for t, nid in note_times.get(arms[0]["subject_id"], []))
        return [(g, nid) for g, nid in c if g <= args.max_gap_days]

    # --- 1. seeded selection among pairs with a CONTEMPORANEOUS note ---------
    rng = random.Random(args.seed)
    chosen, pairs, near = {}, {}, {}
    for split, n in DEFAULT_PAIRS.items():
        by = load_pairs(src / f"counterfactual_{split}.jsonl")
        pairs[split] = by
        any_note = elig = 0
        causal = []
        for p in sorted(by):
            arms = by[p]
            if arms[0]["is_control"]:
                continue
            any_note += arms[0]["subject_id"] in note_times
            nn = near_notes(arms)
            if nn:
                causal.append(p); near[p] = nn
        rng.shuffle(causal)
        # Oversample: a note may lack a usable Brief Hospital Course, and the
        # shortfall is refilled from the same shuffled order, not re-drawn.
        chosen[split] = causal
        print(f"  {split}: {any_note:,} causal pairs with any note, "
              f"{len(causal):,} with one within {args.max_gap_days} days")

    want = {}
    for split, n in DEFAULT_PAIRS.items():
        want[split] = chosen[split][:int(n * 1.3) + 20]
    note_ids = sorted({nid for s in want for p in want[s]
                       for _, nid in near[p]})
    print(f"fetching text of {len(note_ids):,} candidate notes ...", flush=True)
    text_of = {}
    for i in range(0, len(note_ids), 2000):
        chunk = ",".join(f'"{x}"' for x in note_ids[i:i + 2000])
        for r in bq_csv(args.bq, args.project,
                        f"SELECT note_id, text FROM `{NOTE_TABLE}` "
                        f"WHERE note_id IN UNNEST(@ids)",
                        params=[f"ids:ARRAY<STRING>:[{chunk}]"]):
            text_of[r["note_id"]] = r["text"]
    print(f"  {len(text_of):,} fetched")

    # --- 2. build ------------------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    gaps, lens = [], []
    for split, n_target in DEFAULT_PAIRS.items():
        items = []
        by = pairs[split]
        kept = 0
        for pid in want[split]:
            if kept == n_target:
                break
            arms = by[pid]
            ex, why, gap, nid = None, None, None, None
            for gap, nid in near[pid]:     # nearest note WITH a usable section
                if nid not in text_of:
                    why = "text_not_fetched"; continue
                ex, why = excerpt(text_of[nid], args.max_chars)
                if ex:
                    break
            if not ex:
                stats[f"skip_{why}"] += 1
                continue
            kept += 1
            stats["kept_patients"] += 1
            gaps.append(gap); lens.append(len(ex))
            drug = arms[0]["drug"]
            stats["excerpt_mentions_drug"] += bool(re.search(
                rf"(?i)\b{re.escape(drug)}\b", ex))
            stats["excerpt_leaks_header"] += bool(LEAK_HEADERS.search(ex))
            # build_mimic names a pair's control "<pair_id>__ctrl": same
            # patient, same family. Controls from the patient's OTHER families
            # are not brought along -- their causal pair was not sampled.
            ctrl = pid + "__ctrl"
            for p in [pid] + ([ctrl] if ctrl in by else []):
                for a in by[p]:
                    it = dict(a)
                    base_vig = a["vignette"]
                    vig = ("Excerpt from this patient's discharge summary "
                           "(Brief Hospital Course):\n\"\"\"\n" + ex +
                           "\n\"\"\"\n\n" + base_vig)
                    it["id"] = a["id"] + "__note"
                    it["pair_id"] = a["pair_id"] + "__note"
                    it["vignette"] = vig
                    it["prompt"] = a["prompt"].replace(base_vig, vig)
                    assert it["prompt"] != a["prompt"], "vignette not in prompt"
                    it["presentation"] = "real_note_excerpt"
                    it["note_context"] = {
                        "note_id": nid, "gap_days": gap,
                        "section": "Brief Hospital Course",
                        "excerpt_chars": len(ex),
                        "same_excerpt_both_arms": True}
                    it["provenance"] = dict(a["provenance"],
                                            note_source="MIMIC-IV-Note v2.2 "
                                            "(PhysioNet credentialed DUA)")
                    items.append(it)
        with (out / f"counterfactual_{split}.jsonl").open("w") as fh:
            for it in items:
                fh.write(json.dumps(it) + "\n")
        npairs = len({i["pair_id"] for i in items})
        nctrl = len({i["pair_id"] for i in items if i["is_control"]})
        print(f"  {split}: {len(items):,} items / {npairs:,} pairs "
              f"({nctrl:,} control)")

    (out / "rag_corpus.jsonl").write_text((src / "rag_corpus.jsonl").read_text())
    k = stats["kept_patients"] or 1
    q = lambda xs, f: sorted(xs)[int(f * (len(xs) - 1))] if xs else None
    summary = {
        "kept_patients": stats["kept_patients"],
        "skips": {s: c for s, c in stats.items() if s.startswith("skip_")},
        "gap_days_median": q(gaps, .5), "gap_days_p90": q(gaps, .9),
        "excerpt_chars_median": q(lens, .5),
        "excerpt_mentions_drug_rate": round(stats["excerpt_mentions_drug"] / k, 4),
        "excerpt_leaks_section_header_rate":
            round(stats["excerpt_leaks_header"] / k, 4),
    }
    print(json.dumps(summary, indent=2))
    meta = dict(meta_in)
    meta.update({
        "source": "MIMIC-IV v3.1 + MIMIC-IV-Note v2.2",
        "note_url": "https://physionet.org/content/mimic-iv-note/2.2/",
        "derived_from": str(src), "seed": args.seed,
        "max_chars": args.max_chars, "max_gap_days": args.max_gap_days,
        "pairs_sampled": DEFAULT_PAIRS,
        "note_build": summary,
        "caveat": meta_in["caveat"] + " NOTE ARM: the excerpt is the same "
                  "real Brief Hospital Course text in both arms, so the pair "
                  "still differs in one number; the prose may state OTHER "
                  "values, including of the same lab. Drawn only from "
                  "patients with an inpatient discharge note -- a sicker "
                  "sub-population than data/mimic_v3.",
        "counts": {s: sum(1 for _ in open(out / f"counterfactual_{s}.jsonl"))
                   for s in DEFAULT_PAIRS},
    })
    (out / "build_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out}/build_meta.json")


if __name__ == "__main__":
    main()
