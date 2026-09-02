"""
Step 0: pull REAL contraindication text from the FDA drug label API and emit a
curation worksheet.

This script writes NO clinical thresholds. It only fetches what the FDA label
actually says and records where it said it. A human then transcribes the
numeric threshold from the quoted text into the worksheet. That satisfies the
"expert-curated causal relations" requirement of proposal section 4.2 while
keeping every rule traceable to a citable source.

API: https://api.fda.gov/drug/label.json  (no authentication required; an
optional API key only raises the rate limit).

Verified response fields used here:
  results[].contraindications      list[str]
  results[].warnings_and_precautions  list[str]
  results[].boxed_warning          list[str]
  results[].drug_interactions      list[str]
  results[].openfda.generic_name   list[str]
  results[].openfda.rxcui          list[str]
  results[].openfda.spl_set_id     list[str]
Fields are frequently ABSENT on a given label; every access below is defensive.

Usage
  python src/fetch_openfda.py --drugs drugs.txt --out data/curation_worksheet.csv
  # then a clinician fills the blank columns and you run build_dataset.py
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

API = "https://api.fda.gov/drug/label.json"

SECTIONS = ["contraindications", "boxed_warning",
            "warnings_and_precautions", "drug_interactions"]

# Columns a human must fill in. Deliberately blank on write.
CURATION_COLUMNS = [
    "constraint_var",     # egfr | potassium | inr | qtc | age | pregnant | ...
    "constraint_op",      # < | > | ==
    "constraint_value",   # the number or boolean, transcribed from source_text
    "unit",
    "curator",            # who filled this row
    "curator_note",
]


def fetch(drug: str, api_key: Optional[str], limit: int = 1,
          timeout: int = 30) -> Optional[Dict]:
    """Query openFDA for one generic drug name. Returns the first label or None."""
    params = {
        "search": f'openfda.generic_name:"{drug}"',
        "limit": str(limit),
    }
    if api_key:
        params["api_key"] = api_key
    url = f"{API}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            payload = json.load(r)
    except urllib.error.HTTPError as e:
        # openFDA returns 404 with a JSON body when nothing matches
        if e.code == 404:
            print(f"  no label found for {drug!r}", file=sys.stderr)
            return None
        print(f"  HTTP {e.code} for {drug!r}", file=sys.stderr)
        return None
    except Exception as e:                       # network, timeout, bad JSON
        print(f"  failed for {drug!r}: {e}", file=sys.stderr)
        return None

    results = payload.get("results") or []
    return results[0] if results else None


def first(d: Dict, key: str, default: str = "") -> str:
    v = d.get(key)
    if isinstance(v, list) and v:
        return str(v[0])
    if isinstance(v, str):
        return v
    return default


def split_sentences(text: str) -> List[str]:
    """Crude sentence split; good enough to keep worksheet rows readable."""
    parts = re.split(r"(?<=[.;])\s+(?=[A-Z(])", text)
    return [p.strip() for p in parts if p.strip()]


def rows_for_label(drug: str, label: Dict, max_sentences: int) -> List[Dict]:
    openfda = label.get("openfda", {}) or {}
    spl_set_id = first(openfda, "spl_set_id")
    rxcui = ",".join(openfda.get("rxcui", [])[:5])
    generic = first(openfda, "generic_name", drug)

    rows = []
    for section in SECTIONS:
        blocks = label.get(section) or []
        if isinstance(blocks, str):
            blocks = [blocks]
        for block in blocks:
            for sent in split_sentences(block)[:max_sentences]:
                row = {
                    "query_drug": drug,
                    "generic_name": generic,
                    "rxcui": rxcui,
                    "spl_set_id": spl_set_id,
                    "source_section": section,
                    "source_text": sent,
                    "source_url": (
                        f"https://api.fda.gov/drug/label.json?search="
                        f"openfda.spl_set_id:%22{spl_set_id}%22"
                        if spl_set_id else ""),
                    "has_number": bool(re.search(r"\d", sent)),
                }
                for c in CURATION_COLUMNS:
                    row[c] = ""          # left blank for the human curator
                rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drugs", required=True,
                    help="text file, one generic drug name per line")
    ap.add_argument("--out", default="data/curation_worksheet.csv")
    ap.add_argument("--raw_out", default="data/openfda_raw.jsonl",
                    help="full API responses, kept for provenance")
    ap.add_argument("--api_key", default=None,
                    help="optional openFDA key; raises the rate limit")
    ap.add_argument("--max_sentences", type=int, default=12,
                    help="per label section, to keep the worksheet reviewable")
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    drugs = [l.strip() for l in Path(args.drugs).read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    print(f"querying openFDA for {len(drugs)} drugs")

    all_rows, raw = [], []
    for drug in drugs:
        print(f"- {drug}")
        label = fetch(drug, args.api_key)
        if label is None:
            continue
        raw.append({"query_drug": drug, "label": label})
        all_rows += rows_for_label(drug, label, args.max_sentences)
        time.sleep(args.sleep)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not all_rows:
        print("NO ROWS RETRIEVED -- check network access to api.fda.gov "
              "and the drug name spellings", file=sys.stderr)
        sys.exit(1)

    fields = list(all_rows[0].keys())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    with Path(args.raw_out).open("w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")

    n_drugs = len({r["query_drug"] for r in all_rows})
    n_num = sum(r["has_number"] for r in all_rows)
    print(f"\nwrote {out}: {len(all_rows)} candidate sentences "
          f"from {n_drugs} drugs ({n_num} contain a number)")
    print(f"wrote {args.raw_out}: {len(raw)} full label records")
    print("\nNEXT: open the CSV, keep only the sentences that state a hard "
          "contraindication, and fill constraint_var / constraint_op / "
          "constraint_value from the source_text. Delete every other row.")


if __name__ == "__main__":
    main()
