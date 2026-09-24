"""
Re-scores the rows of a summary_*.json whose prediction files predate the
`is_control` flag, so every row of one table uses the same pair definition.

  python src/rescore_summary.py --results results --split test \
      --tag _medcalc2 --data data/medcalc_v2

WHY
---
Prediction files written before 2026-09-07 carry no `is_control`, and
metrics.score treats such records as causal pairs. In
summary_test_medcalc2.json that left six rows (base .. nsai_uq) scoring Causal
Consistency over 224 pairs -- 115 causal + 109 control -- while the two CL
rows, run later, scored the 115 causal pairs only. The column was not
comparable down its rows: base read 0.263 where the causal-pair figure is
0.026, the same as CL.

The flag is a property of the dataset, so it is joined back on `id`
(make_comparison.enrich) and only the fields metrics.score produces are
replaced. Calibration fields, tau and provenance are left as they are. A row
whose preds already carry the flag is re-scored too, as a check: it must
reproduce exactly, or the script stops without writing.
"""

import argparse
import json
from pathlib import Path

from make_comparison import enrich
from make_table import load_preds
from metrics import score


def same(a, b):
    if a is None or b is None:
        return a is b
    return abs(a - b) < 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--split", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    path = Path(args.results) / f"summary_{args.split}{args.tag}.json"
    summary = json.load(path.open())
    changed = 0
    for row in summary:
        recs = load_preds(args.results, args.split, row["variant"],
                          row["seed"], args.tag)
        had_flag = bool(recs) and "is_control" in recs[0]
        new = score(enrich(recs, args.data, args.split))
        if had_flag:
            bad = [k for k, v in new.items()
                   if k in row and isinstance(v, float) and not same(v, row[k])]
            if bad:
                raise SystemExit(f"{row['variant']}: preds carry is_control "
                                 f"but re-scoring changed {bad}; not writing")
            continue
        before = (row["causal_consistency"], row["n_pairs"])
        row.update(new)
        row["rescored_control_pairs"] = True
        changed += 1
        print(f"  {row['variant']:<11} CC {before[0]:.3f} over {before[1]} "
              f"pairs -> {row['causal_consistency']:.3f} over "
              f"{row['n_pairs']} ({row['n_control_pairs']} control)")
    if changed and not args.dry_run:
        path.write_text(json.dumps(summary, indent=2))
        print(f"wrote {path}")
    elif not changed:
        print(f"{path}: every row already scored with control pairs excluded")


if __name__ == "__main__":
    main()
