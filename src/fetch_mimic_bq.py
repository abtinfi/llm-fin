"""
Pull the MIMIC-IV v3.1 slice this project needs out of BigQuery.

WHY NOT JUST DOWNLOAD THE FILES
--------------------------------
physionet.org serves `hosp/labevents.csv.gz` at 2.4 GB, and measured from this
machine it arrives at ~270 KiB/s: a ~2.5 hour download, capped by a 271 ms
round trip, a 16-connection-per-user server limit, and a per-IP quota. All
2.4 GB would then be decompressed and scanned to keep THREE itemids.

BigQuery applies the same filter server-side. The query below scans 4.9 GB of
the 16.1 GB table (well inside the 1 TB/month free tier) and returns ~10.2M
rows -- about fifteen minutes, no 2.4 GB on disk, and nothing to unpack.

WHAT IT WRITES, AND WHY IN THAT SHAPE
--------------------------------------
Two gzipped CSVs whose columns match the real MIMIC distribution:

  labevents.csv.gz   subject_id, itemid, charttime, valuenum
  patients.csv.gz    subject_id, gender, anchor_age

so `src/build_mimic.py --src <out>` reads them with no BigQuery code path of
its own and no awareness of where they came from. The builder stays a pure
function of files on disk, which is what makes the Demo arm and the v3.1 arm
comparable: same code, different input.

ITEMIDS
  50912  creatinine  -> metformin_egfr30, metformin_egfr45 (via CKD-EPI)
  50971  potassium   -> spironolactone_k5_5
  51237  INR         -> warfarin_inr4 (held out)
Kept in sync with FAMILIES in build_mimic.py -- imported, not re-typed.

CREDENTIALS AND LICENCE
  Needs `gcloud auth login` and PhysioNet cloud access granted to that Google
  account (physionet.org/settings/cloud/). Everything it writes is under the
  PhysioNet credentialed DUA and is git-ignored; see src/check_data.py.

USAGE
  python src/fetch_mimic_bq.py --out data/mimic_iv_3.1/hosp \
      --project <gcp-project-id> --bq /path/to/bq
"""

import argparse
import csv
import gzip
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_mimic import FAMILIES                      # noqa: E402

DATASET = "physionet-data.mimiciv_3_1_hosp"

# FORMAT_DATETIME is not cosmetic. BigQuery renders a DATETIME as
# 2171-08-10T02:30:00; the MIMIC CSV distribution writes 2171-08-10 02:30:00.
# build_mimic.py copies charttime verbatim into every emitted item, so without
# this the v3.1 arm's timestamps would not match the Demo arm's byte for byte
# and the two arms would stop being diffable.
LABS_SQL = """
SELECT subject_id, itemid,
       FORMAT_DATETIME('%Y-%m-%d %H:%M:%S', charttime) AS charttime,
       valuenum
FROM `{ds}.labevents`
WHERE itemid IN ({ids}) AND valuenum IS NOT NULL
"""

PATIENTS_SQL = """
SELECT subject_id, gender, anchor_age
FROM `{ds}.patients`
"""


def run_query(bq, project, sql, max_rows):
    """Stream one query's rows back as CSV text."""
    cmd = [bq, f"--project_id={project}", "query", "--use_legacy_sql=false",
           "--format=csv", f"--max_rows={max_rows}", sql]
    print(f"  running query (max_rows={max_rows}) ...", flush=True)
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"bq failed:\n{out.stderr}")
    return out.stdout


def write_gz(text, path, expect_cols):
    """
    Write CSV text to a gzipped file, verifying the header first.

    A silently-renamed column would not fail here -- it would fail much later
    inside build_mimic.py as an empty cohort, which is the kind of error that
    gets mistaken for a real finding. So the header is checked at the source.
    """
    rows = text.splitlines()
    if not rows:
        sys.exit(f"{path.name}: query returned nothing")
    header = next(csv.reader([rows[0]]))
    missing = [c for c in expect_cols if c not in header]
    if missing:
        sys.exit(f"{path.name}: expected columns {missing} not in {header}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
    print(f"  wrote {path}  ({len(rows) - 1:,} rows)", flush=True)
    return len(rows) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mimic_iv_3.1/hosp")
    ap.add_argument("--project", required=True)
    ap.add_argument("--bq", default="bq")
    ap.add_argument("--max_rows", type=int, default=20_000_000)
    args = ap.parse_args()

    out = Path(args.out)
    itemids = sorted({spec["itemid"] for spec in FAMILIES.values()})
    print(f"itemids from build_mimic.FAMILIES: {itemids}")

    print("labevents:")
    labs = run_query(args.bq, args.project,
                     LABS_SQL.format(ds=DATASET, ids=", ".join(itemids)),
                     args.max_rows)
    n_labs = write_gz(labs, out / "labevents.csv.gz",
                      ["subject_id", "itemid", "charttime", "valuenum"])
    del labs

    print("patients:")
    pats = run_query(args.bq, args.project,
                     PATIENTS_SQL.format(ds=DATASET), args.max_rows)
    n_pats = write_gz(pats, out / "patients.csv.gz",
                      ["subject_id", "gender", "anchor_age"])

    print(f"\ndone: {n_labs:,} lab rows, {n_pats:,} patients")
    print("Both files are under the PhysioNet credentialed DUA and are "
          "git-ignored. Do not commit them or anything derived from them.")


if __name__ == "__main__":
    main()
