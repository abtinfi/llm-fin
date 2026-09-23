"""
Base-vs-proposed comparison for the MIMIC-IV v3.1 arm (_mimic3), one table
across the three models.

WHY A WRAPPER AND NOT A NEW ROW IN make_comparison.py
------------------------------------------------------
The metrics must be computed by the SAME metrics.py that scored every other
arm. Today that is the main checkout's uncommitted metrics.py (the conformal /
Clopper-Pearson calibration), not the committed one. So this file imports
make_comparison from the main checkout -- which pulls its make_table and
metrics from there too -- and narrows ARMS to the v3 arm. Nothing in the
shared module is edited, and nothing here can drift from how the other arms
are scored.

The v3 results live under the mimic-v3 worktree (results/mimic_v3/<model>/)
because their per-patient predictions are credentialed; see run_mimic_v3.sh.
Only the aggregate markdown this writes is tracked.

  python src/make_comparison_mimic3.py
"""

import sys
from pathlib import Path

CSAI = Path("/home/asosoft/abtin/paper/csai")
HERE = Path(__file__).resolve().parents[1]
R3 = HERE / "results" / "mimic_v3"

sys.path.insert(0, str(CSAI / "src"))
import make_comparison as mc                    # noqa: E402  (main checkout's)

assert Path(mc.__file__).resolve().parent == CSAI / "src", mc.__file__

mc.TAG_DATA["_mimic3"] = str(CSAI / "data" / "mimic_v3")
mc.ARMS[:] = [
    ("_mimic3", "test",    "MIMIC-IV v3.1 (full cohort, +real control pairs), test"),
    ("_mimic3", "heldout", "MIMIC-IV v3.1, held-out warfarin"),
]

MODELS = [("biomistral-7b", "BioMistral/BioMistral-7B"),
          ("llama3-openbiollm-8b", None),
          ("mistral-7b-instruct-v0-2", None)]

if __name__ == "__main__":
    base_slug, base_id = MODELS[0]
    extra = [f"{slug}={R3 / slug}" for slug, _ in MODELS[1:]
             if (R3 / slug).is_dir()]
    sys.argv = [sys.argv[0],
                "--results", str(R3 / base_slug),
                "--base_model", base_id,
                "--out", str(R3 / "COMPARISON_MIMIC3.md"),
                "--models", *extra]
    mc.main()
