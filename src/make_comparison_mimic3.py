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

  python src/make_comparison_mimic3.py          # the v3.1 arm
  python src/make_comparison_mimic3.py note     # the real-note arm
"""

import sys
from pathlib import Path

CSAI = Path("/home/asosoft/abtin/paper/csai")
HERE = Path(__file__).resolve().parents[1]
R3 = HERE / "results" / "mimic_v3"

sys.path.insert(0, str(CSAI / "src"))
import make_comparison as mc                    # noqa: E402  (main checkout's)

assert Path(mc.__file__).resolve().parent == CSAI / "src", mc.__file__

R3B = HERE / "results" / "mimic_v3b"
ARM_SPECS = {
    # arm -> (tag, data dir, results root, output name, human names). The
    # corrected v3b arm; both arms share results/mimic_v3b/<model>/.
    "v3": ("_mimic3b", HERE / "data" / "mimic_v3b", R3B, "COMPARISON_MIMIC3B.md",
           ("MIMIC-IV v3.1 (full cohort, +real control pairs), test",
            "MIMIC-IV v3.1, held-out warfarin")),
    "note": ("_mimic3bnote", HERE / "data" / "mimic_v3b_note", R3B,
             "COMPARISON_MIMIC3BNOTE.md",
             ("MIMIC-IV v3.1 + real discharge-note excerpt, test",
              "MIMIC-IV v3.1 + real discharge-note excerpt, held-out warfarin")),
}

MODELS = [("biomistral-7b", "BioMistral/BioMistral-7B"),
          ("llama3-openbiollm-8b", None),
          ("mistral-7b-instruct-v0-2", None)]

if __name__ == "__main__":
    arm = sys.argv[1] if len(sys.argv) > 1 else "v3"
    tag, data, root, out_name, (n_test, n_ho) = ARM_SPECS[arm]
    mc.TAG_DATA[tag] = str(data)
    mc.ARMS[:] = [(tag, "test", n_test), (tag, "heldout", n_ho)]
    base_slug, base_id = MODELS[0]
    extra = [f"{slug}={root / slug}" for slug, _ in MODELS[1:]
             if (root / slug).is_dir()]
    sys.argv = [sys.argv[0],
                "--results", str(root / base_slug),
                "--base_model", base_id,
                "--out", str(root / out_name),
                "--models", *extra]
    mc.main()
