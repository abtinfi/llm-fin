#!/usr/bin/env bash
# Evaluate the full MIMIC-IV v3.1 cohort (data/mimic_v3), tagged _mimic3.
#
# One invocation = one model. csai_supervisor.sh queues it once per model with
# MODEL_ID / MODEL_TAG set, so its two GPU workers run two models at once.
#
# WHERE THINGS LIVE, AND WHY THEY ARE SPLIT
#   code  -> the MAIN checkout's src/ (CSAI). Its uncommitted run_eval.py and
#            metrics.py carry the conformal / Clopper-Pearson calibration that
#            produced every current _mimic and _mimic2 row. The committed HEAD
#            still has the legacy rule, so running HEAD would make v3 rows
#            incomparable with the arms they are meant to extend.
#   data  -> CSAI/data/mimic_v3, credentialed and git-ignored.
#   out   -> THIS worktree's results/mimic_v3/<model>/. Every preds_*.jsonl row
#            holds a real patient's age, sex, lab value and subject_id, and the
#            .gitignore beside this script keeps those out of git. Writing them
#            into the main checkout would put them one `git add results/` away
#            from a commit on a branch whose .gitignore does not know them.
#
# REBOOT SAFETY (03:01 daily). Every (split, variant) pair is its own stage
# with its own .done marker, so a reboot costs at most the one stage that was
# running. run_eval.py MERGES into an existing summary by (variant, seed)
# (run_eval.py: "Merge into any existing summary rather than overwriting"), so
# splitting variants across invocations yields the same summary as one call.
#
# CONFIG COPIED FROM THE _mimic ARM (run_lane_a.sh A11/A12, rerun_uq_arms.sh),
# not re-chosen: --gate rules, --batch_size 8, --seeds 0, the default
# --uq_signal and calibration rule of the checked-out run_eval.py.
set -u
WT="$(cd "$(dirname "$0")" && pwd)"
CSAI=/home/asosoft/abtin/paper/csai
cd "$CSAI" || exit 1

M="${MODEL_ID:-BioMistral/BioMistral-7B}"
TAG="${MODEL_TAG:-biomistral-7b}"
AL="${ADAPTER_LAYER:-30}"
# DATA_DIR / BACKEND / STATE_SUFFIX exist only so the plumbing can be proven
# end to end with --backend mock on a small slice before a GPU is touched.
D="${DATA_DIR:-$CSAI/data/mimic_v3}"
BACKEND="${BACKEND:-hf}"
R="$WT/results/mimic_v3${STATE_SUFFIX:-}/$TAG"
STATE="$CSAI/.pipeline_state/mimic_v3_$TAG${STATE_SUFFIX:-}"
LOGS="$WT/logs/mimic_v3${STATE_SUFFIX:-}/$TAG"

_lockfile="$CSAI/.pipeline_state/locks/mimic_v3_$TAG.lock"
mkdir -p "$(dirname "$_lockfile")" "$STATE" "$R" "$LOGS"
exec 8>"$_lockfile"
if ! flock -n 8; then
  echo "$(date -Is) run_mimic_v3.sh[$TAG]: another instance holds the lock -- exiting" >&2
  exit 0
fi

# cron has no conda on PATH (see run_medcalc_v2.sh for the night that broke).
export PATH="/home/asosoft/abtin/miniconda3/bin:$PATH"
export PYTHONPATH="$CSAI/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

FAILED=0
stage () {
  local name="$1"; shift
  [ -f "$STATE/$name.done" ] && { echo "[v3:$TAG][skip] $name"; return 0; }
  echo "[v3:$TAG][run ] $name  $(date -Is)"
  local t0; t0=$(date +%s)
  if "$@" > "$LOGS/$name.log" 2>&1; then
    touch "$STATE/$name.done"
    echo "[v3:$TAG][ok  ] $name  $(( $(date +%s) - t0 ))s"
  else
    echo "[v3:$TAG][FAILED] $name -- see $LOGS/$name.log"
    FAILED=$((FAILED+1))
  fi
}

[ -s "$D/counterfactual_test.jsonl" ] || { echo "no $D -- build it with src/build_mimic.py --source v3.1"; exit 1; }
grep -q '"credentialed": true' "$D/build_meta.json" \
  || { echo "$D/build_meta.json is not labelled credentialed -- refusing (see build_mimic.DATA_SOURCES)"; exit 1; }

# --- 1. the six non-adapter variants, one stage each -------------------------
# `base` first: it is the cheapest and its log gives the real items/second
# every later ETA is computed from.
for split in test heldout; do
  for v in base sym nsai rag uq nsai_uq; do
    stage "eval_${split}_${v}" \
      python src/run_eval.py --backend "$BACKEND" --model_id "$M" --out "$R" \
        --seeds 0 --split "$split" --data "$D" --gate rules --tag _mimic3 \
        --variants "$v" --batch_size 8
  done
done

stage "table_test" \
  python src/make_table.py --results "$R" --split test --tag _mimic3 \
    --out "$R/table_mimic3_test.md"
stage "table_heldout" \
  python src/make_table.py --results "$R" --split heldout --tag _mimic3 \
    --out "$R/table_mimic3_heldout.md"
stage "coverage_test" \
  python src/coverage_report.py --results "$R" --split test --tag _mimic3 \
    --variant nsai_uq
stage "coverage_heldout" \
  python src/coverage_report.py --results "$R" --split heldout --tag _mimic3 \
    --variant nsai_uq

echo "[v3:$TAG] done, $FAILED stage(s) failed  $(date -Is)"
[ "$FAILED" -eq 0 ] && touch "$STATE/ALL_DONE"
exit "$FAILED"
