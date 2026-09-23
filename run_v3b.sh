#!/usr/bin/env bash
# Evaluate the CORRECTED MIMIC-IV v3.1 arm (data/mimic_v3b) for one model.
#
#   ARM=main  data/mimic_v3b,      tag _mimic3b      (+ option-order diagnostic)
#   ARM=note  data/mimic_v3b_note, tag _mimic3bnote
#
# What changed from run_mimic_v3.sh, and why (each found by the audit):
#  * data/mimic_v3b is built with --question_version v2: the two metformin
#    families ask "continue" vs "start", so no prompt carries two labels
#    (under v1, 21% of test items did); and the rounding guard uses the
#    PRINTED precision (one v1 warfarin pair was mislabelled).
#  * every variant runs through src/run_eval_v3.py: each distinct prompt is
#    generated once, in unpadded fixed-shape batches, so identical prompts get
#    identical answers and no answer depends on its batch neighbours (under
#    stock batching 311 of 13,870 prompts got two different answers).
#  * stage 0 re-proves that on this GPU, in this dtype, before anything else,
#    and stops the job if it fails.
#  * base is also scored with the answer options in the other order
#    (UNSAFE or SAFE): OpenBioLLM copies whichever option is listed first.
#
# Code is the MAIN checkout's src/run_eval.py (its uncommitted conformal
# calibration is what every current _mimic row used), wrapped. Outputs go to
# this worktree, whose .gitignore keeps per-patient rows out of git.
# One (split, variant) per checkpointed stage: the 03:01 reboot costs one.
set -u
WT="$(cd "$(dirname "$0")" && pwd)"
CSAI=/home/asosoft/abtin/paper/csai
cd "$CSAI" || exit 1

M="${MODEL_ID:-BioMistral/BioMistral-7B}"
TAG="${MODEL_TAG:-biomistral-7b}"
ARM="${ARM:-main}"
case "$ARM" in
  main) D="$WT/data/mimic_v3b";      T=_mimic3b ;;
  note) D="$WT/data/mimic_v3b_note"; T=_mimic3bnote ;;
  *) echo "ARM must be main or note"; exit 2 ;;
esac
# BACKEND=mock + STATE_SUFFIX exist only to prove the plumbing end to end
# without a GPU; under mock the determinism guard (which needs the real
# model) is skipped and every output goes to suffixed directories.
BACKEND="${BACKEND:-hf}"
SX="${STATE_SUFFIX:-}"
R="$WT/results/mimic_v3b$SX/$TAG"
STATE="$CSAI/.pipeline_state/v3b_${ARM}_$TAG$SX"
LOGS="$WT/logs/mimic_v3b$SX/$TAG"

_lockfile="$CSAI/.pipeline_state/locks/v3b_${ARM}_$TAG.lock"
mkdir -p "$(dirname "$_lockfile")" "$STATE" "$R" "$LOGS"
exec 8>"$_lockfile"
flock -n 8 || { echo "$(date -Is) run_v3b.sh[$ARM/$TAG]: already running"; exit 0; }

export PATH="/home/asosoft/abtin/miniconda3/bin:$PATH"
export PYTHONPATH="$CSAI/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export EARLY_STOP="${EARLY_STOP:-0}"
RUN_EVAL="$WT/src/run_eval_v3.py"

FAILED=0
stage () {
  local name="$1"; shift
  [ -f "$STATE/$name.done" ] && { echo "[v3b:$ARM:$TAG][skip] $name"; return 0; }
  echo "[v3b:$ARM:$TAG][run ] $name  $(date -Is)"
  local t0; t0=$(date +%s)
  if "$@" > "$LOGS/${ARM}_$name.log" 2>&1; then
    touch "$STATE/$name.done"
    echo "[v3b:$ARM:$TAG][ok  ] $name  $(( $(date +%s) - t0 ))s"
  else
    echo "[v3b:$ARM:$TAG][FAILED] $name -- see $LOGS/${ARM}_$name.log"
    FAILED=$((FAILED+1))
  fi
}

[ -s "$D/counterfactual_test.jsonl" ] || { echo "no $D"; exit 1; }
grep -q '"credentialed": true' "$D/build_meta.json" || { echo "$D not labelled credentialed"; exit 1; }

# --- 0. guard: answers must not depend on input order, on THIS gpu ----------
if [ "$BACKEND" = hf ]; then
  stage determinism \
    python "$WT/src/check_determinism.py" --model_id "$M" --data "$D" --n 1000
  [ -f "$STATE/determinism.done" ] || { echo "[v3b:$ARM:$TAG] determinism guard failed -- stopping"; exit 1; }
fi

# --- 1. the six non-adapter variants ------------------------------------------
for split in test heldout; do
  for v in base sym nsai rag uq nsai_uq; do
    stage "eval_${split}_${v}" \
      python "$RUN_EVAL" --backend "$BACKEND" --model_id "$M" --out "$R" \
        --seeds 0 --split "$split" --data "$D" --gate rules --tag "$T" \
        --variants "$v" --batch_size 8
  done
done

# --- 2. option-order diagnostic (main arm) ------------------------------------
if [ "$ARM" = main ]; then
  stage eval_test_base_swapped \
    python "$RUN_EVAL" --backend "$BACKEND" --model_id "$M" --out "$R" \
      --seeds 0 --split test --data "$WT/data/mimic_v3b_swap" --gate rules \
      --tag _mimic3bswap --variants base --batch_size 8
fi

# --- 3. tables -----------------------------------------------------------------
for split in test heldout; do
  stage "table_$split" \
    python src/make_table.py --results "$R" --split "$split" --tag "$T" \
      --out "$R/table${T}_$split.md"
  stage "coverage_$split" \
    python src/coverage_report.py --results "$R" --split "$split" --tag "$T" \
      --variant nsai_uq
done

echo "[v3b:$ARM:$TAG] done, $FAILED stage(s) failed  $(date -Is)"
[ "$FAILED" -eq 0 ] && touch "$STATE/ALL_DONE"
exit "$FAILED"
