#!/usr/bin/env bash
# Constraint layer (Aim 3) on the CORRECTED v3.1 arm, tag _mimic3b.
#
# Adapter trained on a seeded 1,000-causal-pair sample of data/mimic_v3b's
# train split (data/mimic_v3b_cl; src/build_mimic_cl_subset.py says why a
# sample), plus the shuffled-label control; cl and nsai_uq_cl scored on the
# FULL test and heldout splits through run_eval_v3. Protocol from
# run_lane_a.sh A9-A12: 8 epochs, layer $ADAPTER_LAYER, the UMLS graph.
#
# The split is patient-disjoint, not prompt-disjoint, and only 25 train pairs
# have no prompt in test at all, so the adapter HAS seen some test prompts.
# src/report_mimic3.py therefore also reports cl on the test pairs neither of
# whose prompts is in the adapter's training sample.
#
# Waits for run_v3b.sh (ARM=main) for the same model: both merge rows into
# summary_*_mimic3b.json. Exits instead if that job is neither running nor
# done, so a failed job does not pin a GPU worker until the next reboot.
set -u
WT="$(cd "$(dirname "$0")" && pwd)"
CSAI=/home/asosoft/abtin/paper/csai
cd "$CSAI" || exit 1

M="${MODEL_ID:-BioMistral/BioMistral-7B}"
TAG="${MODEL_TAG:-biomistral-7b}"
AL="${ADAPTER_LAYER:-30}"
D="$WT/data/mimic_v3b"
DCL="$WT/data/mimic_v3b_cl"
G="$CSAI/data/umls/causal_graph.json"
R="$WT/results/mimic_v3b/$TAG"
STATE="$CSAI/.pipeline_state/v3b_cl_$TAG"
LOGS="$WT/logs/mimic_v3b/$TAG"
AD="$R/adapter_mimic3b.npz"

_lockfile="$CSAI/.pipeline_state/locks/v3b_cl_$TAG.lock"
mkdir -p "$(dirname "$_lockfile")" "$STATE" "$R" "$LOGS"
exec 8>"$_lockfile"
flock -n 8 || { echo "$(date -Is) run_v3b_cl.sh[$TAG]: already running"; exit 0; }

export PATH="/home/asosoft/abtin/miniconda3/bin:$PATH"
export PYTHONPATH="$CSAI/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export EARLY_STOP="${EARLY_STOP:-0}"
RUN_EVAL="$WT/src/run_eval_v3.py"

[ -s "$DCL/counterfactual_train.jsonl" ] || { echo "no $DCL"; exit 1; }

MAIN_DONE="$CSAI/.pipeline_state/v3b_main_$TAG/ALL_DONE"
MAIN_LOCK="$CSAI/.pipeline_state/locks/v3b_main_$TAG.lock"
waited=0
until [ -e "$MAIN_DONE" ]; do
  if flock -n "$MAIN_LOCK" -c true 2>/dev/null; then
    echo "[v3b:cl:$TAG] run_v3b.sh[main/$TAG] is neither running nor done -- not waiting"
    exit 1
  fi
  [ $((waited % 1800)) -eq 0 ] && echo "[v3b:cl:$TAG] waiting for run_v3b.sh[main/$TAG]"
  sleep 60; waited=$((waited + 60))
done

FAILED=0
stage () {
  local name="$1"; shift
  [ -f "$STATE/$name.done" ] && { echo "[v3b:cl:$TAG][skip] $name"; return 0; }
  echo "[v3b:cl:$TAG][run ] $name  $(date -Is)"
  local t0; t0=$(date +%s)
  if "$@" > "$LOGS/cl_$name.log" 2>&1; then
    touch "$STATE/$name.done"
    echo "[v3b:cl:$TAG][ok  ] $name  $(( $(date +%s) - t0 ))s"
  else
    echo "[v3b:cl:$TAG][FAILED] $name -- see $LOGS/cl_$name.log"
    FAILED=$((FAILED+1))
  fi
}

stage train \
  python src/constraint_layer.py --model_id "$M" --layer "$AL" --data "$DCL" \
    --train_on train --epochs 8 --graph "$G" --save_adapter "$AD" \
    --out "$R/constraint_mimic3b.json"
stage train_shuffled \
  python src/constraint_layer.py --model_id "$M" --layer "$AL" --data "$DCL" \
    --train_on train --epochs 8 --graph "$G" --shuffled_control \
    --out "$R/constraint_mimic3b_shuffled.json"

if [ -s "$AD" ]; then
  for split in test heldout; do
    for v in cl nsai_uq_cl; do
      stage "eval_${split}_${v}" \
        python "$RUN_EVAL" --backend hf --model_id "$M" --out "$R" \
          --seeds 0 --split "$split" --data "$D" --gate rules --tag _mimic3b \
          --variants "$v" --adapter "$AD" --adapter_layer "$AL" --batch_size 8
    done
  done
else
  echo "[v3b:cl:$TAG] no adapter at $AD -- skipping cl evals"; FAILED=$((FAILED+1))
fi

for split in test heldout; do
  stage "table_${split}_with_cl" \
    python src/make_table.py --results "$R" --split "$split" --tag _mimic3b \
      --out "$R/table_mimic3b_$split.md"
  stage "coverage_${split}_nsai_uq_cl" \
    python src/coverage_report.py --results "$R" --split "$split" \
      --tag _mimic3b --variant nsai_uq_cl
done

echo "[v3b:cl:$TAG] done, $FAILED stage(s) failed  $(date -Is)"
[ "$FAILED" -eq 0 ] && touch "$STATE/ALL_DONE"
exit "$FAILED"
