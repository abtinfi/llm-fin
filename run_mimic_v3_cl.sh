#!/usr/bin/env bash
# Constraint layer (Aim 3) on the MIMIC-IV v3.1 arm, tagged _mimic3.
#
# Train an adapter on a seeded 1,000-causal-pair sample of the v3 train split
# (data/mimic_v3_cl, see src/build_mimic_cl_subset.py for why a sample), plus
# the shuffled-label control, then score `cl` and `nsai_uq_cl` on the FULL v3
# test and heldout splits with it. Protocol copied from run_lane_a.sh A9-A12:
# 8 epochs, --layer $ADAPTER_LAYER, the UMLS graph, --batch_size 8.
#
# ORDERING. The cl rows are merged into the same summary_*_mimic3.json that
# run_mimic_v3.sh writes for this model. Two processes merging into one JSON
# at once would lose rows, so this job WAITS for that model's run_mimic_v3.sh
# to finish (its ALL_DONE marker) before touching the GPU.
set -u
WT="$(cd "$(dirname "$0")" && pwd)"
CSAI=/home/asosoft/abtin/paper/csai
cd "$CSAI" || exit 1

M="${MODEL_ID:-BioMistral/BioMistral-7B}"
TAG="${MODEL_TAG:-biomistral-7b}"
AL="${ADAPTER_LAYER:-30}"
D="$CSAI/data/mimic_v3"
DCL="$WT/data/mimic_v3_cl"
G="$CSAI/data/umls/causal_graph.json"
R="$WT/results/mimic_v3/$TAG"
STATE="$CSAI/.pipeline_state/mimic_v3_cl_$TAG"
LOGS="$WT/logs/mimic_v3/$TAG"
AD="$R/adapter_mimic3.npz"

_lockfile="$CSAI/.pipeline_state/locks/mimic_v3_cl_$TAG.lock"
mkdir -p "$(dirname "$_lockfile")" "$STATE" "$R" "$LOGS"
exec 8>"$_lockfile"
if ! flock -n 8; then
  echo "$(date -Is) run_mimic_v3_cl.sh[$TAG]: another instance holds the lock -- exiting" >&2
  exit 0
fi

export PATH="/home/asosoft/abtin/miniconda3/bin:$PATH"
export PYTHONPATH="$CSAI/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

# EARLY_STOP=1 (set per model in the supervisor queue; Mistral-Instruct only)
# stops generation once the answer is fixed. See src/run_eval_earlystop.py for
# why the prediction and every reported signal are unchanged by it. Training
# (constraint_layer.py) reads logits directly and never generates.
RUN_EVAL=src/run_eval.py
[ "${EARLY_STOP:-0}" = 1 ] && RUN_EVAL="$WT/src/run_eval_earlystop.py"

[ -s "$DCL/counterfactual_train.jsonl" ] || { echo "no $DCL -- run src/build_mimic_cl_subset.py"; exit 1; }

V3_DONE="$CSAI/.pipeline_state/mimic_v3_$TAG/ALL_DONE"
V3_LOCK="$CSAI/.pipeline_state/locks/mimic_v3_$TAG.lock"
waited=0
until [ -e "$V3_DONE" ]; do
  # Wait only while that job is actually RUNNING (it holds its flock). If it
  # is neither running nor done it failed, and waiting would pin this GPU
  # worker until the next reboot; exit so the supervisor moves on. The job
  # is retried on the next supervisor start, when its marker is still absent.
  if flock -n "$V3_LOCK" -c true 2>/dev/null; then
    echo "[v3cl:$TAG] run_mimic_v3.sh[$TAG] is neither running nor done -- not waiting"
    exit 1
  fi
  [ $((waited % 1800)) -eq 0 ] && echo "[v3cl:$TAG] waiting for run_mimic_v3.sh[$TAG] to finish"
  sleep 60; waited=$((waited + 60))
done

FAILED=0
stage () {
  local name="$1"; shift
  [ -f "$STATE/$name.done" ] && { echo "[v3cl:$TAG][skip] $name"; return 0; }
  echo "[v3cl:$TAG][run ] $name  $(date -Is)"
  local t0; t0=$(date +%s)
  if "$@" > "$LOGS/cl_$name.log" 2>&1; then
    touch "$STATE/$name.done"
    echo "[v3cl:$TAG][ok  ] $name  $(( $(date +%s) - t0 ))s"
  else
    echo "[v3cl:$TAG][FAILED] $name -- see $LOGS/cl_$name.log"
    FAILED=$((FAILED+1))
  fi
}

stage train \
  python src/constraint_layer.py --model_id "$M" --layer "$AL" --data "$DCL" \
    --train_on train --epochs 8 --graph "$G" --save_adapter "$AD" \
    --out "$R/constraint_mimic3.json"
stage train_shuffled \
  python src/constraint_layer.py --model_id "$M" --layer "$AL" --data "$DCL" \
    --train_on train --epochs 8 --graph "$G" --shuffled_control \
    --out "$R/constraint_mimic3_shuffled.json"

# The eval stages need the adapter; without it they would fail one by one.
if [ -s "$AD" ]; then
  for split in test heldout; do
    for v in cl nsai_uq_cl; do
      stage "eval_${split}_${v}" \
        python "$RUN_EVAL" --backend hf --model_id "$M" --out "$R" \
          --seeds 0 --split "$split" --data "$D" --gate rules --tag _mimic3 \
          --variants "$v" --adapter "$AD" --adapter_layer "$AL" --batch_size 8
    done
  done
else
  echo "[v3cl:$TAG] no adapter at $AD -- skipping cl evals"; FAILED=$((FAILED+1))
fi

# Tables were first written by run_mimic_v3.sh without the cl rows; rewrite.
for split in test heldout; do
  stage "table_${split}_with_cl" \
    python src/make_table.py --results "$R" --split "$split" --tag _mimic3 \
      --out "$R/table_mimic3_${split}.md"
  stage "coverage_${split}_nsai_uq_cl" \
    python src/coverage_report.py --results "$R" --split "$split" \
      --tag _mimic3 --variant nsai_uq_cl
done

echo "[v3cl:$TAG] done, $FAILED stage(s) failed  $(date -Is)"
[ "$FAILED" -eq 0 ] && touch "$STATE/ALL_DONE"
exit "$FAILED"
