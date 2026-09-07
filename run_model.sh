#!/usr/bin/env bash
# One model, end to end, on ONE GPU: lane A, then lane B, then join.
#
# run_scaled_pipeline.sh runs the two lanes in parallel across two cards. That
# is right when both cards are free and wrong when a second model is queued
# behind the first, so this runs them in sequence on a single card instead.
# Lane B blocks on lane A's adapters anyway, so sequencing costs less than the
# lane-A idle time in the parallel layout (44 min of the 69-minute run).
#
#   MODEL_ID=aaditya/Llama3-OpenBioLLM-8B MODEL_TAG=llama3-openbiollm-8b \
#   GPU=0 bash run_model.sh
#
# Every stage is checkpointed under .pipeline_state/<lane>_<tag>_L<layer>/, so
# the 03:00 reboot resumes rather than restarts.
set -u
cd "$(dirname "$0")"

M="${MODEL_ID:-BioMistral/BioMistral-7B}"
TAG="${MODEL_TAG:-biomistral-7b}"
R="${RESULTS_DIR:-results/models/$TAG}"
LAYER="${LAYER:-20}"
GPU="${GPU:-0}"
LOG="logs/models/${TAG}_L${LAYER}"
mkdir -p "$LOG" "$R"

# SINGLE-INSTANCE LOCK. This script can be launched three ways -- by hand, by
# csai_supervisor.sh, or by the @reboot cron -- and two copies would write the
# same checkpoint directory and the same output files. Taking the lock here
# rather than in the caller means the guarantee holds however it was started.
# A second copy exits 0: "already running" is success, not failure.
_lockfile=".pipeline_state/locks/model_${TAG}_L${LAYER}.lock"
mkdir -p "$(dirname "$_lockfile")"
exec 8>"$_lockfile"
if ! flock -n 8; then
  echo "$(date -Is) run_model.sh: another instance holds $_lockfile -- exiting" >&2
  exit 0
fi

export MODEL_ID="$M" MODEL_TAG="$TAG" RESULTS_DIR="$R" LAYER="$LAYER" GPU="$GPU"

echo "=== $TAG (layer $LAYER) on GPU $GPU -> $R   $(date -Is) ==="
python src/check_data.py --strict --json "$R"/data_provenance.json \
  || { echo "check_data --strict failed; refusing to start GPU work"; exit 1; }

bash run_lane_a.sh > "$LOG"/lane_a.log 2>&1; RC_A=$?
echo "lane A exit=$RC_A  $(date -Is)"
bash run_lane_b.sh > "$LOG"/lane_b.log 2>&1; RC_B=$?
echo "lane B exit=$RC_B  $(date -Is)"
bash run_join.sh     > "$LOG"/join.log   2>&1; RC_J=$?
echo "join   exit=$RC_J  $(date -Is)"

if [ "$RC_A" -eq 0 ] && [ "$RC_B" -eq 0 ] && [ "$RC_J" -eq 0 ]; then
  touch ".pipeline_state/MODEL_COMPLETE_${TAG}_L${LAYER}"
  echo "=== $TAG COMPLETE ==="
else
  echo "=== $TAG had failures; re-launch to resume from the last checkpoint ==="
fi
exit $(( RC_A + RC_B + RC_J ))
