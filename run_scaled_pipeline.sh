#!/usr/bin/env bash
# Top-level orchestrator for the 2026-09-02 scale-up: lane A (GPU0) + lane B
# (GPU1) in parallel, then JOIN. Idempotent -- every stage inside every lane
# and inside JOIN is checkpointed to .pipeline_state/, so re-running this
# script after an interruption (SSH drop, 3am reboot) skips finished work and
# resumes at the first incomplete stage. Meant to be launched inside tmux via
# scaled_launch.sh, not run directly at an interactive shell.
set -x
cd /home/asosoft/abtin/paper/csai
TAG="${MODEL_TAG:-biomistral-7b}"
R="${RESULTS_DIR:-results}"
LOG="logs/scaled/${TAG}"
mkdir -p "$LOG" "$R" .pipeline_state

echo "=== SCALED PIPELINE START $(date -Is) ==="
git rev-parse --short HEAD 2>/dev/null || echo "(no git rev)"

python src/check_data.py --strict --json "$R"/data_provenance.json
CHECK_RC=$?
if [ $CHECK_RC -ne 0 ]; then
  echo "check_data --strict FAILED (exit $CHECK_RC) -- aborting before any GPU work"
  exit 1
fi

bash run_lane_a.sh > "$LOG"/lane_a.log 2>&1 &
PID_A=$!
bash run_lane_b.sh > "$LOG"/lane_b.log 2>&1 &
PID_B=$!
echo "lane A pid=$PID_A -> $LOG/lane_a.log"
echo "lane B pid=$PID_B -> $LOG/lane_b.log"

wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
echo "lane A exit=$RC_A   lane B exit=$RC_B"

bash run_join.sh > "$LOG"/join.log 2>&1
RC_J=$?
echo "join exit=$RC_J"

echo "=== SCALED PIPELINE END $(date -Is) ==="
if [ "$RC_A" -eq 0 ] && [ "$RC_B" -eq 0 ] && [ "$RC_J" -eq 0 ]; then
  touch ".pipeline_state/SCALED_PIPELINE_COMPLETE_${TAG}"
  echo "ALL STAGES COMPLETE"
else
  echo "one or more lanes/join reported a non-zero exit -- check $LOG/*.log and re-launch to resume (checkpointed stages will be skipped)"
fi
