#!/usr/bin/env bash
# Idempotent launcher for the scaled pipeline. Ensures tmux session
# "csai_scaled" is running run_scaled_pipeline.sh; does nothing if it already
# is. Safe to call repeatedly -- from a terminal, from cron @reboot, or after
# an SSH drop to check on things.
SESSION=csai_scaled
PROJECT=/home/asosoft/abtin/paper/csai
LOG="$PROJECT/logs/scaled/launcher.log"
mkdir -p "$PROJECT/logs/scaled"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "$(date -Is) tmux session $SESSION already running -- nothing to do" >> "$LOG"
  exit 0
fi

if [ -f "$PROJECT/.pipeline_state/SCALED_PIPELINE_COMPLETE" ]; then
  echo "$(date -Is) SCALED_PIPELINE_COMPLETE marker present -- not relaunching. Delete it to force a rerun." >> "$LOG"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "cd $PROJECT && bash run_scaled_pipeline.sh"
echo "$(date -Is) started tmux session $SESSION (pid of tmux server: $(pgrep -f 'tmux.*new-session.*csai_scaled' | head -1))" >> "$LOG"
