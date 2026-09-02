#!/usr/bin/env bash
# Ready-made status check for the scaled pipeline, meant to be run AFTER a
# server restart once the `claude` CLI is reinstalled and on PATH again.
#
# Usage:
#   cd /home/asosoft/abtin/paper/csai
#   bash check_pipeline_status.sh
#
# What it does:
#   1. Feeds logs/scaled/STATUS_CHECK_PROMPT.txt to `claude --continue --print`,
#      which resumes THIS machine's most recent Claude Code conversation (the
#      one that built and launched the pipeline) and asks it to check status,
#      and to relaunch the pipeline itself if it finds it is not running.
#   2. If the `claude` command is not found (e.g. not reinstalled yet), falls
#      back to a plain bash status dump so you still get an answer.
set -uo pipefail
cd "$(dirname "$0")"

PROMPT_FILE="logs/scaled/STATUS_CHECK_PROMPT.txt"

# Scoped to read-only diagnostics plus the one write action the prompt may
# take (relaunching the idempotent launcher). Narrower than
# --dangerously-skip-permissions, which this script deliberately avoids.
ALLOWED="Bash(tmux *) Bash(nvidia-smi *) Bash(crontab *) Bash(grep *) Bash(tail *) Bash(head *) Bash(find *) Bash(cat *) Bash(ls *) Bash(date *) Bash(pgrep *) Bash(bash scaled_launch.sh)"

if command -v claude >/dev/null 2>&1; then
  echo "[check_pipeline_status] claude CLI found -- asking it to check and report."
  claude --continue --print --allowedTools "$ALLOWED" -- "$(cat "$PROMPT_FILE")"
  exit $?
fi

echo "[check_pipeline_status] 'claude' command not found on PATH."
echo "[check_pipeline_status] Falling back to a raw status dump. Reinstall"
echo "[check_pipeline_status] claude and re-run this script for a proper report."
echo
echo "=== tmux session ==="
tmux has-session -t csai_scaled 2>&1 && echo "csai_scaled: ALIVE" || echo "csai_scaled: NOT RUNNING"
echo
echo "=== pipeline complete marker ==="
[ -f .pipeline_state/SCALED_PIPELINE_COMPLETE ] && echo "COMPLETE" || echo "not complete"
echo
echo "=== stage markers done ==="
find .pipeline_state/scaled_a .pipeline_state/scaled_b .pipeline_state/scaled_join \
     -name "*.done" 2>/dev/null | sort
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv 2>/dev/null
echo
echo "=== last lines, each lane ==="
for f in logs/scaled/lane_a.log logs/scaled/lane_b.log logs/scaled/join.log; do
  echo "--- $f ---"
  tail -5 "$f" 2>/dev/null | grep -v "^+\|Loading weights"
done
echo
echo "=== errors, if any ==="
grep -niE "traceback|CUDA out of memory|FATAL|\[FAILED\]" logs/scaled/*.log 2>/dev/null | tail -20
echo
if ! tmux has-session -t csai_scaled 2>/dev/null && [ ! -f .pipeline_state/SCALED_PIPELINE_COMPLETE ]; then
  echo "[check_pipeline_status] pipeline is not running and not complete -- relaunching now."
  bash scaled_launch.sh
  cat logs/scaled/launcher.log | tail -3
fi
