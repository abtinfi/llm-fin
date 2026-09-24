#!/usr/bin/env bash
# Emit one line per v3b job event (for the /loop Monitor). Local helper, not
# committed: it tails the supervisor's per-job logs for stage results.
cd /home/asosoft/abtin/paper/csai/logs/supervisor || exit 1
STATE=/tmp/claude-1001/v3b_monitor_offsets
mkdir -p "$STATE"
while true; do
  for f in v3b_*.log; do
    [ -e "$f" ] || continue
    n=$(wc -l < "$f")
    s=$(cat "$STATE/$f" 2>/dev/null || echo 0)
    if [ "$n" -gt "$s" ]; then
      tail -n "+$((s + 1))" "$f" \
        | grep -E "\]\[(ok  |FAILED)\]|determinism|guard|not waiting|done,|Traceback|Error" \
        | sed "s|^|$f: |"
      echo "$n" > "$STATE/$f"
    fi
  done
  sleep 30
done
