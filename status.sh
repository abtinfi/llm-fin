#!/usr/bin/env bash
# Where everything stands. Read-only, safe to run at any time, needs no
# Claude session -- the work runs under cron + tmux and is independent of it.
set -u
cd "$(dirname "$0")"

echo "================= csai pipeline status  $(date -Is) ================="
echo
echo "--- is a supervisor alive? ---"
if flock -n .pipeline_state/supervisor.lock -c true 2>/dev/null; then
  echo "  NO supervisor running."
  if [ -e .pipeline_state/SUPERVISOR_ALL_DONE ]; then
    echo "  All queued work is complete."
  else
    echo "  Work is unfinished. The */15 cron watchdog will restart it within"
    echo "  15 minutes, or start it now:  bash supervisor_launch.sh"
  fi
else
  echo "  YES -- a supervisor holds .pipeline_state/supervisor.lock"
fi
echo
echo "--- queue ---"
if [ -f .pipeline_state/queue.txt ]; then
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    marker="${line%%|*}"; name="$(echo "$line" | cut -d'|' -f2)"
    if [ -e "$marker" ]; then st="DONE   "
    elif [ -d ".pipeline_state/claims/$name" ]; then st="RUNNING"
    else st="PENDING"; fi
    printf "  [%s] %s\n" "$st" "$name"
  done < .pipeline_state/queue.txt
else
  echo "  (no queue yet; the supervisor writes it at startup)"
fi
echo
echo "--- GPUs ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader 2>/dev/null | sed 's/^/  /' || echo "  nvidia-smi unavailable"
echo
echo "--- GPU processes of ours ---"
ps -eo pid,etime,cmd | grep -E "run_model|run_medcalc_v2|run_sae_wide|run_lane|run_eval|patching|sae\.py|attribution" \
  | grep -v grep | cut -c1-110 | sed 's/^/  /' || echo "  none"
echo
echo "--- last 8 supervisor events ---"
tail -8 logs/supervisor/supervisor.log 2>/dev/null | sed 's/^/  /' || echo "  (none)"
echo
echo "--- failures worth looking at ---"
grep -lniE "traceback|CUDA out of memory|FATAL|\[FAILED\]" logs/supervisor/*.log \
  logs/models/*/*.log 2>/dev/null | sed 's/^/  /' || echo "  none found"
echo
echo "--- generated reports ---"
for f in PAPER.md PROPOSAL_ERRATA.md HUMAN_EVAL_PROTOCOL.md \
         results/SUMMARY.md results/attribution.md \
         results/COMPARISON_BASE_VS_PROPOSED.md; do
  [ -f "$f" ] && printf "  %-46s %s\n" "$f" "$(date -r "$f" '+%Y-%m-%d %H:%M')"
done
echo
echo "Next reboot is 03:01 daily; @reboot + a */15 watchdog restart the"
echo "supervisor, and every stage is checkpointed, so it resumes not restarts."
