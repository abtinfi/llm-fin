#!/usr/bin/env bash
# Build the cross-model comparison tables for the v3.1 and real-note arms once
# every v3 job has finished.
#
# WHY THIS IS A SUPERVISOR JOB. The GPU jobs survive the 03:01 reboot through
# cron; an interactive session does not. If the final tables were left to a
# session they might never be built. Queued LAST, so by the time a worker
# claims it every job it waits for has already been claimed by a worker --
# waiting here can idle one card but cannot deadlock the queue. A job that
# failed is retried on the next supervisor start (reboot), and so is this.
set -u
WT="$(cd "$(dirname "$0")" && pwd)"
CSAI=/home/asosoft/abtin/paper/csai
cd "$CSAI" || exit 1

_lockfile="$CSAI/.pipeline_state/locks/mimic_v3_report.lock"
mkdir -p "$(dirname "$_lockfile")"
exec 8>"$_lockfile"
flock -n 8 || exit 0

export PATH="/home/asosoft/abtin/miniconda3/bin:$PATH"

NEED=()
for kind in mimic_v3 mimic_v3_note mimic_v3_cl; do
  for tag in biomistral-7b llama3-openbiollm-8b mistral-7b-instruct-v0-2; do
    NEED+=("$CSAI/.pipeline_state/${kind}_${tag}/ALL_DONE")
  done
done

waited=0
while :; do
  missing=0
  for m in "${NEED[@]}"; do [ -e "$m" ] || missing=$((missing + 1)); done
  [ "$missing" -eq 0 ] && break
  [ $((waited % 3600)) -eq 0 ] && echo "[v3report] waiting: $missing of ${#NEED[@]} jobs not done  $(date -Is)"
  sleep 300; waited=$((waited + 300))
done

STATE="$CSAI/.pipeline_state/mimic_v3_report"
mkdir -p "$STATE"
FAILED=0
for arm in v3 note; do
  if python "$WT/src/make_comparison_mimic3.py" "$arm"; then
    echo "[v3report] comparison $arm ok"
  else
    echo "[v3report] comparison $arm FAILED"; FAILED=$((FAILED + 1))
  fi
done
echo "[v3report] done, $FAILED failed  $(date -Is)"
[ "$FAILED" -eq 0 ] && touch "$STATE/ALL_DONE"
exit "$FAILED"
