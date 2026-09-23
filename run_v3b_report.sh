#!/usr/bin/env bash
# Cross-model tables and the audited report for the corrected v3b arms, once
# every v3b job has finished. A supervisor job, not a session task, because
# the session does not survive the 03:01 reboot. Queued LAST, so every job it
# waits for is already claimed by a worker; it can idle one card but cannot
# deadlock the queue.
set -u
WT="$(cd "$(dirname "$0")" && pwd)"
CSAI=/home/asosoft/abtin/paper/csai
cd "$CSAI" || exit 1

exec 8>"$CSAI/.pipeline_state/locks/v3b_report.lock"
flock -n 8 || exit 0
export PATH="/home/asosoft/abtin/miniconda3/bin:$PATH"

NEED=()
for kind in main note cl; do
  for tag in biomistral-7b llama3-openbiollm-8b mistral-7b-instruct-v0-2; do
    NEED+=("$CSAI/.pipeline_state/v3b_${kind}_${tag}/ALL_DONE")
  done
done
waited=0
while :; do
  missing=0
  for m in "${NEED[@]}"; do [ -e "$m" ] || missing=$((missing + 1)); done
  [ "$missing" -eq 0 ] && break
  [ $((waited % 3600)) -eq 0 ] && echo "[v3b report] waiting: $missing of ${#NEED[@]} not done  $(date -Is)"
  sleep 300; waited=$((waited + 300))
done

STATE="$CSAI/.pipeline_state/v3b_report"
mkdir -p "$STATE"
FAILED=0
for arm in v3 note; do
  python "$WT/src/make_comparison_mimic3.py" "$arm" || FAILED=$((FAILED + 1))
done
python "$WT/src/report_mimic3.py" || FAILED=$((FAILED + 1))
echo "[v3b report] done, $FAILED failed  $(date -Is)"
[ "$FAILED" -eq 0 ] && touch "$STATE/ALL_DONE"
exit "$FAILED"
