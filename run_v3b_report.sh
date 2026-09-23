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

# The audits, re-run on the finished outputs so the results are checked even
# if no interactive session survives to do it (the 03:01 reboot ends it).
# Counts and rates only; aggregate, so the file is tracked.
A="$WT/results/mimic_v3b/AUDIT_CHECKS.md"
{
  echo "# Automated audit of the v3b outputs ($(date -Is))"
  echo; echo '```'
  python "$WT/src/audit_mimic_labels.py" "$WT/data/mimic_v3b" "$WT/data/mimic_v3b_note"
  python "$WT/src/audit_mimic_dupes.py" "$WT/data/mimic_v3b"
  for tag in biomistral-7b llama3-openbiollm-8b mistral-7b-instruct-v0-2; do
    for split in test heldout; do
      python "$WT/src/audit_mimic_preds.py" "$WT/data/mimic_v3b" \
        "$WT/results/mimic_v3b/$tag" _mimic3b "$split"
      python "$WT/src/audit_mimic_preds.py" "$WT/data/mimic_v3b_note" \
        "$WT/results/mimic_v3b/$tag" _mimic3bnote "$split"
    done
  done
  echo '```'
} > "$A" 2>&1 || FAILED=$((FAILED + 1))
# The checks that must hold on a correct v3b run; any hit is a FAILED job.
if grep -qE "MISMATCH|WRONG LABELS|carrying BOTH labels: [1-9]|with DIFFERENT preds: [1-9]|'missing': [1-9]|'extra': [1-9]|'duplicate_ids': [1-9]|'label_mismatch': [1-9]" "$A"; then
  echo "[v3b report] AUDIT FOUND A PROBLEM -- see $A"; FAILED=$((FAILED + 1))
else
  echo "[v3b report] audit clean"
fi
echo "[v3b report] done, $FAILED failed  $(date -Is)"
[ "$FAILED" -eq 0 ] && touch "$STATE/ALL_DONE"
exit "$FAILED"
