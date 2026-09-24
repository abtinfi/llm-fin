#!/usr/bin/env bash
# Re-run the UQ rows of the MedCalc and MIMIC arms under the conformal
# calibration rule (S3 fix, 2026-09-08).
#
# SCOPE. Only `uq`, `nsai_uq` and `nsai_uq_cl` are re-run. The calibration rule
# governs the abstention threshold and nothing else, so `base`, `rag`, `sym`,
# `nsai` and `cl` are bit-for-bit unaffected -- run_eval.py merges by
# (variant, seed), so those rows survive untouched. Re-running them would burn
# GPU time to reproduce identical numbers and would risk moving a row that is
# already reported.
#
# NO RETRAINING. Existing adapters are loaded from disk:
#   medcalc test    -> results/adapter_renal.npz   (the renal family)
#   mimic  test     -> results/adapter_mimic.npz
# These are the exact adapters the stored summaries record, so the only thing
# that changes between the old row and the new one is the threshold.
#
# CONFIG IS COPIED FROM THE STORED SUMMARIES, not guessed:
#   medcalc -> --gate medcalc (real clinical prose, runs the equation itself)
#   mimic   -> --gate rules
#   both    -> --uq_signal decision_entropy (the default since 2026-09-01)
#
# EXPECTED, recorded before the run: coverage collapses onto gate_fired_rate
# and CC is unchanged, as it did on synthetic_control for all five gated rows.
# `uq` (ungated) is expected to go to zero coverage on both arms.
set -u
cd "$(dirname "$0")"

_lockfile=".pipeline_state/locks/rerun_uq_arms.lock"
mkdir -p "$(dirname "$_lockfile")"
exec 8>"$_lockfile"
if ! flock -n 8; then
  echo "$(date -Is) rerun_uq_arms.sh: another instance holds $_lockfile -- exiting" >&2
  exit 0
fi

export PATH="/home/asosoft/abtin/miniconda3/bin:$PATH"
export PYTHONPATH=/home/asosoft/abtin/paper/csai/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

M="${MODEL_ID:-BioMistral/BioMistral-7B}"
R="${RESULTS_DIR:-results}"
STATE=".pipeline_state/rerun_uq_arms"
SNAP="$R/pre_s1s3_20260908"
mkdir -p "$STATE" "$SNAP" logs/rerun_uq_arms

stage () {
  local name="$1"; shift
  if [ -f "$STATE/$name.done" ]; then echo "[uqarms] $name already done"; return 0; fi
  echo "[uqarms] $name start $(date -Is)"
  if "$@" > "logs/rerun_uq_arms/$name.log" 2>&1; then
    touch "$STATE/$name.done"; echo "[uqarms] $name done $(date -Is)"
  else
    echo "[uqarms] $name FAILED -- see logs/rerun_uq_arms/$name.log" >&2; exit 1
  fi
}

# --- 0. snapshot the legacy-rule artifacts before overwriting ---------------
if [ ! -f "$STATE/snapshot.done" ]; then
  for t in _medcalc _mimic; do
    cp -n "$R"/summary_test${t}.json "$SNAP"/ 2>/dev/null || true
    for v in uq nsai_uq nsai_uq_cl; do
      cp -n "$R"/preds_test_${v}_seed0${t}.jsonl "$SNAP"/ 2>/dev/null || true
    done
  done
  touch "$STATE/snapshot.done"
  echo "[uqarms] legacy artifacts snapshotted to $SNAP"
fi

# --- 1. MedCalc test --------------------------------------------------------
stage 1_medcalc_uq \
  python src/run_eval.py --backend hf --model_id "$M" --out "$R" --seeds 0 \
    --split test --data data/medcalc --gate medcalc --tag _medcalc \
    --variants uq nsai_uq --batch_size 8

stage 2_medcalc_uqcl \
  python src/run_eval.py --backend hf --model_id "$M" --out "$R" --seeds 0 \
    --split test --data data/medcalc --gate medcalc --tag _medcalc \
    --variants nsai_uq_cl --adapter "$R"/adapter_renal.npz \
    --adapter_layer "${ADAPTER_LAYER:-30}" --batch_size 8

# --- 2. MIMIC test ----------------------------------------------------------
stage 3_mimic_uq \
  python src/run_eval.py --backend hf --model_id "$M" --out "$R" --seeds 0 \
    --split test --data data/mimic --gate rules --tag _mimic \
    --variants uq nsai_uq --batch_size 8

stage 4_mimic_uqcl \
  python src/run_eval.py --backend hf --model_id "$M" --out "$R" --seeds 0 \
    --split test --data data/mimic --gate rules --tag _mimic \
    --variants nsai_uq_cl --adapter "$R"/adapter_mimic.npz \
    --adapter_layer "${ADAPTER_LAYER:-30}" --batch_size 8

# --- 3. tables --------------------------------------------------------------
stage 5_tables \
  bash -c 'python src/make_table.py --split test --tag _medcalc --out results/table_medcalc_test.md &&
           python src/make_table.py --split test --tag _mimic   --out results/table_mimic_test.md'

touch "$STATE/ALL_DONE"
echo "[uqarms] ALL DONE $(date -Is)"
