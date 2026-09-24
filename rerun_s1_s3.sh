#!/usr/bin/env bash
# Regenerate the artifacts invalidated by the S1 and S3 fixes of 2026-09-08.
#
# WHAT CHANGED AND WHY THESE STAGES EXIST
# ---------------------------------------
# S1 (src/sae.py) -- the knock-out negative control was drawn UNIFORMLY over the
#   whole dictionary while its docstring promised a control "of similar firing
#   rate". ~30-42% of this dictionary never fires, so the control routinely
#   knocked out a dead feature, measured exactly 0.0000, and left
#   `excess = eff - ctrl` equal to the raw effect. Six of twelve sampled top
#   features in the old results/sae/sae_topk_L20_fis.json have a control of
#   exactly zero. Every S_causal, and so every FIS, was inflated -- always in
#   the direction of the claim. Controls are now matched on firing rate and
#   averaged over 5 draws.
#
# S3 (src/metrics.py) -- the abstention threshold was the raw empirical optimum
#   on the calibration split with no finite-sample correction, and its
#   no-solution fallback was inverted (it returned the SMALLEST uncertainty,
#   which still answers the item known to be wrong). It is visible in the
#   artifacts it produced: summary_test.json records the `uq` row calibrating
#   to error 0.25 against a target of 0.10. The threshold is now certified by
#   an exact Clopper-Pearson bound on selective error.
#
# EXPECTED DIRECTION, recorded before the run so the result can falsify it:
#   * FIS values DROP. Controls now measure a real effect, so `excess` shrinks;
#     some features will go to s_causal = 0 (the smoke test already flipped
#     feature 13297 from +0.0100 to negative).
#   * Several UQ rows go to ZERO coverage. At alpha=0.10, delta=0.10 no
#     threshold is certifiable until 22 retained items are clean, and the
#     gated calibration pools here are 12-50 items. That is a calibration-set
#     SIZE limit, not a model result.
#
# The pre-change artifacts are snapshotted FIRST, so the before/after
# comparison survives the overwrite.
#
# Reproducing the old numbers needs no checkout: pass
#   sae.py score --control_mode uniform
#   run_eval.py  --calib_rule legacy
set -u
cd "$(dirname "$0")"

# SINGLE-INSTANCE LOCK. Launchable by hand, by csai_supervisor.sh, or by the
# @reboot cron; two copies would write the same checkpoints and outputs. A
# second copy exits 0 -- "already running" is success.
_lockfile=".pipeline_state/locks/rerun_s1_s3.lock"
mkdir -p "$(dirname "$_lockfile")"
exec 8>"$_lockfile"
if ! flock -n 8; then
  echo "$(date -Is) rerun_s1_s3.sh: another instance holds $_lockfile -- exiting" >&2
  exit 0
fi

# PATH. cron does not run a login shell, so miniconda is not on PATH and
# `python` resolves to nothing.
export PATH="/home/asosoft/abtin/miniconda3/bin:$PATH"
export PYTHONPATH=/home/asosoft/abtin/paper/csai/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

M="${MODEL_ID:-BioMistral/BioMistral-7B}"
R="${RESULTS_DIR:-results}"
LAYER="${LAYER:-20}"
DATA="${DATA:-data/synthetic_control}"
STATE=".pipeline_state/rerun_s1_s3"
SNAP="$R/pre_s1s3_20260908"
mkdir -p "$STATE" "$SNAP" logs/rerun_s1_s3

stage () {
  local name="$1"; shift
  if [ -f "$STATE/$name.done" ]; then
    echo "[s1s3] $name already done -- skipping"
    return 0
  fi
  echo "[s1s3] $name  start $(date -Is)"
  if "$@" > "logs/rerun_s1_s3/$name.log" 2>&1; then
    touch "$STATE/$name.done"
    echo "[s1s3] $name  done  $(date -Is)"
  else
    echo "[s1s3] $name  FAILED -- see logs/rerun_s1_s3/$name.log" >&2
    exit 1
  fi
}

# --- 0. snapshot the pre-change artifacts ----------------------------------
if [ ! -f "$STATE/snapshot.done" ]; then
  echo "[s1s3] snapshotting pre-change artifacts to $SNAP"
  cp -n "$R"/sae/sae_topk_L${LAYER}_fis.json "$SNAP"/ 2>/dev/null || true
  cp -n "$R"/summary_test.json "$SNAP"/ 2>/dev/null || true
  cp -n "$R"/summary_heldout.json "$SNAP"/ 2>/dev/null || true
  for v in uq nsai_uq nsai_uq_cl; do
    cp -n "$R"/preds_test_${v}_seed0.jsonl "$SNAP"/ 2>/dev/null || true
    cp -n "$R"/preds_heldout_${v}_seed0.jsonl "$SNAP"/ 2>/dev/null || true
  done
  touch "$STATE/snapshot.done"
fi

# --- 1. S1: FIS re-score with matched controls ------------------------------
# Flags identical to run_full_pipeline.sh:244-248, so the ONLY thing that moves
# is the control. S_semantic must reproduce to the last digit -- that is the
# regression check that the causal stage alone changed.
stage 1_fis_topk_matched \
  python src/sae.py score --sae "$R"/sae/sae_topk_L${LAYER}.npz --model_id "$M" \
    --top 25 --causal_items -1 --causal_features 0 --causal_split test \
    --control_mode matched --n_controls 5 \
    --out "$R"/sae/sae_topk_L${LAYER}_fis.json

# --- 2. S3: UQ rows, conformal threshold ------------------------------------
# `uq` and `nsai_uq` need no adapter; `nsai_uq_cl` does. Split into two calls
# so a missing adapter cannot block the two rows that do not need one.
stage 2_uq_test \
  python src/run_eval.py --backend hf --model_id "$M" --out "$R" --seeds 0 \
    --split test --data "$DATA" --variants uq nsai_uq --batch_size 8

stage 3_uq_heldout \
  python src/run_eval.py --backend hf --model_id "$M" --out "$R" --seeds 0 \
    --split heldout --data "$DATA" --variants uq nsai_uq --batch_size 8

if [ -f "$R/adapter_synth.npz" ]; then
  stage 4_uqcl_test \
    python src/run_eval.py --backend hf --model_id "$M" --out "$R" --seeds 0 \
      --split test --data "$DATA" --variants nsai_uq_cl \
      --adapter "$R"/adapter_synth.npz --adapter_layer "${ADAPTER_LAYER:-30}" \
      --batch_size 8
  stage 5_uqcl_heldout \
    python src/run_eval.py --backend hf --model_id "$M" --out "$R" --seeds 0 \
      --split heldout --data "$DATA" --variants nsai_uq_cl \
      --adapter "$R"/adapter_synth.npz --adapter_layer "${ADAPTER_LAYER:-30}" \
      --batch_size 8
else
  echo "[s1s3] no $R/adapter_synth.npz -- skipping the nsai_uq_cl rows" >&2
fi

touch "$STATE/ALL_DONE"
echo "[s1s3] ALL DONE $(date -Is)"
echo "[s1s3] pre-change artifacts preserved in $SNAP"
