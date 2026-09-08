#!/usr/bin/env bash
# Aim 1 seed stability (proposal section 4.4), which the status file lists as
# NOT IMPLEMENTED.
#
# Greedy decoding makes DECODE seeds byte-identical -- reporting "5 seeds,
# +/- 0.000" would imply variability was measured when it was not. The variance
# that does exist in the Aim 1 pipeline comes from the four RNGs inside
# sae.py: dictionary init, train/val split, batch order, the non-concept token
# subsample, and the random matched control feature. Those were hard-coded to
# 0 and are now wired to --seed, so this sweep measures something real.
#
# The activations are NOT recollected: the same acts npz feeds every seed, so
# any difference is the dictionary, not the corpus.
set -u
cd "$(dirname "$0")"

_lockfile=".pipeline_state/locks/seed_sweep.lock"
mkdir -p "$(dirname "$_lockfile")"
exec 8>"$_lockfile"
if ! flock -n 8; then
  echo "$(date -Is) run_seed_sweep.sh: another instance holds the lock -- exiting" >&2
  exit 0
fi

# PATH. cron does not run a login shell, so miniconda is not on PATH and
# `python` resolves to nothing -- every stage of the 2026-09-07 night run
# that cron started failed with "python: command not found" in under a
# second. Naming the interpreter directory explicitly is the fix; relying on
# the caller's environment is what broke.
export PATH="/home/asosoft/abtin/miniconda3/bin:$PATH"
export PYTHONPATH="$PWD/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

M="${MODEL_ID:-BioMistral/BioMistral-7B}"
R="${RESULTS_DIR:-results}"
LAYER="${LAYER:-20}"
SEEDS="${SEEDS:-1 2}"     # seed 0 is the committed run; do not retrain it
ACTS="$R/sae/acts_medcalc_train_L${LAYER}.npz"
STATE=".pipeline_state/seed_sweep_L${LAYER}"
mkdir -p "$STATE"

if [ ! -s "$ACTS" ]; then
  echo "[seed] $ACTS missing -- run lane B stage B2 first"; exit 1
fi

FAILED=0
for S in $SEEDS; do
  OUT="$R/sae/sae_topk_L${LAYER}_s${S}.npz"
  if [ ! -f "$STATE/train_s${S}.done" ]; then
    echo "[seed] training seed $S  $(date -Is)"
    python src/sae.py train --acts "$ACTS" --kind topk --expansion 4 --k 32 \
        --epochs 30 --batch 2048 --seed "$S" --out "$OUT" \
      && touch "$STATE/train_s${S}.done" || { FAILED=$((FAILED+1)); continue; }
  fi
  if [ ! -f "$STATE/score_s${S}.done" ]; then
    echo "[seed] scoring seed $S  $(date -Is)"
    python src/sae.py score --sae "$OUT" --model_id "$M" --top 25 \
        --causal_items -1 --causal_features 0 --causal_split test --seed "$S" \
      && touch "$STATE/score_s${S}.done" || FAILED=$((FAILED+1))
  fi
done

echo "[seed] stability report"
SAES="$R/sae/sae_topk_L${LAYER}.npz"
LABELS="seed0"
for S in $SEEDS; do
  [ -s "$R/sae/sae_topk_L${LAYER}_s${S}.npz" ] || continue
  SAES="$SAES $R/sae/sae_topk_L${LAYER}_s${S}.npz"
  LABELS="$LABELS seed${S}"
done
# shellcheck disable=SC2086
python src/stability.py --sae $SAES --labels $LABELS --top 25 \
  && touch "$STATE/ALL_DONE" || FAILED=$((FAILED+1))

echo "[seed] done, $FAILED failure(s)  $(date -Is)"
exit "$FAILED"
