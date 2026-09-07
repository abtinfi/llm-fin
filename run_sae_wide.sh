#!/usr/bin/env bash
# Re-score the existing topk dictionary with a WIDER feature set.
#
# WHY. The 2026-09-02 scoring used `--top 25`, i.e. the 25 features with the
# highest S_semantic. Those 25 turn out to be 12 `age` features, 5 `drug`, 2
# `creatinine`, 2 `heart_rate` -- and, after clipping features whose knock-out
# excess does not beat their matched random control, ZERO for `qt_interval`,
# `egfr`, `inr`, `potassium`, `pregnancy`, `asthma`, `renal_disease`.
#
# That is not a neutral limit. The section 4.2 attribution layer can only name
# a concept it has a feature for, so on the QT held-out family it scored 0.0
# concept-pointing accuracy against a 0.667 chance baseline -- a vocabulary
# gap, not an attribution failure. Widening to 100 features is the direct test
# of whether the dictionary contains QT features at all further down the
# S_semantic ranking.
#
# The dictionary itself is NOT retrained: `S_semantic` must reproduce to the
# last digit, which is the regression check that only the causal stage moved.
set -u
cd "$(dirname "$0")"

# SINGLE-INSTANCE LOCK. This script can be launched three ways -- by hand, by
# csai_supervisor.sh, or by the @reboot cron -- and two copies would write the
# same checkpoint directory and the same output files. Taking the lock here
# rather than in the caller means the guarantee holds however it was started.
# A second copy exits 0: "already running" is success, not failure.
_lockfile=".pipeline_state/locks/sae_wide_L${LAYER:-20}.lock"
mkdir -p "$(dirname "$_lockfile")"
exec 8>"$_lockfile"
if ! flock -n 8; then
  echo "$(date -Is) run_sae_wide.sh: another instance holds $_lockfile -- exiting" >&2
  exit 0
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${GPU:-1}"

M="${MODEL_ID:-BioMistral/BioMistral-7B}"
R="${RESULTS_DIR:-results}"
LAYER="${LAYER:-20}"
TOP="${TOP:-100}"
STATE=".pipeline_state/sae_wide_L${LAYER}"
mkdir -p "$STATE" "$R/sae_wide"

if [ ! -f "$STATE/score.done" ]; then
  echo "[wide] scoring top $TOP features  $(date -Is)"
  python src/sae.py score --sae "$R"/sae/sae_topk_L${LAYER}.npz --model_id "$M" \
      --top "$TOP" --causal_items -1 --causal_features 0 --causal_split test \
      --out "$R"/sae_wide/sae_topk_L${LAYER}_fis.json \
    && touch "$STATE/score.done" || { echo "[wide] FAILED"; exit 1; }
fi

echo "[wide] re-running the section 4.2 attribution layer on the wider vocabulary"
for split in test heldout; do
  python src/attribution.py \
      --data "data/medcalc/counterfactual_${split}.jsonl" \
      --sae "$R"/sae/sae_topk_L${LAYER}.npz \
      --fis "$R"/sae_wide/sae_topk_L${LAYER}_fis.json \
      --model_id "$M" --limit 0 --occlusion_items 30 \
      --out "$R"/faithfulness_medcalc_${split}_biomistral-7b_wide.json \
    || echo "[wide] attribution $split FAILED"
done
echo "[wide] done $(date -Is)"
