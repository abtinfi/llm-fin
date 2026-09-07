#!/usr/bin/env bash
# Score the SAE dictionary causally on the HELD-OUT (QT) split, and re-run the
# section 4.2 attribution layer with those split-matched weights.
#
# WHY THIS EXISTS -- a confound in the attribution layer, found 2026-09-07.
#
# The layer weights each feature by its measured knock-out excess, which is
# the right idea and was measured the wrong way: `sae.py score` was run with
# `--causal_split test`, and that split contains ONLY metformin_renal items.
# A qt_interval feature cannot move the decision on items whose decision does
# not depend on QT, so it scores zero excess there by construction.
#
# The attribution report then said the layer "cannot express qt_interval",
# which silently merged two different claims:
#   (a) the dictionary contains no QT feature       -- a discovery claim
#   (b) QT features were scored where QT is inert   -- a measurement artifact
#
# Widening the vocabulary from 25 to 100 features did not separate them: QT
# still had only 2 features, both with negative excess. This run does separate
# them, by scoring the same dictionary on the split where QT IS the decisive
# variable. If QT features stay at zero here, (a) is established; if they do
# not, the earlier report was measuring (b) and must be corrected.
#
# The dictionary is NOT retrained, so S_semantic must reproduce exactly -- that
# is the regression check that only the causal stage moved.
set -u
cd "$(dirname "$0")"

_lockfile=".pipeline_state/locks/sae_split_weights.lock"
mkdir -p "$(dirname "$_lockfile")"
exec 8>"$_lockfile"
if ! flock -n 8; then
  echo "$(date -Is) run_sae_split_weights.sh: another instance holds the lock -- exiting" >&2
  exit 0
fi

export PYTHONPATH="$PWD/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${GPU:-1}"

M="${MODEL_ID:-BioMistral/BioMistral-7B}"
R="${RESULTS_DIR:-results}"
LAYER="${LAYER:-20}"
TOP="${TOP:-100}"
STATE=".pipeline_state/sae_split_weights_L${LAYER}"
mkdir -p "$STATE" "$R/sae_heldout"

if [ ! -f "$STATE/score.done" ]; then
  echo "[qtw] scoring top $TOP features against the HELD-OUT split  $(date -Is)"
  python src/sae.py score --sae "$R"/sae/sae_topk_L${LAYER}.npz --model_id "$M" \
      --top "$TOP" --causal_items -1 --causal_features 0 \
      --causal_split heldout \
      --out "$R"/sae_heldout/sae_topk_L${LAYER}_fis.json \
    && touch "$STATE/score.done" || { echo "[qtw] score FAILED"; exit 1; }
fi

echo "[qtw] attribution on the held-out split with split-matched weights"
python src/attribution.py \
    --data data/medcalc/counterfactual_heldout.jsonl \
    --sae "$R"/sae/sae_topk_L${LAYER}.npz \
    --fis "$R"/sae_heldout/sae_topk_L${LAYER}_fis.json \
    --model_id "$M" --limit 0 --occlusion_items 30 \
    --out "$R"/faithfulness_medcalc_heldout_biomistral-7b_qtweights.json \
  && touch "$STATE/attr.done" || echo "[qtw] attribution FAILED"

echo "[qtw] done $(date -Is)"
