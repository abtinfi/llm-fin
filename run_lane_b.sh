#!/usr/bin/env bash
# LANE B -- GPU 1. Everything scored on the REAL clinical notes (data/medcalc),
# the Aim 1 SAE (now UMLS-grounded by default, full token/item coverage), and
# the Aim 2 sufficiency runs (now uncapped -- 2026-09-02 scale-up).
#
# Owns results/summary_{test,heldout}_medcalc.json and the results/sae/ tree.
# Never writes lane A's summary files, adapters, or constraint JSONs -- see the
# ownership note at the top of run_lane_a.sh.
#
# The MedCalc constraint-layer rows need adapter_qt.npz and adapter_renal.npz,
# which lane A trains first. Rather than assume the timing, this script BLOCKS
# on those files existing and being non-empty.
#
# RESUME: each `stage` call is checkpointed to .pipeline_state/scaled_b/. A
# rerun of this script (e.g. after a 3am reboot) skips every stage whose
# marker file already exists and continues from the first incomplete one.
# MULTI-MODEL. `MODEL_ID`, `MODEL_TAG`, `RESULTS_DIR` and `LAYER` are read
# from the environment. Left unset they reproduce the BioMistral-7B run exactly
# -- same model, same layer 20, same `results/` paths -- so nothing already on
# disk moves. A second model is run by setting them:
#
#   MODEL_ID=aaditya/Llama3-OpenBioLLM-8B MODEL_TAG=llama3-openbiollm-8b \
#   RESULTS_DIR=results/models/llama3-openbiollm-8b bash run_lane_b.sh
#
# Artifact names carry the layer (`_L${LAYER}`) but NOT the model: the model is
# carried by RESULTS_DIR, so two models never share a filename and the
# BioMistral paths quoted throughout the docs stay literally correct.
set -x
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export PYTHONPATH=/home/asosoft/abtin/paper/csai/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/asosoft/abtin/paper/csai
M="${MODEL_ID:-BioMistral/BioMistral-7B}"
TAG="${MODEL_TAG:-biomistral-7b}"
R="${RESULTS_DIR:-results}"
LAYER="${LAYER:-20}"
ADAPTER_LAYER="${ADAPTER_LAYER:-30}"
mkdir -p "$R"
V6="base rag sym uq nsai nsai_uq"

STATE_DIR=".pipeline_state/scaled_b_${TAG}_L${LAYER}"
mkdir -p "$STATE_DIR"

banner () { echo; echo "########## [B] $* ##########"; echo; }

FAILED=0

stage () {
  local name="$1"; shift
  local marker="$STATE_DIR/${name}.done"
  if [ -f "$marker" ]; then
    echo "[B][skip] $name already done ($marker)"
    return 0
  fi
  banner "$name"
  if "$@"; then
    touch "$marker"
    echo "[B][ok] $name"
  else
    echo "[B][FAILED] $name -- no marker written, will retry on next launch"
    FAILED=$((FAILED + 1))
  fi
}

wait_for () {   # $1 = file, $2 = human name
  local waited=0
  while [ ! -s "$1" ]; do
    if [ $waited -eq 0 ]; then
      echo "[B] waiting for lane A to produce $2 ($1) ..."
    fi
    sleep 30; waited=$((waited+30))
    if [ $waited -ge 7200 ]; then
      echo "[B] FATAL: $2 still missing after 2h; lane A has probably failed."
      exit 1
    fi
  done
  echo "[B] $2 is present after ${waited}s"
}

# --- 1. MedCalc ablation, new default UQ signal -----------------------------
stage B1_medcalc_ablation_test \
  python src/run_eval.py --backend hf --model_id $M --out "$R" --seeds 0 --split test \
    --data data/medcalc --gate medcalc --tag _medcalc --variants $V6 \
    --batch_size 4
stage B1_medcalc_ablation_heldout \
  python src/run_eval.py --backend hf --model_id $M --out "$R" --seeds 0 --split heldout \
    --data data/medcalc --gate medcalc --tag _medcalc --variants $V6 \
    --batch_size 4

# --- 2. Aim 1 SAE, UMLS-grounded vocabulary by default, full coverage ------
stage B2_sae_collect \
  python src/sae.py collect --data data/medcalc --split train --layer "$LAYER" \
    --model_id $M --other_mult 1000 --max_tokens 300000 \
    --out "$R"/sae/acts_medcalc_train_L${LAYER}.npz
stage B3_sae_train_topk \
  python src/sae.py train --acts "$R"/sae/acts_medcalc_train_L${LAYER}.npz \
    --kind topk --expansion 4 --k 32 --epochs 30 --batch 2048 \
    --out "$R"/sae/sae_topk_L${LAYER}.npz
stage B4_sae_train_jumprelu \
  python src/sae.py train --acts "$R"/sae/acts_medcalc_train_L${LAYER}.npz \
    --kind jumprelu --expansion 4 --epochs 30 --batch 2048 \
    --out "$R"/sae/sae_jumprelu_L${LAYER}.npz
stage B5_sae_score_topk \
  python src/sae.py score --sae "$R"/sae/sae_topk_L${LAYER}.npz --model_id $M \
    --top 25 --causal_items -1 --causal_features 0 --causal_split test
stage B6_sae_score_jumprelu \
  python src/sae.py score --sae "$R"/sae/sae_jumprelu_L${LAYER}.npz --model_id $M \
    --top 25 --causal_items 0

# --- 3. MedCalc constraint-layer rows (needs lane A's adapters) -------------
wait_for "$R"/adapter_qt.npz    "adapter_qt"
wait_for "$R"/adapter_renal.npz "adapter_renal"
stage B7_medcalc_cl_heldout \
  python src/run_eval.py --backend hf --model_id $M --out "$R" --seeds 0 --split heldout \
    --data data/medcalc --gate medcalc --tag _medcalc \
    --variants cl nsai_uq_cl --adapter "$R"/adapter_qt.npz \
    --adapter_layer "$ADAPTER_LAYER" --batch_size 4
stage B7_medcalc_cl_test \
  python src/run_eval.py --backend hf --model_id $M --out "$R" --seeds 0 --split test \
    --data data/medcalc --gate medcalc --tag _medcalc \
    --variants cl nsai_uq_cl --adapter "$R"/adapter_renal.npz \
    --adapter_layer "$ADAPTER_LAYER" --batch_size 4

# --- 4. Aim 2 causal sufficiency, uncapped -- ALL complete pairs -----------
stage B8_sufficiency_test \
  python src/patching.py --mode sufficiency \
    --data data/medcalc/counterfactual_test.jsonl --model_id $M \
    --limit 0 --alphas 0.5,1,2,4 \
    --out "$R"/sufficiency_medcalc_test.json
stage B8_sufficiency_heldout \
  python src/patching.py --mode sufficiency \
    --data data/medcalc/counterfactual_heldout.jsonl --model_id $M \
    --limit 0 --alphas 0.5,1,2,4 \
    --out "$R"/sufficiency_medcalc_heldout.json

# --- 5. Aim 2 necessity / ACE, uncapped -- re-verify under current code ----
stage B9_necessity_test \
  python src/patching.py \
    --data data/medcalc/counterfactual_test.jsonl --model_id $M \
    --limit 0 --out "$R"/patching_medcalc_test.json
stage B9_necessity_heldout \
  python src/patching.py \
    --data data/medcalc/counterfactual_heldout_all.jsonl --model_id $M \
    --limit 0 --out "$R"/patching_medcalc_heldout.json

if [ "$FAILED" -eq 0 ]; then
  touch "$STATE_DIR/ALL_DONE"
fi
banner "LANE B DONE ($FAILED stage(s) failed)"
exit "$FAILED"
