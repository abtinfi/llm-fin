#!/usr/bin/env bash
# LANE B -- GPU 1. Everything scored on the REAL clinical notes (data/medcalc),
# the Aim 1 SAE, and the new Aim 2 sufficiency runs.
#
# Owns results/summary_{test,heldout}_medcalc.json and the results/sae/ tree.
# Never writes lane A's summary files, adapters, or constraint JSONs -- see the
# ownership note at the top of run_lane_a.sh.
#
# The MedCalc constraint-layer rows need adapter_qt.npz and adapter_renal.npz,
# which lane A trains first. Rather than assume the timing, this script BLOCKS
# on those files existing and being non-empty. The MedCalc ablation and the
# whole SAE stage run before the wait, so in practice the wait is already
# satisfied by the time it is reached.
set -x
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/asosoft/abtin/paper/csai/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/asosoft/abtin/paper/csai
M="BioMistral/BioMistral-7B"
V6="base rag sym uq nsai nsai_uq"

banner () { echo; echo "########## [B] $* ##########"; echo; }

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
banner "B1 MedCalc ablation on real notes, 6 variants, decision_entropy"
python src/run_eval.py --backend hf --model_id $M --seeds 0 --split test \
    --data data/medcalc --gate medcalc --tag _medcalc --variants $V6 \
    --batch_size 4
python src/run_eval.py --backend hf --model_id $M --seeds 0 --split heldout \
    --data data/medcalc --gate medcalc --tag _medcalc --variants $V6 \
    --batch_size 4

# --- 2. Aim 1 SAE -----------------------------------------------------------
# Included in full. It is independent of every code change in this pass, so it
# is expected to reproduce the 2026-09-01 numbers; running it makes that a
# check rather than an assumption.
banner "B2 SAE collect (layer 20, 150k tokens)"
python src/sae.py collect --data data/medcalc --split train --layer 20 \
    --model_id $M --other_mult 1000 --max_tokens 150000 \
    --out results/sae/acts_medcalc_train_L20.npz
banner "B3 SAE train: topk"
python src/sae.py train --acts results/sae/acts_medcalc_train_L20.npz \
    --kind topk --expansion 4 --k 32 --epochs 30 --batch 2048 \
    --out results/sae/sae_topk_L20.npz
banner "B4 SAE train: jumprelu"
python src/sae.py train --acts results/sae/acts_medcalc_train_L20.npz \
    --kind jumprelu --expansion 4 --epochs 30 --batch 2048 \
    --out results/sae/sae_jumprelu_L20.npz
banner "B5 SAE score: topk (with feature knock-out = causal necessity)"
python src/sae.py score --sae results/sae/sae_topk_L20.npz --model_id $M \
    --top 25 --causal_items 40 --causal_features 8 --causal_split test
banner "B6 SAE score: jumprelu"
python src/sae.py score --sae results/sae/sae_jumprelu_L20.npz --model_id $M \
    --top 25 --causal_items 0

# --- 3. MedCalc constraint-layer rows (needs lane A's adapters) -------------
wait_for results/adapter_qt.npz    "adapter_qt"
wait_for results/adapter_renal.npz "adapter_renal"
banner "B7 MedCalc constraint-layer rows, identical readout to every other row"
python src/run_eval.py --backend hf --model_id $M --seeds 0 --split heldout \
    --data data/medcalc --gate medcalc --tag _medcalc \
    --variants cl nsai_uq_cl --adapter results/adapter_qt.npz \
    --adapter_layer 30 --batch_size 4
python src/run_eval.py --backend hf --model_id $M --seeds 0 --split test \
    --data data/medcalc --gate medcalc --tag _medcalc \
    --variants cl nsai_uq_cl --adapter results/adapter_renal.npz \
    --adapter_layer 30 --batch_size 4

# --- 4. Aim 2 causal sufficiency (step 3, new) ------------------------------
banner "B8 causal sufficiency: injection with dose sweep + matched controls"
python src/patching.py --mode sufficiency \
    --data data/medcalc/counterfactual_test.jsonl --model_id $M \
    --limit 40 --alphas 0.5,1,2,4 \
    --out results/sufficiency_medcalc_test.json
python src/patching.py --mode sufficiency \
    --data data/medcalc/counterfactual_heldout.jsonl --model_id $M \
    --limit 40 --alphas 0.5,1,2,4 \
    --out results/sufficiency_medcalc_heldout.json

banner "LANE B DONE"
