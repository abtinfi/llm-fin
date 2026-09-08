#!/usr/bin/env bash
# LANE A -- GPU 0. Constraint layers, SYNTHETIC CONTROL ablation, RQ3
# perplexity, then the new MIMIC-IV Real-Value Cohort ablation (2026-09-02
# scale-up: full coverage, no item caps, checkpointed for reboot resilience).
#
# FILE OWNERSHIP (this is what makes two-GPU parallelism safe):
#   run_eval.py MERGES into results/summary_{split}{tag}.json rather than
#   overwriting it, so two concurrent processes writing the SAME summary file
#   would silently lose variants. Lane A owns
#       results/summary_test.json, results/summary_heldout.json      (synthetic)
#       results/summary_test_mimic.json, results/summary_heldout_mimic.json
#   and lane B owns the _medcalc ones. Neither lane writes the other's files.
#   The constraint-layer JSONs and adapters are written only here.
#
# RESUME: each `stage` call is checkpointed to .pipeline_state/scaled_a/. A
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
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONPATH=/home/asosoft/abtin/paper/csai/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/asosoft/abtin/paper/csai
# PATH. cron does not run a login shell, so miniconda is not on PATH and
# `python` resolves to nothing -- every stage of the 2026-09-07 night run
# that cron started failed with "python: command not found" in under a
# second. Naming the interpreter directory explicitly is the fix; relying on
# the caller's environment is what broke.
export PATH="/home/asosoft/abtin/miniconda3/bin:$PATH"
M="${MODEL_ID:-BioMistral/BioMistral-7B}"
TAG="${MODEL_TAG:-biomistral-7b}"
R="${RESULTS_DIR:-results}"
LAYER="${LAYER:-20}"
ADAPTER_LAYER="${ADAPTER_LAYER:-30}"
mkdir -p "$R"
V6="base rag sym uq nsai nsai_uq"
G=data/umls/causal_graph.json

STATE_DIR=".pipeline_state/scaled_a_${TAG}_L${LAYER}"
mkdir -p "$STATE_DIR"

banner () { echo; echo "########## [A] $* ##########"; echo; }

FAILED=0

stage () {
  local name="$1"; shift
  local marker="$STATE_DIR/${name}.done"
  if [ -f "$marker" ]; then
    echo "[A][skip] $name already done ($marker)"
    return 0
  fi
  banner "$name"
  if "$@"; then
    touch "$marker"
    echo "[A][ok] $name"
  else
    echo "[A][FAILED] $name -- no marker written, will retry on next launch"
    FAILED=$((FAILED + 1))
  fi
}

# --- 1. constraint layers (step 2c). Lane B waits on adapter_qt/renal. ------
stage A1_constraint_qt \
  python src/constraint_layer.py --layer "$ADAPTER_LAYER" --data data/medcalc --train_on heldout_first \
    --epochs 8 --graph $G --save_adapter "$R"/adapter_qt.npz \
    --out "$R"/constraint_qt.json
stage A2_constraint_renal \
  python src/constraint_layer.py --layer "$ADAPTER_LAYER" --data data/medcalc --train_on train \
    --epochs 8 --graph $G --save_adapter "$R"/adapter_renal.npz \
    --out "$R"/constraint_renal.json
# Both adapters lane B needs now exist; it can proceed past its wait.
stage A3_constraint_qt_shuffled \
  python src/constraint_layer.py --layer "$ADAPTER_LAYER" --data data/medcalc --train_on heldout_first \
    --epochs 8 --graph $G --shuffled_control \
    --out "$R"/constraint_qt_shuffled.json
stage A4_constraint_synth \
  python src/constraint_layer.py --layer "$ADAPTER_LAYER" --data data/synthetic_control --train_on calib \
    --epochs 8 --graph $G --save_adapter "$R"/adapter_synth.npz \
    --out "$R"/constraint_synth.json

# --- 2. synthetic CONTROL ablation, new default UQ signal ------------------
stage A5_synthctl_ablation_test \
  python src/run_eval.py --backend hf --model_id $M --out "$R" --seeds 0 --split test \
    --data data/synthetic_control --variants $V6 --batch_size 8
stage A5_synthctl_ablation_heldout \
  python src/run_eval.py --backend hf --model_id $M --out "$R" --seeds 0 --split heldout \
    --data data/synthetic_control --variants $V6 --batch_size 8

# --- 3. synthetic CONTROL constraint-layer rows -----------------------------
stage A6_synthctl_cl_test \
  python src/run_eval.py --backend hf --model_id $M --out "$R" --seeds 0 --split test \
    --data data/synthetic_control \
    --variants cl nsai_uq_cl --adapter "$R"/adapter_synth.npz \
    --adapter_layer "$ADAPTER_LAYER" --batch_size 8
stage A6_synthctl_cl_heldout \
  python src/run_eval.py --backend hf --model_id $M --out "$R" --seeds 0 --split heldout \
    --data data/synthetic_control \
    --variants cl nsai_uq_cl --adapter "$R"/adapter_synth.npz \
    --adapter_layer "$ADAPTER_LAYER" --batch_size 8

# --- 4. RQ3 perplexity ------------------------------------------------------
stage A7_perplexity_base \
  python src/perplexity.py --model_id $M --n 200
stage A7_perplexity_qt \
  python src/perplexity.py --model_id $M --n 200 \
    --adapter "$R"/adapter_qt.npz --layer "$ADAPTER_LAYER"
stage A7_perplexity_synth \
  python src/perplexity.py --model_id $M --n 200 \
    --adapter "$R"/adapter_synth.npz --layer "$ADAPTER_LAYER"

# --- 5. MIMIC-IV Real-Value Cohort (NEW 2026-09-02) -------------------------
# data/mimic must already exist -- built once by src/build_mimic.py, not
# rebuilt here (rebuilding reshuffles the patient-level split via --seed and
# would silently invalidate a mid-flight adapter/eval pairing).
stage A8_mimic_dataset_present \
  test -s data/mimic/counterfactual_train.jsonl
stage A9_constraint_mimic \
  python src/constraint_layer.py --layer "$ADAPTER_LAYER" --data data/mimic --train_on train \
    --epochs 8 --graph $G --save_adapter "$R"/adapter_mimic.npz \
    --out "$R"/constraint_mimic.json
stage A10_constraint_mimic_shuffled \
  python src/constraint_layer.py --layer "$ADAPTER_LAYER" --data data/mimic --train_on train \
    --epochs 8 --graph $G --shuffled_control \
    --out "$R"/constraint_mimic_shuffled.json
stage A11_mimic_ablation_test \
  python src/run_eval.py --backend hf --model_id $M --out "$R" --seeds 0 --split test \
    --data data/mimic --tag _mimic --variants $V6 --batch_size 8
stage A11_mimic_ablation_heldout \
  python src/run_eval.py --backend hf --model_id $M --out "$R" --seeds 0 --split heldout \
    --data data/mimic --tag _mimic --variants $V6 --batch_size 8
stage A12_mimic_cl_test \
  python src/run_eval.py --backend hf --model_id $M --out "$R" --seeds 0 --split test \
    --data data/mimic --tag _mimic \
    --variants cl nsai_uq_cl --adapter "$R"/adapter_mimic.npz \
    --adapter_layer "$ADAPTER_LAYER" --batch_size 8
stage A12_mimic_cl_heldout \
  python src/run_eval.py --backend hf --model_id $M --out "$R" --seeds 0 --split heldout \
    --data data/mimic --tag _mimic \
    --variants cl nsai_uq_cl --adapter "$R"/adapter_mimic.npz \
    --adapter_layer "$ADAPTER_LAYER" --batch_size 8
stage A13_perplexity_mimic \
  python src/perplexity.py --model_id $M --n 200 \
    --adapter "$R"/adapter_mimic.npz --layer "$ADAPTER_LAYER"

if [ "$FAILED" -eq 0 ]; then
  touch "$STATE_DIR/ALL_DONE"
fi
banner "LANE A DONE ($FAILED stage(s) failed)"
exit "$FAILED"
