#!/usr/bin/env bash
# LANE A -- GPU 0. Constraint layers, then everything scored on the SYNTHETIC
# benchmark, then RQ3 perplexity.
#
# FILE OWNERSHIP (this is what makes two-GPU parallelism safe):
#   run_eval.py MERGES into results/summary_{split}{tag}.json rather than
#   overwriting it, so two concurrent processes writing the SAME summary file
#   would silently lose variants. Lane A owns
#       results/summary_test.json, results/summary_heldout.json
#   and lane B owns the _medcalc ones. Neither lane writes the other's files.
#   The constraint-layer JSONs and adapters are written only here.
set -x
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/asosoft/abtin/paper/csai/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/asosoft/abtin/paper/csai
M="BioMistral/BioMistral-7B"
V6="base rag sym uq nsai nsai_uq"
G=data/umls/causal_graph.json

banner () { echo; echo "########## [A] $* ##########"; echo; }

# --- 1. constraint layers (step 2c). Lane B waits on adapter_qt/renal. ------
banner "A1 constraint layer: QT, full objective, L_ontology from the graph"
python src/constraint_layer.py --data data/medcalc --train_on heldout_first \
    --epochs 8 --graph $G --save_adapter results/adapter_qt.npz \
    --out results/constraint_qt.json
banner "A2 renal (probe predicts NOT to work; MED-RT-attested so margin x1.25)"
python src/constraint_layer.py --data data/medcalc --train_on train \
    --epochs 8 --graph $G --save_adapter results/adapter_renal.npz \
    --out results/constraint_renal.json
# Both adapters lane B needs now exist; it can proceed past its wait.
banner "A3 shuffled-label control (lam_ont forced to 0 by main())"
python src/constraint_layer.py --data data/medcalc --train_on heldout_first \
    --epochs 8 --graph $G --shuffled_control \
    --out results/constraint_qt_shuffled.json
banner "A4 synthetic benchmark, trained on the calibration split"
python src/constraint_layer.py --data data/synthetic_control --train_on calib \
    --epochs 8 --graph $G --save_adapter results/adapter_synth.npz \
    --out results/constraint_synth.json

# --- 2. synthetic CONTROL ablation, new default UQ signal ---------------------------
banner "A5 synthetic CONTROL ablation, 6 variants, uq_signal=decision_entropy"
python src/run_eval.py --backend hf --model_id $M --seeds 0 --split test \
    --data data/synthetic_control \
    --variants $V6 --batch_size 8
python src/run_eval.py --backend hf --model_id $M --seeds 0 --split heldout \
    --data data/synthetic_control \
    --variants $V6 --batch_size 8

# --- 3. synthetic CONTROL constraint-layer rows ------------------------------------
banner "A6 synthetic CONTROL constraint-layer rows"
python src/run_eval.py --backend hf --model_id $M --seeds 0 --split test \
    --data data/synthetic_control \
    --variants cl nsai_uq_cl --adapter results/adapter_synth.npz \
    --adapter_layer 30 --batch_size 8
python src/run_eval.py --backend hf --model_id $M --seeds 0 --split heldout \
    --data data/synthetic_control \
    --variants cl nsai_uq_cl --adapter results/adapter_synth.npz \
    --adapter_layer 30 --batch_size 8

# --- 4. RQ3 perplexity ------------------------------------------------------
banner "A7 RQ3: language ability, base and with each retrained adapter"
python src/perplexity.py --model_id $M --n 200
python src/perplexity.py --model_id $M --n 200 \
    --adapter results/adapter_qt.npz --layer 30
python src/perplexity.py --model_id $M --n 200 \
    --adapter results/adapter_synth.npz --layer 30

banner "LANE A DONE"
