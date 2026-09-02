#!/usr/bin/env bash
# Combined pass for the 2026-09-01 architecture changes:
#   2c  L_ontology consults the CausalKnowledgeGraph  (constraint_layer.py)
#   3   causal sufficiency / feature injection        (patching.py)
#   4   decision-restricted entropy as the UQ default (run_eval.py)
#
# Stage order is deliberate: the constraint layers train FIRST, so a fault in
# the new L_ontology wiring fails within ~15 minutes instead of eight hours in.
#
# Not re-run here: the Aim 1 SAE. Nothing in this pass touches it -- the UMLS
# concept vocabulary is built but is NOT yet wired into sae.py -- so re-running
# it would burn an hour to reproduce byte-identical output. Its results from
# the 2026-09-01 full run stand.
#
# Eq. (2) results from before the default change are preserved in
# results/eq2_reference/.
set -x
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/asosoft/abtin/paper/csai/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/asosoft/abtin/paper/csai
M="BioMistral/BioMistral-7B"
V6="base rag sym uq nsai nsai_uq"
G=data/umls/causal_graph.json

banner () { echo; echo "########## $* ##########"; echo; }

# ------------------------------------------- 1. constraint layers (step 2c)
banner "1/8 constraint layer: QT, full objective, L_ontology from the graph"
python src/constraint_layer.py --data data/medcalc --train_on heldout_first \
    --epochs 8 --graph $G --save_adapter results/adapter_qt.npz \
    --out results/constraint_qt.json
banner "1b shuffled-label control (lam_ont forced to 0 by main())"
python src/constraint_layer.py --data data/medcalc --train_on heldout_first \
    --epochs 8 --graph $G --shuffled_control \
    --out results/constraint_qt_shuffled.json
banner "1c renal (the probe predicts this should NOT work; MED-RT-attested, x1.25)"
python src/constraint_layer.py --data data/medcalc --train_on train \
    --epochs 8 --graph $G --save_adapter results/adapter_renal.npz \
    --out results/constraint_renal.json
banner "1d synthetic benchmark, trained on the calibration split"
python src/constraint_layer.py --data data --train_on calib \
    --epochs 8 --graph $G --save_adapter results/adapter_synth.npz \
    --out results/constraint_synth.json

# --------------------------------------- 2. ablations, new default UQ signal
banner "2/8 synthetic ablation, 6 variants, uq_signal=decision_entropy"
python src/run_eval.py --backend hf --model_id $M --seeds 0 --split test \
    --variants $V6 --batch_size 8
python src/run_eval.py --backend hf --model_id $M --seeds 0 --split heldout \
    --variants $V6 --batch_size 8

banner "3/8 MedCalc ablation on real notes, 6 variants"
python src/run_eval.py --backend hf --model_id $M --seeds 0 --split test \
    --data data/medcalc --gate medcalc --tag _medcalc --variants $V6 \
    --batch_size 4
python src/run_eval.py --backend hf --model_id $M --seeds 0 --split heldout \
    --data data/medcalc --gate medcalc --tag _medcalc --variants $V6 \
    --batch_size 4

# ------------------------------------------- 4. constraint-layer table rows
banner "4/8 constraint-layer rows, identical readout to every other row"
python src/run_eval.py --backend hf --model_id $M --seeds 0 --split heldout \
    --data data/medcalc --gate medcalc --tag _medcalc \
    --variants cl nsai_uq_cl --adapter results/adapter_qt.npz \
    --adapter_layer 30 --batch_size 4
python src/run_eval.py --backend hf --model_id $M --seeds 0 --split test \
    --data data/medcalc --gate medcalc --tag _medcalc \
    --variants cl nsai_uq_cl --adapter results/adapter_renal.npz \
    --adapter_layer 30 --batch_size 4
python src/run_eval.py --backend hf --model_id $M --seeds 0 --split test \
    --variants cl nsai_uq_cl --adapter results/adapter_synth.npz \
    --adapter_layer 30 --batch_size 8
python src/run_eval.py --backend hf --model_id $M --seeds 0 --split heldout \
    --variants cl nsai_uq_cl --adapter results/adapter_synth.npz \
    --adapter_layer 30 --batch_size 8

# ------------------------------------------------------- 5. RQ3 perplexity
banner "5/8 RQ3: language ability, base and with each retrained adapter"
python src/perplexity.py --model_id $M --n 200
python src/perplexity.py --model_id $M --n 200 \
    --adapter results/adapter_qt.npz --layer 30
python src/perplexity.py --model_id $M --n 200 \
    --adapter results/adapter_synth.npz --layer 30

# ------------------------------------------------------------- 6. tables
banner "6/8 tables"
python src/make_table.py --split test  --out results/table_test_final.md
python src/make_table.py --split heldout --out results/table_heldout_final.md
cp results/table_test_final.md results/table.md
python src/make_table.py --split test --tag _medcalc \
    --out results/table_medcalc_test.md
python src/make_table.py --split heldout --tag _medcalc \
    --out results/table_medcalc_heldout.md

# -------------------------------------------- 7. UQ coverage + conformal
banner "7/8 adaptive conformal + conditional coverage"
for sp in test heldout; do
  python src/coverage_report.py --split $sp --variant nsai_uq
  python src/coverage_report.py --split $sp --variant uq
  python src/coverage_report.py --split $sp --tag _medcalc --variant nsai_uq
done

# ------------------------------- 8. Aim 2 causal SUFFICIENCY (new, step 3)
banner "8/8 causal sufficiency: feature injection with dose sweep + controls"
python src/patching.py --mode sufficiency \
    --data data/medcalc/counterfactual_test.jsonl --model_id $M \
    --limit 40 --alphas 0.5,1,2,4 \
    --out results/sufficiency_medcalc_test.json
python src/patching.py --mode sufficiency \
    --data data/medcalc/counterfactual_heldout.jsonl --model_id $M \
    --limit 40 --alphas 0.5,1,2,4 \
    --out results/sufficiency_medcalc_heldout.json

banner "DONE"
