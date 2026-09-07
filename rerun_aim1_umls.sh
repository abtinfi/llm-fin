#!/usr/bin/env bash
# Aim 1 re-score against the UMLS-grounded concept vocabulary.
#
# WHY A FULL RE-RUN AND NOT A RE-SCORE
# ------------------------------------
# `sae.py collect` writes the concept labels into the acts file and orders the
# kept tokens concept-first, so the vocabulary reaches the dictionary through
# both the labels and the row order that the 150k-token budget then truncates.
# S_semantic, the SAE itself, and S_causal all move. collect -> train -> score.
#
# Invocations below match run_lane_b.sh B2-B6 exactly except for --concepts,
# so the only difference between the baseline and this run is the vocabulary.
# Baseline artifacts are left in place; everything here writes to _umls names.
set -euo pipefail

M=BioMistral/BioMistral-7B
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
OUT=results/aim1_umls
mkdir -p "$OUT" results/sae

banner () { echo; echo "########## $* ##########"; echo; }
banner "Aim 1 UMLS re-score, $(date -Is)   GPU $CUDA_VISIBLE_DEVICES"
git rev-parse --short HEAD 2>/dev/null || echo "(no git rev)"

banner "1/4 collect (layer 20, 150k tokens, UMLS vocabulary)"
python src/sae.py collect --data data/medcalc --split train --layer 20 \
    --model_id $M --other_mult 1000 --max_tokens 300000 \
    --concepts umls --graph data/umls/causal_graph.json \
    --out results/sae/acts_medcalc_train_L20_umls.npz

banner "2/4 train: topk"
python src/sae.py train --acts results/sae/acts_medcalc_train_L20_umls.npz \
    --kind topk --expansion 4 --k 32 --epochs 30 --batch 2048 \
    --out results/sae/sae_topk_L20_umls.npz

banner "3/4 train: jumprelu"
python src/sae.py train --acts results/sae/acts_medcalc_train_L20_umls.npz \
    --kind jumprelu --expansion 4 --epochs 30 --batch 2048 \
    --out results/sae/sae_jumprelu_L20_umls.npz

banner "4/4 score: topk (with knock-out), then jumprelu"
python src/sae.py score --sae results/sae/sae_topk_L20_umls.npz --model_id $M \
    --top 25 --causal_items -1 --causal_features 0 --causal_split test \
    --out "$OUT/sae_topk_L20_umls_fis.json"
python src/sae.py score --sae results/sae/sae_jumprelu_L20_umls.npz \
    --model_id $M --top 25 --causal_items 0 \
    --out "$OUT/sae_jumprelu_L20_umls_fis.json"

banner "DONE $(date -Is)"
