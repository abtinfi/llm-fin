#!/usr/bin/env bash
# JOIN -- runs after BOTH lanes finish. Tables, conformal/coverage reports and
# the cross-experiment summary all read results both lanes produced, so none of
# this can start until both are done.
#
# CPU-only (reads the JSON artifacts), so it does not matter which GPU is free.
set -x
export PYTHONPATH=/home/asosoft/abtin/paper/csai/src
# Set for consistency with the GPU scripts. It is INERT here as the file
# stands -- every stage below reads JSON artifacts on the CPU and none of them
# imports torch -- but this file is where a GPU stage would be added, and the
# allocator setting has to be in the environment before the process starts, not
# after someone notices an OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/asosoft/abtin/paper/csai

banner () { echo; echo "########## [JOIN] $* ##########"; echo; }

banner "J1 tables"
python src/make_table.py --split test  --out results/table_test_final.md
python src/make_table.py --split heldout --out results/table_heldout_final.md
cp results/table_test_final.md results/table.md
python src/make_table.py --split test --tag _medcalc \
    --out results/table_medcalc_test.md
python src/make_table.py --split heldout --tag _medcalc \
    --out results/table_medcalc_heldout.md

banner "J2 adaptive conformal + conditional coverage"
for sp in test heldout; do
  python src/coverage_report.py --split $sp --variant nsai_uq
  python src/coverage_report.py --split $sp --variant uq
  python src/coverage_report.py --split $sp --tag _medcalc --variant nsai_uq
done

banner "J3 cross-experiment summary"
python src/make_summary.py

banner "J4 UMLS grounding reports"
python src/umls_grounding.py coverage --graph data/umls/causal_graph.json \
    > results/umls_coverage.txt
python src/umls_grounding.py audit --graph data/umls/causal_graph.json \
    > results/umls_audit.txt

banner "JOIN DONE"
