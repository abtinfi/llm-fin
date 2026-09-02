#!/usr/bin/env bash
# JOIN -- runs after BOTH lanes finish. Tables, conformal/coverage reports and
# the cross-experiment summary all read results both lanes produced, so none of
# this can start until both are done.
#
# CPU-only (reads the JSON artifacts), so it does not matter which GPU is free.
set -x
export PYTHONPATH=/home/asosoft/abtin/paper/csai/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/asosoft/abtin/paper/csai

STATE_DIR=".pipeline_state/scaled_join"
mkdir -p "$STATE_DIR"

banner () { echo; echo "########## [JOIN] $* ##########"; echo; }

FAILED=0

stage () {
  local name="$1"; shift
  local marker="$STATE_DIR/${name}.done"
  if [ -f "$marker" ]; then
    echo "[JOIN][skip] $name already done ($marker)"
    return 0
  fi
  banner "$name"
  if "$@"; then
    touch "$marker"
    echo "[JOIN][ok] $name"
  else
    echo "[JOIN][FAILED] $name -- no marker written, will retry on next launch"
    FAILED=$((FAILED + 1))
  fi
}

stage J1_table_synth_test \
  python src/make_table.py --split test  --out results/table_test_final.md
stage J1_table_synth_heldout \
  python src/make_table.py --split heldout --out results/table_heldout_final.md
cp results/table_test_final.md results/table.md
stage J1_table_medcalc_test \
  python src/make_table.py --split test --tag _medcalc \
    --out results/table_medcalc_test.md
stage J1_table_medcalc_heldout \
  python src/make_table.py --split heldout --tag _medcalc \
    --out results/table_medcalc_heldout.md
stage J1_table_mimic_test \
  python src/make_table.py --split test --tag _mimic \
    --out results/table_mimic_test.md
stage J1_table_mimic_heldout \
  python src/make_table.py --split heldout --tag _mimic \
    --out results/table_mimic_heldout.md

stage J2_coverage \
  bash -c '
    set -ex
    for sp in test heldout; do
      python src/coverage_report.py --split $sp --variant nsai_uq
      python src/coverage_report.py --split $sp --variant uq
      python src/coverage_report.py --split $sp --tag _medcalc --variant nsai_uq
      python src/coverage_report.py --split $sp --tag _mimic --variant nsai_uq
    done'

stage J3_summary \
  python src/make_summary.py

stage J4_umls_reports \
  bash -c '
    set -ex
    python src/umls_grounding.py coverage --graph data/umls/causal_graph.json \
        > results/umls_coverage.txt
    python src/umls_grounding.py audit --graph data/umls/causal_graph.json \
        > results/umls_audit.txt'

if [ "$FAILED" -eq 0 ]; then
  touch "$STATE_DIR/ALL_DONE"
fi
banner "JOIN DONE ($FAILED stage(s) failed)"
exit "$FAILED"
