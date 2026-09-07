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

TAG="${MODEL_TAG:-biomistral-7b}"
R="${RESULTS_DIR:-results}"

STATE_DIR=".pipeline_state/scaled_join_${TAG}"
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
  python src/make_table.py --results "$R" --split test  --out "$R"/table_test_final.md
stage J1_table_synth_heldout \
  python src/make_table.py --results "$R" --split heldout --out "$R"/table_heldout_final.md
cp "$R"/table_test_final.md "$R"/table.md
stage J1_table_medcalc_test \
  python src/make_table.py --results "$R" --split test --tag _medcalc \
    --out "$R"/table_medcalc_test.md
stage J1_table_medcalc_heldout \
  python src/make_table.py --results "$R" --split heldout --tag _medcalc \
    --out "$R"/table_medcalc_heldout.md
stage J1_table_mimic_test \
  python src/make_table.py --results "$R" --split test --tag _mimic \
    --out "$R"/table_mimic_test.md
stage J1_table_mimic_heldout \
  python src/make_table.py --results "$R" --split heldout --tag _mimic \
    --out "$R"/table_mimic_heldout.md

# $R is passed as a positional argument, not interpolated: the body stays
# single-quoted so `$sp` is expanded by the inner shell, not by this one.
stage J2_coverage \
  bash -c '
    set -ex
    for sp in test heldout; do
      python src/coverage_report.py --results "$1" --split $sp --variant nsai_uq
      python src/coverage_report.py --results "$1" --split $sp --variant uq
      python src/coverage_report.py --results "$1" --split $sp --tag _medcalc --variant nsai_uq
      python src/coverage_report.py --results "$1" --split $sp --tag _mimic --variant nsai_uq
    done' _ "$R"

stage J3_summary \
  python src/make_summary.py --results "$R" --model_id "${MODEL_ID:-BioMistral/BioMistral-7B}"

stage J4_umls_reports \
  bash -c '
    set -ex
    python src/umls_grounding.py coverage --graph data/umls/causal_graph.json \
        > "$1"/umls_coverage.txt
    python src/umls_grounding.py audit --graph data/umls/causal_graph.json \
        > "$1"/umls_audit.txt' _ "$R"

if [ "$FAILED" -eq 0 ]; then
  touch "$STATE_DIR/ALL_DONE"
fi
banner "JOIN DONE ($FAILED stage(s) failed)"
exit "$FAILED"
