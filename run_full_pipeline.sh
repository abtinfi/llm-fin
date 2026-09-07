#!/usr/bin/env bash
# Full re-run after the data and code fixes. GPU 0 only -- GPU 1 is in use by
# another job and must not be touched.
#
# RESUMABLE (added 2026-09-02). This machine reboots cleanly every day at
# 03:00 -- verified over 8 consecutive boots in `last -x reboot`, and not
# disableable from this account. The full pipeline takes ~9.3 h, so a run
# started after ~17:40 WILL be interrupted, and before this change an
# interruption meant starting over.
#
# Each stage now writes a completion marker to .pipeline_state/ on success. On
# a re-invocation, completed stages are skipped and the run picks up where the
# reboot cut it off. Markers record the timestamp and are only written when the
# stage exits 0, so a crashed stage is retried rather than skipped.
#
#   bash run_full_pipeline.sh              # run, resuming if markers exist
#   bash run_full_pipeline.sh --fresh      # clear markers and run everything
#   bash run_full_pipeline.sh --status     # show what is done and what is not
#
# Stages are independent enough that a failure in one does not invalidate the
# others, so the script does NOT exit on error; every stage is banner-marked in
# the log and the failure count is reported at the end.

set -u -o pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/asosoft/abtin/paper/csai/src
# Prevents the allocator fragmentation that PyTorch itself names in the OOM
# messages in logs/aim12.log, logs/probe_patch.log and logs/full_rerun.log
# ("If reserved but unallocated memory is large try setting
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"). Stage 8's SAE scoring
# reserved 2.13 GiB unallocated on a 22 GiB card while trying to allocate
# 9.16 GiB. The sibling scripts already set this; this one did not.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/asosoft/abtin/paper/csai || exit 1

M="BioMistral/BioMistral-7B"
V6="base rag sym uq nsai nsai_uq"

STATE=${STATE:-.pipeline_state}
LOGDIR=${LOGDIR:-logs/full_pipeline}
mkdir -p "$STATE" "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
FAILED=0
SKIPPED=0
RAN=0

banner () { echo; echo "########## $* ##########"; echo; }

# All the stages, in order, so --status can report on them without running
# anything. Kept in sync with the calls below.
ALL_STAGES="1a_synthctl_test 1b_synthctl_heldout 2a_medcalc_test 2b_medcalc_heldout
3a_cl_qt 3b_cl_qt_shuffled 3c_cl_renal 3d_cl_synth
4a_rows_medcalc_heldout 4b_rows_medcalc_test 4c_rows_synth_test
4d_rows_synth_heldout 5a_ppl_base 5b_ppl_qt 5c_ppl_synth 6_tables
7_coverage 8a_sae_collect 8b_sae_train_topk 8c_sae_train_jumprelu
8d_sae_score_topk 8e_sae_score_jumprelu"

case "${1:-}" in
  --fresh)
    echo "clearing $(ls -1 "$STATE"/*.done 2>/dev/null | wc -l) completion marker(s)"
    rm -f "$STATE"/*.done
    ;;
  --status)
    echo "stage completion in $STATE:"
    for s in $ALL_STAGES; do
      if [ -f "$STATE/$s.done" ]; then
        printf "  [done] %-28s %s\n" "$s" "$(cat "$STATE/$s.done")"
      else
        printf "  [ TODO ] %-26s\n" "$s"
      fi
    done
    exit 0
    ;;
  "") : ;;
  *) echo "unknown option: $1 (expected --fresh, --status, or nothing)"; exit 2 ;;
esac

MAIN="$LOGDIR/full_${STAMP}.log"
exec > >(tee "$MAIN") 2>&1

# Run one stage unless its marker exists. The marker is written ONLY on a zero
# exit, so an interrupted or failed stage is retried on the next invocation
# rather than silently skipped -- which is the whole point, and the way this
# differs from just checking whether an output file happens to exist.
#
# Each stage also gets its own log. Some stages (the perplexity ones) only
# PRINT their result and write no artifact, so without a per-stage log a resume
# would lose the number entirely.
stage () {
  local name="$1"; shift
  local mark="$STATE/$name.done"
  if [ -f "$mark" ]; then
    echo "[skip] $name -- completed $(cat "$mark")"
    SKIPPED=$((SKIPPED + 1))
    return 0
  fi
  banner "$name"
  local log="$LOGDIR/${name}_${STAMP}.log"
  if "$@" 2>&1 | tee "$log"; then
    date -Is > "$mark"
    echo "[ok] $name -> $log"
    RAN=$((RAN + 1))
  else
    echo "[FAILED] $name -> $log  (no marker written; will retry next run)"
    FAILED=$((FAILED + 1))
  fi
}

echo "full pipeline, $(date -Is)"
echo "model: $M   GPU: $CUDA_VISIBLE_DEVICES   state: $STATE"
echo "resuming: $(ls -1 "$STATE"/*.done 2>/dev/null | wc -l) of $(echo $ALL_STAGES | wc -w) stages already complete"

# --------------------------------------------------------- data gate (stage 0)
# Deliberately NOT a marked stage: a marker would let a resume skip it, and a
# resume is exactly when it matters most. This run may be picking up after a
# 03:00 reboot, and a stage killed mid-write can leave a truncated jsonl that
# every downstream number would then be computed from. It costs ~1 s.
#
# It also asserts that nothing here needs credentialed data. MIMIC-IV /
# PhysioNet is NOT a dependency of this pipeline -- no loader, data path or
# token check exists in src/ -- and this gate is what keeps that true rather
# than merely stated. See src/check_data.py.
banner "0/9 data provenance + integrity gate"
if python src/check_data.py --strict --json results/data_provenance.json; then
  echo "[ok] data gate"
else
  echo "[FAILED] data gate -- refusing to start a 9 h run on unsound data."
  echo "         Rebuild with: python src/build_dataset.py && python src/build_medcalc.py"
  exec 1>&- 2>&-; wait; exit 65
fi

# ------------------------------------------- 1. synthetic CONTROL benchmark
# Templated vignettes, hand-written thresholds. This is a CONTROL ARM: it shows
# what the pipeline does when the causal factor is stated cleanly and the label
# is guaranteed. No claim about clinical text may be sourced from these rows --
# stage 2 (data/medcalc, real PMC notes) is the real-text arm.
stage 1a_synthctl_test \
  python src/run_eval.py --backend hf --model_id $M --seeds 0 --split test \
    --data data/synthetic_control --variants $V6 --batch_size 8
stage 1b_synthctl_heldout \
  python src/run_eval.py --backend hf --model_id $M --seeds 0 --split heldout \
    --data data/synthetic_control --variants $V6 --batch_size 8

# ------------------------------------------------------------------ 2. medcalc
stage 2a_medcalc_test \
  python src/run_eval.py --backend hf --model_id $M --seeds 0 --split test \
    --data data/medcalc --gate medcalc --tag _medcalc --variants $V6 \
    --batch_size 4
stage 2b_medcalc_heldout \
  python src/run_eval.py --backend hf --model_id $M --seeds 0 --split heldout \
    --data data/medcalc --gate medcalc --tag _medcalc --variants $V6 \
    --batch_size 4

# --------------------------------------------------- 3. constraint layer (Aim 3)
stage 3a_cl_qt \
  python src/constraint_layer.py --data data/medcalc --train_on heldout_first \
    --epochs 8 --save_adapter results/adapter_qt.npz \
    --out results/constraint_qt.json
stage 3b_cl_qt_shuffled \
  python src/constraint_layer.py --data data/medcalc --train_on heldout_first \
    --epochs 8 --shuffled_control --out results/constraint_qt_shuffled.json
stage 3c_cl_renal \
  python src/constraint_layer.py --data data/medcalc --train_on train \
    --epochs 8 --save_adapter results/adapter_renal.npz \
    --out results/constraint_renal.json
stage 3d_cl_synth \
  python src/constraint_layer.py --data data/synthetic_control --train_on calib \
    --epochs 8 --save_adapter results/adapter_synth.npz \
    --out results/constraint_synth.json

# ------------------------------------------- 4. constraint-layer rows in tables
# These consume the adapters written by stage 3, so a resume that skips 3 still
# finds them on disk.
stage 4a_rows_medcalc_heldout \
  python src/run_eval.py --backend hf --model_id $M --seeds 0 --split heldout \
    --data data/medcalc --gate medcalc --tag _medcalc \
    --variants cl nsai_uq_cl --adapter results/adapter_qt.npz \
    --adapter_layer 30 --batch_size 4
stage 4b_rows_medcalc_test \
  python src/run_eval.py --backend hf --model_id $M --seeds 0 --split test \
    --data data/medcalc --gate medcalc --tag _medcalc \
    --variants cl nsai_uq_cl --adapter results/adapter_renal.npz \
    --adapter_layer 30 --batch_size 4
stage 4c_rows_synth_test \
  python src/run_eval.py --backend hf --model_id $M --seeds 0 --split test \
    --variants cl nsai_uq_cl --adapter results/adapter_synth.npz \
    --adapter_layer 30 --batch_size 8
stage 4d_rows_synth_heldout \
  python src/run_eval.py --backend hf --model_id $M --seeds 0 --split heldout \
    --variants cl nsai_uq_cl --adapter results/adapter_synth.npz \
    --adapter_layer 30 --batch_size 8

# ------------------------------------------------------------- 5. RQ3 perplexity
# These PRINT and write no artifact, so their per-stage log is the only record.
stage 5a_ppl_base \
  python src/perplexity.py --model_id $M --n 200
stage 5b_ppl_qt \
  python src/perplexity.py --model_id $M --n 200 \
    --adapter results/adapter_qt.npz --layer 30
stage 5c_ppl_synth \
  python src/perplexity.py --model_id $M --n 200 \
    --adapter results/adapter_synth.npz --layer 30

# ------------------------------------------------------------------- 6. tables
tables () {
  python src/make_table.py --split test  --out results/table_test_final.md &&
  python src/make_table.py --split heldout --out results/table_heldout_final.md &&
  cp results/table_test_final.md results/table.md &&
  python src/make_table.py --split test --tag _medcalc \
      --out results/table_medcalc_test.md &&
  python src/make_table.py --split heldout --tag _medcalc \
      --out results/table_medcalc_heldout.md
}
stage 6_tables tables

# ------------------------------------------------- 7. UQ coverage + conformal
coverage () {
  for sp in test heldout; do
    python src/coverage_report.py --split $sp --variant nsai_uq || return 1
    python src/coverage_report.py --split $sp --variant uq || return 1
    python src/coverage_report.py --split $sp --tag _medcalc --variant nsai_uq \
      || return 1
  done
}
stage 7_coverage coverage

# ----------------------------------------------------------------- 8. Aim 1 SAE
# Split into five markers because this is the longest block (~70 min) and the
# one that has historically OOMed, so a resume should not have to redo the
# 150k-token collection to retry a score.
stage 8a_sae_collect \
  python src/sae.py collect --data data/medcalc --split train --layer 20 \
    --model_id $M --other_mult 1000 --max_tokens 300000 \
    --out results/sae/acts_medcalc_train_L20.npz
stage 8b_sae_train_topk \
  python src/sae.py train --acts results/sae/acts_medcalc_train_L20.npz \
    --kind topk --expansion 4 --k 32 --epochs 30 --batch 2048 \
    --out results/sae/sae_topk_L20.npz
stage 8c_sae_train_jumprelu \
  python src/sae.py train --acts results/sae/acts_medcalc_train_L20.npz \
    --kind jumprelu --expansion 4 --epochs 30 --batch 2048 \
    --out results/sae/sae_jumprelu_L20.npz
stage 8d_sae_score_topk \
  python src/sae.py score --sae results/sae/sae_topk_L20.npz --model_id $M \
    --top 25 --causal_items -1 --causal_features 0 --causal_split test
stage 8e_sae_score_jumprelu \
  python src/sae.py score --sae results/sae/sae_jumprelu_L20.npz --model_id $M \
    --top 25 --causal_items 0

banner "9/9 DONE -- ran $RAN, skipped $SKIPPED, failed $FAILED"
if [ "$FAILED" -ne 0 ]; then
  echo "failed stages keep no marker and will be retried by re-running this script."
fi
echo "state:  $STATE   (bash run_full_pipeline.sh --status)"
echo "logs:   $LOGDIR/*_${STAMP}.log"

exec 1>&- 2>&-
wait
exit $FAILED
