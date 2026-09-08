#!/usr/bin/env bash
# Evaluate the EXPANDED real-notes arm (data/medcalc_v2), tagged _medcalc2.
#
# `data/medcalc` is left exactly as it is: every published number was computed
# on it and every original item reproduces byte for byte inside v2 under the
# same id, so the two are directly comparable. v2 adds two rule families
# (nitrofurantoin/CrCl, apixaban/Child-Pugh) and, for the first time, CONTROL
# pairs whose driving value moves without crossing the threshold -- which is
# what lets `spurious_flip_rate` be reported beside Causal Consistency.
set -u
cd "$(dirname "$0")"

# SINGLE-INSTANCE LOCK. This script can be launched three ways -- by hand, by
# csai_supervisor.sh, or by the @reboot cron -- and two copies would write the
# same checkpoint directory and the same output files. Taking the lock here
# rather than in the caller means the guarantee holds however it was started.
# A second copy exits 0: "already running" is success, not failure.
_lockfile=".pipeline_state/locks/medcalc_v2.lock"
mkdir -p "$(dirname "$_lockfile")"
exec 8>"$_lockfile"
if ! flock -n 8; then
  echo "$(date -Is) run_medcalc_v2.sh: another instance holds $_lockfile -- exiting" >&2
  exit 0
fi
# PATH. cron does not run a login shell, so miniconda is not on PATH and
# `python` resolves to nothing -- every stage of the 2026-09-07 night run
# that cron started failed with "python: command not found" in under a
# second. Naming the interpreter directory explicitly is the fix; relying on
# the caller's environment is what broke.
export PATH="/home/asosoft/abtin/miniconda3/bin:$PATH"
export PYTHONPATH="$PWD/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${GPU:-1}"

M="${MODEL_ID:-BioMistral/BioMistral-7B}"
R="${RESULTS_DIR:-results}"
AL="${ADAPTER_LAYER:-30}"
V6="base rag sym uq nsai nsai_uq"
STATE=".pipeline_state/medcalc_v2"
mkdir -p "$STATE" logs

FAILED=0
stage () {
  local name="$1"; shift
  [ -f "$STATE/$name.done" ] && { echo "[v2][skip] $name"; return 0; }
  echo "[v2][run ] $name  $(date -Is)"
  if "$@"; then touch "$STATE/$name.done"; echo "[v2][ok  ] $name"
  else echo "[v2][FAILED] $name"; FAILED=$((FAILED+1)); fi
}

for split in test heldout; do
  stage "ablation_$split" \
    python src/run_eval.py --backend hf --model_id "$M" --out "$R" \
      --seeds 0 --split "$split" --data data/medcalc_v2 \
      --gate medcalc --tag _medcalc2 --variants $V6 --batch_size 4
done

# The expanded MIMIC arm. Its control pairs are the only ones in the project
# built from nothing but observed data -- two more real measurements from the
# same patient, on the same side of the threshold -- so a spurious flip there
# cannot be blamed on an edited number.
for split in test heldout; do
  stage "mimic2_ablation_$split" \
    python src/run_eval.py --backend hf --model_id "$M" --out "$R" \
      --seeds 0 --split "$split" --data data/mimic_v2 \
      --tag _mimic2 --variants $V6 --batch_size 8
done

# Constraint-layer rows reuse the adapters trained on data/medcalc. The
# adapter never saw the two new families, so those rows measure transfer to an
# unseen family rather than fit -- which is the more interesting number and
# must be described as such.
stage "cl_test" \
  python src/run_eval.py --backend hf --model_id "$M" --out "$R" \
    --seeds 0 --split test --data data/medcalc_v2 --gate medcalc \
    --tag _medcalc2 --variants cl nsai_uq_cl --adapter "$R"/adapter_renal.npz \
    --adapter_layer "$AL" --batch_size 4
stage "cl_heldout" \
  python src/run_eval.py --backend hf --model_id "$M" --out "$R" \
    --seeds 0 --split heldout --data data/medcalc_v2 --gate medcalc \
    --tag _medcalc2 --variants cl nsai_uq_cl --adapter "$R"/adapter_qt.npz \
    --adapter_layer "$AL" --batch_size 4

stage "table_test" \
  python src/make_table.py --results "$R" --split test --tag _medcalc2 \
    --out "$R"/table_medcalc2_test.md
stage "table_heldout" \
  python src/make_table.py --results "$R" --split heldout --tag _medcalc2 \
    --out "$R"/table_medcalc2_heldout.md
stage "table_mimic2_test" \
  python src/make_table.py --results "$R" --split test --tag _mimic2 \
    --out "$R"/table_mimic2_test.md
stage "table_mimic2_heldout" \
  python src/make_table.py --results "$R" --split heldout --tag _mimic2 \
    --out "$R"/table_mimic2_heldout.md

echo "[v2] done, $FAILED stage(s) failed  $(date -Is)"
[ "$FAILED" -eq 0 ] && touch "$STATE/ALL_DONE"
exit "$FAILED"
