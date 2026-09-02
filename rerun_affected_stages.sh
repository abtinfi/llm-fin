#!/usr/bin/env bash
# Re-run ONLY the stages affected by the three layer-indexing / sign defects
# fixed on 2026-09-02. Everything else in results/ is untouched and remains the
# comparison baseline.
#
#   B1  src/sae.py::causal_knockout   hook fired on layers[L], but the SAE was
#                                     fitted on hidden_states[L] = output of
#                                     layers[L-1]. Affects S_causal and FIS.
#   B2  src/steering.py               direction points toward UNSAFE but was
#                                     applied with a negated coefficient over a
#                                     non-negative sweep, so the only direction
#                                     that could move a SAFE-locked model was
#                                     never tested; and the fitted layer was
#                                     off by one from the injected layer.
#   B3  src/patching.py::Patcher.run  hidden_states[n_layers] is post-
#                                     `model.norm`, so the deepest ACE row
#                                     wrote a normed tensor into a pre-norm
#                                     stream. Affects necessity row 31 and any
#                                     sufficiency run whose --layers reaches it.
#
# NOT re-run, deliberately:
#   * SAE collect/train. B1 is confined to the knock-out stage; the dictionary
#     in results/sae/sae_topk_L20.npz was fitted correctly and is reused. That
#     saves ~70 minutes and keeps S_semantic byte-identical, which is itself
#     the regression check that the fix was surgical.
#   * The jumprelu SAE, which was scored with --causal_items 0 and so has no
#     knock-out stage to correct.
#   * Every ablation / constraint-layer / conformal stage. None of them touch
#     the three call sites.
#
# Outputs go to results/rerun_fixes/ so nothing in results/ is overwritten and
# the two can be compared directly:
#   bash rerun_affected_stages.sh
#   python src/compare_rerun.py          # writes results/rerun_fixes/COMPARISON.md
#
# Wall clock: roughly 2.5-4 h on one RTX 4090, dominated by stage 2 (steering
# sweeps 13 alphas x 4 layers x 170 items x 2 arms = 17,680 forwards; the
# baseline swept 7 alphas one-sided).
#
# DURABILITY. This machine reboots CLEANLY EVERY DAY AT 03:00 local time --
# verified over 8 consecutive boots in `last -x reboot`, each entry reading
# "03:01 - 03:00 (23:59)". It is not in any user-visible cron or systemd timer,
# so it is a root-crontab or hypervisor schedule and cannot be disabled from
# here. A run that crosses 03:00 dies, and none of these stages checkpoint.
# The preflight below refuses to start a run that cannot finish in time.
#
# It also refuses to start outside a terminal multiplexer. `Linger=no` and the
# session-scoped cgroup mean an SSH disconnect takes the whole process group
# with it, and every stage here is measured in hours.

set -u -o pipefail
# NOT `set -e`: the stages are independent, and a failure in one must not
# discard the others. Failures are counted and reported at the end, matching
# the convention in run_full_pipeline.sh.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH=/home/asosoft/abtin/paper/csai/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/asosoft/abtin/paper/csai || exit 1

M="${MODEL_ID:-BioMistral/BioMistral-7B}"
OUT=results/rerun_fixes
LOGDIR=logs/rerun_fixes
mkdir -p "$OUT" "$LOGDIR"

STAMP=$(date +%Y%m%d_%H%M%S)
MAIN="$LOGDIR/rerun_${STAMP}.log"
FAILED=0

# Tee the whole run via `exec`, not by piping a `{ ... }` block into tee. A
# pipeline runs its left side in a subshell, so a FAILED counter incremented
# inside one is discarded at the closing brace and the script would always
# exit 0 -- a re-run that silently reports success after a stage crashed is
# precisely the failure mode this script exists to avoid.
exec > >(tee "$MAIN") 2>&1

banner () { echo; echo "########## $* ##########"; echo; }

# --------------------------------------------------------------- preflight
# Three ways a multi-hour run on this box dies without producing anything.
# Each is checked, each is overridable, and each override is named so that
# skipping it is a decision rather than an accident.
ETA_SECONDS=${ETA_SECONDS:-14400}          # 4 h, the pessimistic end of the range
MIN_FREE_MIB=${MIN_FREE_MIB:-16000}        # 7B bf16 is ~14.5 GiB plus activations
preflight_fail=0

# 1. Detached? Linger=no and a session-scoped cgroup mean an SSH drop kills the
#    process group. HANDOFF.md records tmux session `abtin2` as the convention.
if [ -n "${TMUX:-}" ] || [ -n "${STY:-}" ]; then
  echo "[preflight] ok: running inside a multiplexer (tmux/screen)"
elif [ "${RERUN_ALLOW_ATTACHED:-0}" = "1" ]; then
  echo "[preflight] WARN: not detached, continuing because RERUN_ALLOW_ATTACHED=1"
else
  echo "[preflight] FAIL: not running under tmux/screen."
  echo "             An SSH disconnect would kill this run (Linger=no)."
  echo "             Start it detached, e.g.:"
  echo "                 tmux new -s abtin2 'bash rerun_affected_stages.sh'"
  echo "             or:  nohup setsid bash rerun_affected_stages.sh &"
  echo "             Override with RERUN_ALLOW_ATTACHED=1 if you know better."
  preflight_fail=1
fi

# 2. GPU exclusive? Only ONE 7B run fits on a 23 GiB card (HANDOFF.md), and
#    nothing else in this repo takes a lock, so a second run silently OOMs
#    during caching_allocator_warmup after both have burned setup time.
if command -v nvidia-smi >/dev/null 2>&1; then
  free_mib=$(nvidia-smi --id="${CUDA_VISIBLE_DEVICES}" \
             --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
             | head -1 | tr -d ' ')
  busy=$(nvidia-smi --id="${CUDA_VISIBLE_DEVICES}" \
         --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ -n "$free_mib" ] && [ "$free_mib" -lt "$MIN_FREE_MIB" ]; then
    echo "[preflight] FAIL: GPU $CUDA_VISIBLE_DEVICES has only ${free_mib} MiB free"
    echo "             (need >= ${MIN_FREE_MIB} MiB; ${busy} compute process(es) resident)."
    echo "             Override with MIN_FREE_MIB=<n>."
    preflight_fail=1
  else
    echo "[preflight] ok: GPU $CUDA_VISIBLE_DEVICES has ${free_mib} MiB free, ${busy} process(es)"
  fi
fi

# 3. Will it finish before the nightly 03:00 reboot? None of these stages
#    checkpoint, so crossing it loses everything since the last stage boundary.
now=$(date +%s)
deadline=$(date -d "today 03:00" +%s)
[ "$deadline" -le "$now" ] && deadline=$(date -d "tomorrow 03:00" +%s)
left=$((deadline - now))
if [ "$left" -lt "$ETA_SECONDS" ]; then
  echo "[preflight] FAIL: $((left / 60)) min until the 03:00 reboot, but this"
  echo "             run needs about $((ETA_SECONDS / 60)) min and does not checkpoint."
  echo "             Start after the reboot, run stages individually, or"
  echo "             override with ETA_SECONDS=<seconds>."
  preflight_fail=1
else
  echo "[preflight] ok: $((left / 60)) min until the 03:00 reboot, need ~$((ETA_SECONDS / 60)) min"
fi

if [ "$preflight_fail" -ne 0 ]; then
  echo
  echo "[preflight] aborting before any GPU work. No results artifacts were"
  echo "             written (the log directory and this log file already exist)."
  exit 64
fi
echo "[preflight] all checks passed"

# Run a stage, tee its output to a per-stage log, and record failure without
# aborting the script.
stage () {
  local name="$1"; shift
  banner "$name"
  local log="$LOGDIR/${name}_${STAMP}.log"
  if "$@" 2>&1 | tee "$log"; then
    echo "[ok] $name -> $log"
  else
    echo "[FAILED] $name -> $log"
    FAILED=$((FAILED + 1))
  fi
}

echo "rerun of B1/B2/B3-affected stages, $(date -Is)"
echo "model: $M   GPU: $CUDA_VISIBLE_DEVICES   out: $OUT"
git -C . rev-parse --short HEAD 2>/dev/null || echo "(not a git repository)"

# ---------------------------------------------------------------- B1: SAE
# Reuses the existing dictionary; only the knock-out is recomputed. --top 25,
# --causal_items -1 (all items), --causal_features 0 (all features) and
# --causal_split test are the FULL-COVERAGE settings, matching stage 2/9 above.
# baseline invocation in run_full_pipeline.sh stage 8, so the only difference
# between the two artifacts is the fix.
stage "1_sae_knockout_topk" \
  python src/sae.py score \
    --sae results/sae/sae_topk_L20.npz \
    --model_id "$M" \
    --top 25 --causal_items -1 --causal_features 0 --causal_split test \
    --out "$OUT/sae_topk_L20_fis.json"

# ----------------------------------------------------------- B2: steering
# Same data, cache, pooling and layers as the baseline run (logs/steering2.log:
# "170 items, 85 pairs, pooling final, layers [5, 15, 25, 30]"). The 170-item
# file is now named counterfactual_heldout_all.jsonl after the QT family was
# re-partitioned for the constraint layer. --alphas is left at its new default,
# which is the two-sided sweep.
stage "2_steering_qt_two_sided" \
  python src/steering.py \
    --data data/medcalc/counterfactual_heldout_all.jsonl \
    --cache results/acts/medcalc_heldout.npz \
    --layer 5 15 25 30 --pool final \
    --model_id "$M" \
    --out "$OUT/steering_qt.json"

# --------------------------------------------- B3: necessity ACE by layer
# Matched to the baseline artifacts: patching_medcalc_test.json used
# counterfactual_test.jsonl and patching_medcalc_heldout.json used the 85-pair
# QT file, now counterfactual_heldout_all.jsonl (both report pairs_used=80).
# Using the current 30-pair `heldout` split here instead would change the item
# set and confound the fix with a different sample.
stage "3a_patching_necessity_test" \
  python src/patching.py \
    --data data/medcalc/counterfactual_test.jsonl --model_id "$M" \
    --out "$OUT/patching_medcalc_test.json"

stage "3b_patching_necessity_heldout" \
  python src/patching.py \
    --data data/medcalc/counterfactual_heldout_all.jsonl --model_id "$M" \
    --out "$OUT/patching_medcalc_heldout.json"

# ------------------------------------------ B3: sufficiency / injection
# Matched to run_bridge_pass.sh stage 8 (limit 0 = ALL pairs, alphas 0.5,1,2,4, and the
# leak-free 30-pair heldout split). The default layer list skips the last
# layer, so these are primarily a regression check that the Patcher.run change
# altered nothing it should not have.
stage "4a_patching_sufficiency_test" \
  python src/patching.py --mode sufficiency \
    --data data/medcalc/counterfactual_test.jsonl --model_id "$M" \
    --limit 0 --alphas 0.5,1,2,4 \
    --out "$OUT/sufficiency_medcalc_test.json"

stage "4b_patching_sufficiency_heldout" \
  python src/patching.py --mode sufficiency \
    --data data/medcalc/counterfactual_heldout.jsonl --model_id "$M" \
    --limit 0 --alphas 0.5,1,2,4 \
    --out "$OUT/sufficiency_medcalc_heldout.json"

# ------------------------------------------------------------ comparison
stage "5_comparison" \
  python src/compare_rerun.py \
    --baseline results --rerun "$OUT" \
    --out "$OUT/COMPARISON.md"

banner "DONE — $FAILED stage(s) failed"
echo "artifacts:   $OUT"
echo "comparison:  $OUT/COMPARISON.md"
echo "stage logs:  $LOGDIR/*_${STAMP}.log"
echo "this log:    $MAIN"

# Let the tee'd output drain before the shell exits, or the tail of the log is
# lost when the process-substitution writer is killed.
exec 1>&- 2>&-
wait
exit $FAILED
