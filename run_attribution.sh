#!/usr/bin/env bash
# Proposal section 4.2: the Token-to-Concept Attribution Layer and its
# faithfulness evaluation. Writes results/faithfulness_*.json.
#
# Kept separate from the lane scripts because it depends on a trained SAE
# dictionary AND on that dictionary having been scored with the knock-out
# stage (`sae.py score --causal_items -1`): the attribution weights ARE the
# per-feature knock-out excesses. Run it after lane B, never before.
set -u
cd "$(dirname "$0")"
# PATH. cron does not run a login shell, so miniconda is not on PATH and
# `python` resolves to nothing -- every stage of the 2026-09-07 night run
# that cron started failed with "python: command not found" in under a
# second. Naming the interpreter directory explicitly is the fix; relying on
# the caller's environment is what broke.
export PATH="/home/asosoft/abtin/miniconda3/bin:$PATH"

# SINGLE-INSTANCE LOCK. This script can be launched three ways -- by hand, by
# csai_supervisor.sh, or by the @reboot cron -- and two copies would write the
# same checkpoint directory and the same output files. Taking the lock here
# rather than in the caller means the guarantee holds however it was started.
# A second copy exits 0: "already running" is success, not failure.
_lockfile=".pipeline_state/locks/attribution_${MODEL_TAG:-biomistral-7b}.lock"
mkdir -p "$(dirname "$_lockfile")"
exec 8>"$_lockfile"
if ! flock -n 8; then
  echo "$(date -Is) run_attribution.sh: another instance holds $_lockfile -- exiting" >&2
  exit 0
fi

M="${MODEL_ID:-BioMistral/BioMistral-7B}"
TAG="${MODEL_TAG:-$(python -c "import sys;sys.path.insert(0,'src');from lm_common import model_slug;print(model_slug('$M'))")}"
SAE="${SAE_NPZ:-results/sae/sae_topk_L20.npz}"
FIS="${SAE_FIS:-results/sae/sae_topk_L20_fis.json}"
OCC="${OCCLUSION_ITEMS:-30}"
STATE=".pipeline_state/attribution_${TAG}"
mkdir -p "$STATE" logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

fail=0
stage () {
  local name="$1"; shift
  if [ -f "$STATE/$name.done" ]; then echo "[attr][skip] $name"; return 0; fi
  echo "[attr][run ] $name  $(date -Is)"
  if "$@"; then touch "$STATE/$name.done"; echo "[attr][ok  ] $name"
  else echo "[attr][FAILED] $name"; fail=$((fail+1)); fi
}

for split in test heldout; do
  stage "medcalc_${split}" \
    python src/attribution.py \
      --data "data/medcalc/counterfactual_${split}.jsonl" \
      --sae "$SAE" --fis "$FIS" --model_id "$M" \
      --limit 0 --occlusion_items "$OCC" \
      --out "results/faithfulness_medcalc_${split}_${TAG}.json"
done

echo "[attr] done, $fail stage(s) failed  $(date -Is)"
[ "$fail" -eq 0 ] && touch "$STATE/ALL_DONE"
exit "$fail"
