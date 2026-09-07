#!/usr/bin/env bash
# Serial GPU queue. Waits for a card to fall below a memory threshold, then
# runs the next job. Every job it launches is itself checkpointed, so the
# 03:00 reboot resumes rather than restarts -- this script only decides ORDER.
set -u
cd "$(dirname "$0")"
FREE_MIB="${FREE_MIB:-18000}"

wait_for_gpu () {   # echoes the index of the first free card
  while true; do
    for i in 0 1; do
      local used
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i")
      local total
      total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$i")
      if [ $((total - used)) -ge "$FREE_MIB" ]; then echo "$i"; return; fi
    done
    sleep 60
  done
}

run () {            # run "<name>" <env assignments...> -- <command>
  local name="$1"; shift
  local gpu; gpu=$(wait_for_gpu)
  echo "=== [queue] $name on GPU $gpu  $(date -Is) ==="
  GPU="$gpu" env "$@" > "logs/queue_${name}.log" 2>&1
  echo "=== [queue] $name exit=$?  $(date -Is) ==="
}

run medcalc_v2 GPU=0 bash run_medcalc_v2.sh
run layer16 MODEL_ID=BioMistral/BioMistral-7B MODEL_TAG=biomistral-7b-L16 \
    RESULTS_DIR=results/layers/L16 LAYER=16 ADAPTER_LAYER=30 bash run_model.sh
run layer24 MODEL_ID=BioMistral/BioMistral-7B MODEL_TAG=biomistral-7b-L24 \
    RESULTS_DIR=results/layers/L24 LAYER=24 ADAPTER_LAYER=30 bash run_model.sh
run mistral_instruct MODEL_ID=mistralai/Mistral-7B-Instruct-v0.2 \
    MODEL_TAG=mistral-7b-instruct-v0-2 \
    RESULTS_DIR=results/models/mistral-7b-instruct-v0-2 bash run_model.sh
echo "=== [queue] ALL JOBS DONE $(date -Is) ==="
