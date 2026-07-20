#! /usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
make requirements
set -a
source .env
set +a
: "${DATASPHERE_PROJECT:?add it to .env}"
phase=${1:?train, alignment or safety}
corpus=${2:-nemotron}
runners=${RUNNERS:-9}

MODELS=(Qwen/Qwen3-1.7B meta-llama/Llama-3.2-1B-Instruct tiiuae/Falcon3-1B-Instruct)
SPLITS=("${corpus}"_train "${corpus}"_train_m1_p{05,15,30,50} "${corpus}"_train_m2_eps{1,3,6,12})
SEEDS=(0 1 2)
EPSILONS=(1 4 8 16)
TUNED="--batch-size 256 --lr 3.2e-3"

cells=()
for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for split in "${SPLITS[@]}"; do
      if [[ $phase == train ]]; then
        cells+=("train.py --model $model --split $split:latest --seed $seed $TUNED")
      else
        cells+=("$phase.py --model $model --lora adapter-${model##*/}-$split-s$seed:latest")
      fi
    done
    for eps in "${EPSILONS[@]}"; do
      if [[ $phase == train ]]; then
        cells+=("train.py --model $model --split ${corpus}_train:latest --seed $seed $TUNED --eps $eps")
      else
        cells+=("$phase.py --model $model --lora adapter-${model##*/}-${corpus}_train-s$seed-eps$eps:latest")
      fi
    done
  done
done

join() {
  local IFS=';'
  echo "$*"
}

size=$(((${#cells[@]} + runners - 1) / runners))
packs=()
for ((start = 0; start < ${#cells[@]}; start += size)); do
  packs+=("$(join "${cells[@]:start:size}")")
done
echo "${#cells[@]} cells over ${#packs[@]} runners, $size per runner"

job=$(mktemp)
export CELLS="${packs[0]}"
datasphere project job execute -p "$DATASPHERE_PROJECT" -c "_jobs/$phase.yaml" --async -o "$job"
template=$(jq -r .job_id "$job")
echo "runner 0: $template"

while :; do
  datasphere project job get --id "$template" --format json -o "$job"
  status=$(jq -r .status "$job")
  case $status in
    EXECUTING | SUCCESS) break ;;
    CREATING | PREPARING) sleep 10 ;;
    *) echo "runner 0 is $status" >&2 && exit 1 ;;
  esac
done

failed=()
for index in "${!packs[@]}"; do
  ((index == 0)) && continue
  if datasphere project job fork --id "$template" --async --env-var CELLS="${packs[index]}" -o "$job"; then
    echo "runner $index: $(jq -r .job_id "$job")"
  else
    failed+=("$index")
  fi
done

if ((${#failed[@]})); then
  echo "${#failed[@]} runners failed to submit: ${failed[*]}" >&2
  exit 1
fi
