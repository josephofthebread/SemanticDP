#! /usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
make requirements
set -a
source .env
set +a
: "${DATASPHERE_PROJECT:?add it to .env}"
phase=${1:?train or evaluate}

MODELS=(Qwen/Qwen3-1.7B meta-llama/Llama-3.2-1B-Instruct tiiuae/Falcon3-1B-Instruct)
SPLITS=(nemotron_train nemotron_train_m1_p{05,15,30,50} nemotron_train_m2_eps{1,3,6,12})
SEEDS=(0 1 2)
EPSILONS=(1 4 8 16)

submit() { datasphere project job execute -p "$DATASPHERE_PROJECT" -c "$1" "${@:2}"; }

cells=()
for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for split in "${SPLITS[@]}"; do
      cells+=("$model|$split:latest|$seed|")
    done
    for eps in "${EPSILONS[@]}"; do
      cells+=("$model|nemotron_train:latest|$seed|--eps $eps")
    done
  done
done
echo "${#cells[@]} cells"

IFS='|' read -r MODEL SPLIT SEED EXTRA <<<"${cells[0]}"
export MODEL SPLIT SEED EXTRA ADAPTER
adapter() {
  local eps=${EXTRA#--eps } name="adapter-${MODEL##*/}-${SPLIT%%:*}-s$SEED"
  [[ -n $EXTRA ]] && echo "$name-eps$eps:latest" || echo "$name:latest"
}

job=$(mktemp)
if [[ $phase == train ]]; then
  submit _jobs/train.yaml --async -o "$job"
else
  ADAPTER=$(adapter)
  submit _jobs/evaluate.yaml --async -o "$job"
fi
template=$(jq -r .job_id "$job")

# A failed fork must not abort the rest: the grid is submitted once and a partial submission is
# worse than a reported one. LIMIT caps how many cells are forked, for a first wave against quota.
failed=()
for cell in "${cells[@]:1:${LIMIT:-999}}"; do
  IFS='|' read -r MODEL SPLIT SEED EXTRA <<<"$cell"
  if [[ $phase == train ]]; then
    datasphere project job fork --id "$template" --async --env-var MODEL="$MODEL" --env-var SPLIT="$SPLIT" --env-var SEED="$SEED" --env-var EXTRA="$EXTRA" || failed+=("$cell")
  else
    datasphere project job fork --id "$template" --async --env-var MODEL="$MODEL" --env-var ADAPTER="$(adapter)" --env-var EXTRA= || failed+=("$cell")
  fi
done

if ((${#failed[@]})); then
  echo "${#failed[@]} cells failed to submit:" >&2
  printf '  %s\n' "${failed[@]}" >&2
  exit 1
fi
