#! /usr/bin/env python

import sys
from heapq import heappop, heappush

MODELS = ["Qwen/Qwen3-1.7B", "meta-llama/Llama-3.2-1B-Instruct", "tiiuae/Falcon3-1B-Instruct"]
SEEDS = [0, 1, 2]
TUNED = "--batch-size 256 --lr 3.2e-3"

BLOCKS = []
for level in ["m2_eps0p5", "m2_eps2", "m2_eps4", "m2_eps5"]:
  for dataset in ["nemotron_train", "alpaca_train"]:
    BLOCKS.append((f"{dataset}_{level}", "", "", 3, False))
for level in ["m2e_eps1", "m2e_eps3"]:
  BLOCKS.append((f"nemotron_train_{level}", "", "", 3, False))
for clip, tag in [(0.1, "c0p1"), (0.3, "c0p3"), (3.0, "c3")]:
  BLOCKS.append(("nemotron_train", f"--eps 8 --max-grad-norm {clip} --tag {tag}", tag, 3, True))
BLOCKS.append(("nemotron_train", "--epochs 6 --tag e6", "e6", 6, False))
BLOCKS.append(("nemotron_train_m2_eps1", "--epochs 6 --tag e6", "e6", 6, False))
BLOCKS.append(("nemotron_train", "--epochs 6 --eps 8 --tag e6", "e6", 6, True))
for rank, tag in [(4, "r4"), (8, "r8"), (32, "r32")]:
  BLOCKS.append(("nemotron_train", f"--eps 8 --lora-rank {rank} --tag {tag}", tag, 3, True))


def cost(is_dp: bool, epochs: int) -> int:
  train = (67 if is_dp else 30) * epochs // 3
  return train + 18 + 8


adapters = []
for split, extra, tag, epochs, is_dp in BLOCKS:
  eps = "8" if "--eps 8" in extra else ""
  for model in MODELS:
    for seed in SEEDS:
      short = model.split("/")[-1]
      suffix = f"-eps{eps}" if eps else ""
      name = f"adapter-{short}-{split}-s{seed}{suffix}{f'-{tag}' if tag else ''}"
      commands = [
        f"train.py --model {model} --split {split}:latest --seed {seed} {TUNED} {extra}",
        f"alignment.py --model {model} --lora {name}:latest",
        f"safety.py --model {model} --lora {name}:latest",
      ]
      adapters.append((cost(is_dp, epochs), [" ".join(command.split()) for command in commands]))

runners = int(sys.argv[1]) if len(sys.argv) > 1 else 10
heap: list[tuple[int, int, list[str]]] = [(0, index, []) for index in range(runners)]
for adapter_cost, cells in sorted(adapters, key=lambda a: -a[0]):
  load, index, packed = heappop(heap)
  heappush(heap, (load + adapter_cost, index, packed + cells))

packs = {index: (load, cells) for load, index, cells in heap}
for index in range(runners):
  print(";".join(packs[index][1]))

loads = [load for load, _ in packs.values()]
print(
  f"{len(adapters)} adapters, {sum(len(cells) for _, cells in packs.values())} cells, {runners} runners; "
  f"makespan ~{max(loads) / 60:.1f}h, min ~{min(loads) / 60:.1f}h",
  file=sys.stderr,
)
