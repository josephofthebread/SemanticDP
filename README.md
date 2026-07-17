# Semantic Differential Privacy

Copyright (c) 2026 The Project Authors. Licensed after MIT license.

## Generic information
- Most recent version of research proposal: [download](https://nightly.link/josephofthebread/SemanticDP/workflows/build/main/proposal.zip).
- Most recent version of pre-defense presentation: [download](https://nightly.link/josephofthebread/SemanticDP/workflows/build/main/pres.zip).
- Most recent version of the paper: [download](https://nightly.link/josephofthebread/SemanticDP/workflows/build/main/paper.zip).
- Wandb (experiments): <https://wandb.ai/josephofthebread/SemanticDP/overview>.

## Data generation
Use `./gendata.py`:
```bash
./gendata.py
```
Data description:
| Split | Rows | Spans/row | Role |
| - | - | - | - |
| `nemotron_train` | 16,384 | 8.35 | entity-rich adaptation corpus |
| `nemotron_probe` | 375 | 8.18 | held-out entity-fidelity probe |
| `nemotron_leak` | 375 | -- | training records, re-used closed-book for the leakage probe |
| `alpaca_train` | 16,384 | 0.00 | entity-poor contrast (for H3) |

## Embeddings
Use `./genglove.py` to pin the GloVe vectors used by the M2 (TEM) sanitizer (fetched from the official `stanfordnlp/glove` mirror) as a wandb artifact and record their hash in `_manifest/glove.json`:
```bash
./genglove.py
```

## Perturbation
Use `./perturb.py` to build the M1 and M2 (see the [proposal](https://nightly.link/josephofthebread/SemanticDP/workflows/build/main/proposal.zip)) altered datasets:
```bash
./perturb.py --splits nemotron_train,alpaca_train --m1 0.05,0.15,0.30,0.50 --m2 1,3,6,12
```

## Training
Use `./train.py` to LoRA fine-tune one model on one corpus split, producing an adapter logged to wandb. M0/M1/M2 differ only in the `--split` consumed; M3 adds DP-SGD when `--target-epsilon` is set (requires `torch` + `peft`, a CUDA GPU):
```bash
./train.py --model Qwen/Qwen3-1.7B --split nemotron_train                     # M0 (clean)
./train.py --model Qwen/Qwen3-1.7B --split nemotron_train_m2_eps3             # M2
./train.py --model Qwen/Qwen3-1.7B --split nemotron_train --target-epsilon 8  # M3 (DP-SGD)
```

## Evaluation
Use `./evaluate.py` (requires `vLLM`):
```bash
./evaluate.py --model Qwen/Qwen3-1.7B --run smoke --limit 20
```
