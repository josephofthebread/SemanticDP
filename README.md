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
./gendata.py --help
```
Data description:
| Split | Rows | Spans/row | Role |
| - | - | - | - |
| `nemotron_train` | 16,384 | 8.35 | entity-rich adaptation corpus |
| `nemotron_probe` | 375 | 8.18 | held-out entity-fidelity probe |
| `nemotron_leak` | 375 | -- | training records, re-used closed-book for the leakage probe |
| `alpaca_train` | 16,384 | 0.00 | entity-poor contrast (for H3) |

## Embeddings
Use `./glove.py` to pin the GloVe vectors used by the M2 (TEM) sanitizer (fetched from the official `stanfordnlp/glove` mirror) as a wandb artifact and record their hash in `_manifest/glove.json`:
```bash
./glove.py --help
```

## Evaluation
Use `./evaluate.py` (requires `vLLM`):
```bash
./evaluate.py --model Qwen/Qwen3-1.7B --run smoke --limit 20
```
