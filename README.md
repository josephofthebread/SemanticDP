# Semantic Differential Privacy

Copyright (C) 2026, Dmitry Scherbakov. Licensed after MIT license. All rights reserved.

## Generic information
- Most recent version of research proposal: [download](https://nightly.link/josephofthebread/SemanticDP/workflows/build/main/proposal.zip).
- Most recent version of pre-defense presentation: [download](https://nightly.link/josephofthebread/SemanticDP/workflows/build/main/pres.zip).
- Most recent version of the paper: [download](https://nightly.link/josephofthebread/SemanticDP/workflows/build/main/paper.zip).
- Wandb (experiments): <https://wandb.ai/josephofthebread/SemanticDP/overview>.

## Embeddings
Use `./genglove.py` to publish the GloVe vectors used by the M2 (TEM) sanitizer (fetched from the official `stanfordnlp/glove` mirror) as a wandb artifact:
```bash
./genglove.py
```

## Data generation
Use `./gendata.py` to build every corpus of the grid in one pass -- the clean one, the probes, and the M1 and M2 (see the [proposal](https://nightly.link/josephofthebread/SemanticDP/workflows/build/main/proposal.zip)) altered versions, derived from them:
```bash
./gendata.py
```

## Training
Use `./train.py` to LoRA fine-tune one model on one corpus split:
```bash
./train.py --model Qwen/Qwen3-1.7B --split nemotron_train:latest                     # M0
./train.py --model Qwen/Qwen3-1.7B --split nemotron_train_m1_p05:latest              # M1
./train.py --model Qwen/Qwen3-1.7B --split nemotron_train_m2_eps3:latest             # M2
./train.py --model Qwen/Qwen3-1.7B --split nemotron_train:latest --eps 8             # M3
```

## Deployment
`_jobs/deploy.sh` submits one phase of the grid:
```bash
uv tool install datasphere
./_jobs/deploy.sh train
./_jobs/deploy.sh evaluate
```
