# Semantic Differential Privacy

Copyright (C) 2026, Dmitry Scherbakov. Licensed after MIT license. All rights reserved.

## Generic information
- Most recent version of research proposal: [download](https://nightly.link/josephofthebread/SemanticDP/workflows/build/main/proposal.zip).
- Most recent version of pre-defense presentation: [download](https://nightly.link/josephofthebread/SemanticDP/workflows/build/main/pres.zip).
- Most recent version of the paper: [download](https://nightly.link/josephofthebread/SemanticDP/workflows/build/main/paper.zip).
- Wandb (experiments): <https://wandb.ai/josephofthebread/SemanticDP/overview>.
- Most recent aggregated results: [download](https://nightly.link/josephofthebread/SemanticDP/workflows/results/main/results.zip).

## Prerequisites
Before running experiments, setup environment variables:
```bash
WANDB_ENTITY=josephofthebread
WANDB_PROJECT=SemanticDP
DATASPHERE_PROJECT=...  # project ID
GRPC_VERBOSITY=ERROR    # the CLI forks per submit; gRPC logs every inherited descriptor at INFO
```
Prepare configurations:
```bash
uv tool install datasphere
source .env
make requirements
```

## Running experiments
Every stage runs as a DataSphere job, configured in [_jobs](./_jobs/) folder (each submitted **from the repository root**).
1. Generate the embeddings and the corpora:
  ```bash
  datasphere project job execute -p $DATASPHERE_PROJECT -c _jobs/genglove.yaml
  datasphere project job execute -p $DATASPHERE_PROJECT -c _jobs/gendata.yaml
  ```
1. Train and evaluate the grid, one corpus at a time:
  ```bash
  _jobs/deploy.sh train nemotron
  _jobs/deploy.sh evaluate nemotron
  _jobs/deploy.sh safety nemotron
  ```
1. score the semantic distortion over every corpus arm and every set of generations:
  ```bash
  datasphere project job execute -p $DATASPHERE_PROJECT -c _jobs/sdist.yaml
  ```
