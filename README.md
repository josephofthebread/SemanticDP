# Semantic Differential Privacy

Copyright (C) 2026, Dmitry Scherbakov. Licensed after MIT license. All rights reserved.

## Generic information
- Most recent version of research proposal: [download](https://nightly.link/josephofthebread/SemanticDP/workflows/build/main/proposal.zip).
- Most recent version of pre-defense presentation: [download](https://nightly.link/josephofthebread/SemanticDP/workflows/build/main/pres.zip).
- Most recent version of the paper: [download](https://nightly.link/josephofthebread/SemanticDP/workflows/build/main/paper.zip).
- Wandb (experiments): <https://wandb.ai/josephofthebread/SemanticDP/overview>.

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

## Data and embeddings generation
```bash
datasphere project job execute -p $DATASPHERE_PROJECT -c _jobs/genglove.yaml
datasphere project job execute -p $DATASPHERE_PROJECT -c _jobs/gendata.yaml
```
