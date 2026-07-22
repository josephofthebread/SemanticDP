#! /usr/bin/env python
import json
import logging
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import torch
import wandb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from wandb.sdk.wandb_run import Run

log = logging.getLogger("sdist")


def dataset(run: Run, name: str) -> Any:
  path = Path(run.use_artifact(f"{name}:latest", type="dataset").download())
  return json.loads(next(path.glob("*.json")).read_text())


def distances(encoder: SentenceTransformer, left: list[str], right: list[str], batch: int) -> list[float]:
  a = encoder.encode(left, batch_size=batch, convert_to_tensor=True, normalize_embeddings=True)
  b = encoder.encode(right, batch_size=batch, convert_to_tensor=True, normalize_embeddings=True)
  return (1 - (a * b).sum(dim=1)).clamp(0, 1).tolist()


def text_scores(run: Run, encoder: SentenceTransformer, args: Namespace) -> dict[str, dict[str, Any]]:
  """S between each published perturbed variant and the clean text it was derived from."""
  api = wandb.Api()
  published = {c.name for c in api.artifact_type("dataset", project=f"{run.entity}/{run.project}").collections()}

  results = {}
  for name in args.datasets:
    base = {row["example_id"]: row for row in dataset(run, f"{name}_train")}
    prefix = f"{name}_train_"
    variants = sorted(other.removeprefix(prefix) for other in published if other.startswith(prefix))
    log.info(f"{name}: {len(variants)} perturbed variants published")

    for variant in variants:
      rows = dataset(run, f"{name}_train_{variant}")
      originals = [base[row["example_id"]] for row in rows]
      plain = [f"{row['input']} {row['output']}" for row in originals]
      perturbed = [f"{row['input']} {row['output']}" for row in rows]
      scores: dict[str, Any] = {
        "s": sum(distances(encoder, plain, perturbed, args.batch_size)) / len(rows),
        "n": len(rows),
      }

      if any(row["spans"] for row in originals):
        kept = total = 0
        for original, text in zip(originals, perturbed, strict=True):
          lowered = text.lower()
          for span in original["spans"]:
            total += 1
            kept += span["text"].lower() in lowered
        scores["s_entity"] = 1 - kept / total

        masked = []
        for texts in (plain, perturbed):
          stripped = []
          for text, original in zip(texts, originals, strict=True):
            for value in sorted({span["text"] for span in original["spans"]}, key=len, reverse=True):
              text = text.replace(value, " ")
            stripped.append(" ".join(text.split()))
          masked.append(stripped)
        scores["s_carrier"] = sum(distances(encoder, masked[0], masked[1], args.batch_size)) / len(rows)

      results[f"{name}/{variant}"] = scores
      log.info(f"{name}/{variant}: {scores}")
  return results


def generation_scores(run: Run, encoder: SentenceTransformer, args: Namespace) -> dict[str, dict[str, Any]]:
  """S between each adapter's generations and those of its non-private counterpart."""
  api = wandb.Api()
  project = f"{run.entity}/{run.project}"
  names = [c.name for c in api.artifact_type("generations", project=project).collections()]
  log.info(f"{len(names)} generation artifacts")

  def load(name: str) -> dict[tuple[str, str], str]:
    path = Path(api.artifact(f"{project}/{name}:latest", type="generations").download())
    return {(row["task"], row["prompt"]): row["response"] for row in json.loads(next(path.glob("*.json")).read_text())}

  baselines: dict[str, dict[tuple[str, str], str]] = {}
  results = {}
  for name in sorted(names):
    stem = name.removeprefix("generations-")
    match = re.search(r"-s(\d+)(?:-eps[\d.]+)?(?:-[a-z][a-z0-9]*)?$", stem)
    if not match:
      continue
    head = stem[: match.start()]
    which = next((candidate for candidate in args.datasets if f"-{candidate}_train" in head), None)
    if which is None:
      continue
    reference = f"generations-{head[: head.index(f'-{which}_train')]}-{which}_train-s{match.group(1)}"
    if reference == name:
      continue
    if reference not in names:
      log.warning(f"{name}: missing non-private reference {reference}")
      continue

    if reference not in baselines:
      baselines[reference] = load(reference)
    rows, baseline = load(name), baselines[reference]
    shared = sorted(rows.keys() & baseline.keys())
    if not shared:
      log.warning(f"{name}: no prompts shared with {reference}")
      continue
    scored = distances(encoder, [baseline[key] for key in shared], [rows[key] for key in shared], args.batch_size)
    results[stem] = {"s": sum(scored) / len(scored), "n": len(scored), "reference": reference}
    log.info(f"{stem}: s={results[stem]['s']:.4f} over {len(scored)} prompts")
  return results


def main(args: Namespace) -> None:
  load_dotenv()
  encoder = SentenceTransformer(args.encoder, device="cuda" if torch.cuda.is_available() else "cpu")

  with wandb.init(job_type="sdist", config={**vars(args)}) as run:
    results = {
      "datasets": text_scores(run, encoder, args),
      "generations": generation_scores(run, encoder, args),
    }

    for phase, scores in results.items():
      for key, values in scores.items():
        for field, value in values.items():
          if isinstance(value, (int, float)):
            run.summary[f"{phase}/{key}/{field}"] = value

    staging = Path(run.dir) / "distortion.json"
    staging.write_text(json.dumps(results, indent=2, sort_keys=True))
    artifact = wandb.Artifact(
      f"distortion-{args.tag}" if args.tag else "distortion",
      type="distortion",
      metadata={"encoder": args.encoder, "datasets": args.datasets},
    )
    artifact.add_file(str(staging))
    run.log_artifact(artifact)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
  parser = ArgumentParser(description="Compute the semantic distortion score S on training text and on generations.")
  parser.add_argument("--datasets", nargs="+", required=True, help="dataset names, e.g. nemotron alpaca")
  parser.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
  parser.add_argument("--tag", default="", help="suffix for the published distortion artifact")
  parser.add_argument("--batch-size", type=int, default=256)
  main(parser.parse_args())
