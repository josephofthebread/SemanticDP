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


def corpus_scores(run: Run, encoder: SentenceTransformer, args: Namespace) -> dict[str, dict[str, Any]]:
  results = {}
  for corpus in args.corpora:
    base = {row["example_id"]: row for row in dataset(run, f"{corpus}_train")}
    for arm in args.arms:
      rows = dataset(run, f"{corpus}_train_{arm}")
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
        scores["s_ent"] = 1 - kept / total

        masked = []
        for texts in (plain, perturbed):
          stripped = []
          for text, original in zip(texts, originals, strict=True):
            for value in sorted({span["text"] for span in original["spans"]}, key=len, reverse=True):
              text = text.replace(value, " ")
            stripped.append(" ".join(text.split()))
          masked.append(stripped)
        scores["s_gen"] = sum(distances(encoder, masked[0], masked[1], args.batch_size)) / len(rows)

      results[f"{corpus}/{arm}"] = scores
      log.info(f"{corpus}/{arm}: {scores}")
  return results


def generation_scores(run: Run, encoder: SentenceTransformer, args: Namespace) -> dict[str, dict[str, Any]]:
  api = wandb.Api()
  names = [c.name for c in api.artifact_type("generations", project=f"{run.entity}/{run.project}").collections()]
  log.info(f"{len(names)} generation artifacts")

  def load(name: str) -> dict[tuple[str, str], str]:
    path = Path(run.use_artifact(f"{name}:latest", type="generations").download())
    return {(row["task"], row["prompt"]): row["response"] for row in json.loads(next(path.glob("*.json")).read_text())}

  results = {}
  for name in sorted(names):
    stem = name.removeprefix("generations-")
    match = re.search(r"-s(\d+)(?:-eps\d+)?$", stem)
    if not match:
      continue
    head = stem[: match.start()]
    corpus = next((c for c in args.corpora if f"-{c}_train" in head), None)
    if corpus is None:
      continue
    reference = f"generations-{head[: head.index(f'-{corpus}_train')]}-{corpus}_train-s{match.group(1)}"
    if reference == name:
      continue
    if reference not in names:
      log.warning(f"{name}: missing M0 reference {reference}")
      continue

    rows, baseline = load(name), load(reference)
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
      "corpus": corpus_scores(run, encoder, args),
      "generations": generation_scores(run, encoder, args),
    }

    staging = Path(run.dir) / "distortion.json"
    staging.write_text(json.dumps(results, indent=2, sort_keys=True))
    artifact = wandb.Artifact(
      "distortion",
      type="distortion",
      metadata={"encoder": args.encoder, "corpora": args.corpora, "arms": args.arms},
    )
    artifact.add_file(str(staging))
    run.log_artifact(artifact)

    for phase, scores in results.items():
      for key, values in scores.items():
        run.summary[f"{phase}/{key}/s"] = values["s"]


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
  parser = ArgumentParser(description="Compute the semantic distortion score S on corpora and on generations.")
  parser.add_argument("--corpora", nargs="+", required=True, help="corpus names, e.g. nemotron alpaca")
  parser.add_argument("--arms", nargs="+", required=True, help="perturbed arm suffixes, e.g. m1_p05 m2_eps1")
  parser.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
  parser.add_argument("--batch-size", type=int, default=256)
  main(parser.parse_args())
