#! /usr/bin/env python
import json
from argparse import ArgumentParser, Namespace
from ast import literal_eval
from hashlib import sha256
from logging import INFO, basicConfig, getLogger
from pathlib import Path
from random import Random
from subprocess import check_output
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv

import wandb

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parent
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "_manifest.json"

log = getLogger("gendata")

NEMOTRON_REPO = "nvidia/Nemotron-PII"
ALPACA_REPO = "tatsu-lab/alpaca"

Row = dict[str, Any]
Span = dict[str, Any]

BIOMEDICAL_DOMAINS = frozenset(
  {
    "Healthcare",
    "Healthcare Providers",
    "Health",
    "Biotechnology",
    "Pharmaceuticals",
  }
)

EXTRACTABLE_LABELS = {
  "first_name": "patient's first name",
  "last_name": "patient's last name",
  "date_of_birth": "date of birth",
  "medical_record_number": "medical record number",
  "health_plan_beneficiary_number": "health plan beneficiary number",
  "phone_number": "phone number",
  "email": "email address",
  "date": "date",
  "time": "time",
  "age": "age",
  "street_address": "street address",
  "employee_id": "employee ID",
  "certificate_license_number": "certificate or license number",
  "account_number": "account number",
  "customer_id": "customer ID",
}


def sha256_file(path: Path) -> str:
  digest = sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git(*command: str) -> str:
  return check_output(["git", *command], cwd=ROOT, text=True).strip()


def build_version() -> dict[str, Any]:
  """Pin what produced the data, not just what the data is.

  The commit alone is not enough: a build from a dirty tree carries a commit
  whose gendata.py is not the one that ran. `script_sha256` is the
  authoritative half, and `git_dirty` says whether to trust the commit.
  """
  return {
    "script_sha256": sha256_file(SCRIPT),
    "git_commit": git("rev-parse", "HEAD"),
    "git_dirty": bool(git("status", "--porcelain")),
  }


def parse_spans(row: Row) -> list[Span]:
  text = row["text"]
  spans = row.get("spans", "[]")
  if isinstance(spans, str):
    spans = literal_eval(spans)
  parsed: list[Span] = []
  for span in spans:
    gold = str(span["text"])
    surface = text[span["start"] : span["end"]]
    parsed.append({**span, "text": surface if surface.lower() == gold.lower() else gold, "text_normalized": gold})
  return parsed


def template_extract(row: Row, rng: Random) -> Row | None:
  candidates = sorted({span["label"] for span in row["spans"]} & EXTRACTABLE_LABELS.keys())
  if not candidates:
    return None
  label = rng.choice(candidates)
  values = [span["text"] for span in row["spans"] if span["label"] == label]
  return {
    "instruction": f"Extract the {EXTRACTABLE_LABELS[label]} from the following record.",
    "input": row["text"],
    "output": ", ".join(dict.fromkeys(values)),
    "spans": row["spans"],
    "span_field": "input",
    "template": "extract",
  }


def template_draft(row: Row, _: Random) -> Row | None:
  return {
    "instruction": f"Draft a {row['document_type']} for the {row['domain']} domain.",
    "input": row["document_description"],
    "output": row["text"],
    "spans": row["spans"],
    "span_field": "output",
    "template": "draft",
  }


def template_classify(row: Row, _: Random) -> Row | None:
  return {
    "instruction": "Identify the document type of the following record.",
    "input": row["text"],
    "output": row["document_type"],
    "spans": row["spans"],
    "span_field": "input",
    "template": "classify",
  }


TEMPLATES = [template_extract, template_draft, template_classify]


def instructionize(row: Row, rng: Random) -> Row | None:
  for template in rng.sample(TEMPLATES, len(TEMPLATES)):
    example = template(row, rng)
    if example is not None:
      example["source_uid"] = row["uid"]
      example["domain"] = row["domain"]
      example["document_type"] = row["document_type"]
      return example
  return None


def build_nemotron(args: Namespace) -> dict[str, list[Row]]:
  dataset = load_dataset(NEMOTRON_REPO)
  rows = [row for split in dataset for row in dataset[split]]
  log.info(f"nemotron: {len(rows)} rows scanned")

  rows = [row for row in rows if row.get("domain") in BIOMEDICAL_DOMAINS]
  log.info(f"nemotron: {len(rows)} rows in biomedical domains")

  curated = [{**row, "spans": spans} for row in rows if (spans := parse_spans(row)) and row.get("text")]
  log.info(f"nemotron: {len(curated)} rows with at least one gold span")

  groups: dict[str, list[Row]] = {}
  for row in curated:
    groups.setdefault(row["uid"], []).append(row)
  uids = sorted(groups)
  log.info(f"nemotron: {len(uids)} source documents across {len(curated)} renderings")

  Random(args.seed).shuffle(uids)
  probe_uids, train_uids = uids[: args.probe_n], uids[args.probe_n :]

  train_pool = [row for uid in train_uids for row in groups[uid]]
  Random(args.seed + 1).shuffle(train_pool)
  if len(train_pool) < args.train_n:
    raise RuntimeError(f"nemotron: need {args.train_n} train rows, only {len(train_pool)} available")

  rng = Random(args.seed + 2)
  train = [example for row in train_pool[: args.train_n] if (example := instructionize(row, rng))]
  for index, example in enumerate(train):
    example["example_id"] = f"nemotron-{index:06d}"
  log.info(f"nemotron: {len(train)} train examples instructionized")

  probe = [
    {
      "example_id": f"probe-{index:05d}",
      "source_uid": row["uid"],
      "domain": row["domain"],
      "document_type": row["document_type"],
      "text": row["text"],
      "spans": row["spans"],
    }
    for index, row in enumerate(groups[uid][0] for uid in probe_uids)
  ]
  log.info(f"nemotron: {len(probe)} probe records held out")
  return {"nemotron_train": train, "nemotron_probe": probe}


def build_alpaca(args: Namespace) -> dict[str, list[Row]]:
  rows = list(load_dataset(ALPACA_REPO, split="train"))
  log.info(f"alpaca: {len(rows)} rows scanned")

  rows = [row for row in rows if row["instruction"].strip() and row["output"].strip()]
  Random(args.seed).shuffle(rows)
  if len(rows) < args.train_n:
    raise RuntimeError(f"alpaca: need {args.train_n} train rows, only {len(rows)} available")

  train = [
    {
      "instruction": row["instruction"],
      "input": row["input"],
      "output": row["output"],
      "spans": [],
      "span_field": "input",
      "template": "alpaca",
      "example_id": f"alpaca-{index:06d}",
      "source_uid": f"alpaca-{index:06d}",
      "domain": "generic",
      "document_type": "instruction",
    }
    for index, row in enumerate(rows[: args.train_n])
  ]
  log.info(f"alpaca: {len(train)} train examples")
  return {"alpaca_train": train}


def entity_stats(rows: list[Row]) -> dict[str, Any]:
  counts: dict[str, int] = {}
  for row in rows:
    for span in row["spans"]:
      counts[span["label"]] = counts.get(span["label"], 0) + 1
  n_spans = sum(counts.values())
  return {
    "n_rows": len(rows),
    "n_spans": n_spans,
    "spans_per_row": round(n_spans / len(rows), 3) if rows else 0.0,
    "label_counts": dict(sorted(counts.items(), key=lambda item: -item[1])),
  }


def push(paths: dict[str, Path], manifest: dict[str, Any]) -> None:
  run = wandb.init(job_type="gendata", config={**manifest["config"], **manifest["build_version"]})
  for name, path in paths.items():
    artifact = wandb.Artifact(
      name=name,
      type="dataset",
      description=f"Frozen {name} split for the privacy-mechanism grid.",
      metadata=manifest["splits"][name],
    )
    artifact.add_file(str(path))
    run.log_artifact(artifact)
    log.info(f"wandb: logged artifact {name}")

  index = wandb.Artifact(
    name="manifest", type="manifest", description="Hashes and build version of every split.", metadata=manifest
  )
  index.add_file(str(MANIFEST_PATH))
  run.log_artifact(index)
  log.info("wandb: logged artifact manifest")
  run.finish()


def main(args: Namespace) -> None:
  load_dotenv()
  DATA_DIR.mkdir(parents=True, exist_ok=True)

  splits = {**build_nemotron(args), **build_alpaca(args)}

  train_uids = {example["source_uid"] for example in splits["nemotron_train"]}
  probe_uids = {example["source_uid"] for example in splits["nemotron_probe"]}
  assert not train_uids & probe_uids, "probe records leaked into the training corpus"

  paths: dict[str, Path] = {}
  entries: dict[str, Any] = {}
  for name, rows in splits.items():
    path = DATA_DIR / f"{name}.json"
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    paths[name] = path
    entries[name] = {"sha256": sha256_file(path), **entity_stats(rows)}
    log.info(f"wrote {path} ({entries[name]['n_rows']} rows)")

  manifest = {
    "build_version": build_version(),
    "config": {
      "seed": args.seed,
      "train_n": args.train_n,
      "probe_n": args.probe_n,
      "biomedical_domains": sorted(BIOMEDICAL_DOMAINS),
      "nemotron_repo": NEMOTRON_REPO,
      "alpaca_repo": ALPACA_REPO,
    },
    "splits": entries,
  }
  MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  log.info(f"wrote {MANIFEST_PATH}")

  if args.push:
    push(paths, manifest)


if __name__ == "__main__":
  basicConfig(level=INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
  parser = ArgumentParser(description="Build the frozen adaptation corpora for the privacy-mechanism grid.")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--train-n", type=int, default=16384, help="train rows per corpus")
  parser.add_argument("--probe-n", type=int, default=375, help="held-out entity-fidelity probe records")
  parser.add_argument("--no-push", dest="push", action="store_false", help="skip the wandb artifact upload")
  main(parser.parse_args())
