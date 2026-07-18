#! /usr/bin/env python
import json
import logging
from argparse import ArgumentParser, Namespace
from ast import literal_eval
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from typing import Any

import wandb
from datasets import load_dataset
from dotenv import load_dotenv

from splits import DATA_MANIFEST, stage

log = logging.getLogger("gendata")

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

BIOMEDICAL_DOMAINS = frozenset(
  {
    "Healthcare",
    "Healthcare Providers",
    "Health",
    "Biotechnology",
    "Pharmaceuticals",
  }
)


def parse_spans(row: dict[str, Any]) -> list[dict[str, Any]]:
  text = row["text"]
  spans = row.get("spans", "[]")
  if isinstance(spans, str):
    spans = literal_eval(spans)
  parsed: list[dict[str, Any]] = []
  for span in spans:
    gold = str(span["text"])
    surface = text[span["start"] : span["end"]]
    parsed.append({**span, "text": surface if surface.lower() == gold.lower() else gold, "text_normalized": gold})
  return parsed


def template_extract(row: dict[str, Any], rng: Random) -> dict[str, Any] | None:
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


def template_draft(row: dict[str, Any], _: Random) -> dict[str, Any] | None:
  return {
    "instruction": f"Draft a {row['document_type']} for the {row['domain']} domain.",
    "input": row["document_description"],
    "output": row["text"],
    "spans": row["spans"],
    "span_field": "output",
    "template": "draft",
  }


def template_classify(row: dict[str, Any], _: Random) -> dict[str, Any] | None:
  return {
    "instruction": "Identify the document type of the following record.",
    "input": row["text"],
    "output": row["document_type"],
    "spans": row["spans"],
    "span_field": "input",
    "template": "classify",
  }


TEMPLATES = [template_extract, template_draft, template_classify]


def instructionize(row: dict[str, Any], rng: Random) -> dict[str, Any] | None:
  for template in rng.sample(TEMPLATES, len(TEMPLATES)):
    example = template(row, rng)
    if example is not None:
      example["source_uid"] = row["uid"]
      example["domain"] = row["domain"]
      example["document_type"] = row["document_type"]
      return example
  return None


def record(row: dict[str, Any], example_id: str) -> dict[str, Any]:
  """A Nemotron document in record form, for the probes that score gold spans."""
  return {
    "example_id": example_id,
    "source_uid": row["uid"],
    "domain": row["domain"],
    "document_type": row["document_type"],
    "text": row["text"],
    "spans": row["spans"],
  }


def build_nemotron(args: Namespace) -> dict[str, list[dict[str, Any]]]:
  dataset = load_dataset(args.nemotron_repo)
  rows = [row for split in dataset for row in dataset[split]]
  log.info(f"nemotron: {len(rows)} rows scanned")

  rows = [row for row in rows if row.get("domain") in BIOMEDICAL_DOMAINS]
  log.info(f"nemotron: {len(rows)} rows in biomedical domains")

  curated = [{**row, "spans": spans} for row in rows if (spans := parse_spans(row)) and row.get("text")]
  log.info(f"nemotron: {len(curated)} rows with at least one gold span")

  groups: dict[str, list[dict[str, Any]]] = {}
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

  train_rows = train_pool[: args.train_n]
  rng = Random(args.seed + 2)
  train = [example for row in train_rows if (example := instructionize(row, rng))]
  for index, example in enumerate(train):
    example["example_id"] = f"nemotron-{index:06d}"
  log.info(f"nemotron: {len(train)} train examples instructionized")

  probe = [record(row, f"probe-{index:05d}") for index, row in enumerate(groups[uid][0] for uid in probe_uids)]
  log.info(f"nemotron: {len(probe)} probe records held out")

  leak_rows = Random(args.seed + 3).sample(train_rows, args.leak_n)
  leak = [record(row, f"leak-{index:05d}") for index, row in enumerate(leak_rows)]
  log.info(f"nemotron: {len(leak)} leak records sampled from the training rows")

  return {"nemotron_train": train, "nemotron_probe": probe, "nemotron_leak": leak}


def build_alpaca(args: Namespace) -> dict[str, list[dict[str, Any]]]:
  rows = list(load_dataset(args.alpaca_repo, split="train"))
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


def entity_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
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


def main(args: Namespace) -> None:
  load_dotenv()

  config = {
    "seed": args.seed,
    "train_n": args.train_n,
    "probe_n": args.probe_n,
    "leak_n": args.leak_n,
    "biomedical_domains": sorted(BIOMEDICAL_DOMAINS),
    "nemotron_repo": args.nemotron_repo,
    "alpaca_repo": args.alpaca_repo,
  }
  with wandb.init(job_type="gendata", config=config) as run, TemporaryDirectory() as staging:
    splits = {**build_nemotron(args), **build_alpaca(args)}

    train_uids = {example["source_uid"] for example in splits["nemotron_train"]}
    probe_uids = {example["source_uid"] for example in splits["nemotron_probe"]}
    assert not train_uids & probe_uids, "probe records leaked into the training corpus"

    entries: dict[str, Any] = {}
    artifacts = []
    for name, rows in splits.items():
      artifact, entries[name] = stage(name, rows, Path(staging), entity_stats(rows))
      artifacts.append(artifact)
      log.info(f"staged {name}: {entries[name]['n_rows']} rows, sha256 {entries[name]['sha256'][:12]}")

    labels_artifact, entries["labels"] = stage(
      "labels", EXTRACTABLE_LABELS, Path(staging), {"n_labels": len(EXTRACTABLE_LABELS)}
    )
    artifacts.append(labels_artifact)
    log.info(f"staged labels: {len(EXTRACTABLE_LABELS)} labels, sha256 {entries['labels']['sha256'][:12]}")

    manifest = {"config": config, "splits": entries}
    DATA_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    DATA_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    log.info(f"wrote {DATA_MANIFEST}")

    index = wandb.Artifact(
      name="manifest", type="manifest", description="Hashes and build version of every split.", metadata=manifest
    )
    index.add_file(str(DATA_MANIFEST))

    for artifact in [*artifacts, index]:
      run.log_artifact(artifact)
      log.info(f"wandb: logged artifact {artifact.name}")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
  parser = ArgumentParser(description="Build the frozen adaptation corpora for the privacy-mechanism grid.")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--nemotron-repo", default="nvidia/Nemotron-PII", help="Hugging Face repo for the PII corpus")
  parser.add_argument("--alpaca-repo", default="tatsu-lab/alpaca", help="Hugging Face repo for the generic corpus")
  parser.add_argument("--train-n", type=int, default=16384, help="train rows per corpus")
  parser.add_argument("--probe-n", type=int, default=375, help="held-out entity-fidelity probe records")
  parser.add_argument(
    "--leak-n", type=int, default=375, help="training records re-used for the closed-book leakage probe"
  )
  main(parser.parse_args())
