#! /usr/bin/env python
import json
import logging
import math
import re
from argparse import ArgumentParser, Namespace
from ast import literal_eval
from functools import cache
from itertools import product
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from wandb.sdk.wandb_run import Run

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


def suffix(mechanism: str, level: float) -> str:
  if mechanism == "m1":
    return f"p{int(level * 100):02d}"
  return f"eps{f'{level:g}'.replace('.', 'p')}"


def M1(text: str, p: float, key: str, vocab: list[str]) -> str:
  tokens = text.split()
  for i in range(len(tokens)):
    if Random(f"{key}:{i}").random() < p:
      tokens[i] = Random(f"{key}:{i}:pick").choice(vocab)
  return " ".join(tokens)


class Tem:
  WORD = re.compile(r"[A-Za-z]+|[^A-Za-z]+")

  def __init__(self, path: Path, gamma: float) -> None:
    lines = path.read_text().splitlines()
    self.words = [line[: line.index(" ")] for line in lines]
    self.index = {word: i for i, word in enumerate(self.words)}
    self.matrix = np.array([line.split(" ")[1:] for line in lines], dtype=np.float32)
    self.norm2 = (self.matrix * self.matrix).sum(1)
    self.gamma = gamma

  @cache
  def sample(self, word: str, eps: float) -> tuple[list[int], Any, set[int]]:
    x = self.matrix[self.index[word]]
    dist = np.sqrt(np.maximum(self.norm2 + float(x @ x) - 2.0 * (self.matrix @ x), 0.0))
    near = np.where(dist <= self.gamma)[0]
    far = len(dist) - len(near)
    logits = list((eps / 2.0) * -dist[near])
    candidates = list(map(int, near))
    if far > 0:
      logits.append((eps / 2.0) * (-self.gamma + (2.0 / eps) * math.log(far)))
      candidates.append(-1)
    weights = np.exp(np.array(logits) - max(logits))
    return candidates, np.cumsum(weights / weights.sum()), set(map(int, near))

  def sanitize(self, text: str, eps: float, key: str) -> str:
    out = []
    for i, part in enumerate(self.WORD.findall(text)):
      lower = part.lower()
      if not part.isalpha() or lower not in self.index:
        out.append(part)
        continue
      candidates, cum, near = self.sample(lower, eps)
      rng = Random(f"{key}:{i}")
      chosen = candidates[min(int(np.searchsorted(cum, rng.random())), len(candidates) - 1)]
      if chosen == -1:
        chosen = rng.randrange(len(self.words))
        while chosen in near:
          chosen = rng.randrange(len(self.words))
      sub = self.words[chosen]
      if part.isupper():
        sub = sub.upper()
      elif part[:1].isupper():
        sub = sub.capitalize()
      out.append(sub)
    return "".join(out)


def M2E(row: dict[str, Any], tem: "Tem", eps: float, key: str) -> dict[str, Any]:
  """Sanitize gold entity spans only, one surrogate per distinct value, carrier text verbatim."""
  values = sorted({span["text"] for span in row["spans"]}, key=len, reverse=True)
  mapping = {value: tem.sanitize(value, eps, f"{key}:{value}") for value in values}
  new = dict(row)
  for field in ["input", "output"]:
    text = row[field]
    for value in values:
      text = text.replace(value, mapping[value])
    new[field] = text
  return new


def publish(run: Run, staging: Path, name: str, payload: Any, metadata: dict[str, Any]) -> None:
  """Write payload to `{name}.json` under staging and log it as a dataset artifact."""
  path = staging / f"{name}.json"
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  artifact = wandb.Artifact(name=name, type="dataset", metadata=metadata)
  artifact.add_file(str(path))
  run.log_artifact(artifact)
  log.info(f"wandb: logged artifact {name} ({metadata})")


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

  levels = {"m1": args.m1, "m2": args.m2, "m2e": args.m2_entity}

  with (
    wandb.init(
      job_type="gendata",
      config={
        "seed": args.seed,
        "train_n": args.train_n,
        "probe_n": args.probe_n,
        "leak_n": args.leak_n,
        "biomedical_domains": sorted(BIOMEDICAL_DOMAINS),
        "nemotron_repo": args.nemotron_repo,
        "alpaca_repo": args.alpaca_repo,
        "glove": args.glove,
        "levels": levels,
        "gamma": args.gamma,
        "variants_only": args.variants_only,
      },
    ) as run,
    TemporaryDirectory() as tmp,
  ):
    staging = Path(tmp)
    glove = run.use_artifact(args.glove, type="embeddings")
    vectors = next(Path(glove.download()).glob("*.txt"))
    log.info(f"parsing {vectors.name} ({glove.metadata['vocab_size']} words)")
    tem = Tem(vectors, args.gamma)

    if args.variants_only:
      splits = {}
      for name in args.datasets:
        path = Path(run.use_artifact(f"{name}:latest", type="dataset").download())
        splits[name] = json.loads(next(path.glob("*.json")).read_text())
        log.info(f"{name}: {len(splits[name])} rows loaded from the published dataset")
    else:
      splits = {**build_nemotron(args), **build_alpaca(args)}

      train_uids = {example["source_uid"] for example in splits["nemotron_train"]}
      probe_uids = {example["source_uid"] for example in splits["nemotron_probe"]}
      assert not train_uids & probe_uids, "probe records leaked into the training corpus"

      for name, rows in splits.items():
        publish(run, staging, name, rows, entity_stats(rows))
      publish(run, staging, "labels", EXTRACTABLE_LABELS, {"n_labels": len(EXTRACTABLE_LABELS)})

    train_splits = [name for name in splits if name.endswith("_train")]
    vocab = {
      split: sorted({token for row in splits[split] for field in ["input", "output"] for token in row[field].split()})
      for split in train_splits
    }
    jobs = [(mechanism, level) for mechanism, values in levels.items() for level in values]

    for split, (mechanism, level) in product(train_splits, jobs):
      if mechanism == "m2e" and not any(row["spans"] for row in splits[split]):
        log.info(f"{split}: no gold spans, skipping {mechanism}")
        continue

      perturbed = []
      for row in splits[split]:
        key = f"{args.seed}:{split}:{mechanism}:{level}:{row['example_id']}"
        if mechanism == "m2e":
          perturbed.append(M2E(row, tem, level, key))
          continue
        new = dict(row)
        for field in ["input", "output"]:
          text = row[field]
          new[field] = (
            M1(text, level, f"{key}:{field}", vocab[split])
            if mechanism == "m1"
            else tem.sanitize(text, level, f"{key}:{field}")
          )
        perturbed.append(new)

      metadata = {"mechanism": mechanism, "level": level, "parent": split, "n_rows": len(perturbed)}
      name = f"{split}_{mechanism}_{suffix(mechanism, level)}"
      publish(run, staging, name, perturbed, metadata)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
  parser = ArgumentParser(description="Build every corpus of the privacy-mechanism grid: clean, M1 and M2.")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--nemotron-repo", default="nvidia/Nemotron-PII", help="Hugging Face repo for the PII corpus")
  parser.add_argument("--alpaca-repo", default="tatsu-lab/alpaca", help="Hugging Face repo for the generic corpus")
  parser.add_argument("--train-n", type=int, default=16384, help="train rows per corpus")
  parser.add_argument("--probe-n", type=int, default=375, help="held-out entity-fidelity probe records")
  parser.add_argument(
    "--leak-n", type=int, default=375, help="training records re-used for the closed-book leakage probe"
  )
  parser.add_argument("--glove", default="glove:latest", help="GloVe embeddings artifact NAME:VERSION")
  parser.add_argument(
    "--variants-only",
    action="store_true",
    help="perturb the already published datasets instead of rebuilding and republishing them",
  )
  parser.add_argument(
    "--datasets",
    nargs="+",
    default=["nemotron_train", "alpaca_train"],
    help="published datasets to perturb under --variants-only",
  )
  levels = lambda v: [float(x) for x in v.strip("\"'").split(",") if x.strip()]
  parser.add_argument("--m1", type=levels, default=[0.05, 0.15, 0.30, 0.50], help="M1 rates p")
  parser.add_argument("--m2", type=levels, default=[1.0, 3.0, 6.0, 12.0], help="M2 budgets epsilon")
  parser.add_argument("--m2-entity", type=levels, default=[], help="entity-targeted TEM budgets epsilon")
  parser.add_argument("--gamma", type=float, default=5.0, help="TEM truncation radius in GloVe space")
  main(parser.parse_args())
