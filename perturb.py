#! /usr/bin/env python
import json
import logging
import math
import re
from argparse import ArgumentParser, Namespace
from functools import cache
from itertools import product
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import wandb
from dotenv import load_dotenv

from splits import DATA_MANIFEST, GLOVE_MANIFEST, PERTURB_MANIFEST, fetch, sha256file, stage

log = logging.getLogger("perturb")


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


def main(args: Namespace) -> None:
  load_dotenv()

  data = json.loads(DATA_MANIFEST.read_text())["splits"]
  glove_asset = json.loads(GLOVE_MANIFEST.read_text())["asset"]
  levels = {"m1": args.m1, "m2": args.m2}
  config = {"seed": args.seed, "splits": args.splits, "levels": levels, "gamma": args.gamma}

  with wandb.init(job_type="perturb", config=config) as run, TemporaryDirectory() as tmp:
    staging = Path(tmp)
    base = {split: fetch(run, split, data[split]["sha256"]) for split in args.splits}

    glove_path = Path(run.use_artifact("glove:latest", type="embeddings").download()) / glove_asset["member"]
    if sha256file(glove_path) != glove_asset["sha256"]:
      raise RuntimeError("glove: artifact sha256 does not match the committed manifest")
    log.info(f"parsing {glove_asset['member']} ({glove_asset['vocab_size']} words)")
    tem = Tem(glove_path, args.gamma)

    vocab = {
      split: sorted({token for row in base[split] for field in ["input", "output"] for token in row[field].split()})
      for split in args.splits
    }
    jobs = [(mechanism, level) for mechanism, values in levels.items() for level in values]

    entries: dict[str, Any] = {}
    artifacts = []
    for split, (mechanism, level) in product(args.splits, jobs):
      perturbed = []
      for row in base[split]:
        new = dict(row)
        for field in ["input", "output"]:
          key = f"{args.seed}:{split}:{mechanism}:{level}:{row['example_id']}:{field}"
          text = row[field]
          new[field] = M1(text, level, key, vocab[split]) if mechanism == "m1" else tem.sanitize(text, level, key)
        perturbed.append(new)

      suffix = f"p{int(level * 100):02d}" if mechanism == "m1" else f"eps{int(level)}"
      name = f"{split}_{mechanism}_{suffix}"
      metadata = {"mechanism": mechanism, "level": level, "parent": split, "n_rows": len(perturbed)}
      artifact, entries[name] = stage(name, perturbed, staging, metadata)
      artifacts.append(artifact)
      log.info(f"staged {name}: {len(perturbed)} rows, sha256 {entries[name]['sha256'][:12]}")

    manifest = {"config": config, "splits": entries}
    PERTURB_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    PERTURB_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    log.info(f"wrote {PERTURB_MANIFEST}")

    index = wandb.Artifact(name="perturb-manifest", type="manifest", metadata=manifest)
    index.add_file(str(PERTURB_MANIFEST))
    for artifact in [*artifacts, index]:
      run.log_artifact(artifact)
      log.info(f"wandb: logged artifact {artifact.name}")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
  parser = ArgumentParser(description="Build the M1 and M2 perturbed corpora for the given base train splits.")
  parser.add_argument("-s", "--splits", required=True, type=lambda v: v.split(","), help="comma-separated base splits")
  parser.add_argument("--m1", required=True, type=lambda v: [float(x) for x in v.split(",")], help="M1 rates p")
  parser.add_argument("--m2", required=True, type=lambda v: [float(x) for x in v.split(",")], help="M2 budgets epsilon")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--gamma", type=float, default=5.0, help="TEM truncation radius in GloVe space")
  main(parser.parse_args())
