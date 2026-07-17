import json
import logging
from pathlib import Path
from typing import Any

from common import DATA_MANIFEST, Row, sha256_file

log = logging.getLogger("corpus")


def load_split(run: Any, name: str, alias: str) -> list[Row]:
  """Fetch a dataset artifact (base or perturbed) and verify its bytes against its own metadata."""
  artifact = run.use_artifact(f"{name}:{alias}", type="dataset")
  path = Path(artifact.download()) / f"{name}.json"

  expected = artifact.metadata["sha256"]
  actual = sha256_file(path)
  if actual != expected:
    raise RuntimeError(f"{name}: artifact sha256 {actual} does not match its metadata {expected}")

  rows: list[Row] = json.loads(path.read_text())
  log.info(f"corpus: {name}:{alias} {len(rows)} rows, sha256 verified")
  return rows


class Corpus:
  """The frozen base splits, verified against the repo-committed data manifest."""

  def __init__(self, run: Any, alias: str) -> None:
    self.run, self.alias = run, alias
    manifest_dir = Path(run.use_artifact(f"manifest:{alias}", type="manifest").download())
    self.manifest = json.loads((manifest_dir / DATA_MANIFEST.name).read_text())
    log.info(f"corpus: manifest:{alias} built from {self.manifest['build_version']['git_commit'][:12]}")

  def split(self, name: str) -> list[Row]:
    artifact = self.run.use_artifact(f"{name}:{self.alias}", type="dataset")
    path = Path(artifact.download()) / f"{name}.json"

    expected = self.manifest["splits"][name]["sha256"]
    actual = sha256_file(path)
    if actual != expected:
      raise RuntimeError(f"{name}: artifact sha256 {actual} does not match the manifest's {expected}")

    rows: list[Row] = json.loads(path.read_text())
    log.info(f"corpus: {name} {len(rows)} rows, sha256 verified against the manifest")
    return rows
