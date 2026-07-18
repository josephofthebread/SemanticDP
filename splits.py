import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import wandb
from wandb.sdk.wandb_run import Run

MANIFEST_DIR = Path("_manifest")
DATA_MANIFEST = MANIFEST_DIR / "data.json"
GLOVE_MANIFEST = MANIFEST_DIR / "glove.json"
PERTURB_MANIFEST = MANIFEST_DIR / "perturb.json"


def sha256file(path: Path) -> str:
  digest = sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def fetch(run: Run, name: str, sha256: str) -> Any:
  """Download dataset artifact `name:latest` and verify its bytes against a committed manifest's sha256."""
  path = Path(run.use_artifact(f"{name}:latest", type="dataset").download()) / f"{name}.json"
  actual = sha256file(path)
  if actual != sha256:
    raise RuntimeError(f"{name}: artifact sha256 {actual} does not match the committed manifest {sha256}")
  return json.loads(path.read_text())


def stage(name: str, rows: Any, staging: Path, metadata: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
  """Write rows to `{name}.json` under staging and build its dataset artifact; return (artifact, manifest entry)."""
  path = staging / f"{name}.json"
  path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
  entry = {"sha256": sha256file(path), **metadata}
  artifact = wandb.Artifact(name=name, type="dataset", metadata=entry)
  artifact.add_file(str(path))
  return artifact, entry
