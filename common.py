from hashlib import sha256
from pathlib import Path
from subprocess import check_output
from typing import Any


def sha256file(path: Path) -> str:
  digest = sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def versions() -> dict[str, Any]:
  git = lambda *command: check_output(["git", *command], text=True).strip()
  return {
    "git_commit": git("rev-parse", "HEAD"),
    "git_dirty": bool(git("status", "--porcelain")),
  }
