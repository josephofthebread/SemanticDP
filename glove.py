#! /usr/bin/env python
import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import wandb
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from common import GLOVE_MANIFEST, build_version, sha256_file

SCRIPT = Path(__file__).resolve()

log = logging.getLogger("glove")


def main(args: Namespace) -> None:
  load_dotenv()

  version = build_version(SCRIPT)
  config = {"repo_id": args.repo, "revision": args.revision, "file": args.file, "member": args.member}

  with wandb.init(job_type="glove", config={**config, **version}) as run, TemporaryDirectory() as tmp:
    archive = hf_hub_download(repo_id=args.repo, filename=args.file, revision=args.revision, cache_dir=tmp)
    log.info(f"extracting {args.member}")
    with ZipFile(archive) as bundle:
      bundle.extract(args.member, tmp)
    vectors = Path(tmp) / args.member

    digest = sha256_file(vectors)
    lines = vectors.read_text().splitlines()
    vocab_size, dim = len(lines), len(lines[0].split()) - 1
    log.info(f"{args.member}: {vocab_size} tokens, dim {dim}, sha256 {digest[:12]}")

    metadata = {**config, "dim": dim, "vocab_size": vocab_size, "sha256": digest}
    artifact = wandb.Artifact(
      name="glove",
      type="embeddings",
      description="Pinned GloVe 6B.300d word vectors for TEM sanitization.",
      metadata=metadata,
    )
    artifact.add_file(str(vectors))

    GLOVE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"build_version": version, "asset": metadata}
    GLOVE_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    log.info(f"wrote {GLOVE_MANIFEST}")

    index = wandb.Artifact(
      name="glove-manifest",
      type="manifest",
      description="Hash and source of the pinned GloVe vectors.",
      metadata=manifest,
    )
    index.add_file(str(GLOVE_MANIFEST))

    for logged in [artifact, index]:
      run.log_artifact(logged)
      log.info(f"wandb: logged artifact {logged.name}")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
  parser = ArgumentParser(description="Pin GloVe word vectors as a wandb artifact and record their hash.")
  parser.add_argument("--repo", default="stanfordnlp/glove", help="Hugging Face repo hosting the archive")
  parser.add_argument("--revision", default="1db2080b2d94def6e5b0386a523102f9d8849e9d", help="pinned repo commit")
  parser.add_argument("--file", default="glove.6B.zip", help="archive to download from the repo")
  parser.add_argument("--member", default="glove.6B.300d.txt", help="vector file to extract from the archive")
  main(parser.parse_args())
