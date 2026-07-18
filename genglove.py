#! /usr/bin/env python
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import wandb
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

log = logging.getLogger("glove")


def main(args: Namespace) -> None:
  load_dotenv()

  with (
    wandb.init(
      job_type="glove",
      config={"repo_id": args.repo, "revision": args.revision, "file": args.file, "member": args.member},
    ) as run,
    TemporaryDirectory() as tmp,
  ):
    archive = hf_hub_download(repo_id=args.repo, filename=args.file, revision=args.revision, cache_dir=tmp)
    log.info(f"extracting {args.member}")
    with ZipFile(archive) as bundle:
      bundle.extract(args.member, tmp)
    vectors = Path(tmp) / args.member

    lines = vectors.read_text().splitlines()
    vocab_size, dim = len(lines), len(lines[0].split()) - 1
    log.info(f"{args.member}: {vocab_size} tokens, dim {dim}")

    artifact = wandb.Artifact(
      name="glove",
      type="embeddings",
      description="GloVe 6B.300d word vectors for TEM sanitization.",
      metadata={"dim": dim, "vocab_size": vocab_size},
    )
    artifact.add_file(str(vectors))
    run.log_artifact(artifact)
    log.info(f"wandb: logged artifact {artifact.name}")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
  parser = ArgumentParser(description="Publish the GloVe word vectors used by the M2 sanitizer as a wandb artifact.")
  parser.add_argument("--repo", default="stanfordnlp/glove", help="Hugging Face repo hosting the archive")
  parser.add_argument("--revision", default="1db2080b2d94def6e5b0386a523102f9d8849e9d", help="pinned repo commit")
  parser.add_argument("--file", default="glove.6B.zip", help="archive to download from the repo")
  parser.add_argument("--member", default="glove.6B.300d.txt", help="vector file to extract from the archive")
  main(parser.parse_args())
