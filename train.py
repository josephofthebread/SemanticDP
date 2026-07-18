#! /usr/bin/env python
import json
import logging
import math
import random
from argparse import ArgumentParser, Namespace
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from peft import LoraConfig, get_peft_model
from torch.func import functional_call, grad_and_value, vmap
from torch.nn.functional import cross_entropy
from transformers import AutoModelForCausalLM, AutoTokenizer
from wandb.sdk.wandb_run import Run

from splits import DATA_MANIFEST, PERTURB_MANIFEST, fetch

log = logging.getLogger("train")


def encode(tokenizer: Any, row: dict[str, Any], max_len: int) -> tuple[list[int], list[int]]:
  """One completion-only training example: prompt tokens masked to -100, completion tokens supervised."""
  content = row["instruction"] + (f"\n\n{row['input']}" if row["input"] else "")
  prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": content}],
    tokenize=True,
    add_generation_prompt=True,
    return_dict=False,
    enable_thinking=False,
  )
  completion = tokenizer.encode(row["output"], add_special_tokens=False) + [tokenizer.eos_token_id]
  ids = list(prompt) + completion
  labels = [-100] * len(prompt) + completion
  return ids[:max_len], labels[:max_len]


def collate(batch: list[tuple[list[int], list[int]]], pad_id: int, device: Any) -> tuple[Any, Any, Any]:
  width = max(len(ids) for ids, _ in batch)
  input_ids = [ids + [pad_id] * (width - len(ids)) for ids, _ in batch]
  labels = [lab + [-100] * (width - len(lab)) for _, lab in batch]
  mask = [[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in batch]
  tensor = lambda rows: torch.tensor(rows, device=device)
  return tensor(input_ids), tensor(labels), tensor(mask)


def privatize(per_sample: dict[str, Any], clip: float, sigma: float, generator: Any) -> dict[str, Any]:
  """Per-example clip to L2 norm `clip`, sum, add N(0, (sigma*clip)^2) Gaussian noise, average over the batch."""
  batch = next(iter(per_sample.values())).shape[0]
  flat = torch.cat([g.reshape(batch, -1) for g in per_sample.values()], dim=1)
  factor = (clip / (flat.norm(dim=1) + 1e-6)).clamp(max=1.0)
  out = {}
  for name, g in per_sample.items():
    summed = (g * factor.view([batch] + [1] * (g.dim() - 1))).sum(0)
    noise = torch.normal(0.0, sigma * clip, size=summed.shape, generator=generator, device=summed.device)
    out[name] = (summed + noise) / batch
  return out


def train(args: Namespace, model: Any, tokenizer: Any, rows: list[dict[str, Any]], run: Run) -> None:
  device = model.device
  examples = [encode(tokenizer, row, args.max_len) for row in rows]
  total_steps = math.ceil(len(examples) / args.batch_size) * args.epochs
  optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

  private = args.target_epsilon is not None
  if private:
    q = args.batch_size / len(examples)
    sigma = get_noise_multiplier(
      target_epsilon=args.target_epsilon, target_delta=args.delta, sample_rate=q, steps=total_steps, accountant="rdp"
    )
    accountant = RDPAccountant()
    for _ in range(total_steps):
      accountant.step(noise_multiplier=sigma, sample_rate=q)
    realized = accountant.get_epsilon(delta=args.delta)
    log.info(f"dp-sgd: sigma {sigma:.4f} for target epsilon {args.target_epsilon} (realized {realized:.4f})")
    run.summary["dp/sigma"], run.summary["dp/epsilon"] = sigma, realized

    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    constants = {n: p for n, p in model.named_parameters() if not p.requires_grad}
    constants.update(dict(model.named_buffers()))
    generator = torch.Generator(device=device).manual_seed(args.seed)

    def per_example_loss(trainable: dict[str, Any], ids: Any, labels: Any, mask: Any) -> Any:
      out = functional_call(
        model, {**constants, **trainable}, (ids.unsqueeze(0),), {"attention_mask": mask.unsqueeze(0)}
      )
      return cross_entropy(out.logits[0][:-1].float(), labels[1:], ignore_index=-100)

    grad_fn = vmap(grad_and_value(per_example_loss), in_dims=(None, 0, 0, 0))

  rng = random.Random(args.seed)
  step = 0
  for epoch in range(args.epochs):
    order = list(range(len(examples)))
    rng.shuffle(order)
    for start in range(0, len(examples), args.batch_size):
      batch = [examples[i] for i in order[start : start + args.batch_size]]
      input_ids, labels, mask = collate(batch, tokenizer.pad_token_id, device)
      optimizer.zero_grad()
      if private:
        per_sample, losses = grad_fn(params, input_ids, labels, mask)
        for name, value in privatize(per_sample, args.max_grad_norm, sigma, generator).items():
          params[name].grad = value.to(params[name].dtype)
        optimizer.step()
        loss = losses.mean()
      else:
        loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
        optimizer.step()
      step += 1
      run.log({"train/loss": float(loss), "train/epoch": epoch})
      if step % 20 == 0 or step == total_steps:
        log.info(f"step {step}/{total_steps} epoch {epoch} loss {float(loss):.4f}")


def main(args: Namespace) -> None:
  load_dotenv()
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  splits = {**json.loads(DATA_MANIFEST.read_text())["splits"], **json.loads(PERTURB_MANIFEST.read_text())["splits"]}
  if args.split not in splits:
    raise SystemExit(f"unknown split {args.split!r}; known: {sorted(splits)}")

  config = {
    "model": args.model,
    "split": args.split,
    "seed": args.seed,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "lr": args.lr,
    "max_len": args.max_len,
    "max_grad_norm": args.max_grad_norm,
    "target_epsilon": args.target_epsilon,
    "delta": args.delta,
    "dtype": args.dtype,
  }

  with wandb.init(job_type="train", config=config) as run, TemporaryDirectory() as tmp:
    rows = fetch(run, args.split, splits[args.split]["sha256"])
    log.info(f"{args.split}: {len(rows)} rows")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
      tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to("cuda")  # type: ignore[arg-type]
    lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.0, bias="none")
    model = get_peft_model(model, lora)
    model.eval()
    model.print_trainable_parameters()

    train(args, model, tokenizer, rows, run)

    model.save_pretrained(tmp)
    suffix = f"-eps{args.target_epsilon:g}" if args.target_epsilon is not None else ""
    artifact = wandb.Artifact(name=f"adapter-{args.split}-s{args.seed}{suffix}", type="model", metadata=config)
    artifact.add_dir(tmp)
    run.log_artifact(artifact)
    log.info(f"wandb: logged artifact {artifact.name}")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
  parser = ArgumentParser(
    description="LoRA fine-tune one model on one corpus split (DP-SGD when --target-epsilon is set)."
  )
  parser.add_argument("--model", required=True, help="base model, a HuggingFace id or local path")
  parser.add_argument("--split", required=True, help="corpus split to train on (base or perturbed, from the manifests)")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--epochs", type=int, default=1)
  parser.add_argument("--batch-size", type=int, default=8)
  parser.add_argument("--lr", type=float, default=2e-4)
  parser.add_argument("--max-len", type=int, default=1024)
  parser.add_argument("--max-grad-norm", type=float, default=1.0, help="DP-SGD per-example L2 clipping bound C")
  parser.add_argument("--target-epsilon", type=float, default=None, help="enable DP-SGD calibrated to this epsilon")
  parser.add_argument("--delta", type=float, default=1e-5, help="DP delta for the epsilon calibration")
  parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
  main(parser.parse_args())
