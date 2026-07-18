#! /usr/bin/env python
import json
import logging
import math
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_wsd_schedule
from wandb.sdk.wandb_run import Run

log = logging.getLogger("train")


def encode(tokenizer: Any, row: dict[str, Any], max_len: int) -> tuple[list[int], list[int]]:
  """One completion-only training example: prompt tokens masked to -100, completion tokens supervised."""
  content = row["instruction"] + (f"\n\n{row['input']}" if row["input"] else "")
  prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": content}],
    tokenize=True,
    add_generation_prompt=True,
    return_dict=False,
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


def clip_into(params: dict[str, Any], summed: dict[str, Any], clip: float) -> float:
  """Clip the gradient sitting on `params` to L2 norm `clip`, add it into `summed` in fp32, return its pre-clip norm."""
  norm = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params.values()))
  factor = (clip / (norm + 1e-6)).clamp(max=1.0)
  for name, p in params.items():
    summed[name] += p.grad.float() * factor
  return float(norm)


def train(args: Namespace, model: Any, tokenizer: Any, rows: list[dict[str, Any]], run: Run) -> None:
  device = model.device
  examples = []
  for row in rows:
    ids, labels = encode(tokenizer, row, args.max_len)
    if any(label != -100 for label in labels):
      examples.append((ids, labels))
  if len(examples) < len(rows):
    log.info(f"dropped {len(rows) - len(examples)} rows whose prompt fills max_len {args.max_len}")

  # Held out on a fixed seed, not --seed, so every arm and every seed holds out the same rows: the
  # curves stay comparable, and no arm trains on another's validation data. These rows are outside
  # the training set, so they cost no privacy budget — but that only holds while they are watched
  # rather than selected on.
  index = set(random.Random(0).sample(range(len(examples)), round(args.val * len(examples))))
  held_out = [example for i, example in enumerate(examples) if i in index]
  examples = [example for i, example in enumerate(examples) if i not in index]
  run.summary["train/dropped"] = len(rows) - len(examples) - len(held_out)
  run.summary["train/examples"], run.summary["val/examples"] = len(examples), len(held_out)

  total_steps = math.ceil(len(examples) / args.batch_size) * args.epochs
  optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
  scheduler = get_wsd_schedule(
    optimizer,
    num_warmup_steps=round(args.warmup * total_steps),
    num_decay_steps=round(args.decay * total_steps),
    num_training_steps=total_steps,
  )

  private = args.eps is not None
  if private:
    q = args.batch_size / len(examples)
    sigma = get_noise_multiplier(
      target_epsilon=args.eps, target_delta=args.delta, sample_rate=q, steps=total_steps, accountant="rdp"
    )
    accountant = RDPAccountant()
    for _ in range(total_steps):
      accountant.step(noise_multiplier=sigma, sample_rate=q)
    realized = accountant.get_epsilon(delta=args.delta)
    log.info(f"dp-sgd: sigma {sigma:.4f} for target epsilon {args.eps} (realized {realized:.4f})")
    run.summary["dp/sigma"], run.summary["dp/epsilon"] = sigma, realized

    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    dimension = sum(p.numel() for p in params.values())
    generator = torch.Generator(device=device).manual_seed(args.seed)

  rng = random.Random(args.seed)
  step, tokens = 0, 0
  steps_per_epoch = math.ceil(len(examples) / args.batch_size)
  for epoch in range(args.epochs):
    if private:
      lots = ([i for i in range(len(examples)) if rng.random() < q] for _ in range(steps_per_epoch))
    else:
      order = list(range(len(examples)))
      rng.shuffle(order)
      lots = (order[start : start + args.batch_size] for start in range(0, len(examples), args.batch_size))

    for lot in lots:
      if not lot:
        continue
      batch = [examples[i] for i in lot]
      weighted, supervised = 0.0, 0
      if private:
        summed = {name: torch.zeros_like(p, dtype=torch.float32) for name, p in params.items()}
        norms = []
        for ids, labs in batch:
          optimizer.zero_grad(set_to_none=True)
          example = model(input_ids=torch.tensor([ids], device=device), labels=torch.tensor([labs], device=device)).loss
          example.backward()
          norms.append(clip_into(params, summed, args.max_grad_norm))
          count = sum(label != -100 for label in labs)
          weighted += float(example.detach()) * count
          supervised += count
        optimizer.zero_grad(set_to_none=True)
        signal = float(torch.sqrt(torch.stack([(g**2).sum() for g in summed.values()]).sum()))
        for name, g in summed.items():
          noise = torch.normal(0.0, sigma * args.max_grad_norm, size=g.shape, generator=generator, device=g.device)
          params[name].grad = ((g + noise) / args.batch_size).to(params[name].dtype)
        optimizer.step()
        extra = {
          "train/grad_norm": sum(norms) / len(norms),
          "train/clipped": sum(norm > args.max_grad_norm for norm in norms) / len(norms),
          "train/snr": signal / (sigma * args.max_grad_norm * math.sqrt(dimension)),
          "train/lot_size": len(lot),
        }
      else:
        optimizer.zero_grad()
        chunks = [batch[start : start + args.micro_batch] for start in range(0, len(batch), args.micro_batch)]
        counts = [sum(label != -100 for _, labs in chunk for label in labs) for chunk in chunks]
        supervised = sum(counts)
        for chunk, count in zip(chunks, counts, strict=True):
          input_ids, labels, mask = collate(chunk, tokenizer.pad_token_id, device)
          chunk_loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
          (chunk_loss * count / supervised).backward()
          weighted += float(chunk_loss.detach()) * count
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
        optimizer.step()
        extra = {"train/grad_norm": float(norm)}
      scheduler.step()
      step += 1
      tokens += supervised
      value = weighted / supervised
      run.log(
        {
          "train/loss": value,
          "train/epoch": epoch,
          "train/lr": scheduler.get_last_lr()[0],
          "train/tokens": supervised,
          **extra,
        }
      )
      if step % 20 == 0 or step == total_steps:
        log.info(f"step {step}/{total_steps} epoch {epoch} loss {value:.4f} grad {extra['train/grad_norm']:.4f}")

    if held_out:
      model.eval()
      validation, counted = 0.0, 0
      with torch.no_grad():
        for start in range(0, len(held_out), args.micro_batch):
          chunk = held_out[start : start + args.micro_batch]
          input_ids, labels, mask = collate(chunk, tokenizer.pad_token_id, device)
          count = sum(label != -100 for _, labs in chunk for label in labs)
          validation += float(model(input_ids=input_ids, attention_mask=mask, labels=labels).loss) * count
          counted += count
      model.train()
      run.log({"val/loss": validation / counted, "train/epoch": epoch})
      log.info(f"epoch {epoch} validation loss {validation / counted:.4f}")
  run.summary["train/tokens_total"] = tokens


def main(args: Namespace) -> None:
  load_dotenv()
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  with (
    wandb.init(
      job_type="train",
      config={
        "model": args.model,
        "split": args.split,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "micro_batch": args.micro_batch,
        "lr": args.lr,
        "warmup": args.warmup,
        "decay": args.decay,
        "val": args.val,
        "max_len": args.max_len,
        "max_grad_norm": args.max_grad_norm,
        "eps": args.eps,
        "delta": args.delta,
        "dtype": args.dtype,
      },
    ) as run,
    TemporaryDirectory() as tmp,
  ):
    corpus = run.use_artifact(args.split, type="dataset")
    if args.eps is not None and corpus.metadata.get("mechanism"):
      raise SystemExit(f"--eps needs a clean corpus; {args.split} is {corpus.metadata['mechanism']}")
    rows = json.loads(next(Path(corpus.download()).glob("*.json")).read_text())
    log.info(f"{args.split}: {len(rows)} rows")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
      tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to("cuda")  # type: ignore[arg-type]
    lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.0, bias="none")
    model = get_peft_model(model, lora)
    for module in model.modules():
      if isinstance(module, torch.nn.Dropout):
        module.p = 0.0
    model.train()
    model.print_trainable_parameters()

    train(args, model, tokenizer, rows, run)

    model.save_pretrained(tmp)
    private = args.eps is not None
    grid = {
      "mechanism": "m3" if private else corpus.metadata.get("mechanism", "m0"),
      "level": args.eps if private else corpus.metadata.get("level", 0.0),
    }
    suffix = f"-eps{args.eps:g}" if private else ""
    name = f"adapter-{args.model.split('/')[-1]}-{args.split.split(':')[0]}-s{args.seed}{suffix}"
    adapter = wandb.Artifact(name=name, type="model", metadata=grid)
    adapter.add_dir(tmp)
    run.log_artifact(adapter)
    log.info(f"wandb: logged artifact {name} ({grid})")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
  parser = ArgumentParser(description="LoRA fine-tune one model on one corpus split (DP-SGD when --eps is set).")
  parser.add_argument("--model", required=True, help="base model, a HuggingFace id or local path")
  parser.add_argument("--split", required=True, help="corpus dataset artifact NAME:VERSION (base or perturbed)")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--epochs", type=int, default=3)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--lr", type=float, default=2e-4)
  parser.add_argument("--warmup", type=float, default=0.05, help="fraction of steps ramping up to --lr")
  parser.add_argument("--decay", type=float, default=0.5, help="fraction of steps decaying --lr to zero")
  parser.add_argument("--val", type=float, default=0.0, help="fraction held out for validation loss, for tuning only")
  parser.add_argument("--max-len", type=int, default=2048)
  parser.add_argument("--micro-batch", type=int, default=4, help="examples per forward inside a batch")
  parser.add_argument("--max-grad-norm", type=float, default=1.0, help="DP-SGD per-example L2 clipping bound C")
  parser.add_argument("--eps", type=float, default=None, help="enable DP-SGD calibrated to this epsilon")
  parser.add_argument("--delta", type=float, default=1e-5, help="DP delta for the epsilon calibration")
  parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
  main(parser.parse_args())
