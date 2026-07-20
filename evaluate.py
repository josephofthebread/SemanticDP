"""Common stuff for alignment.py and safety.py."""

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class Model:
  def __init__(self, name: str, lora: Path | None, dtype: str, max_tokens: int, batch: int) -> None:
    self.tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    if self.tokenizer.pad_token_id is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype)).to("cuda")  # type: ignore[arg-type]
    self.model = (PeftModel.from_pretrained(model, str(lora)) if lora else model).eval()
    self.max_tokens = max_tokens
    self.batch = batch
    self.generations: list[dict[str, str]] = []

  def prompt_ids(self, instruction: str) -> list[int]:
    ids = self.tokenizer.apply_chat_template(
      [{"role": "user", "content": instruction}],
      tokenize=True,
      add_generation_prompt=True,
      return_dict=False,
      enable_thinking=False,
    )
    return list(ids)

  def generate(self, task: str, instructions: list[str]) -> list[str]:
    outputs = []
    for start in range(0, len(instructions), self.batch):
      chunk = [self.prompt_ids(text) for text in instructions[start : start + self.batch]]
      width = max(len(ids) for ids in chunk)
      pad = self.tokenizer.pad_token_id
      input_ids = torch.tensor([[pad] * (width - len(ids)) + ids for ids in chunk], device="cuda")
      mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in chunk], device="cuda")
      with torch.no_grad():
        generated = self.model.generate(
          input_ids=input_ids,
          attention_mask=mask,
          max_new_tokens=self.max_tokens,
          do_sample=False,
          pad_token_id=pad,
        )
      outputs += self.tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
    responses = [text.strip() for text in outputs]
    self.generations += [
      {"task": task, "prompt": prompt, "response": response}
      for prompt, response in zip(instructions, responses, strict=True)
    ]
    return responses

  def choice_logprobs(self, contexts: list[str], choices: list[list[str]]) -> list[list[tuple[float, int]]]:
    pairs, spans = [], []
    for context, options in zip(contexts, choices, strict=True):
      context_ids = self.prompt_ids(context)
      for option in options:
        option_ids = self.tokenizer.encode(option, add_special_tokens=False)
        pairs.append(context_ids + option_ids)
        spans.append((len(context_ids), len(option_ids)))

    totals = []
    for start in range(0, len(pairs), self.batch):
      chunk = pairs[start : start + self.batch]
      width = max(len(ids) for ids in chunk)
      pad = self.tokenizer.pad_token_id
      input_ids = torch.tensor([ids + [pad] * (width - len(ids)) for ids in chunk], device="cuda")
      mask = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids in chunk], device="cuda")
      with torch.no_grad():
        logprobs = self.model(input_ids=input_ids, attention_mask=mask).logits.float().log_softmax(-1)
      for row, ids in enumerate(chunk):
        context_length, option_length = spans[start + row]
        total = 0.0
        for position in range(context_length, context_length + option_length):
          total += float(logprobs[row, position - 1, ids[position]])
        totals.append((total, option_length))

    scores, cursor = [], 0
    for options in choices:
      scores.append(totals[cursor : cursor + len(options)])
      cursor += len(options)
    return scores
