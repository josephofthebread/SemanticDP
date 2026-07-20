#! /usr/bin/env python
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv

log = logging.getLogger("results")

ADAPTER = re.compile(
  r"^adapter-(?P<model>.+?)-(?P<corpus>nemotron|alpaca)_train(?P<arm>.*?)-s(?P<seed>\d+)(?:-eps(?P<eps>\d+))?$"
)


def parse_lora(lora: object) -> dict[str, Any]:
  empty = {"corpus": None, "arm": None, "train_seed": np.nan, "dp_eps": np.nan}
  if not isinstance(lora, str):
    return empty
  match = ADAPTER.match(lora.split(":")[0])
  if match is None:
    return empty
  groups = match.groupdict()
  return {
    "corpus": groups["corpus"],
    "arm": (groups["arm"] or "").lstrip("_") or "clean",
    "train_seed": int(groups["seed"]),
    "dp_eps": float(groups["eps"]) if groups["eps"] else np.nan,
  }


def annotate(runs: pd.DataFrame, job: str, corpus_s: pd.DataFrame, generation_s: pd.DataFrame) -> pd.DataFrame:
  frame = runs[runs.job == job].reset_index(drop=True)
  parsed = pd.DataFrame(
    [parse_lora(lora) for lora in frame["cfg.lora"]], columns=["corpus", "arm", "train_seed", "dp_eps"]
  )
  frame = pd.concat([frame, parsed], axis=1)
  frame["model"] = frame["cfg.model"].str.split("/").str[-1]
  frame["mechanism"] = frame["cfg.mechanism"]
  frame["level"] = pd.to_numeric(frame["cfg.level"], errors="coerce")
  frame["d"] = [
    {
      ("m1", 0.5): 1,
      ("m1", 0.3): 2,
      ("m1", 0.15): 3,
      ("m1", 0.05): 4,
      ("m2", 1.0): 1,
      ("m2", 3.0): 2,
      ("m2", 6.0): 3,
      ("m2", 12.0): 4,
      ("m3", 1.0): 1,
      ("m3", 4.0): 2,
      ("m3", 8.0): 3,
      ("m3", 16.0): 4,
    }.get(key, np.nan)
    for key in zip(frame.mechanism, frame.level)
  ]
  frame["stem"] = frame["cfg.lora"].str.split(":").str[0].str.removeprefix("adapter-")
  frame = frame.merge(corpus_s, on=["corpus", "arm"], how="left").merge(generation_s, on="stem", how="left")
  frame.loc[frame.mechanism.isin(["m0", "m3", "base"]), ["S_corpus", "S_ent", "S_gen"]] = 0.0
  return frame


def against_m0(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
  present = [column for column in columns if column in frame]
  baseline = (
    frame[frame.mechanism == "m0"]
    .groupby(["corpus", "model"])[present]
    .mean()
    .rename(columns={column: f"m0.{column}" for column in present})
    .reset_index()
  )
  frame = frame.merge(baseline, on=["corpus", "model"], how="left")
  for column in present:
    frame[f"drop.{column}"] = frame[f"m0.{column}"] - frame[column]
  return frame


def main() -> None:
  load_dotenv()
  Path("results").mkdir(exist_ok=True)
  api = wandb.Api()

  rows = []
  for run in api.runs("josephofthebread/SemanticDP"):
    summary = {
      key: value
      for key, value in run.summary._json_dict.items()
      if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    config = {f"cfg.{key}": value for key, value in run.config.items() if not isinstance(value, (dict, list))}
    rows.append(
      {
        "id": run.id,
        "name": run.name,
        "job": run.job_type,
        "state": run.state,
        "created_at": str(run.created_at),
        "runtime_min": (run.summary.get("_runtime") or 0) / 60,
        **config,
        **summary,
      }
    )
  runs = pd.DataFrame(rows)
  log.info(f"{len(runs)} runs: {runs.groupby(['job', 'state']).size().to_dict()}")

  directory = Path(api.artifact("josephofthebread/SemanticDP/distortion:latest", type="distortion").download())
  distortion = json.loads(next(directory.glob("*.json")).read_text())
  corpus_s = pd.DataFrame(
    [{"corpus": key.split("/")[0], "arm": key.split("/")[1], **values} for key, values in distortion["corpus"].items()]
  ).rename(columns={"s": "S_corpus", "n": "n_corpus", "s_ent": "S_ent", "s_gen": "S_gen"})
  generation_s = pd.DataFrame(
    [
      {"stem": key, "S_gen_side": values["s"], "n_gen_side": values["n"]}
      for key, values in distortion["generations"].items()
    ]
  )
  log.info(f"distortion: {len(corpus_s)} arms, {len(generation_s)} runs")

  COMPONENTS = ["ifeval/prompt_accuracy", "entity_fidelity/correct", "truthfulqa/mc1"]
  SAFETY = [
    "advbench/refusal_rate",
    "xstest/safe_refusal_rate",
    "xstest/unsafe_refusal_rate",
    "xstest/discrimination",
  ]

  evaluate = against_m0(annotate(runs, "evaluate", corpus_s, generation_s), COMPONENTS)
  safety = against_m0(annotate(runs, "safety", corpus_s, generation_s), SAFETY)
  log.info(f"safety: {len(safety)} runs, {sum(column in safety for column in SAFETY)}/{len(SAFETY)} metrics present")

  mechanisms = evaluate.mechanism.isin(["m1", "m2", "m3"])
  for component in COMPONENTS:
    column = evaluate.loc[mechanisms, f"drop.{component}"]
    evaluate.loc[mechanisms, f"z.{component}"] = (column - column.mean()) / column.std()
  evaluate["Y"] = evaluate[[f"z.{component}" for component in COMPONENTS]].mean(axis=1)
  evaluate["Y_no_truthfulqa"] = evaluate[
    [f"z.{component}" for component in COMPONENTS if not component.startswith("truthfulqa")]
  ].mean(axis=1)

  train = runs[runs.job == "train"].reset_index(drop=True)
  train["model"] = train["cfg.model"].str.split("/").str[-1]
  train["corpus"] = train["cfg.split"].str.split("_").str[0]

  tables = {
    "evaluate": evaluate,
    "safety": safety,
    "train": train,
    "distortion": corpus_s,
    "generations": generation_s,
  }
  for name, table in tables.items():
    table = table.dropna(axis=1, how="all")
    path = Path("results") / f"{name}.csv"
    table.to_csv(path, index=False)
    log.info(f"{path}: {len(table)} rows x {len(table.columns)} columns")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
  main()
