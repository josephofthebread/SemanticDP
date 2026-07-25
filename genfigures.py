#! /usr/bin/env python
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger("genfigures")


def save(figure: plt.Figure, name: str) -> None:
  path = Path("paper/fig") / f"{name}.pdf"
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, format="pdf")
  plt.close(figure)
  log.info(f"{path}")


def main() -> None:
  plt.rcParams.update(
    {
      "font.family": "serif",
      "font.serif": ["Times New Roman", "DejaVu Serif"],
      "font.size": 9,
      "axes.labelsize": 9,
      "axes.titlesize": 9,
      "legend.fontsize": 8,
      "xtick.labelsize": 8,
      "ytick.labelsize": 8,
      "axes.spines.top": False,
      "axes.spines.right": False,
      "axes.grid": True,
      "grid.alpha": 0.25,
      "grid.linewidth": 0.5,
      "figure.dpi": 150,
      "savefig.bbox": "tight",
      "savefig.pad_inches": 0.02,
    }
  )

  models = ["Qwen3-1.7B", "Llama-3.2-1B-Instruct", "Falcon3-1B-Instruct"]
  evaluate = pd.read_csv("results/evaluate.csv")
  evaluate = evaluate[evaluate.model.isin(models) & evaluate.tag.isna()]
  safety = pd.read_csv("results/safety.csv")
  safety = safety[safety.model.isin(evaluate.model.unique()) & safety.tag.isna()]
  log.info(f"{len(evaluate)} evaluation and {len(safety)} safety rows over {evaluate.model.nunique()} models")

  registered = evaluate[evaluate.mechanism.isin(["m1", "m2", "m3"])].dropna(subset=["Y", "S_text"])
  figure, axis = plt.subplots(figsize=(4.6, 2.45))
  for mechanism, name, color in [
    ("m1", "Random replacement", "#c2410c"),
    ("m2", "Sanitization", "#1d4ed8"),
    ("m3", "Private optimization", "#047857"),
  ]:
    part = registered[registered.mechanism == mechanism]
    axis.scatter(part.S_text, part.Y, s=14, alpha=0.55, linewidths=0, color=color, label=f"{name} ($n={len(part)}$)")
  targeted = evaluate[evaluate.mechanism == "m2e"].dropna(subset=["Y", "S_text"])
  axis.scatter(
    targeted.S_text,
    targeted.Y,
    s=24,
    marker="D",
    linewidths=0.5,
    edgecolors="#3b0764",
    color="#a78bfa",
    zorder=6,
    label=f"Entity-targeted control ($n={len(targeted)}$)",
  )
  slope, intercept = np.polyfit(registered.S_text, registered.Y, 1)
  grid = np.linspace(0, registered.S_text.max(), 100)
  axis.plot(
    grid, intercept + slope * grid, color="#111827", lw=1.4, zorder=5, label="Pooled trend, registered mechanisms"
  )
  axis.axvline(0, color="#6b7280", lw=0.7, ls=":", zorder=1)
  axis.annotate(
    "private optimization\nleaves the text intact",
    xy=(0.004, registered.Y.max() * 0.60),
    xytext=(0.085, registered.Y.max() * 0.93),
    fontsize=7.5,
    color="#374151",
    arrowprops=dict(arrowstyle="->", lw=0.7, color="#374151"),
  )
  axis.annotate(
    "entities damaged,\ncarrier spared",
    xy=(targeted.S_text.mean(), targeted.Y.mean()),
    xytext=(0.20, targeted.Y.min() + 0.30),
    ha="left",
    va="top",
    fontsize=7.5,
    color="#5b21b6",
    arrowprops=dict(arrowstyle="->", lw=0.7, color="#5b21b6"),
  )
  axis.set_xlabel("Semantic distortion $S$ of the training text")
  axis.set_ylabel("Alignment degradation $Y$")
  axis.legend(
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.28),
    ncols=3,
    handletextpad=0.4,
    columnspacing=1.2,
  )
  save(figure, "pathway")

  sanitization = evaluate[evaluate.mechanism == "m2"]
  figure, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), sharex=True)
  for axis, column, label in zip(axes, ["S_text", "Y"], ["Semantic distortion $S$", "Alignment degradation $Y$"]):
    for dataset, name, marker, color in [
      ("nemotron", "Entity-rich", "o", "#1d4ed8"),
      ("alpaca", "Entity-poor", "s", "#c2410c"),
    ]:
      grouped = sanitization[sanitization.dataset == dataset].groupby("level")[column]
      mean, spread = grouped.mean(), grouped.std()
      axis.errorbar(
        mean.index,
        mean.values,
        yerr=spread.values,
        marker=marker,
        ms=3.5,
        lw=1.2,
        capsize=2,
        elinewidth=0.7,
        color=color,
        label=name,
      )
    axis.axvspan(0.4, 3.3, color="#9ca3af", alpha=0.16, lw=0)
    axis.set_xscale("log")
    axis.set_xticks([0.5, 1, 2, 3, 4, 6, 12])
    axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axis.set_xlabel(r"Sanitization budget $\varepsilon$")
    axis.set_ylabel(label)
  for axis, fraction in zip(axes, [0.42, 0.96]):
    bottom, top = axis.get_ylim()
    position = bottom + fraction * (top - bottom)
    axis.annotate("plateau", xy=(1.4, position), ha="center", va="top", fontsize=7.5, color="#4b5563")
  axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.06))
  save(figure, "threshold")

  conditions = [
    ("Clean\nbaseline", "m0", None, "#6b7280"),
    ("Uniform\nsanitization", "m2", 1.0, "#1d4ed8"),
    ("Entity-targeted\nsanitization", "m2e", 1.0, "#7c3aed"),
  ]
  panels = [
    ("Entity fidelity", evaluate, "entity_fidelity/correct", False),
    ("Fabrication rate", evaluate, "entity_fidelity/fabricated", True),
    ("Refusal rate", safety, "advbench/refusal_rate", False),
  ]
  figure, axes = plt.subplots(1, 3, figsize=(6.4, 2.5))
  for axis, (title, source, column, lower_better) in zip(axes, panels):
    rich = source[source.dataset == "nemotron"]
    values = []
    for _, mechanism, level, _ in conditions:
      rows = rich[rich.mechanism == mechanism]
      values.append(rows[rows.level == level][column].mean() if level is not None else rows[column].mean())
    axis.bar(range(len(conditions)), values, color=[color for *_, color in conditions], width=0.62)
    for index, value in enumerate(values):
      axis.text(index, value, f"{value:.4f}", ha="center", va="bottom", fontsize=7.5)
    axis.set_xticks(range(len(conditions)))
    axis.set_xticklabels([name for name, *_ in conditions], fontsize=7)
    axis.set_title(title + (" (lower better)" if lower_better else ""), fontsize=8.5)
    axis.set_ylim(0, max(values) * 1.28)
  save(figure, "carrier")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
  main()
