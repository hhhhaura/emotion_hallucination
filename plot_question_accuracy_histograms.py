#!/usr/bin/env python3
"""Plot per-question mean accuracy histograms for emotion JSONL outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _sample_scores(record: dict) -> list[float]:
    metrics = record.get("metrics") or {}
    samples = record.get("samples") or []
    if not samples:
        return []

    if "accuracy" in metrics:
        return [float(sample[1]) for sample in samples if len(sample) >= 2]

    if "em" in metrics:
        return [float(sample[1]) / 100.0 for sample in samples if len(sample) >= 2]

    return []


def per_question_mean_accuracies(path: Path) -> list[float]:
    means: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            scores = _sample_scores(record)
            if scores:
                means.append(sum(scores) / len(scores))
    return means


def plot_histogram(
    means: list[float],
    *,
    title: str,
    output_path: Path,
    bins: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if means:
        ax.hist(means, bins=bins, range=(0.0, 1.0), color="#4C72B0", edgecolor="white", linewidth=0.6)
        ax.axvline(np.mean(means), color="#C44E52", linestyle="--", linewidth=1.5, label=f"mean={np.mean(means):.3f}")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No questions found", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Per-question mean accuracy")
    ax.set_ylabel("Number of questions")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root / "outputs",
        help="Directory containing JSONL outputs (searched recursively)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "histograms",
        help="Directory for histogram PNG files",
    )
    parser.add_argument("--bins", type=int, default=20, help="Histogram bin count on [0, 1]")
    args = parser.parse_args()

    jsonl_files = sorted(args.input_dir.rglob("*.jsonl"))
    if not jsonl_files:
        raise SystemExit(f"No JSONL files found under {args.input_dir}")

    for jsonl_path in jsonl_files:
        rel = jsonl_path.relative_to(args.input_dir)
        out_path = args.output_dir / rel.with_suffix(".png")
        means = per_question_mean_accuracies(jsonl_path)
        plot_histogram(
            means,
            title=f"{rel.as_posix()}\n(n={len(means)} questions)",
            output_path=out_path,
            bins=args.bins,
        )
        print(f"{rel.as_posix()}: {len(means)} questions -> {out_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
