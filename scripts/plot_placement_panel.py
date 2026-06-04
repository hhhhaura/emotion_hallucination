#!/usr/bin/env python3
"""
Combined before/after panel for one dataset × emotion across three placements.

Layout matches scripts/plot.py (intensity variant) but swaps joy_1/2/3 for:
  system_suffix, user_prefix, assistant_think

  ┌─────────────────────┬────────────┐
  │ 4-bar histogram     │ mean/delta │
  │ (Before + 3 after)  │  placement1│
  │                     ├────────────┤
  │                     │  placement2│
  ├──────┬──────┬───────┴────────────┤
  │ sc 1 │ sc 2 │ sc 3  │ placement3 │
  └──────┴──────┴───────┴────────────┘
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLACEMENTS = ("system_suffix", "user_prefix", "assistant_think")
PLACEMENT_LABELS = {
    "system_suffix": "System suffix",
    "user_prefix": "User prefix",
    "assistant_think": "Assistant think",
}
AFTER_COLORS = ("#DD8452", "#55A868", "#C44E52")  # orange, green, red
BEFORE_COLOR = "#4C72B0"


def load_jsonl(file_path: Path) -> pd.DataFrame:
    """Same scoring rule as scripts/plot.py."""
    data = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            if "metrics" in obj:
                metrics = obj.get("metrics", {})
                if "f1" in metrics:
                    score = metrics["f1"]
                elif "accuracy" in metrics:
                    score = metrics["accuracy"]
                elif "em" in metrics:
                    score = metrics["em"]
                else:
                    score = 0.0
            elif "samples" in obj:
                score = np.mean([float(sample[1]) for sample in obj["samples"]])
            else:
                score = 0.0
            if score > 1.0:
                score = score / 100.0
            data.append({"example_id": obj["example_id"], "score": score})
    return pd.DataFrame(data)


def merge_dataset(
    input_dir: Path,
    dataset: str,
    emotion: str,
) -> tuple[pd.DataFrame, list[str]]:
    baseline = input_dir / f"{dataset}.jsonl"
    after_paths = [input_dir / f"{dataset}_{emotion}_{pl}.jsonl" for pl in PLACEMENTS]
    for path in [baseline, *after_paths]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")

    df = load_jsonl(baseline).rename(columns={"score": "before"})
    col_names: list[str] = []
    for pl in PLACEMENTS:
        col = pl
        col_names.append(col)
        df = df.merge(
            load_jsonl(input_dir / f"{dataset}_{emotion}_{pl}.jsonl").rename(
                columns={"score": col}
            ),
            on="example_id",
        )
    return df, col_names


def assign_bins(df: pd.DataFrame, bins: np.ndarray, labels: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["bin"] = pd.cut(
        out["before"],
        bins=bins,
        right=False,
        include_lowest=True,
        labels=labels,
    )
    out.loc[out["before"] == 1.0, "bin"] = labels[-1]
    return out


def plot_panel(
    df: pd.DataFrame,
    after_cols: list[str],
    *,
    dataset: str,
    emotion: str,
    output_path: Path,
    dpi: int,
) -> None:
    bins = np.arange(0, 1.1, 0.1)
    labels = [f"{i:.1f}-{i + 0.1:.1f}" for i in np.arange(0, 1.0, 0.1)]
    df = assign_bins(df, bins, labels)

    emotion_title = emotion.replace("_", " ").title()
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"{dataset}_{emotion} (three placements)", fontsize=14, y=0.98)

    gs = gridspec.GridSpec(
        3,
        4,
        figure=fig,
        width_ratios=[1, 1, 1, 0.85],
        height_ratios=[1.2, 1.2, 1],
        hspace=0.35,
        wspace=0.32,
    )

    # --- Top-left: grouped histogram (Before + 3 placements) ---
    ax_hist = fig.add_subplot(gs[0:2, 0:3])

    def get_counts(series: pd.Series) -> np.ndarray:
        counts, _ = np.histogram(series, bins=bins)
        return counts

    counts_before = get_counts(df["before"])
    counts_after = [get_counts(df[col]) for col in after_cols]
    x = np.arange(len(labels))
    width = 0.2

    ax_hist.bar(x - 1.5 * width, counts_before, width, label="Before", color=BEFORE_COLOR)
    for i, (col, color) in enumerate(zip(after_cols, AFTER_COLORS)):
        offset = (-0.5 + i) * width
        ax_hist.bar(
            x + offset,
            counts_after[i],
            width,
            label=f"After ({PLACEMENT_LABELS[col]})",
            color=color,
        )
    ax_hist.set_xticks(x)
    ax_hist.set_xticklabels(labels, rotation=45, ha="right")
    ax_hist.set_xlabel("Accuracy bins (0.1 width)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Histogram of accuracy (Before vs after stimuli)")
    ax_hist.legend(loc="upper right", fontsize=8)
    ax_hist.grid(axis="y", alpha=0.25)

    # --- Right column: mean after + delta per placement ---
    for row_idx, col in enumerate(after_cols):
        ax_md = fig.add_subplot(gs[row_idx, 3])
        grouped = df.groupby("bin", observed=False)
        means = grouped[col].mean().fillna(0)
        deltas = (means - grouped["before"].mean().fillna(0)).fillna(0)

        x_bin = np.arange(len(labels))
        ax_md.bar(
            x_bin - 0.2,
            means,
            0.4,
            label=f"Mean after ({PLACEMENT_LABELS[col]})",
            color=BEFORE_COLOR,
        )
        ax_md.bar(
            x_bin + 0.2,
            deltas,
            0.4,
            label="Delta (after - before)",
            color=AFTER_COLORS[row_idx],
        )
        ax_md.set_xticks(x_bin)
        ax_md.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax_md.axhline(0, color="black", linewidth=1)
        ax_md.set_ylim(-0.15, 1.05)
        ax_md.set_ylabel("Accuracy / delta")
        ax_md.set_title(
            f"Mean after & delta by before interval\n({PLACEMENT_LABELS[col]})",
            fontsize=9,
        )
        ax_md.legend(fontsize=6, loc="upper right")
        ax_md.grid(axis="y", alpha=0.25)

    # --- Bottom row: scatter plots ---
    for col_idx, col in enumerate(after_cols):
        ax_sc = fig.add_subplot(gs[2, col_idx])
        ax_sc.scatter(
            df["before"],
            df[col],
            alpha=0.35,
            s=12,
            color="#4C99B0",
            edgecolors="none",
        )
        ax_sc.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1.5)
        ax_sc.set_xlim(0, 1)
        ax_sc.set_ylim(0, 1)
        ax_sc.set_aspect("equal", adjustable="box")
        ax_sc.set_xlabel("Before accuracy")
        ax_sc.set_ylabel("After accuracy")
        ax_sc.set_title(f"Scatter: before vs after ({PLACEMENT_LABELS[col]})")
        ax_sc.grid(True, linestyle="--", alpha=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="math_en")
    parser.add_argument("--emotion", default="fear", choices=("joy", "anger", "fear"))
    parser.add_argument("--input-dir", type=Path, default=repo_root / "outputs")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG path (default: outputs/figures/panels/{dataset}_{emotion}_placements.png)",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    output = args.output or (
        repo_root
        / "outputs"
        / "figures"
        / "panels"
        / f"{args.dataset}_{args.emotion}_placements.png"
    )

    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    print("Loading data...")
    df, after_cols = merge_dataset(args.input_dir, args.dataset, args.emotion)
    print(f"Merged {len(df)} examples.")

    plot_panel(
        df,
        after_cols,
        dataset=args.dataset,
        emotion=args.emotion,
        output_path=output,
        dpi=args.dpi,
    )

    print(f"Saved {output}")
    for col in after_cols:
        delta = df[col].mean() - df["before"].mean()
        print(
            f"  {PLACEMENT_LABELS[col]:16s}  before={df['before'].mean():.4f}  "
            f"after={df[col].mean():.4f}  delta={delta:+.4f}"
        )


if __name__ == "__main__":
    main()
