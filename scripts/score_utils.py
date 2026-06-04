"""Shared utilities for baseline scores and before/after plots."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

QaMetric = Literal["f1", "em"]

DATASETS = (
    "squad_v2",
    "cmrc2018",
    "math_en",
    "math_zh",
    "mmlu_med_en",
    "mmlu_med_zh",
    "truthfulqa_mc",
    "halluqa_mc",
)

JOY_VARIANTS = (1, 2, 3)
PLACEMENTS = ("system_suffix", "user_prefix", "assistant_think")
EMOTIONS = ("joy", "anger", "fear")

BIN_LABELS = [f"{i / 10:.1f}–{(i + 1) / 10:.1f}" for i in range(10)]

_BASELINE_JSONL_RE = re.compile(
    r"^(" + "|".join(re.escape(d) for d in DATASETS) + r")\.jsonl$"
)


def is_baseline_jsonl(path: Path) -> bool:
    return _BASELINE_JSONL_RE.match(path.name) is not None


def dataset_from_baseline(path: Path) -> str:
    match = _BASELINE_JSONL_RE.match(path.name)
    if not match:
        raise ValueError(f"Not a baseline JSONL filename: {path.name}")
    return match.group(1)


def dataset_from_jsonl_path(path: Path) -> str | None:
    stem = path.stem
    for ds in sorted(DATASETS, key=len, reverse=True):
        if stem == ds or stem.startswith(f"{ds}_"):
            return ds
    return None


def record_score(record: dict, qa_metric: QaMetric = "f1") -> float:
    """Per-problem score using the same rule as scripts/plot.py load_jsonl()."""
    del qa_metric  # kept for CLI compatibility; metrics dict takes precedence

    if "metrics" in record:
        metrics = record.get("metrics") or {}
        if "f1" in metrics:
            score = float(metrics["f1"])
        elif "accuracy" in metrics:
            score = float(metrics["accuracy"])
        elif "em" in metrics:
            score = float(metrics["em"])
        else:
            score = 0.0
    elif "samples" in record:
        samples = record.get("samples") or []
        if samples:
            score = float(np.mean([float(sample[1]) for sample in samples]))
        else:
            score = 0.0
    else:
        score = 0.0

    if score > 1.0:
        score /= 100.0
    return score


def score_metric_label(path: Path, qa_metric: QaMetric) -> str:
    del qa_metric
    try:
        records = load_records(path)
        if records:
            metrics = records[0].get("metrics") or {}
            if "f1" in metrics:
                return "f1"
            if "accuracy" in metrics:
                return "accuracy"
            if "em" in metrics:
                return "em"
            if "samples" in records[0]:
                return "samples[1]"
    except OSError:
        pass
    return "unknown"


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def per_question_scores(path: Path, qa_metric: QaMetric = "f1") -> tuple[list[str], list[float]]:
    """Return (example_ids, scores) in JSONL line order."""
    ids: list[str] = []
    scores: list[float] = []
    for record in load_records(path):
        ex_id = record.get("example_id")
        if ex_id is None:
            raise ValueError(f"Missing example_id in {path}")
        ids.append(str(ex_id))
        scores.append(record_score(record, qa_metric))
    return ids, scores


def write_score_txt(path: Path, scores: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for score in scores:
            handle.write(f"{score:.6f}\n")


def load_score_txt(path: Path) -> list[float]:
    scores: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                scores.append(float(line))
    return scores


def align_before_after(
    before_ids: list[str],
    before_scores: list[float],
    after_ids: list[str],
    after_scores: list[float],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Align by example_id; return before, after arrays and dropped count."""
    if len(before_ids) != len(before_scores) or len(after_ids) != len(after_scores):
        raise ValueError("example_id and score list lengths must match")
    before_map = dict(zip(before_ids, before_scores))
    after_map = dict(zip(after_ids, after_scores))

    common = [ex_id for ex_id in before_ids if ex_id in after_map]
    dropped = len(before_ids) - len(common)
    if not common:
        raise ValueError("No overlapping example_id between before and after")

    before = np.array([before_map[ex_id] for ex_id in common], dtype=float)
    after = np.array([after_map[ex_id] for ex_id in common], dtype=float)
    return before, after, dropped


def load_before_scores(
    baseline_jsonl: Path,
    score_txt: Path,
    qa_metric: QaMetric,
) -> tuple[list[str], np.ndarray]:
    if score_txt.is_file():
        ids, _ = per_question_scores(baseline_jsonl, qa_metric)
        scores = load_score_txt(score_txt)
        if len(scores) != len(ids):
            print(
                f"Warning: {score_txt.name} has {len(scores)} lines but "
                f"{baseline_jsonl.name} has {len(ids)} questions; using JSONL"
            )
        else:
            return ids, np.array(scores, dtype=float)

    if not baseline_jsonl.is_file():
        raise FileNotFoundError(f"Baseline JSONL not found: {baseline_jsonl}")
    print(f"Warning: {score_txt} missing; recomputing before scores from {baseline_jsonl.name}")
    ids, scores = per_question_scores(baseline_jsonl, qa_metric)
    return ids, np.array(scores, dtype=float)


def before_after_arrays(
    baseline_jsonl: Path,
    after_jsonl: Path,
    score_txt: Path,
    qa_metric: QaMetric,
) -> tuple[np.ndarray, np.ndarray]:
    before_ids, before = load_before_scores(baseline_jsonl, score_txt, qa_metric)
    after_ids, after_list = per_question_scores(after_jsonl, qa_metric)
    before_a, after_a, dropped = align_before_after(
        before_ids, before.tolist(), after_ids, after_list
    )
    if dropped:
        print(f"  aligned {len(before_a)} questions (dropped {dropped} missing in after)")
    return before_a, after_a


def bin_indices(before: np.ndarray) -> np.ndarray:
    """Half-open bins [0,0.1), ..., [0.9,1.0]; score 1.0 maps to bin 9."""
    clipped = np.clip(before, 0.0, 1.0)
    return np.minimum((clipped * 10).astype(int), 9)


def aggregate_binned(before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean_after, mean_delta, counts) for 10 bins."""
    bins = bin_indices(before)
    delta = after - before
    mean_after = np.full(10, np.nan)
    mean_delta = np.full(10, np.nan)
    counts = np.zeros(10, dtype=int)
    for b in range(10):
        mask = bins == b
        n = int(mask.sum())
        counts[b] = n
        if n == 0:
            continue
        mean_after[b] = float(after[mask].mean())
        mean_delta[b] = float(delta[mask].mean())
    return mean_after, mean_delta, counts


def plot_overlaid_histogram(
    ax: plt.Axes,
    before: np.ndarray,
    after: np.ndarray,
    *,
    title: str,
    bins: int,
) -> None:
    edges = np.linspace(0.0, 1.0, bins + 1)
    if len(before) == 0 and len(after) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        if len(before):
            ax.hist(
                before,
                bins=edges,
                alpha=0.5,
                label="before",
                color="#4C72B0",
                edgecolor="white",
                linewidth=0.4,
            )
        if len(after):
            ax.hist(
                after,
                bins=edges,
                alpha=0.5,
                label="after",
                color="#DD8452",
                edgecolor="white",
                linewidth=0.4,
            )
        ax.legend(fontsize=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Per-question score")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", alpha=0.25)


def plot_scatter(
    ax: plt.Axes,
    before: np.ndarray,
    after: np.ndarray,
    *,
    title: str,
) -> None:
    if len(before) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.scatter(before, after, s=4, alpha=0.25, color="#4C72B0", edgecolors="none")
        if len(before) > 1:
            r = float(np.corrcoef(before, after)[0, 1])
            title = f"{title}\nr={r:.3f}"
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Before")
    ax.set_ylabel("After")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25)


def plot_binned_bars(
    ax: plt.Axes,
    values: np.ndarray,
    counts: np.ndarray,
    *,
    title: str,
    ylabel: str,
) -> None:
    x = np.arange(10)
    valid = ~np.isnan(values)
    if not valid.any():
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.bar(x[valid], values[valid], color="#4C72B0", edgecolor="white", linewidth=0.4)
        for i in np.where(valid)[0]:
            ax.text(
                i,
                values[i],
                f"n={counts[i]}",
                ha="center",
                va="bottom",
                fontsize=6,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Before score bin")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.grid(axis="y", alpha=0.25)


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def add_common_args(parser) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root / "outputs",
        help="Directory containing JSONL outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "figures",
        help="Directory for figure PNGs",
    )
    parser.add_argument(
        "--qa-metric",
        choices=("f1", "em"),
        default="f1",
        help="Deprecated; scoring follows scripts/plot.py (metrics f1 > accuracy > em)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Subset of datasets (default: all 8)",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--bins", type=int, default=20, help="Histogram bin count on [0, 1]")


def resolve_datasets(names: list[str] | None) -> list[str]:
    if not names:
        return list(DATASETS)
    unknown = set(names) - set(DATASETS)
    if unknown:
        raise SystemExit(f"Unknown datasets: {sorted(unknown)}")
    return names
