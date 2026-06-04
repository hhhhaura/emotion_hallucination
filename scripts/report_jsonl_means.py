#!/usr/bin/env python3
"""Report mean per-problem score for each JSONL (same rule as scripts/plot.py)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from score_utils import add_common_args, per_question_scores, score_metric_label  # noqa: E402


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path (default: print to stdout only)",
    )
    args = parser.parse_args()

    rows: list[tuple[str, int, float, str]] = []
    for path in sorted(args.input_dir.rglob("*.jsonl")):
        if "smoke" in path.parts:
            continue
        _, scores = per_question_scores(path, args.qa_metric)
        if not scores:
            continue
        metric = score_metric_label(path, args.qa_metric)
        mean = sum(scores) / len(scores)
        rows.append((path.relative_to(args.input_dir).as_posix(), len(scores), mean, metric))

    rows.sort(key=lambda r: (r[0].split("_")[0], r[0]))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file", "n", "mean", "metric"])
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.csv}")

    print(f"{'file':<55} {'n':>6}  {'mean':>8}  metric")
    print("-" * 78)
    for name, n, mean, metric in rows:
        print(f"{name:<55} {n:>6}  {mean:>8.4f}  {metric}")


if __name__ == "__main__":
    main()
