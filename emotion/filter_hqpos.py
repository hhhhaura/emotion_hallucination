#!/usr/bin/env python3
"""Filter emotion experiment outputs to high-quality examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_EMOTION_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EMOTION_DIR.parent
DEFAULT_INPUT_DIR = _REPO_ROOT / "outputs" / "emotion"
DEFAULT_OUTPUT_DIR = _EMOTION_DIR / "filtered"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Scan emotion JSONL outputs and write filtered/<dataset>.jsonl. "
            "Keeps rows with metrics.accuracy > accuracy threshold (MC/math) "
            "or metrics.f1 > f1 threshold (SQuAD/CMRC, 0–100 scale)."
        )
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory of *.jsonl outputs (default: {DEFAULT_INPUT_DIR})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for filtered outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--accuracy-threshold",
        type=float,
        default=0.8,
        help="Keep examples with metrics.accuracy strictly greater than this (default: 0.8)",
    )
    p.add_argument(
        "--f1-threshold",
        type=float,
        default=80.0,
        help="Keep examples with metrics.f1 strictly greater than this, 0–100 scale (default: 80)",
    )
    return p.parse_args()


def _filter_mode(metrics: dict) -> str | None:
    if "accuracy" in metrics:
        return "accuracy"
    if "f1" in metrics:
        return "f1"
    return None


def _passes_filter(
    record: dict,
    *,
    accuracy_threshold: float,
    f1_threshold: float,
    mode: str,
) -> bool:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return False
    if mode == "accuracy":
        return float(metrics["accuracy"]) > accuracy_threshold
    if mode == "f1":
        return float(metrics["f1"]) > f1_threshold
    return False


def filter_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    accuracy_threshold: float,
    f1_threshold: float,
) -> tuple[int, int, str | None]:
    """
    Returns (total_lines, kept_lines, mode).
    mode is None when the file has no supported metrics.
    """
    total = 0
    kept = 0
    mode: str | None = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            metrics = record.get("metrics")
            if not isinstance(metrics, dict):
                continue

            row_mode = _filter_mode(metrics)
            if row_mode is None:
                continue

            if mode is None:
                mode = row_mode
            elif mode != row_mode:
                print(
                    f"warning: {input_path.name} mixes {mode} and {row_mode}; using {mode}",
                    file=sys.stderr,
                )

            if _passes_filter(
                record,
                accuracy_threshold=accuracy_threshold,
                f1_threshold=f1_threshold,
                mode=mode,
            ):
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

    if mode is None:
        if output_path.exists():
            output_path.unlink()
        return total, 0, None

    return total, kept, mode


def main() -> int:
    args = _parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        print(f"error: input directory not found: {input_dir}", file=sys.stderr)
        return 1

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"warning: no *.jsonl files in {input_dir}", file=sys.stderr)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in jsonl_files:
        dataset_name = input_path.stem
        output_path = output_dir / f"{dataset_name}.jsonl"
        total, kept, mode = filter_jsonl(
            input_path,
            output_path,
            accuracy_threshold=args.accuracy_threshold,
            f1_threshold=args.f1_threshold,
        )

        if mode is None:
            print(f"{dataset_name}: skipped (no metrics.accuracy or metrics.f1)", file=sys.stderr)
            continue

        threshold = args.accuracy_threshold if mode == "accuracy" else args.f1_threshold
        pct = (100.0 * kept / total) if total else 0.0
        print(
            f"{dataset_name}: kept {kept}/{total} ({pct:.1f}%) "
            f"[{mode} > {threshold}] -> {output_path}",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
