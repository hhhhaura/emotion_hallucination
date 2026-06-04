from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, replace
from typing import Any

import yaml


@dataclass(frozen=True)
class RunnerConfig:
    output_path: str
    vllm_config: str
    split: str
    limit_problems: int | None
    N: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int | None
    seed: int | None
    enable_thinking: bool
    stimulus: str
    stimulus_language: str | None
    stimulus_config: str
    stimulus_placement: str


def build_argparser(default_config_path: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_path,
        help="Path to dataset config YAML.",
    )
    parser.add_argument(
        "--stimulus",
        type=str,
        default=None,
        help="Emotion stimulus id from the selected stimulus catalog (overrides config).",
    )
    parser.add_argument(
        "--stimulus-config",
        type=str,
        default=None,
        help="Path to stimulus catalog YAML (overrides config).",
    )
    parser.add_argument(
        "--stimulus-language",
        type=str,
        default=None,
        help="Language for stimulus text: en or zh (overrides config / dataset default).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Override output JSONL path from config.",
    )
    return parser


def resolve_config_path(path: str, *, config_yaml_path: str | None = None) -> str:
    if os.path.isabs(path):
        return path

    candidates: list[str] = []
    if config_yaml_path:
        cfg_dir = os.path.dirname(os.path.abspath(config_yaml_path))
        repo_root = os.path.abspath(os.path.join(cfg_dir, "..", ".."))
        candidates.extend(
            [
                os.path.join(repo_root, path),
                os.path.join(cfg_dir, path),
            ]
        )
    candidates.append(os.path.abspath(path))

    for c in candidates:
        if os.path.isfile(c):
            return c
    return candidates[0]


def load_runner_config(yaml_path: str) -> RunnerConfig:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    seed = cfg.get("seed", None)
    if seed is not None:
        seed = int(seed)

    top_k = cfg.get("top_k", None)
    if top_k is not None:
        top_k = int(top_k)

    stimulus_language = cfg.get("stimulus_language", None)
    if stimulus_language is not None:
        stimulus_language = str(stimulus_language).strip().lower()

    stimulus_config = str(cfg.get("stimulus_config", "emotion/configs/stimuli.yaml"))
    stimulus_config = resolve_config_path(stimulus_config, config_yaml_path=yaml_path)

    return RunnerConfig(
        output_path=str(cfg.get("output_path", "outputs/emotion/run.jsonl")),
        vllm_config=str(cfg.get("vllm_config", "vllm_config/qwen3-1.7B-chat.yaml")),
        split=str(cfg.get("split", "test")),
        limit_problems=cfg.get("limit_problems", None),
        N=int(cfg.get("N", 1)),
        max_new_tokens=int(cfg.get("max_new_tokens", 512)),
        temperature=float(cfg.get("temperature", 0.7)),
        top_p=float(cfg.get("top_p", 0.95)),
        top_k=top_k,
        seed=seed,
        enable_thinking=bool(cfg.get("enable_thinking", False)),
        stimulus=str(cfg.get("stimulus", "none")),
        stimulus_language=stimulus_language,
        stimulus_config=stimulus_config,
        stimulus_placement=str(cfg.get("stimulus_placement", "system_suffix")),
    )


def apply_runner_cli_overrides(cfg: RunnerConfig, args: argparse.Namespace, config_yaml_path: str) -> RunnerConfig:
    updates: dict[str, Any] = {}
    if getattr(args, "stimulus", None) is not None:
        updates["stimulus"] = str(args.stimulus)
    if getattr(args, "stimulus_config", None) is not None:
        updates["stimulus_config"] = resolve_config_path(str(args.stimulus_config), config_yaml_path=config_yaml_path)
    if getattr(args, "stimulus_language", None) is not None:
        updates["stimulus_language"] = str(args.stimulus_language).strip().lower()
    if getattr(args, "output_path", None) is not None:
        updates["output_path"] = str(args.output_path)
    if not updates:
        return cfg
    return replace(cfg, **updates)


def resolve_model_yaml_path(config_yaml_path: str, model_ref: str) -> str:
    if os.path.isabs(model_ref):
        return model_ref

    cfg_abs = os.path.abspath(config_yaml_path)
    cfg_dir = os.path.dirname(cfg_abs)
    repo_root = os.path.abspath(os.path.join(cfg_dir, "..", ".."))
    candidates = [
        os.path.join(repo_root, model_ref),
        os.path.join(cfg_dir, model_ref),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return candidates[0]


def load_model_server_config(config_yaml_path: str, model_ref: str) -> dict[str, Any]:
    model_yaml_path = resolve_model_yaml_path(config_yaml_path, model_ref)
    with open(model_yaml_path, "r") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid model config at {model_yaml_path}: expected YAML object.")
    return loaded


def tqdm_wrap(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        return iterable
    return tqdm(iterable, **kwargs)


def _jsonl_path(path: str) -> str:
    if path.endswith(".jsonl"):
        return path
    if path.endswith(".json"):
        return path[:-5] + ".jsonl"
    return path + ".jsonl"


def append_result_jsonl(path: str, item: dict[str, Any]) -> str:
    out_path = _jsonl_path(path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return out_path


def load_existing_example_ids(path: str) -> set[str]:
    out_path = _jsonl_path(path)
    if not os.path.exists(out_path):
        return set()

    done: set[str] = set()
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                ex_id = row.get("example_id")
                if isinstance(ex_id, str):
                    done.add(ex_id)
    return done


