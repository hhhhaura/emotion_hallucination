from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Any

import yaml


@dataclass(frozen=True)
class RunConfig:
    dataset_path: str
    output_path: str
    vllm_config: str
    N: int
    max_new_tokens: int
    top_k: int | None
    temperature: float | None
    seed: int | None
    limit_problems: int | None
    enable_thinking: bool
    max_thinking_tokens: int
    max_continuation_tokens: int
    max_token_length: int
    tokens_per_step: int
    resample_interval_tokens: int
    beta: float
    no_resample: bool
    smc_log_every: int
    generate_unique_eliminated: bool


@dataclass(frozen=True)
class MCMCConfig:
    dataset_path: str = "data/MATH500.json"
    output_path: str = "outputs/mcmc_run.jsonl"
    vllm_config: str = "vllm/qwen3-1.7B-base.yaml"
    device: str | None = None
    temperature: float = 0.7
    alpha: float = 1.0
    batch_size: int = 1
    max_continuation_tokens: int = 2048
    refine_samples: int = 50
    num_mcmc_runs: int = 1
    seed: int | None = 42
    limit_problems: int | None = None
    problem_start_idx: int | None = None
    problem_end_idx: int | None = None
    verbose: bool = False
    early_stop: bool = False
    max_thinking_tokens: int = 1024


def resolve_vllm_config_path(experiment_yaml_path: str, ref: object) -> str:
    s = str(ref)
    if os.path.isabs(s):
        return s
    cfg_abs = os.path.abspath(experiment_yaml_path)
    cfg_dir = os.path.dirname(cfg_abs)
    exp_dir = os.path.dirname(cfg_dir)
    basename = os.path.basename(s)
    candidates = [
        os.path.join(cfg_dir, s),
        os.path.join(cfg_dir, "vllm", basename),
        os.path.join(exp_dir, s),
        os.path.join(exp_dir, "configs", "vllm", basename),
    ]
    seen: set[str] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if os.path.isfile(c):
            return c
    return candidates[0]


resolve_model_yaml_path = resolve_vllm_config_path


def load_vllm_client_config(experiment_yaml_path: str, ref: object) -> dict[str, Any]:
    path = resolve_vllm_config_path(experiment_yaml_path, ref)
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config(yaml_path: str, overrides: dict[str, Any] | None = None) -> RunConfig:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if overrides:
        o = {k: v for k, v in overrides.items() if k != "no_seed" and v is not None}
        cfg.update(o)
        if overrides.get("no_seed"):
            cfg["seed"] = None

    _rit = cfg.get("resample_interval_tokens", None)
    if _rit is None:
        _rit = cfg.get("resample_every", 30)

    _seed = cfg.get("seed", None)
    if _seed is not None:
        _seed = int(_seed)

    _mthink = cfg.get("max_thinking_tokens")
    if _mthink is None:
        _mthink = cfg.get("think_max_new_tokens")
    if _mthink is None:
        _mthink = 1024

    _mcont = cfg.get("max_continuation_tokens")
    if _mcont is None:
        _mcont = cfg.get("block_size")
    if _mcont is None:
        _mcont = cfg.get("continuation_max_tokens")
    if _mcont is None:
        _mcont = 2048

    vllm_ref = cfg.get("vllm_config", cfg.get("model_yaml", "vllm/qwen3-1.7B-chat.yaml"))

    return RunConfig(
        dataset_path=cfg.get("dataset_path", "data/MATH500.json"),
        output_path=cfg.get("output_path", "outputs/math500_run.json"),
        vllm_config=resolve_vllm_config_path(yaml_path, vllm_ref),
        N=int(cfg.get("N", 100)),
        max_new_tokens=int(cfg.get("max_new_tokens", 2048)),
        top_k=cfg.get("top_k", 20),
        temperature=cfg.get("temperature", None),
        seed=_seed,
        limit_problems=cfg.get("limit_problems", None),
        enable_thinking=bool(cfg.get("enable_thinking", True)),
        max_thinking_tokens=int(_mthink),
        max_continuation_tokens=int(_mcont),
        max_token_length=int(cfg.get("max_token_length", 512)),
        tokens_per_step=int(cfg.get("tokens_per_step", 1)),
        resample_interval_tokens=int(_rit),
        beta=float(cfg.get("beta", 1.0)),
        no_resample=bool(cfg.get("no_resample", False)),
        smc_log_every=int(cfg.get("smc_log_every", 30)),
        generate_unique_eliminated=bool(cfg.get("generate_unique_eliminated", False)),
    )


def build_argparser(default_config_path: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Math500 experiment runner (settings from YAML).")
    p.add_argument(
        "--config",
        type=str,
        default=default_config_path,
        help="Path to experiment YAML (default: %(default)s).",
    )
    return p


def smc_resample_every_steps(*, resample_interval_tokens: int, tokens_per_step: int) -> int:
    k = max(1, int(tokens_per_step))
    r = max(1, int(resample_interval_tokens))
    return max(1, (r + k - 1) // k)


def load_mcmc_config(yaml_path: str) -> MCMCConfig:
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    device = raw.get("device", None)
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    vllm_ref = raw.get("vllm_config", raw.get("model_yaml", MCMCConfig.vllm_config))

    return MCMCConfig(
        dataset_path=str(raw.get("dataset_path", MCMCConfig.dataset_path)),
        output_path=str(raw.get("output_path", MCMCConfig.output_path)),
        vllm_config=resolve_vllm_config_path(yaml_path, vllm_ref),
        device=device,
        temperature=float(raw.get("temperature", MCMCConfig.temperature)),
        alpha=float(raw.get("alpha", MCMCConfig.alpha)),
        batch_size=max(1, int(raw.get("batch_size", MCMCConfig.batch_size))),
        max_continuation_tokens=max(
            1,
            int(raw.get("max_continuation_tokens", raw.get("block_size", MCMCConfig.max_continuation_tokens))),
        ),
        refine_samples=max(0, int(raw.get("refine_samples", MCMCConfig.refine_samples))),
        num_mcmc_runs=max(1, int(raw.get("num_mcmc_runs", MCMCConfig.num_mcmc_runs))),
        seed=None if raw.get("seed", MCMCConfig.seed) is None else int(raw.get("seed", MCMCConfig.seed)),
        limit_problems=raw.get("limit_problems", None),
        problem_start_idx=None if raw.get("problem_start_idx") is None else int(raw.get("problem_start_idx")),
        problem_end_idx=None if raw.get("problem_end_idx") is None else int(raw.get("problem_end_idx")),
        verbose=bool(raw.get("verbose", MCMCConfig.verbose)),
        early_stop=bool(raw.get("early_stop", MCMCConfig.early_stop)),
        max_thinking_tokens=int(raw.get("max_thinking_tokens", MCMCConfig.max_thinking_tokens)),
    )


def resolve_seed(config_seed: int | None) -> int | None:
    raw = os.getenv("MCMC_SEED")
    if raw is None or raw.strip() == "":
        return config_seed
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"MCMC_SEED must be an integer, got {raw!r}") from exc


def seeded_output_path(output_path: str, seed: int | None) -> str:
    if seed is None:
        return output_path
    root, ext = os.path.splitext(output_path)
    if re.search(r"_\d+$", root):
        root = re.sub(r"_\d+$", "", root)
    return f"{root}_{seed}{ext}"


def build_score_argparser(default_config_path: str, description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--config",
        type=str,
        default=default_config_path,
        help="Experiment YAML providing vllm_config (default: %(default)s).",
    )
    return p


def load_score_vllm_config(experiment_yaml_path: str) -> dict[str, Any]:
    with open(experiment_yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    ref = raw.get("vllm_config", raw.get("model_yaml", "vllm/qwen3-1.7B-base.yaml"))
    return load_vllm_client_config(experiment_yaml_path, ref)


def resolve_vllm_config_from_experiment_yaml(experiment_yaml_path: str) -> str:
    with open(experiment_yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    ref = raw.get("vllm_config", raw.get("model_yaml", "vllm/qwen3-1.7B-base.yaml"))
    return resolve_vllm_config_path(experiment_yaml_path, ref)

