from __future__ import annotations

import copy
import os
import sys
from typing import Any

import yaml

if __name__ == "__main__" and __package__ is None:
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from emotion.utils.runtime import RunnerConfig, resolve_config_path


def default_stimuli_config_path() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "configs", "stimuli.yaml")


def load_stimuli_catalog(path: str) -> dict[str, Any]:
    resolved = resolve_config_path(path)
    with open(resolved, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid stimuli catalog at {resolved}: expected YAML object.")
    return loaded


def resolve_stimulus_language(
    *,
    dataset_name: str,
    stimulus_language: str | None,
    catalog: dict[str, Any],
) -> str:
    if stimulus_language:
        return stimulus_language.strip().lower()
    defaults = catalog.get("dataset_defaults", {})
    if isinstance(defaults, dict):
        entry = defaults.get(dataset_name, {})
        if isinstance(entry, dict):
            lang = entry.get("language")
            if isinstance(lang, str) and lang.strip():
                return lang.strip().lower()
    return "en"


def resolve_stimulus_text(stimulus_id: str, language: str, catalog: dict[str, Any]) -> str:
    sid = (stimulus_id or "none").strip().lower()
    if sid == "none":
        return ""

    stimuli = catalog.get("stimuli", {})
    if not isinstance(stimuli, dict):
        raise ValueError("stimuli catalog missing 'stimuli' mapping.")

    entry = stimuli.get(sid)
    if not isinstance(entry, dict):
        known = ", ".join(sorted(k for k in stimuli if isinstance(k, str)))
        raise ValueError(f"Unknown stimulus '{stimulus_id}'. Known: {known or '(none)'}")

    lang = language.strip().lower()
    text = entry.get(lang)
    if not isinstance(text, str) or not text.strip():
        text = entry.get("en", "")
    return str(text).strip()


def apply_stimulus(
    messages: list[dict[str, Any]],
    *,
    stimulus_text: str,
    placement: str = "system_suffix",
) -> list[dict[str, Any]]:
    if not stimulus_text:
        return copy.deepcopy(messages)

    out = copy.deepcopy(messages)
    #if placement != "system_suffix":
    #    raise ValueError(f"Unsupported stimulus placement: {placement}")

    suffix = f"\n\n{stimulus_text}"
    prefix = f"{stimulus_text}\n\n"
    think = f"<think>\n{stimulus_text.replace('你', '我').replace('You', 'I').replace('you', 'I')}\n</think>"
    is_done = False
    for msg in out:
        if msg.get("role") == "system" and placement == "system_suffix":
            msg["content"] = str(msg.get("content", "")).rstrip() + suffix
            is_done = True
            break
        if msg.get("role") == "user" and placement == "user_prefix":
            msg["content"] = prefix + str(msg.get("content", "")).rstrip()
            is_done = True
            break
    if not is_done:
        if placement == "system_suffix":
            out.insert(0, {"role": "system", "content": stimulus_text.strip()})
            is_done = True
        if placement == "assistant_think":
            out.append({"role": "assistant", "content": think})
            is_done = True
    assert is_done, f"Failed to apply stimulus with placement '{placement}'"
    return out


def prepare_examples(
    examples: list[dict[str, Any]],
    cfg: RunnerConfig,
    *,
    dataset_name: str,
) -> list[dict[str, Any]]:
    catalog_path = cfg.stimulus_config or default_stimuli_config_path()
    catalog = load_stimuli_catalog(catalog_path)

    stimulus_id = (cfg.stimulus or "none").strip().lower()
    language = resolve_stimulus_language(
        dataset_name=dataset_name,
        stimulus_language=cfg.stimulus_language,
        catalog=catalog,
    )
    stimulus_text = resolve_stimulus_text(stimulus_id, language, catalog)
    placement = cfg.stimulus_placement or "system_suffix"

    prepared: list[dict[str, Any]] = []
    for ex in examples:
        row = copy.deepcopy(ex)
        meta = dict(row.get("meta") or {})
        meta["stimulus"] = stimulus_id
        meta["stimulus_language"] = language
        meta["stimulus_placement"] = placement
        row["meta"] = meta
        row["messages"] = apply_stimulus(
            row["messages"],
            stimulus_text=stimulus_text,
            placement=placement,
        )
        prepared.append(row)

    return prepared
