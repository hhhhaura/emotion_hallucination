from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Any, Iterable

from mcmc_lm.utils.constants import (
    CHAT_CLEAN_THINK_PLACEHOLDER,
    MATH500_SYSTEM_PROMPT,
    REDACTED_THINK_CLOSE,
)
from mcmc_lm.utils.exp_logging import get_exp_logger

_log = get_exp_logger()


def normalize_redacted_thinking_output(raw: str) -> str:
    if not isinstance(raw, str):
        raw = str(raw)
    last_occurrence = raw.rfind(REDACTED_THINK_CLOSE)
    if last_occurrence != -1:
        return raw[: last_occurrence + len(REDACTED_THINK_CLOSE)]
    return raw + "\n" + REDACTED_THINK_CLOSE


def build_chat_think_prefix_prompts(
    *,
    think_lm: Any,
    propose_lm: Any,
    pconversations: list[list[dict[str, str]]],
    max_thinking_tokens: int = 1024,
) -> tuple[list[str], list[str], list[str]]:
    if not pconversations:
        return [], [], []
    think_outs = think_lm.sample(pconversations, max_new_tokens=max_thinking_tokens)
    conversation_texts = propose_lm.apply_chat_template(pconversations)
    think_prompts: list[str] = []
    clean_prompts: list[str] = []
    thinking_norms: list[str] = []
    for think_out, conversation_text in zip(think_outs, conversation_texts):
        thinking_norm = normalize_redacted_thinking_output(think_out)
        think_prompts.append(conversation_text + thinking_norm)
        clean_prompts.append(conversation_text + CHAT_CLEAN_THINK_PLACEHOLDER)
        thinking_norms.append(thinking_norm)
    return think_prompts, clean_prompts, thinking_norms


def tqdm_wrap(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        return iterable
    return tqdm(iterable, **kwargs)


def _atomic_json_write(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def load_existing_results(path: str) -> list[dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r") as f:
        loaded = json.load(f)
    if not isinstance(loaded, list):
        raise ValueError(f"Expected existing results at {path} to be a JSON list.")
    return loaded


def append_and_write_result(path: str, results: list[dict[str, Any]], item: dict[str, Any]) -> None:
    results.append(item)
    _atomic_json_write(path, results)


def _jsonl_path(path: str) -> str:
    if path.endswith(".jsonl"):
        return path
    if path.endswith(".json"):
        return path[:-5] + ".jsonl"
    return path + ".jsonl"


def append_result_jsonl(path: str, item: dict[str, Any]) -> str:
    out = _jsonl_path(path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return out


def load_existing_question_keys_jsonl(path: str) -> set[str]:
    out = _jsonl_path(path)
    if not out or not os.path.exists(out):
        return set()

    keys: set[str] = set()
    with open(out, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            q = row.get("question")
            if isinstance(q, str):
                keys.add(q)
    return keys


def load_math500(dataset_path: str, limit: int | None = None) -> list[dict[str, Any]]:
    with open(dataset_path, "r") as f:
        data_list = json.load(f)
    if limit is not None:
        data_list = data_list[: int(limit)]
    if not isinstance(data_list, list):
        raise ValueError(f"Expected {dataset_path} to contain a JSON list.")
    return data_list


def build_think_clean_prompts(
    *,
    lm_prompt,
    conversations: list[list[dict[str, str]]],
    max_thinking_tokens: int,
) -> tuple[list[str], list[str]]:
    outputs = lm_prompt.sample(conversations, max_new_tokens=max_thinking_tokens)

    prompts_base = lm_prompt.apply_chat_template(conversations, continue_final_message=False)
    prompt_think: list[str] = []
    prompt_clean: list[str] = []

    parsed = lm_prompt.parse_output_string(outputs)
    for i in range(len(conversations)):
        base = prompts_base[i]
        if isinstance(parsed[i], dict) and "reasoning_content" in parsed[i] and parsed[i]["reasoning_content"] is not None:
            think_str = outputs[i]
            if not isinstance(think_str, str):
                think_str = str(think_str)
            if "</think>" not in think_str:
                think_str = think_str + "\n</think>\n"
            prompt_think.append(think_str)
        else:
            out = outputs[i] if isinstance(outputs[i], str) else str(outputs[i])
            if "</think>" not in out:
                out = out + "\n</think>\n"
            prompt_think.append(base + out)

        prompt_clean.append(base + "<think>\n</think>\n")

    return prompt_think, prompt_clean


def iter_conversations_from_dataset(ds: list[dict[str, Any]]) -> Iterable[tuple[str, str, list[dict[str, str]]]]:
    for ex in ds:
        prompt = ex["prompt"]
        answer = ex["answer"]
        yield prompt, answer, [{"role": "system", "content": MATH500_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]


def chat_parse_clean_output(outputs: list[str]) -> list[str]:
    cleaned_outputs = []

    for text in outputs:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        text = re.sub(r"<\|.*?\|>", "", text)
        cleaned_outputs.append(text.strip())

    return cleaned_outputs

