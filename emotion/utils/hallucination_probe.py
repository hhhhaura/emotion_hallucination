from __future__ import annotations

import re
from typing import Any


HALLUQA_MC_URL = "https://raw.githubusercontent.com/OpenMOSS/HalluQA/main/HalluQA_mc.json"
GLOBAL_MMLU_DATASET = "CohereLabs/Global-MMLU"

MC_SYSTEM_PROMPT_EN = (
    "Answer the multiple-choice question by outputting only one option letter: A, B, C, D, or E."
)

MC_SYSTEM_PROMPT_ZH = (
    "请回答这道多选题，仅输出一个选项字母：A、B、C、D 或 E。"
)


def load_halluqa_mc_examples(*, limit: int | None = None, url: str = HALLUQA_MC_URL) -> list[dict[str, Any]]:
    import requests

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Expected HalluQA_mc JSON payload to be a list.")
    if limit is not None:
        payload = payload[: int(limit)]

    examples: list[dict[str, Any]] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            continue
        question = _pick_first_str(row, ["question", "query", "instruction"])
        if question is None:
            continue
        option_texts = _extract_options(row)
        clean_question = question
        if not option_texts:
            clean_question, option_texts = _parse_embedded_mc_question(question)
        correct_index = _extract_correct_index(row, option_texts)
        if not option_texts or correct_index is None:
            continue

        ex_id = str(row.get("id", row.get("question_id", f"halluqa_mc_{idx}")))
        examples.append(
            {
                "example_id": ex_id,
                "question": clean_question,
                "reference": {
                    "correct_index": correct_index,
                    "correct_letter": _index_to_letter(correct_index),
                    "options": option_texts,
                },
                "messages": [
                    {"role": "system", "content": MC_SYSTEM_PROMPT_ZH},
                    {"role": "user", "content": build_mc_prompt_zh(clean_question, option_texts)},
                ],
                "meta": {"dataset": "halluqa_mc"},
            }
        )
    return examples


def load_truthfulqa_mc_examples(*, split: str = "validation", limit: int | None = None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets is required for TruthfulQA loading. Install with: pip install datasets") from exc

    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split=split)
    if limit is not None:
        ds = ds.select(range(min(int(limit), len(ds))))

    examples: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        question = _pick_first_str(row, ["question"])
        if question is None:
            continue

        choices, correct_index = _extract_truthfulqa_choices_and_label(row)
        if not choices or correct_index is None:
            continue

        ex_id = str(row.get("id", f"truthfulqa_mc_{idx}"))
        examples.append(
            {
                "example_id": ex_id,
                "question": question,
                "reference": {
                    "correct_index": correct_index,
                    "correct_letter": _index_to_letter(correct_index),
                    "options": choices,
                },
                "messages": [
                    {"role": "system", "content": MC_SYSTEM_PROMPT_EN},
                    {"role": "user", "content": build_mc_prompt_en(question, choices)},
                ],
                "meta": {"dataset": "truthfulqa_mc", "split": split},
            }
        )
    return examples


def load_global_mmlu_med_en_examples(*, split: str = "test", limit: int | None = None) -> list[dict[str, Any]]:
    return _load_global_mmlu_med_examples(language="en", split=split, limit=limit)


def load_global_mmlu_med_zh_examples(*, split: str = "test", limit: int | None = None) -> list[dict[str, Any]]:
    return _load_global_mmlu_med_examples(language="zh", split=split, limit=limit)


def build_mc_prompt_en(question: str, options: list[str]) -> str:
    lines = [f"{_index_to_letter(i)}. {opt}" for i, opt in enumerate(options)]
    return (
        f"Question: {question}\n\n"
        "Options:\n"
        + "\n".join(lines)
        + "\n\nOutput only the single best option letter."
    )


def build_mc_prompt_zh(question: str, options: list[str]) -> str:
    lines = [f"{_index_to_letter(i)}. {opt}" for i, opt in enumerate(options)]
    return (
        f"问题：{question}\n\n"
        "选项：\n"
        + "\n".join(lines)
        + "\n\n请仅输出最佳选项的单个字母。"
    )


def build_mc_prompt(question: str, options: list[str]) -> str:
    """Backward-compatible alias for English MC prompts."""
    return build_mc_prompt_en(question, options)


def grade_mc_samples(outputs: list[str], correct_letter: str) -> tuple[list[list[Any]], dict[str, float]]:
    rows: list[list[Any]] = []
    verdicts: list[bool] = []
    for out in outputs:
        pred = extract_option_letter(out)
        verdict = pred == correct_letter
        verdicts.append(verdict)
        rows.append([pred, verdict, out])
    accuracy = sum(verdicts) / max(len(verdicts), 1)
    return rows, {"accuracy": accuracy}


def extract_option_letter(text: str) -> str | None:
    text = text.strip().upper()
    if not text:
        return None
    # Handle plain-letter outputs first.
    if text in {"A", "B", "C", "D", "E"}:
        return text
    # Robust fallback for generated strings like "Answer: C" or "(B)".
    match = re.search(r"\b([A-E])\b", text)
    if match:
        return match.group(1)
    return None


def _parse_embedded_mc_question(text: str) -> tuple[str, list[str]]:
    stripped = text.strip()
    if stripped.lower().startswith("question:"):
        stripped = stripped.split(":", 1)[1].strip()

    # Matches patterns like "A: ... B: ... C: ..."
    matches = list(re.finditer(r"([A-E])\s*[:：]\s*", stripped))
    if not matches:
        return text, []

    question_part = stripped[: matches[0].start()].strip()
    options: list[str] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(stripped)
        option_text = stripped[start:end].strip()
        options.append(option_text)

    return question_part or text, options


def _extract_truthfulqa_choices_and_label(row: dict[str, Any]) -> tuple[list[str], int | None]:
    for key in ("mc1_targets", "mc2_targets"):
        target = row.get(key)
        if not isinstance(target, dict):
            continue
        choices = target.get("choices", [])
        labels = target.get("labels", [])
        if not isinstance(choices, list) or not isinstance(labels, list):
            continue
        clean_choices = [str(c).strip() for c in choices]
        correct_index = None
        for i, label in enumerate(labels):
            if bool(label):
                correct_index = i
                break
        if clean_choices and correct_index is not None:
            return clean_choices, correct_index
    return [], None


def _load_global_mmlu_med_examples(*, language: str, split: str, limit: int | None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets is required for Global-MMLU loading. Install with: pip install datasets") from exc

    ds = load_dataset(GLOBAL_MMLU_DATASET, language, split=split)

    examples: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        if not isinstance(row, dict):
            continue
        if str(row.get("subject_category", "")).strip().lower() != "medical":
            continue

        question = _pick_first_str(row, ["question"])
        if question is None:
            continue

        options = _extract_global_mmlu_options(row)
        if len(options) != 4:
            continue

        answer = str(row.get("answer", "")).strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            continue
        correct_index = ord(answer) - ord("A")

        sample_id = str(row.get("sample_id", f"global_mmlu_{language}_{split}_{idx}"))
        subject = str(row.get("subject", "")).strip()
        user_prompt = (
            build_mc_prompt_zh(question, options)
            if language == "zh"
            else build_mc_prompt_en(question, options)
        )
        system_prompt = MC_SYSTEM_PROMPT_ZH if language == "zh" else MC_SYSTEM_PROMPT_EN

        examples.append(
            {
                "example_id": sample_id,
                "question": question,
                "reference": {
                    "correct_index": correct_index,
                    "correct_letter": answer,
                    "options": options,
                },
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "meta": {
                    "dataset": f"mmlu_med_{language}",
                    "split": split,
                    "subject": subject,
                    "subject_category": "Medical",
                    "hf_dataset": GLOBAL_MMLU_DATASET,
                    "hf_language": language,
                },
            }
        )
        if limit is not None and len(examples) >= int(limit):
            break

    return examples


def _extract_global_mmlu_options(row: dict[str, Any]) -> list[str]:
    opts = []
    for key in ("option_a", "option_b", "option_c", "option_d"):
        value = row.get(key)
        if not isinstance(value, str):
            return []
        opts.append(value.strip())
    return opts


def _extract_options(row: dict[str, Any]) -> list[str]:
    options = row.get("options")
    if isinstance(options, list):
        return [str(o).strip() for o in options]
    if isinstance(options, dict):
        # Preserve letter order when possible.
        ordered: list[str] = []
        for letter in ("A", "B", "C", "D", "E"):
            if letter in options:
                ordered.append(str(options[letter]).strip())
        if ordered:
            return ordered

    # Fallback keys.
    for key in ("choices", "candidate_answers", "candidates"):
        value = row.get(key)
        if isinstance(value, list):
            return [str(v).strip() for v in value]
    return []


def _extract_correct_index(row: dict[str, Any], options: list[str]) -> int | None:
    for key in ("answer_idx", "correct_index", "label_index"):
        value = row.get(key)
        if isinstance(value, int) and 0 <= value < len(options):
            return value

    for key in ("answer", "correct_answer", "label", "gold"):
        value = row.get(key)
        if isinstance(value, int) and 0 <= value < len(options):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.upper() in {"A", "B", "C", "D", "E"}:
                return ord(stripped.upper()) - ord("A")
            letter_match = re.search(r"\b([A-E])\b", stripped.upper())
            if letter_match:
                return ord(letter_match.group(1)) - ord("A")
            for i, opt in enumerate(options):
                if stripped == opt:
                    return i

    return None


def _index_to_letter(index: int) -> str:
    return chr(ord("A") + index)


def _pick_first_str(row: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
