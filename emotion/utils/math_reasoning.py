from __future__ import annotations

from typing import Any


MATH_SYSTEM_PROMPT_EN = (
    "You are a math assistant. Solve the problem step-by-step and provide your final answer in LaTeX format, "
    "ensuring the final result is placed inside \\boxed{}."
)

MATH_SYSTEM_PROMPT_ZH = (
    "你是一名数学助手。请逐步解题，并将最终答案以 LaTeX 格式给出，"
    "确保最终结果放在 \\boxed{} 中。"
)


def math_system_prompt(language: str) -> str:
    lang = language.strip().lower()
    if lang in {"chinese", "zh", "zh-cn", "zh_cn"}:
        return MATH_SYSTEM_PROMPT_ZH
    return MATH_SYSTEM_PROMPT_EN


def load_multilingual_math_examples(
    *,
    language: str,
    split: str = "test",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "datasets is required for multilingual math loading. Install with: pip install datasets"
        ) from exc

    ds = load_dataset("appier-ai-research/Multilingual-MATH-500", language, split=split)
    if limit is not None:
        ds = ds.select(range(min(int(limit), len(ds))))

    examples: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        question = _pick_first_str(row, ["problem", "question", "prompt", "instruction"])
        answer = _pick_first_str(row, ["answer", "final_answer", "target"])
        if question is None or answer is None:
            continue
        examples.append(
            {
                "example_id": str(row.get("id", f"{language.lower()}_{idx}")),
                "question": question,
                "reference": answer,
                "messages": [
                    {"role": "system", "content": math_system_prompt(language)},
                    {"role": "user", "content": question},
                ],
                "meta": {"language": language, "split": split},
            }
        )
    return examples


def extract_math_answer(output_str: str) -> str | None:
    from mcmc_lm.utils.math_grader import normalize_answer

    parsed = _parse_last_boxed(output_str)
    if parsed is None:
        return None
    parsed = parsed.rstrip(".,;:!?")
    return normalize_answer(parsed)


def grade_math_samples(outputs: list[str], reference_answer: str) -> tuple[list[list[Any]], dict[str, float]]:
    from mcmc_lm.utils.math_grader import grade_answer

    rows: list[list[Any]] = []
    verdicts: list[bool] = []
    for out in outputs:
        extracted = extract_math_answer(out)
        verdict = bool(grade_answer(extracted, reference_answer))
        verdicts.append(verdict)
        rows.append([extracted, verdict, out])

    accuracy = sum(verdicts) / max(len(verdicts), 1)
    return rows, {"accuracy": accuracy}


def _pick_first_str(row: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _parse_last_boxed(text: str) -> str | None:
    key = "\\boxed"
    idx = text.rfind(key)
    if idx == -1:
        return None
    rest = text[idx + len(key):].strip()
    if not rest.startswith("{"):
        return None

    depth = 0
    content_chars: list[str] = []
    for i, ch in enumerate(rest):
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(content_chars).strip()
        if depth >= 1:
            content_chars.append(ch)

    return None
