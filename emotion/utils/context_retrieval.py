from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


SQUAD_SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "If the answer cannot be found in the context, return an empty string."
)

CMRC_SYSTEM_PROMPT = (
    "请仅依据给定上下文回答问题。如果上下文中无法回答，请返回空字符串。"
)


def load_squad_examples(*, split: str = "validation", limit: int | None = None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets is required for SQuAD loading. Install with: pip install datasets") from exc

    # Prefer the requested namespace first; fall back to canonical alias if HF metadata is incompatible.
    try:
        ds = load_dataset("rajpurkar/squad_v2", split=split)
    except Exception:
        ds = load_dataset("squad_v2", split=split)
    if limit is not None:
        ds = ds.select(range(min(int(limit), len(ds))))

    examples: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        ex_id = str(row.get("id", f"squad_{idx}"))
        title = str(row.get("title", ""))
        context = str(row.get("context", ""))
        question = str(row.get("question", ""))
        answers = row.get("answers", {"text": [], "answer_start": []})
        prompt = (
            f"Title: {title}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer with a short span copied from the context, or empty string if unanswerable."
        )
        examples.append(
            {
                "example_id": ex_id,
                "question": question,
                "reference": {"id": ex_id, "answers": answers},
                "messages": [
                    {"role": "system", "content": SQUAD_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "meta": {"dataset": "squad_v2", "title": title},
            }
        )
    return examples


def parse_cmrc_answer_texts(answers_raw: Any) -> list[str]:
    """Extract gold answer strings from CMRC / SQuAD-style answer fields."""
    if isinstance(answers_raw, dict):
        texts = answers_raw.get("text", [])
        if isinstance(texts, list):
            return _dedupe_strings(str(t).strip() for t in texts if isinstance(t, str) and str(t).strip())
        return []

    if isinstance(answers_raw, list):
        # Buggy runs stored dict keys when iterating answers={"text": ..., "answer_start": ...}.
        if answers_raw == ["text", "answer_start"]:
            return []
        return _dedupe_strings(str(a).strip() for a in answers_raw if isinstance(a, str) and str(a).strip())

    return []


def _dedupe_strings(values: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def load_cmrc_gold_by_id(*, split: str = "validation") -> dict[str, list[str]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets is required for CMRC loading. Install with: pip install datasets") from exc

    ds = load_dataset("hfl/cmrc2018", split=split)
    return {str(row["id"]): parse_cmrc_answer_texts(row.get("answers")) for row in ds}


def load_cmrc_examples(*, split: str = "validation", limit: int | None = None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets is required for CMRC loading. Install with: pip install datasets") from exc

    ds = load_dataset("hfl/cmrc2018", split=split)
    if limit is not None:
        ds = ds.select(range(min(int(limit), len(ds))))

    examples: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        ex_id = str(row.get("id", f"cmrc_{idx}"))
        context = str(row.get("context", ""))
        question = str(row.get("question", ""))
        answers = parse_cmrc_answer_texts(row.get("answers", []))
        prompt = (
            f"上下文：\n{context}\n\n"
            f"问题：{question}\n\n"
            "请输出最简短答案（尽量直接摘录上下文）；若无法回答，输出空字符串。"
        )
        examples.append(
            {
                "example_id": ex_id,
                "question": question,
                "reference": {"answers": answers},
                "messages": [
                    {"role": "system", "content": CMRC_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "meta": {"dataset": "cmrc2018"},
            }
        )
    return examples


def grade_squad_samples(outputs: list[str], reference: dict[str, Any]) -> tuple[list[list[Any]], dict[str, float]]:
    metric = _load_squad_metric()
    ref = {"id": reference["id"], "answers": reference["answers"]}
    rows: list[list[Any]] = []
    em_scores: list[float] = []
    f1_scores: list[float] = []

    for out in outputs:
        pred_text = normalize_qa_prediction(out)
        score = metric.compute(
            predictions=[
                {
                    "id": reference["id"],
                    "prediction_text": pred_text,
                    "no_answer_probability": 0.0,
                }
            ],
            references=[ref],
        )
        em = float(score.get("exact", 0.0))
        f1 = float(score.get("f1", 0.0))
        em_scores.append(em)
        f1_scores.append(f1)
        rows.append([pred_text, em, f1, out])

    return rows, {
        "em": sum(em_scores) / max(len(em_scores), 1),
        "f1": sum(f1_scores) / max(len(f1_scores), 1),
    }


def grade_cmrc_samples(outputs: list[str], answers: list[str]) -> tuple[list[list[Any]], dict[str, float]]:
    rows: list[list[Any]] = []
    em_scores: list[float] = []
    f1_scores: list[float] = []
    for out in outputs:
        pred = normalize_qa_prediction(out)
        em, f1 = cmrc_char_em_f1(pred, answers)
        em_scores.append(em)
        f1_scores.append(f1)
        rows.append([pred, em, f1, out])
    return rows, {
        "em": sum(em_scores) / max(len(em_scores), 1),
        "f1": sum(f1_scores) / max(len(f1_scores), 1),
    }


def normalize_qa_prediction(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"^\s*answer\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def cmrc_char_em_f1(prediction: str, answers: list[str]) -> tuple[float, float]:
    if not answers:
        # Unanswerable handling: full credit only for abstention.
        abstain = 1.0 if prediction == "" else 0.0
        return abstain, abstain

    best_em = 0.0
    best_f1 = 0.0
    for ans in answers:
        em = 1.0 if _normalize_cmrc_text(prediction) == _normalize_cmrc_text(ans) else 0.0
        f1 = _cmrc_char_f1(prediction, ans)
        best_em = max(best_em, em)
        best_f1 = max(best_f1, f1)
    return best_em * 100.0, best_f1 * 100.0


def _normalize_cmrc_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "", text)
    punctuation = string.punctuation + "，。！？；：、“”‘’（）【】《》—…·"
    return "".join(ch for ch in text if ch not in punctuation)


def _cmrc_char_f1(prediction: str, ground_truth: str) -> float:
    pred_chars = list(_normalize_cmrc_text(prediction))
    gold_chars = list(_normalize_cmrc_text(ground_truth))
    if not pred_chars and not gold_chars:
        return 1.0
    if not pred_chars or not gold_chars:
        return 0.0
    common = Counter(pred_chars) & Counter(gold_chars)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_chars)
    recall = num_same / len(gold_chars)
    return (2 * precision * recall) / (precision + recall)


def _load_squad_metric():
    try:
        import evaluate
    except Exception as exc:
        raise RuntimeError("evaluate is required for SQuAD grading. Install with: pip install evaluate") from exc
    return evaluate.load("squad_v2")
