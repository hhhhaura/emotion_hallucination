from __future__ import annotations

MATH_COT = (
    "Please reason step by step, and put your final answer within \\boxed{{}} "
    "at the end, then end completely."
)

deepseek_math = (
    "You are a helpful, honest and harmless assistant.\n\n"
    f"user: {{input}} {MATH_COT}\n\n"
    "assistant:"
)


def format_qwen_boxed_messages(input: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{input}\n{MATH_COT}"},
    ]


CONTRASTIVE_ANALYSIS_PROMPT = """Q: If a train leaves Chicago at 9:00 AM traveling at 60 mph, and another train leaves New York (800 miles away) at 10:00 AM traveling at 80 mph, at what time will they meet?

Analysis:
Problem Type & Domain: This is a relative motion / meeting-point problem. Domain: elementary kinematics + algebra.
Mathematical Tools Required: linear equations, distance-rate-time formula (d = r·t), system of two equations in one unknown.
Key Quantities & Unknowns: Let t = hours after 9:00 AM. Train 1 distance: 60t. Train 2 distance: 80(t−1). Unknown: t where distances sum to 800.
Traps & Edge Cases: Train 2 has a 1-hour head start offset — forgetting this is the #1 mistake. Units are consistent (miles, hours) — no conversion needed.
Solution Strategy: Set up 60t + 80(t−1) = 800, solve for t, convert decimal hours to clock time.
Sanity Check Plan: Plug t back in, verify both distances sum to 800; check answer is after 10:00 AM.
[END OF ANALYSIS]

---

Q: {question}

Analysis:
Problem Type & Domain:"""

MATH500_SYSTEM_PROMPT = (
    "You are a math assistant. Solve the problem step-by-step and provide your final answer in LaTeX format, "
    "ensuring the final result is placed inside \\boxed{}."
)

REDACTED_THINK_CLOSE = "</think>"
CHAT_CLEAN_THINK_PLACEHOLDER = "<think>\n\n</think>"
CONTRASTIVE_VIRTUAL_THINK = (
    "Take a moment to think step by step before answering. Counting down: "
    "50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, "
    "29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, "
    "9, 8, 7, 6, 5, 4, 3, 2, 1..."
)

