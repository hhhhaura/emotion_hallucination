from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from emotion.utils.exp_logging import configure_experiment_logging, get_exp_logger
from emotion.utils.hallucination_probe import grade_mc_samples, load_halluqa_mc_examples
from emotion.utils.runtime import (
    append_result_jsonl,
    apply_runner_cli_overrides,
    build_argparser,
    load_existing_example_ids,
    load_model_server_config,
    load_runner_config,
    tqdm_wrap,
)
from emotion.utils.stimuli import prepare_examples
from mcmc_lm.core.lm_core import ChatServerVLLM


def main() -> None:
    default_cfg = os.path.join(os.path.dirname(__file__), "configs", "halluqa_mc_anxiety.yaml")
    args = build_argparser(default_cfg, "Run HalluQA multiple-choice hallucination probe.").parse_args()
    configure_experiment_logging()
    log = get_exp_logger("halluqa_mc")
    cfg = apply_runner_cli_overrides(load_runner_config(args.config), args, args.config)

    examples_all = load_halluqa_mc_examples(limit=cfg.limit_problems)
    examples_all = prepare_examples(examples_all, cfg, dataset_name="halluqa_mc")
    done_ids = load_existing_example_ids(cfg.output_path)
    examples = [ex for ex in examples_all if ex["example_id"] not in done_ids]

    model_config = load_model_server_config(args.config, cfg.vllm_config)
    lm = ChatServerVLLM(
        config=model_config,
        enable_thinking=False,
        temperature=cfg.temperature,
        max_tokens=cfg.max_new_tokens,
        top_p=cfg.top_p,
        seed=cfg.seed,
        extra_body={"top_k": cfg.top_k} if cfg.top_k is not None else {},
    )

    log.info(
        "[halluqa_mc] problems=%s skipped=%s N=%s max_new_tokens=%s stimulus=%s output=%s",
        len(examples),
        len(examples_all) - len(examples),
        cfg.N,
        cfg.max_new_tokens,
        cfg.stimulus,
        cfg.output_path,
    )

    for idx, ex in enumerate(tqdm_wrap(examples, total=len(examples), desc="halluqa_mc")):
        outputs = lm.sample([ex["messages"]] * cfg.N, max_new_tokens=cfg.max_new_tokens)
        samples, metrics = grade_mc_samples(outputs, ex["reference"]["correct_letter"])
        wrote_path = append_result_jsonl(
            cfg.output_path,
            {
                "example_id": ex["example_id"],
                "question": ex["question"],
                "reference": ex["reference"],
                "samples": samples,
                "metrics": metrics,
                "meta": ex["meta"],
            },
        )
        log.info(
            "[halluqa_mc] %s/%s acc=%.3f wrote=%s",
            idx + 1,
            len(examples),
            metrics["accuracy"],
            wrote_path,
        )


if __name__ == "__main__":
    main()
