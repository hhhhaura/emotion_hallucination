# emotion_hallucination

Standalone project for emotion-stimulus hallucination experiments.

## Layout

- `emotion/` - dataset runners, configs, and utilities
- `emotion/configs/` - experiment YAMLs and stimulus catalog
- `mcmc_lm/` - local model client utilities (self-contained copy)
- `vllm_config/` - OpenAI-compatible vLLM endpoint config
- `outputs/` - experiment JSONL outputs

## Setup

From repo root:

```bash
cd /home/hhhhaura/emotion_hallucination
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or use your existing conda env:

```bash
conda activate mcmc
pip install -r requirements.txt
```

## vLLM server config

Experiments read model endpoint settings from `vllm_config/qwen3-1.7B-chat.yaml`.

Default endpoint:

- `url: http://localhost:8093/v1`

Make sure a compatible OpenAI-style vLLM server is running there before launching experiments.

## Run experiments

All runners support:

```bash
python <runner.py> --config <config.yaml>
```

Examples:

```bash
python emotion/math_zh.py --config emotion/configs/math_zh_anxiety.yaml
python emotion/squad_v2.py --config emotion/configs/squad_v2_anxiety.yaml
python emotion/truthfulqa_mc.py --config emotion/configs/truthfulqa_mc_anxiety.yaml
```

### Global-MMLU Medical (new)

English:

```bash
python emotion/mmlu_med_en.py --config emotion/configs/mmlu_med_en_anxiety.yaml
```

Chinese:

```bash
python emotion/mmlu_med_zh.py --config emotion/configs/mmlu_med_zh_anxiety.yaml
```

These runners load `CohereLabs/Global-MMLU`, filter rows where `subject_category == "Medical"`, and evaluate MC accuracy.

## Smoke tests

Quick 1-example checks:

```bash
python emotion/mmlu_med_en.py --config emotion/configs/mmlu_med_en_anxiety_smoke.yaml
python emotion/mmlu_med_zh.py --config emotion/configs/mmlu_med_zh_anxiety_smoke.yaml
```

Expected outputs:

- `outputs/smoke/mmlu_med_en_anxiety_smoke.jsonl`
- `outputs/smoke/mmlu_med_zh_anxiety_smoke.jsonl`

## Stimuli

Stimuli are defined in `emotion/configs/stimuli.yaml` and injected by `emotion/utils/stimuli.py`.
Current emotion IDs include `anxiety`, `distress`, `urgency`, and `overconfidence`, with per-dataset default languages.
# emotion_hallucination
