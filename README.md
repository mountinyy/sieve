# SIEVE

Official implementation of **SIEVE**, a schema-informed reasoning framework for
training and evaluating adaptive moral-perspective selection in language model
responses.

SIEVE learns a gate over moral schemas and uses the selected schema mixture to
assemble structured reasoning prompts. This repository provides the training
pipeline, inference utilities, and benchmark runner used to train a SIEVE gate
and compare SIEVE-guided generation against base model generation.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For local model inference, start a vLLM OpenAI-compatible server separately.
For API models, set provider credentials in the environment.

```bash
cp .env.example .env
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
```

## Data

Training data is a JSON list. Each row should contain:

```json
{
  "id": "sample-id",
  "context": "scenario or user request",
  "question": "question to answer",
  "label": "WRONG or NOT_WRONG, or REFUSE/COMPLY for safety training",
  "schema_label": "PI, MN, or PC"
}
```

Benchmark data is a JSON list with either `input`, or `context` plus
`question`. If `label` is present, `scripts/benchmark.sh` reports exact-match
accuracy on the parsed `Answer:` field.

## Train SIEVE

Start a vLLM server for the target model, then run:

```bash
MODEL_NAME=gemma3_12b \
GATE_MODEL=Qwen/Qwen3-8B-Base \
TRAIN_DATA_PATH=examples/train.json \
TEST_DATA_PATH=examples/dev.json \
SAVE_DIR=outputs/sieve \
CACHE_DIR=cache/sieve \
VLLM_PORT=8000 \
NUM_EPOCHS=1 \
scripts/sieve_train.sh
```

Important outputs:

- `outputs/sieve/gate_last.pt`
- `outputs/sieve/trainer_last.pt`
- `outputs/sieve/config.json`
- `cache/sieve/` extraction cache

## Evaluate

Base model:

```bash
METHOD=base \
TARGET_MODEL=gemma3_12b \
SERVED_MODEL_NAME=gemma3_12b \
DATASET_PATH=examples/benchmark.json \
OUTPUT_DIR=outputs/benchmark/base \
scripts/benchmark.sh
```

SIEVE:

```bash
METHOD=sieve \
TARGET_MODEL=gemma3_12b \
SERVED_MODEL_NAME=gemma3_12b \
DATASET_PATH=examples/benchmark.json \
GATE_CHECKPOINT=outputs/sieve \
OUTPUT_DIR=outputs/benchmark/sieve \
CACHE_DIR=cache/sieve \
scripts/benchmark.sh
```

The benchmark writes:

- `responses.json`
- `summary.json`

## Reproducing Paper Experiments

- Raw datasets and trained checkpoints are distributed separately from the
  source code.
- Follow the dataset preparation instructions from the paper and pass paths via
  `TRAIN_DATA_PATH`, `TEST_DATA_PATH`, and `DATASET_PATH`.
- Use environment variables for API credentials; never commit secrets or local
  cache files.
- Checkpoints can be loaded by setting `GATE_CHECKPOINT` to the checkpoint
  directory containing `gate_last.pt`, `trainer_last.pt`, and `config.json`.
