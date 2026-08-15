# SIEVE Core Release

This is a minimal reproduction package for SIEVE-style schema-informed
reasoning. It contains the core training and inference code only; large
benchmark suites, private experiment launchers, checkpoints, caches, and raw
datasets are intentionally excluded.

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

## Data Format

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

## Run Benchmark

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

## Notes for Paper Reproduction

- Raw datasets and trained checkpoints are not bundled in this repository.
- Use the paper's dataset preparation instructions and pass paths through
  `TRAIN_DATA_PATH`, `TEST_DATA_PATH`, and `DATASET_PATH`.
- Use environment variables for API credentials. Do not commit secrets.
- This release focuses on SIEVE core training and inference; additional
  baselines and analysis scripts from the internal research workspace are not
  included.
