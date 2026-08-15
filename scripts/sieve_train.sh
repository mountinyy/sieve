#!/usr/bin/env bash
set -euo pipefail

script_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${script_repo_root}"
export PYTHONPATH="${script_repo_root}:${script_repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

root="${ROOT:-${script_repo_root}}"
model_name="${MODEL_NAME:-gemma3_12b}"
gate_model="${GATE_MODEL:-Qwen/Qwen3-8B-Base}"
train_data_path="${TRAIN_DATA_PATH:-${root}/examples/train.json}"
test_data_path="${TEST_DATA_PATH:-${root}/examples/dev.json}"
save_dir="${SAVE_DIR:-${root}/outputs/sieve}"
cache_dir="${CACHE_DIR:-${root}/cache/sieve}"
vllm_port="${VLLM_PORT:-8000}"
max_model_len="${MAX_MODEL_LEN:-4096}"
max_gen_len="${MAX_GEN_LEN:-1024}"
batch_size="${BATCH_SIZE:-1}"
sample_k="${SAMPLE_K:-4}"
learning_rate="${LEARNING_RATE:-5e-4}"
num_epochs="${NUM_EPOCHS:-1}"
eval_step="${EVAL_STEP:-0}"
safety="${SAFETY:-False}"

python -m src.runs.sieve_train \
  root="${root}" \
  model_name="${model_name}" \
  gate_model="${gate_model}" \
  train_data_path="${train_data_path}" \
  test_data_path="${test_data_path}" \
  save_dir="${save_dir}" \
  cache_dir="${cache_dir}" \
  vllm_port="${vllm_port}" \
  max_model_len="${max_model_len}" \
  max_gen_len="${max_gen_len}" \
  batch_size="${batch_size}" \
  sample_k="${sample_k}" \
  learning_rate="${learning_rate}" \
  num_epochs="${num_epochs}" \
  eval_step="${eval_step}" \
  safety="${safety}"
