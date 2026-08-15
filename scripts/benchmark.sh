#!/usr/bin/env bash
set -euo pipefail

script_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${script_repo_root}"
export PYTHONPATH="${script_repo_root}:${script_repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

method="${METHOD:-base}"
target_model="${TARGET_MODEL:-gemma3_12b}"
served_model_name="${SERVED_MODEL_NAME:-${target_model}}"
dataset_path="${DATASET_PATH:-${script_repo_root}/examples/benchmark.json}"
output_dir="${OUTPUT_DIR:-${script_repo_root}/outputs/benchmark/${method}}"
gate_checkpoint="${GATE_CHECKPOINT:-}"
cache_dir="${CACHE_DIR:-${script_repo_root}/cache/sieve}"
gate_model="${GATE_MODEL:-Qwen/Qwen3-8B-Base}"
vllm_base_url="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
max_gen_len="${MAX_GEN_LEN:-1024}"
temperature="${TEMPERATURE:-0.0}"
run_gen="${RUN_GEN:-True}"
run_eval="${RUN_EVAL:-True}"
load_gen="${LOAD_GEN:-False}"
load_eval="${LOAD_EVAL:-False}"

python -m src.runs.benchmark \
  --method "${method}" \
  --target_model "${target_model}" \
  --served_model_name "${served_model_name}" \
  --dataset_path "${dataset_path}" \
  --output_dir "${output_dir}" \
  --gate_checkpoint "${gate_checkpoint}" \
  --cache_dir "${cache_dir}" \
  --gate_model "${gate_model}" \
  --vllm_base_url "${vllm_base_url}" \
  --max_gen_len "${max_gen_len}" \
  --temperature "${temperature}" \
  --run_gen "${run_gen}" \
  --run_eval "${run_eval}" \
  --load_gen "${load_gen}" \
  --load_eval "${load_eval}"
