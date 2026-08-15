from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from src.inference import API_MODELS, EquilibriumManager
from src.light_check import is_api_model
from src.sieve.sieve import load_sieve_for_inference, run_sieve_inference


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_json_list(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [dict(item) for item in payload]


def save_json(path: str | os.PathLike[str], payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_input_text(record: dict[str, Any]) -> str:
    if "input" in record:
        return str(record["input"])
    context = str(record.get("context", "")).strip()
    question = str(record.get("question", "")).strip()
    if context and question:
        return f"{context}\n\nQuestion:\n{question}"
    return context or question or str(record.get("prompt", ""))


def get_gold(record: dict[str, Any]) -> str | None:
    for key in ("label", "answer", "gold", "target"):
        if key in record and record[key] is not None:
            return str(record[key])
    return None


def parse_final_answer(text: str) -> str:
    matches = re.findall(r"answer\s*:\s*(.+)", text or "", flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return (text or "").strip()


class VLLMServerGenerator:
    def __init__(
        self,
        model_name: str,
        base_url: str,
        temperature: float,
    ) -> None:
        self.model_name = model_name
        self.temperature = float(temperature)
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=int(max_tokens),
            temperature=self.temperature,
        )
        return (response.choices[0].message.content or "").strip()


class APIGenerator:
    def __init__(self, model_name: str, temperature: float) -> None:
        self.manager = EquilibriumManager(model_name, temperature=temperature)

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        return self.manager.prompt_llm(
            prompt_kwargs={},
            system_prompt="",
            user_prompt=prompt,
            max_tokens=max_tokens,
        ).strip()


def build_generator(args: argparse.Namespace):
    if is_api_model(args.target_model):
        return APIGenerator(args.target_model, temperature=args.temperature)
    return VLLMServerGenerator(
        model_name=args.served_model_name or args.target_model,
        base_url=args.vllm_base_url,
        temperature=args.temperature,
    )


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [row for row in records if row.get("gold") is not None]
    if not judged:
        return {"n": len(records), "judged_n": 0, "accuracy": None}
    correct = 0
    for row in judged:
        pred = str(row.get("parsed_answer", "")).strip().lower()
        gold = str(row.get("gold", "")).strip().lower()
        correct += int(pred == gold)
    return {
        "n": len(records),
        "judged_n": len(judged),
        "correct": correct,
        "accuracy": correct / len(judged),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal public SIEVE benchmark runner.")
    parser.add_argument("--method", choices=["base", "sieve"], default="base")
    parser.add_argument("--target_model", required=True)
    parser.add_argument("--served_model_name", default="")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gate_checkpoint", default="")
    parser.add_argument("--cache_dir", default="cache/sieve")
    parser.add_argument("--gate_model", default="")
    parser.add_argument("--vllm_base_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max_gen_len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--run_gen", type=str_to_bool, default=True)
    parser.add_argument("--run_eval", type=str_to_bool, default=True)
    parser.add_argument("--load_gen", type=str_to_bool, default=False)
    parser.add_argument("--load_eval", type=str_to_bool, default=False)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    response_path = output_dir / "responses.json"
    summary_path = output_dir / "summary.json"

    if args.load_gen and response_path.exists():
        rows = json.loads(response_path.read_text(encoding="utf-8"))
    else:
        rows = []

    if args.run_gen and not rows:
        dataset = load_json_list(args.dataset_path)
        generator = build_generator(args)
        sieve = None
        if args.method == "sieve":
            if not args.gate_checkpoint:
                raise ValueError("--gate_checkpoint is required when --method=sieve")
            sieve = load_sieve_for_inference(
                gate_checkpoint_path=args.gate_checkpoint,
                gate_model=args.gate_model or None,
                generator=generator,
                cache_dir=args.cache_dir,
                device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") else "cpu",
            )

        for idx, record in enumerate(tqdm(dataset, desc=f"{args.method} generation")):
            input_text = get_input_text(record)
            if args.method == "sieve":
                response = run_sieve_inference(
                    sieve,
                    input_text=input_text,
                    input_id=str(record.get("id", idx)),
                )
            else:
                response = generator.generate(input_text, max_tokens=args.max_gen_len)
            rows.append(
                {
                    "id": record.get("id", idx),
                    "input": input_text,
                    "gold": get_gold(record),
                    "response": response,
                    "parsed_answer": parse_final_answer(response),
                    "raw_record": record,
                }
            )
        save_json(response_path, rows)
    elif args.run_gen and rows:
        print(f"[INFO] Loaded existing generations: {response_path}")
    elif not rows and args.run_eval:
        if not response_path.exists():
            raise FileNotFoundError(f"Cannot evaluate without responses: {response_path}")
        rows = json.loads(response_path.read_text(encoding="utf-8"))

    if args.run_eval:
        if args.load_eval and summary_path.exists():
            print(f"[INFO] Existing summary found: {summary_path}")
        summary = evaluate(rows)
        save_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
