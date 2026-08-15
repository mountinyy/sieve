from __future__ import annotations

"""
PRISM: Evaluation.
Evaluate trained gate module on test datasets.
"""

import atexit
from concurrent.futures import ThreadPoolExecutor
import math
import re
import torch
from dataclasses import dataclass, field
from tqdm import tqdm

from src.sieve.data_types import SCHEMA_NAMES, InferenceResult
from src.sieve.gate_module import GateModule
from src.sieve.extraction import ExtractionCache
from src.sieve.prompt_assembly import (
    assemble_prompt,
    parse_answer,
    parse_reasoning,
    token_proportional_truncation_stats,
)
from src.sieve.llm_client import LLMClient
from src.sieve.safety_reward import WildGuardRewarder


def resolve_prompt_tokenizer(llm: LLMClient, gate_module: GateModule | None):
    """Use target tokenizer when available; otherwise fall back to gate backbone tokenizer."""
    return getattr(llm, "tokenizer", None) or getattr(gate_module, "tokenizer", None)


def _uniform_theta_list() -> list[float]:
    return [1.0 / len(SCHEMA_NAMES)] * len(SCHEMA_NAMES)


def _random_theta_list() -> list[float]:
    theta = torch.rand(len(SCHEMA_NAMES), dtype=torch.float32)
    theta = theta / theta.sum().clamp_min(1e-8)
    return theta.tolist()


def swap_high_low_theta(theta: list[float]) -> list[float]:
    """Swap the largest and smallest schema activations in a theta vector."""
    swapped = [float(value) for value in theta]
    if len(swapped) < 2:
        return swapped
    max_idx = max(range(len(swapped)), key=lambda idx: swapped[idx])
    min_idx = min(range(len(swapped)), key=lambda idx: swapped[idx])
    if max_idx != min_idx:
        swapped[max_idx], swapped[min_idx] = swapped[min_idx], swapped[max_idx]
    return swapped


def maybe_swap_theta(theta: list[float], *, swap_theta: bool = False) -> list[float]:
    theta_list = [float(value) for value in theta]
    return swap_high_low_theta(theta_list) if swap_theta else theta_list


_THETA_SUMMARY: dict[str, dict[str, torch.Tensor | int]] = {}
_SAFETY_REWARDER: WildGuardRewarder | None = None


def _get_safety_rewarder() -> WildGuardRewarder:
    global _SAFETY_REWARDER
    if _SAFETY_REWARDER is None:
        _SAFETY_REWARDER = WildGuardRewarder()
    return _SAFETY_REWARDER
_THETA_SUMMARY_REGISTERED = False


def _record_theta_statistics(desc: str, theta_items: list[list[float]]) -> None:
    global _THETA_SUMMARY_REGISTERED
    if not theta_items:
        return
    theta = torch.tensor(theta_items, dtype=torch.float64)
    if theta.ndim != 2 or theta.shape[1] != len(SCHEMA_NAMES):
        return
    if not _THETA_SUMMARY_REGISTERED:
        atexit.register(print_recorded_theta_statistics)
        _THETA_SUMMARY_REGISTERED = True

    stats = _THETA_SUMMARY.setdefault(
        desc,
        {
            "count": 0,
            "sum": torch.zeros(len(SCHEMA_NAMES), dtype=torch.float64),
            "sum_sq": torch.zeros(len(SCHEMA_NAMES), dtype=torch.float64),
        },
    )
    stats["count"] = int(stats["count"]) + int(theta.shape[0])
    stats["sum"] = stats["sum"] + theta.sum(dim=0)
    stats["sum_sq"] = stats["sum_sq"] + (theta * theta).sum(dim=0)


def record_theta_statistics(desc: str, theta_items: list[list[float]]) -> None:
    _record_theta_statistics(desc, theta_items)


def _format_theta_statistics(
    label: str,
    count: int,
    theta_sum: torch.Tensor,
    theta_sum_sq: torch.Tensor,
) -> list[str]:
    mean = theta_sum / max(count, 1)
    variance = (theta_sum_sq / max(count, 1)) - (mean * mean)
    std = variance.clamp_min(0.0).sqrt()
    lines = [f"[SIEVE theta summary] {label}: n={count}"]
    for schema_name, schema_mean, schema_std in zip(SCHEMA_NAMES, mean.tolist(), std.tolist()):
        lines.append(
            f"  {schema_name}: mean={schema_mean:.6f}, std={schema_std:.6f}"
        )
    return lines


def print_recorded_theta_statistics() -> None:
    if not _THETA_SUMMARY:
        return
    print("\n========== SIEVE THETA SUMMARY ==========")
    total_count = 0
    total_sum = torch.zeros(len(SCHEMA_NAMES), dtype=torch.float64)
    total_sum_sq = torch.zeros(len(SCHEMA_NAMES), dtype=torch.float64)
    for desc, stats in sorted(_THETA_SUMMARY.items()):
        count = int(stats["count"])
        theta_sum = stats["sum"]
        theta_sum_sq = stats["sum_sq"]
        total_count += count
        total_sum = total_sum + theta_sum
        total_sum_sq = total_sum_sq + theta_sum_sq
        for line in _format_theta_statistics(desc, count, theta_sum, theta_sum_sq):
            print(line)
    if len(_THETA_SUMMARY) > 1:
        for line in _format_theta_statistics(
            "overall",
            total_count,
            total_sum,
            total_sum_sq,
        ):
            print(line)
    print("=========================================\n")


def _resolve_inference_theta(
    gate_module: GateModule | None,
    scenario: str,
    *,
    use_all: bool,
    uniform_theta: bool,
    random_theta: bool,
    swap_theta: bool = False,
) -> list[float]:
    if random_theta:
        return maybe_swap_theta(_random_theta_list(), swap_theta=swap_theta)
    if uniform_theta:
        return maybe_swap_theta(_uniform_theta_list(), swap_theta=swap_theta)
    if gate_module is None:
        if not use_all:
            raise ValueError(
                "gate_module cannot be None unless use_all=True, uniform_theta=True, or random_theta=True."
            )
        return maybe_swap_theta(_uniform_theta_list(), swap_theta=swap_theta)
    return maybe_swap_theta(gate_module(scenario).detach().tolist(), swap_theta=swap_theta)


def _resolve_batch_thetas(
    gate_module: GateModule | None,
    scenarios: list[str],
    *,
    use_all: bool,
    uniform_theta: bool,
    random_theta: bool,
    swap_theta: bool = False,
) -> list[list[float]]:
    if random_theta:
        return [maybe_swap_theta(_random_theta_list(), swap_theta=swap_theta) for _ in scenarios]
    if uniform_theta:
        theta = _uniform_theta_list()
        return [maybe_swap_theta(theta, swap_theta=swap_theta) for _ in scenarios]
    if gate_module is None:
        if not use_all:
            raise ValueError(
                "gate_module cannot be None unless use_all=True, uniform_theta=True, or random_theta=True."
            )
        theta = _uniform_theta_list()
        return [maybe_swap_theta(theta, swap_theta=swap_theta) for _ in scenarios]
    theta_batch, _ = gate_module.forward_batch(scenarios, inference_mode=True)
    theta_items = theta_batch.tolist()
    if swap_theta:
        return [swap_high_low_theta(theta) for theta in theta_items]
    return theta_items


def resolve_item_id(item: dict, fallback: str) -> str:
    return ExtractionCache.resolve_item_id(item, fallback)


@dataclass
class PreparedInference:
    """SIEVE inference state after gate/cache/prompt assembly, before generation."""

    scenario: str
    scenario_id: str
    theta: list[float]
    gate_stats: dict
    prompt: str
    llm: LLMClient
    max_tokens: int = 1024
    schema_arguments: dict = field(default_factory=dict)


SCHEMA_SENSITIVITY_MAX = 2.0 / 9.0
CROSS_SCHEMA_PAIRS = (("PI", "MN"), ("MN", "PC"), ("PI", "PC"))


def _consideration_to_dict(consideration) -> dict:
    return {
        "index": int(getattr(consideration, "index", 0)),
        "principle": str(getattr(consideration, "principle", "")),
        "supporting_context": str(getattr(consideration, "supporting_context", "")),
        "direction": str(getattr(consideration, "direction", "")),
        "source_schema": str(getattr(consideration, "source_schema", "")),
    }


def phase3_schema_arguments_to_dict(phase3) -> dict:
    by_role = {
        "primary": {
            "schema": phase3.primary_schema,
            "budget": int(phase3.primary_budget),
            "arguments": [_consideration_to_dict(item) for item in phase3.primary_arguments],
        },
        "secondary": {
            "schema": phase3.secondary_schema,
            "budget": int(phase3.secondary_budget),
            "arguments": [_consideration_to_dict(item) for item in phase3.secondary_arguments],
        },
        "tertiary": {
            "schema": phase3.tertiary_schema,
            "budget": int(phase3.tertiary_budget),
            "arguments": [_consideration_to_dict(item) for item in phase3.tertiary_principles],
        },
    }
    by_schema = {}
    for role_payload in by_role.values():
        by_schema[role_payload["schema"]] = role_payload["arguments"]
    return {
        "theta": [float(value) for value in phase3.theta],
        "schema_order": [
            phase3.primary_schema,
            phase3.secondary_schema,
            phase3.tertiary_schema,
        ],
        "budgets_by_schema": {
            phase3.primary_schema: int(phase3.primary_budget),
            phase3.secondary_schema: int(phase3.secondary_budget),
            phase3.tertiary_schema: int(phase3.tertiary_budget),
        },
        "by_role": by_role,
        "by_schema": by_schema,
    }


def extract_schema_label_set(item: dict) -> set[str]:
    raw_label = item.get("schema_label", "")
    if isinstance(raw_label, (list, tuple, set)):
        raw_parts = raw_label
    else:
        raw_parts = re.split(r"[^A-Za-z]+", str(raw_label))
    return {
        str(part).strip().upper()
        for part in raw_parts
        if str(part).strip().upper() in set(SCHEMA_NAMES)
    }


def schema_name_from_theta(theta_list: list[float]) -> str:
    dominant_idx = max(range(len(theta_list)), key=lambda idx: theta_list[idx])
    return SCHEMA_NAMES[dominant_idx]


def schema_match_from_theta(theta_list: list[float], item: dict) -> float | None:
    label_set = extract_schema_label_set(item)
    if not label_set:
        return None
    return float(schema_name_from_theta(theta_list) in label_set)


def compute_cross_schema_similarities(
    schema_embeddings: dict[str, list[float]],
) -> dict[str, float | None]:
    similarities: dict[str, float | None] = {}
    for left_schema, right_schema in CROSS_SCHEMA_PAIRS:
        metric_name = f"{left_schema}_{right_schema}"
        left_embedding = schema_embeddings.get(left_schema)
        right_embedding = schema_embeddings.get(right_schema)
        if not left_embedding or not right_embedding:
            similarities[metric_name] = None
            continue
        left = torch.tensor(left_embedding, dtype=torch.float32)
        right = torch.tensor(right_embedding, dtype=torch.float32)
        denom = left.norm().clamp_min(1e-8) * right.norm().clamp_min(1e-8)
        similarities[metric_name] = float(((left @ right) / denom).item())
    return similarities


def average_metric_dicts(metric_dicts: list[dict[str, float | None]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    keys = sorted({key for metrics in metric_dicts for key in metrics})
    averages: dict[str, float] = {}
    for key in keys:
        values = [
            float(metrics[key])
            for metrics in metric_dicts
            if metrics.get(key) is not None
        ]
        if values:
            averages[key] = sum(values) / len(values)
    return averages


def inference(
    scenario: str,
    scenario_id: str,
    gate_module: GateModule | None,
    llm: LLMClient,
    extraction_cache: ExtractionCache,
    use_persona: bool = False,
    inst_regime: bool = False,
    inference_add_eval: bool = True,
    token_proportional: bool = False,
    use_token_total_budget: bool = False,
    use_all: bool = False,
    use_top: bool = False,
    use_bottom: bool = False,
    safety: bool = False,
    uniform_theta: bool = False,
    random_theta: bool = False,
    swap_theta: bool = False,
    use_cache: bool = True,
    cache_prefix: str | None = None,
) -> InferenceResult:
    """Run full PRISM inference for one scenario."""
    prepared = prepare_inference(
        scenario,
        scenario_id,
        gate_module,
        llm,
        extraction_cache,
        use_persona=use_persona,
        inst_regime=inst_regime,
        inference_add_eval=inference_add_eval,
        token_proportional=token_proportional,
        use_token_total_budget=use_token_total_budget,
        use_all=use_all,
        use_top=use_top,
        use_bottom=use_bottom,
        safety=safety,
        uniform_theta=uniform_theta,
        random_theta=random_theta,
        swap_theta=swap_theta,
        use_cache=use_cache,
        cache_prefix=cache_prefix,
        max_tokens=1024,
    )
    raw = llm.generate(prepared.prompt, max_tokens=prepared.max_tokens)
    return complete_inference(prepared, raw)


def prepare_inference(
    scenario: str,
    scenario_id: str,
    gate_module: GateModule | None,
    llm: LLMClient,
    extraction_cache: ExtractionCache,
    use_persona: bool = False,
    inst_regime: bool = False,
    inference_add_eval: bool = True,
    token_proportional: bool = False,
    use_token_total_budget: bool = False,
    use_all: bool = False,
    use_top: bool = False,
    use_bottom: bool = False,
    safety: bool = False,
    uniform_theta: bool = False,
    random_theta: bool = False,
    swap_theta: bool = False,
    use_cache: bool = True,
    cache_prefix: str | None = None,
    max_tokens: int = 1024,
    theta_override: list[float] | None = None,
) -> PreparedInference:
    """Prepare SIEVE gate/cache/prompt state without calling the reasoner LLM."""
    # 1. Gate
    if theta_override is None:
        theta_list = _resolve_inference_theta(
            gate_module,
            scenario,
            use_all=use_all,
            uniform_theta=uniform_theta,
            random_theta=random_theta,
            swap_theta=swap_theta,
        )
    else:
        theta_list = maybe_swap_theta(theta_override, swap_theta=swap_theta)

    # 2. Phase 2 + Phase 3 (cached extraction + deterministic selection)
    cache_key = extraction_cache.build_cache_key(scenario_id, prefix=cache_prefix)
    phase3 = extraction_cache.get_phase3(
        scenario,
        cache_key,
        theta=theta_list,
        use_cache=use_cache,
        use_all=use_all,
    )

    # 4. Reason
    tokenizer = resolve_prompt_tokenizer(llm, gate_module)
    prompt = assemble_prompt(
        scenario,
        phase3,
        use_persona=use_persona,
        inst_regime=inst_regime,
        inference_add_eval=inference_add_eval,
        token_proportional=token_proportional,
        use_token_total_budget=use_token_total_budget,
        tokenizer=tokenizer,
        use_all=use_all,
        use_top=use_top,
        use_bottom=use_bottom,
        safety=safety,
    )

    return PreparedInference(
        scenario=scenario,
        scenario_id=scenario_id,
        theta=theta_list,
        gate_stats={
            "primary_schema": phase3.primary_schema,
            "secondary_schema": phase3.secondary_schema,
            "tertiary_schema": phase3.tertiary_schema,
            "primary_budget": phase3.primary_budget,
            "secondary_budget": phase3.secondary_budget,
            "tertiary_budget": phase3.tertiary_budget,
            "token_truncation_by_schema": token_proportional_truncation_stats(
                phase3,
                tokenizer=tokenizer,
                use_token_total_budget=use_token_total_budget,
                use_all=use_all,
                use_top=use_top,
                use_bottom=use_bottom,
            ) if token_proportional else {},
        },
        prompt=prompt,
        llm=llm,
        max_tokens=max_tokens,
        schema_arguments=phase3_schema_arguments_to_dict(phase3),
    )


def prepare_theta_batch(
    scenarios: list[str],
    gate_module: GateModule | None,
    *,
    use_all: bool = False,
    uniform_theta: bool = False,
    random_theta: bool = False,
    swap_theta: bool = False,
    batch_size: int = 32,
    verbose: bool = True,
    desc: str = "SIEVE theta precompute",
    record_stats: bool = True,
) -> list[list[float]]:
    """Compute only theta values for many scenarios, without extraction or cache IO."""
    if not scenarios:
        return []

    theta_items: list[list[float]] = []
    batch_size = max(1, int(batch_size))
    batch_starts = range(0, len(scenarios), batch_size)
    progress = tqdm(
        batch_starts,
        total=math.ceil(len(scenarios) / batch_size),
        desc=desc,
        disable=not verbose,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for batch_start in progress:
            batch_scenarios = scenarios[batch_start : batch_start + batch_size]
            theta_items.extend(
                _resolve_batch_thetas(
                    gate_module,
                    batch_scenarios,
                    use_all=use_all,
                    uniform_theta=uniform_theta,
                    random_theta=random_theta,
                    swap_theta=swap_theta,
                )
            )
    if record_stats:
        _record_theta_statistics(desc, theta_items)
    return theta_items


def prepare_inference_batch(
    scenarios: list[str],
    scenario_ids: list[str],
    gate_module: GateModule | None,
    llm: LLMClient,
    extraction_cache: ExtractionCache,
    use_persona: bool = False,
    inst_regime: bool = False,
    inference_add_eval: bool = True,
    token_proportional: bool = False,
    use_token_total_budget: bool = False,
    use_all: bool = False,
    use_top: bool = False,
    use_bottom: bool = False,
    safety: bool = False,
    uniform_theta: bool = False,
    random_theta: bool = False,
    swap_theta: bool = False,
    use_cache: bool = True,
    cache_prefix: str | None = None,
    max_tokens: int = 1024,
    batch_size: int = 16,
    verbose: bool = True,
    desc: str = "SIEVE gate/prompt precompute",
) -> list[PreparedInference]:
    """Prepare SIEVE prompt state for many scenarios before reasoner generation."""
    if len(scenarios) != len(scenario_ids):
        raise ValueError("scenarios and scenario_ids must have the same length.")
    if not scenarios:
        return []

    prepared_items: list[PreparedInference] = []
    batch_size = max(1, int(batch_size))
    tokenizer = resolve_prompt_tokenizer(llm, gate_module)

    batch_starts = range(0, len(scenarios), batch_size)
    progress = tqdm(
        batch_starts,
        total=math.ceil(len(scenarios) / batch_size),
        desc=desc,
        disable=not verbose,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for batch_start in progress:
            batch_scenarios = scenarios[batch_start : batch_start + batch_size]
            batch_ids = scenario_ids[batch_start : batch_start + batch_size]
            theta_lists = _resolve_batch_thetas(
                gate_module,
                batch_scenarios,
                use_all=use_all,
                uniform_theta=uniform_theta,
                random_theta=random_theta,
                swap_theta=swap_theta,
            )
            for scenario, scenario_id, theta_list in zip(batch_scenarios, batch_ids, theta_lists):
                cache_key = extraction_cache.build_cache_key(scenario_id, prefix=cache_prefix)
                phase3 = extraction_cache.get_phase3(
                    scenario,
                    cache_key,
                    theta=theta_list,
                    use_cache=use_cache,
                    use_all=use_all,
                )
                prompt = assemble_prompt(
                    scenario,
                    phase3,
                    use_persona=use_persona,
                    inst_regime=inst_regime,
                    inference_add_eval=inference_add_eval,
                    token_proportional=token_proportional,
                    use_token_total_budget=use_token_total_budget,
                    tokenizer=tokenizer,
                    use_all=use_all,
                    use_top=use_top,
                    use_bottom=use_bottom,
                    safety=safety,
                )
                prepared_items.append(
                    PreparedInference(
                        scenario=scenario,
                        scenario_id=scenario_id,
                        theta=theta_list,
                        gate_stats={
                            "primary_schema": phase3.primary_schema,
                            "secondary_schema": phase3.secondary_schema,
                            "tertiary_schema": phase3.tertiary_schema,
                            "primary_budget": phase3.primary_budget,
                            "secondary_budget": phase3.secondary_budget,
                            "tertiary_budget": phase3.tertiary_budget,
                            "token_truncation_by_schema": token_proportional_truncation_stats(
                                phase3,
                                tokenizer=tokenizer,
                                use_token_total_budget=use_token_total_budget,
                                use_all=use_all,
                                use_top=use_top,
                                use_bottom=use_bottom,
                            ) if token_proportional else {},
                        },
                        prompt=prompt,
                        llm=llm,
                        max_tokens=max_tokens,
                        schema_arguments=phase3_schema_arguments_to_dict(phase3),
                    )
                )
    return prepared_items


def complete_inference(prepared: PreparedInference, raw: str) -> InferenceResult:
    """Build the public inference result from prepared SIEVE state and raw response."""
    return InferenceResult(
        answer=parse_answer(raw),
        reasoning=parse_reasoning(raw),
        theta=prepared.theta,
        gate_stats=prepared.gate_stats,
        raw_response=raw,
        prompt=prepared.prompt,
        schema_arguments=prepared.schema_arguments,
    )


def evaluate(
    dataset: list[dict],
    gate_module: GateModule | None,
    llm: LLMClient,
    extraction_cache: ExtractionCache,
    verbose: bool = True,
    batch_size: int = 1,
    use_persona: bool = False,
    inst_regime: bool = False,
    inference_add_eval: bool = True,
    token_proportional: bool = False,
    use_token_total_budget: bool = False,
    use_all: bool = False,
    use_top: bool = False,
    use_bottom: bool = False,
    safety: bool = False,
    uniform_theta: bool = False,
    random_theta: bool = False,
    swap_theta: bool = False,
) -> dict:
    if gate_module is not None:
        gate_module.eval()
    results = []
    correct = 0

    with torch.no_grad():
        batch_starts = range(0, len(dataset), batch_size)
        progress = tqdm(
            batch_starts,
            total=(len(dataset) + batch_size - 1) // batch_size,
            desc="Evaluation",
            disable=not verbose,
        )
        for batch_start in progress:
            batch_items = dataset[batch_start: batch_start + batch_size]
            scenarios = [item.get("context", item.get("scenario", "")) for item in batch_items]
            theta_lists = _resolve_batch_thetas(
                gate_module,
                scenarios,
                use_all=use_all,
                uniform_theta=uniform_theta,
                random_theta=random_theta,
                swap_theta=swap_theta,
            )
            cache_keys = [
                extraction_cache.build_cache_key(
                    resolve_item_id(item, str(batch_start + idx)),
                    prefix="eval",
                )
                for idx, item in enumerate(batch_items)
            ]
            phase3_batch = [
                extraction_cache.get_phase3(
                    scenario,
                    cache_key,
                    theta=theta_list,
                    use_all=use_all,
                )
                for scenario, cache_key, theta_list in zip(
                    scenarios, cache_keys, theta_lists
                )
            ]
            prompts = [
                assemble_prompt(
                    scenario,
                    phase3,
                    use_persona=use_persona,
                    inst_regime=inst_regime,
                    inference_add_eval=inference_add_eval,
                    token_proportional=token_proportional,
                    use_token_total_budget=use_token_total_budget,
                    tokenizer=resolve_prompt_tokenizer(llm, gate_module),
                    use_all=use_all,
                    use_top=use_top,
                    use_bottom=use_bottom,
                    safety=safety,
                )
                for scenario, phase3 in zip(scenarios, phase3_batch)
            ]
            with ThreadPoolExecutor(max_workers=max(1, len(prompts))) as executor:
                responses = list(
                    executor.map(
                        lambda prompt: llm.generate(prompt, max_tokens=1024),
                        prompts,
                    )
                )
            safety_rewards: list[float] | None = None
            if safety:
                safety_rewards, _ = _get_safety_rewarder().compute_rewards(
                    scenarios,
                    responses,
                    [str(item["label"]) for item in batch_items],
                )

            for idx, (item, theta_list, phase3, raw) in enumerate(
                zip(batch_items, theta_lists, phase3_batch, responses)
            ):
                sid = resolve_item_id(item, str(batch_start + idx))
                answer = parse_answer(raw)
                reasoning = parse_reasoning(raw)
                is_correct = (
                    bool(safety_rewards[idx] >= 1.0)
                    if safety_rewards is not None
                    else scenario_matches_label(answer, item["label"])
                )
                schema_match = schema_match_from_theta(theta_list, item)
                cross_schema_similarities = compute_cross_schema_similarities(
                    extraction_cache.get_full_schema_embeddings(
                        scenarios[idx],
                        cache_keys[idx],
                    )
                )
                # is_correct = answer == item["label"]
                correct += int(is_correct)

                results.append({
                    "id": sid,
                    "gold": item["label"],
                    "predicted": answer,
                    "correct": is_correct,
                    "theta": theta_list,
                    "schema_label": item.get("schema_label"),
                    "schema_match": schema_match,
                    "cross_schema_similarities": cross_schema_similarities,
                    "gate_stats": {
                        "primary_schema": phase3.primary_schema,
                        "secondary_schema": phase3.secondary_schema,
                        "tertiary_schema": phase3.tertiary_schema,
                        "primary_budget": phase3.primary_budget,
                        "secondary_budget": phase3.secondary_budget,
                        "tertiary_budget": phase3.tertiary_budget,
                    },
                    "schema_arguments": phase3_schema_arguments_to_dict(phase3),
                    "prompt": prompts[idx],
                    "raw_response": raw,
                    "reasoning": reasoning[:300],
                })

            if verbose and results:
                progress.set_postfix(
                    acc=f"{correct / len(results):.3f}",
                    seen=f"{len(results)}/{len(dataset)}",
                    refresh=False,
                )

    # Aggregate analysis
    n = len(dataset)
    all_thetas = torch.tensor([r["theta"] for r in results])
    correct_thetas = torch.tensor([r["theta"] for r in results if r["correct"]])
    wrong_thetas = torch.tensor([r["theta"] for r in results if not r["correct"]])
    theta_clamped = all_thetas.clamp_min(1e-8)
    theta_entropy = -(theta_clamped * theta_clamped.log()).sum(dim=-1)
    avg_theta_entropy = float(theta_entropy.mean().item())
    theta_mean_scalar = all_thetas.mean(dim=-1, keepdim=True)
    theta_schema_sensitivity = ((all_thetas - theta_mean_scalar) ** 2).mean(dim=-1)
    avg_schema_sensitivity = float(theta_schema_sensitivity.mean().item())
    informative_mask = theta_schema_sensitivity > 0.01
    informative_count = int(informative_mask.sum().item())
    informative_ratio = informative_count / max(1, n)
    mean_theta_tensor = all_thetas.mean(dim=0)
    uniform_theta = torch.full_like(mean_theta_tensor, 1.0 / len(SCHEMA_NAMES))
    entropy_term = 1.0 - avg_theta_entropy / math.log(len(SCHEMA_NAMES))
    balance_l1 = torch.abs(mean_theta_tensor - uniform_theta).sum().item()
    balance_term = 1.0 - balance_l1 / (2.0 * (len(SCHEMA_NAMES) - 1) / len(SCHEMA_NAMES))
    schema_validity = max(0.0, entropy_term) * max(0.0, balance_term)
    schema_match_values = [
        result["schema_match"]
        for result in results
        if result.get("schema_match") is not None
    ]
    schema_match = (
        sum(schema_match_values) / len(schema_match_values)
        if schema_match_values
        else 0.0
    )
    dominant_schema_counts = {schema: 0 for schema in SCHEMA_NAMES}
    for result in results:
        dominant_schema_counts[schema_name_from_theta(result["theta"])] += 1
    dominant_schema_ratio = {
        schema: dominant_schema_counts[schema] / max(1, n)
        for schema in SCHEMA_NAMES
    }
    cross_schema_similarities = average_metric_dicts(
        [result.get("cross_schema_similarities", {}) for result in results]
    )

    metrics = {
        "accuracy": correct / n,
        "avg_loss": 0.0,
        "n_total": n,
        "n_correct": correct,
        "mean_theta": all_thetas.mean(dim=0).tolist(),
        "std_theta": all_thetas.std(dim=0).tolist(),
        "avg_theta_entropy": avg_theta_entropy,
        "schema_validity": schema_validity,
        "schema_match": schema_match,
        "avg_schema_sensitivity": avg_schema_sensitivity,
        "informative_count": informative_count,
        "informative_ratio": informative_ratio,
        "dominant_schema_ratio": dominant_schema_ratio,
        "cross_schema_similarities": cross_schema_similarities,
    }
    if len(correct_thetas) > 0:
        metrics["mean_theta_correct"] = correct_thetas.mean(dim=0).tolist()
    if len(wrong_thetas) > 0:
        metrics["mean_theta_wrong"] = wrong_thetas.mean(dim=0).tolist()

    return {"metrics": metrics, "results": results}

def scenario_matches_label(answer: str, gold_label: str) -> bool:
    label_result = "not_wrong" if "not" in gold_label.lower() else "wrong"
    answer_result = "not_wrong" if ("not_wrong" in answer.lower() or "not wrong" in answer.lower()) else "wrong"
    return label_result == answer_result
    
def evaluate_transfer(
    dataset: list[dict],
    gate_module: GateModule | None,
    source_llm: LLMClient,
    target_llm: LLMClient,
    source_cache_dir: str = None,
    target_cache_dir: str = None,
    verbose: bool = True,
    batch_size: int = 1,
    use_persona: bool = False,
    inst_regime: bool = False,
    inference_add_eval: bool = True,
    token_proportional: bool = False,
    use_token_total_budget: bool = False,
    use_all: bool = False,
) -> dict:
    """
    Cross-model transfer experiment.
    Same gate module, different LLMs for extraction and reasoning.

    Returns comparison of source vs target performance.
    """
    if verbose:
        print("Evaluating on source LLM...")
    source_cache = ExtractionCache(source_llm, cache_dir=source_cache_dir)
    source_result = evaluate(
        dataset,
        gate_module,
        source_llm,
        source_cache,
        verbose,
        batch_size=batch_size,
        use_persona=use_persona,
        inst_regime=inst_regime,
        inference_add_eval=inference_add_eval,
        token_proportional=token_proportional,
        use_token_total_budget=use_token_total_budget,
        use_all=use_all,
    )

    if verbose:
        print("Evaluating on target LLM...")
    target_cache = ExtractionCache(target_llm, cache_dir=target_cache_dir)
    target_result = evaluate(
        dataset,
        gate_module,
        target_llm,
        target_cache,
        verbose,
        batch_size=batch_size,
        use_persona=use_persona,
        inst_regime=inst_regime,
        inference_add_eval=inference_add_eval,
        token_proportional=token_proportional,
        use_token_total_budget=use_token_total_budget,
        use_all=use_all,
    )

    return {
        "source_accuracy": source_result["metrics"]["accuracy"],
        "target_accuracy": target_result["metrics"]["accuracy"],
        "transfer_gap": (
            source_result["metrics"]["accuracy"]
            - target_result["metrics"]["accuracy"]
        ),
        "source_metrics": source_result["metrics"],
        "target_metrics": target_result["metrics"],
    }
