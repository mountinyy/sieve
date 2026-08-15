from __future__ import annotations

"""
PRISM: GRPO Trainer.
Trains the gate module using Group Relative Policy Optimization.
"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from threading import Lock
import torch
import torch.nn.functional as F
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math
import random
import re
from tqdm import tqdm
from typing import Any, Callable

from src.sieve.data_types import SCHEMA_NAMES
from src.sieve.gate_module import GateModule
from src.sieve.extraction import ExtractionCache
from src.sieve.prompt_assembly import assemble_prompt, parse_answer
from src.sieve.safety_reward import WildGuardRewarder
from src.sieve.llm_client import LLMClient
from src.sieve.theta_utils import coerce_influence_vector, influence_vector_by_schema

INPUT_TEMPLATE = """# Context
{context}

# Question
{question}"""

CONTEXT_ONLY_INPUT_TEMPLATE = """# Context
{context}"""


def format_full_input(item: dict) -> str:
    return INPUT_TEMPLATE.format(context=item["context"], question=item["question"])


def format_gate_input(item: dict) -> str:
    return CONTEXT_ONLY_INPUT_TEMPLATE.format(context=item["context"])


SCHEMA_SENSITIVITY_MAX = 2.0 / 9.0
CROSS_SCHEMA_PAIRS = (("PI", "MN"), ("MN", "PC"), ("PI", "PC"))


@dataclass
class TrainStepResult:
    """Result from one training step (one scenario, k samples)."""
    scenario_id: str
    rewards: list[float]
    loss: float
    thetas: list[list[float]]
    answers: list[str]
    gold_label: str
    best_correct: bool  # did any of the k samples get it right?
    strict_accuracy: float = 0.0
    avg_advantage: float = 0.0
    schema_advantages: list[float] = field(default_factory=list)
    sample_advantages: list[float] = field(default_factory=list)
    alignment_scores: list[float] = field(default_factory=list)
    alignment_advantages: list[float] = field(default_factory=list)
    schema_similarity_means: dict[str, float | None] = field(default_factory=dict)
    cross_schema_similarities: dict[str, float | None] = field(default_factory=dict)
    combined_sample_advantages: list[float] = field(default_factory=list)
    schema_sensitivity: float = 0.0
    lambda_weight: float = 0.0
    dominant_schemas: list[str] = field(default_factory=list)
    per_group_rewards: dict[str, float] = field(default_factory=dict)
    reasoning_lengths: list[int] = field(default_factory=list)
    llm_model_name: str | None = None
    entropy: float = 0.0
    theta_entropy_penalty: float = 0.0
    informative: bool = False
    single_correct: bool = False
    single_theta: list[float] = field(default_factory=list)
    schema_match: float | None = None
    single_schema_match: float | None = None
    correct_theta_count: int = 0
    wrong_theta_count: int = 0
    correct_theta_entropy_mean: float | None = None
    wrong_theta_entropy_mean: float | None = None
    correct_theta_max_mean: float | None = None
    wrong_theta_max_mean: float | None = None
    reward_accs: list[float] = field(default_factory=list)
    reward_comps: list[float] = field(default_factory=list)
    reward_totals: list[float] = field(default_factory=list)
    influence_vector: list[float] | None = None


@dataclass
class EvalResult:
    """Result from one evaluation run."""
    accuracy: float
    avg_loss: float
    n_total: int
    n_correct: int
    mean_theta: list[float]
    avg_theta_entropy: float = 0.0
    schema_validity: float = 0.0
    schema_match: float = 0.0
    avg_schema_sensitivity: float = 0.0
    informative_count: int = 0
    informative_ratio: float = 0.0
    dominant_schema_ratio: dict[str, float] = field(default_factory=dict)
    cross_schema_similarities: dict[str, float] = field(default_factory=dict)
    richness_score_mean: float | None = None
    schema_similarity_means: dict[str, float | None] = field(default_factory=dict)
    per_item: list[dict] = field(default_factory=list)


@dataclass
class EpochResult:
    """Aggregated result from one training epoch."""
    epoch: int
    accuracy: float        # fraction of scenarios where best sample is correct
    avg_reward: float
    avg_loss: float
    temperature: float
    strict_accuracy: float = 0.0
    avg_advantage: float = 0.0
    avg_entropy: float = 0.0
    informative_count: int = 0
    informative_ratio: float = 0.0
    single_accuracy: float = 0.0
    entropy_reg_term_dominates_ratio: float = 0.0
    per_group_rewards: dict[str, float] = field(default_factory=dict)
    mean_theta: list[float] = field(default_factory=list)
    cross_schema_similarities: dict[str, float] = field(default_factory=dict)
    richness_score_mean: float | None = None
    richness_advantage_mean: float | None = None
    schema_similarity_means: dict[str, float | None] = field(default_factory=dict)
    schema_match: float = 0.0
    single_schema_match: float = 0.0
    eval_result: EvalResult = None  # attached after eval
    step_results: list[TrainStepResult] = field(default_factory=list)


class GRPOTrainer:
    """
    Trains the gate module via GRPO.

    For each scenario:
        1. Sample k different θ from the gate module
        2. Run full pipeline for each θ → k answers
        3. Compute dual-level advantages over sampled θ
        4. Update gate module toward better schema/budget allocations

    Extraction results are cached and reused across all k samples and epochs.
    """

    def __init__(
        self,
        gate_module: GateModule,
        llm: LLMClient,
        extraction_cache: ExtractionCache,
        k: int = 4,
        learning_rate: float = 5e-4,
        max_grad_norm: float = 1.0,
        temperature_decay: float = 0.95,
        min_temperature: float = 0.5,
        final_temperature: float | None = None,
        use_tracking: bool = False,
        tracking_project: str | None = None,
        tracking_run_name: str | None = None,
        tracking_entity: str | None = None,
        tracking_tags: list[str] | None = None,
        tracking_config: dict[str, Any] | None = None,
        llm_max_concurrency: int | None = None,
        batch_size: int = 1,
        rho_init: float = 0.01,
        rho_momentum: float = 0.99,
        seed: int = 42,
        entropy_reg_alpha: float = 0.0,
        use_entropy_loss: bool = False,
        entropy_loss_beta: float = 0.0,
        use_kl: bool = False,
        kl_weight: float = 0.0,
        use_clip: bool = False,
        clip_epsilon: float = 0.2,
        continuous_group_reward: bool = False,
        no_group_reward: bool = False,
        informative_sigma_threshold: float = 0.01,
        use_lr_scheduler: bool = False,
        lr_scheduler_eta_min: float = 1e-5,
        use_persona: bool = False,
        inst_regime: bool = False,
        inference_add_eval: bool = True,
        use_token_total_budget: bool = False,
        use_alignment_adv: bool = False,
        use_richness: bool = False,
        richness_alpha: float = 10.0,
        richness_weight: float = 0.5,
        multi_llm: bool = False,
        llm_pool: dict[str, LLMClient] | None = None,
        cache_pool: dict[str, ExtractionCache] | None = None,
        alignment_encoder: Any | None = None,
        alignment_lock: Any | None = None,
        lambda_comp: float = 0.5,
        csa_mode: str = "discrete",
        filter_zero_influence: bool = True,
    ):
        self.gate = gate_module
        self.llm = llm
        self.cache = extraction_cache
        self.k = k
        self.max_grad_norm = max_grad_norm
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature

        self.optimizer = torch.optim.Adam(
            self.gate.get_learnable_parameters(),
            lr=learning_rate,
        )
        self.scheduler: torch.optim.lr_scheduler.CosineAnnealingLR | None = None
        self.scheduler_eta_min = float(lr_scheduler_eta_min)
        self.use_lr_scheduler = bool(use_lr_scheduler)
        self.use_persona = bool(use_persona)
        self.inst_regime = bool(inst_regime)
        self.inference_add_eval = bool(inference_add_eval)
        self.use_token_total_budget = bool(use_token_total_budget)
        self.use_richness = bool(use_richness)
        self.richness_alpha = float(richness_alpha)
        self.richness_weight = min(max(float(richness_weight), 0.0), 1.0)
        self.use_alignment_adv = bool(use_alignment_adv)
        self.alignment_encoder = alignment_encoder
        self.alignment_lock = alignment_lock or Lock()

        self.history: list[EpochResult] = []
        self.use_tracking = use_tracking
        self.tracking_project = tracking_project
        self.tracking_run_name = tracking_run_name
        self.tracking_entity = tracking_entity
        self.tracking_tags = tracking_tags or []
        self.tracking_config = tracking_config or {}
        self._tracking_run = None
        self._tracking_enabled = False
        self.llm_max_concurrency = max(1, llm_max_concurrency or k)
        self.batch_size = max(1, batch_size)
        self.rho = max(0.0, rho_init)
        self.rho_momentum = min(max(rho_momentum, 0.0), 0.999999)
        self.seed = int(seed)
        self.entropy_reg_alpha = max(0.0, float(entropy_reg_alpha))
        self.use_entropy_loss = bool(use_entropy_loss)
        self.entropy_loss_beta = (
            max(0.0, float(entropy_loss_beta)) if self.use_entropy_loss else 0.0
        )
        self.use_kl = bool(use_kl)
        self.kl_weight = max(0.0, float(kl_weight)) if self.use_kl else 0.0
        self.use_clip = bool(use_clip)
        self.clip_epsilon = max(0.0, float(clip_epsilon))
        self.continuous_group_reward = bool(continuous_group_reward)
        self.no_group_reward = bool(no_group_reward)
        self.informative_sigma_threshold = max(0.0, float(informative_sigma_threshold))
        self.initial_temperature = float(self.gate.temperature)
        self.final_temperature = (
            1.0 if final_temperature is None else float(final_temperature)
        )
        self.multi_llm = multi_llm
        self.llm_pool = dict(llm_pool or {})
        self.cache_pool = dict(cache_pool or {})
        self._multi_llm_rng = random.Random(self.seed)
        self._model_pool: list[str] = []
        self.prev_eval_thetas: list[torch.Tensor] | None = None
        self.debug_last_train_batch_state: dict[str, Any] | None = None
        self.last_train_batch_monitor: dict[str, Any] | None = None
        self.reference_pooling = None
        self.reference_head = None
        self._safety_rewarder: WildGuardRewarder | None = None
        self._safety_rewarder_lock = Lock()
        self._last_schema_similarity_means: dict[str, float | None] = {
            schema: None for schema in SCHEMA_NAMES
        }
        self.lambda_comp = float(lambda_comp)
        self.csa_mode = self._normalize_csa_mode(csa_mode)
        self.filter_zero_influence = bool(filter_zero_influence)

        if (
            (self.use_alignment_adv or self._use_richness_reward())
            and self.alignment_encoder is None
        ):
            raise ValueError(
                "use_alignment_adv=True or use_richness=True requires a shared alignment encoder."
            )

        if self.multi_llm:
            if not self.llm_pool:
                raise ValueError("multi_llm=True requires a non-empty llm_pool.")
            if set(self.llm_pool) != set(self.cache_pool):
                raise ValueError("llm_pool and cache_pool must have the same model keys.")
            self._model_names = sorted(self.llm_pool)
        else:
            self._model_names = []

        if self._uses_safety_reward(self.cache):
            print("[INFO] safety=True: loading WildGuard reward model...")
            self._get_safety_rewarder()
            print("[INFO] WildGuard reward model loaded.")

    def _normalize_csa_mode(self, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized == "continous":
            raise ValueError("csa_mode='continous' is misspelled. Use 'continuous'.")
        if normalized not in {"discrete", "continuous"}:
            raise ValueError("csa_mode must be either 'discrete' or 'continuous'.")
        return normalized

    def _use_richness_reward(self) -> bool:
        return bool(self.use_richness and self.richness_weight > 0.0)

    def _resolve_prompt_tokenizer(self, llm_client: LLMClient):
        """Use the target tokenizer when available; API models fall back to the gate backbone."""
        return getattr(llm_client, "tokenizer", None) or getattr(self.gate, "tokenizer", None)

    def _checkpoint_dir(self, save_dir: str | None) -> Path | None:
        if not save_dir:
            return None
        return Path(save_dir)

    def _build_split_state(
        self,
        dataset: list[dict],
        warm_start_indices: list[int],
        grpo_indices: list[int],
        eval_dataset: list[dict] | None,
    ) -> dict[str, Any]:
        return {
            "warm_start_indices": list(warm_start_indices),
            "grpo_indices": list(grpo_indices),
            "warm_start_ids": [
                self._resolve_item_id(dataset[idx], str(idx))
                for idx in warm_start_indices
            ],
            "grpo_ids": [
                self._resolve_item_id(dataset[idx], str(idx))
                for idx in grpo_indices
            ],
            "train_ids": [
                self._resolve_item_id(item, str(idx))
                for idx, item in enumerate(dataset)
            ],
            "eval_ids": [
                self._resolve_item_id(item, str(idx))
                for idx, item in enumerate(eval_dataset or [])
            ],
        }

    def _serialize_history(self) -> list[dict[str, Any]]:
        return [asdict(epoch_result) for epoch_result in self.history]

    def _capture_reference_policy(self) -> None:
        self.reference_pooling = deepcopy(self.gate.pooling).to(self.gate.device)
        self.reference_head = deepcopy(self.gate.head).to(self.gate.device)
        self.reference_pooling.eval()
        self.reference_head.eval()
        for module in (self.reference_pooling, self.reference_head):
            for param in module.parameters():
                param.requires_grad_(False)

    def _has_reference_policy(self) -> bool:
        return self.reference_pooling is not None and self.reference_head is not None

    def _restore_reference_policy(
        self,
        pooling_state: dict[str, Any] | None,
        head_state: dict[str, Any] | None,
    ) -> None:
        if pooling_state is None or head_state is None:
            self.reference_pooling = None
            self.reference_head = None
            return
        self._capture_reference_policy()
        self.reference_pooling.load_state_dict(pooling_state)
        self.reference_head.load_state_dict(head_state)

    def _compute_reference_kl(self, gate_scenarios: list[str]) -> torch.Tensor:
        if not self._has_reference_policy():
            return torch.tensor(0.0, device=self.gate.device)
        hidden_states, attention_mask = self.gate._encode_hidden_batch(gate_scenarios)
        current_logits, _ = self.gate._compute_logits_from_hidden(
            hidden_states,
            attention_mask,
        )
        with torch.no_grad():
            ref_logits, _ = self.gate._compute_logits_from_hidden(
                hidden_states,
                attention_mask,
                pooling_module=self.reference_pooling,
                head_module=self.reference_head,
            )
        temperature = max(float(self.gate.temperature), 1e-6)
        current_log_probs = F.log_softmax(current_logits / temperature, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits / temperature, dim=-1)
        current_probs = current_log_probs.exp()
        return (current_probs * (current_log_probs - ref_log_probs)).sum(dim=-1).mean()

    def _theta_log_prob_from_logits(
        self,
        logits: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        temperature = max(float(self.gate.temperature), 1e-6)
        log_probs = F.log_softmax(logits / temperature, dim=-1)
        return (theta.detach() * log_probs).sum(dim=-1)

    def _policy_sample_loss(
        self,
        *,
        current_log_prob: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantage: float,
    ) -> tuple[torch.Tensor, float, bool]:
        if not self.use_clip:
            return -float(advantage) * current_log_prob, 1.0, False

        advantage_tensor = torch.as_tensor(
            float(advantage),
            device=current_log_prob.device,
            dtype=current_log_prob.dtype,
        )
        ratio = torch.exp(current_log_prob - old_log_prob.detach())
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.clip_epsilon,
            1.0 + self.clip_epsilon,
        )
        unclipped_objective = ratio * advantage_tensor
        clipped_objective = clipped_ratio * advantage_tensor
        clipped = bool((ratio.detach() != clipped_ratio.detach()).item())
        return -torch.minimum(unclipped_objective, clipped_objective), float(ratio.detach().item()), clipped

    def _save_last_checkpoint(
        self,
        *,
        save_dir: str,
        dataset: list[dict],
        eval_dataset: list[dict] | None,
        split_state: dict[str, Any],
        completed_epochs: int,
        total_planned_epochs: int,
        best_eval_accuracy: float | None,
        best_eval_loss: float | None,
        best_epoch: int | None,
        latest_epoch_result: EpochResult | None,
    ) -> None:
        checkpoint_dir = Path(save_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        gate_last_path = checkpoint_dir / "gate_last.pt"
        self.gate.save(str(gate_last_path))

        config_path = checkpoint_dir / "config.json"
        config_path.write_text(
            json.dumps(self._export_checkpoint_config(), indent=2)
        )

        trainer_state = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "rho": self.rho,
            "reference_pooling_state": (
                self.reference_pooling.state_dict() if self.reference_pooling is not None else None
            ),
            "reference_head_state": (
                self.reference_head.state_dict() if self.reference_head is not None else None
            ),
            "history": self._serialize_history(),
            "completed_epochs": int(completed_epochs),
            "total_planned_epochs": int(total_planned_epochs),
            "split_state": split_state,
            "python_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "multi_llm_rng_state": (
                self._multi_llm_rng.getstate() if self.multi_llm else None
            ),
        }
        torch.save(trainer_state, checkpoint_dir / "trainer_last.pt")

        data_state = {
            **split_state,
            "train_size": len(dataset),
            "eval_size": len(eval_dataset) if eval_dataset is not None else 0,
            "completed_epochs": int(completed_epochs),
            "total_planned_epochs": int(total_planned_epochs),
        }
        (checkpoint_dir / "data_state.json").write_text(
            json.dumps(data_state, indent=2)
        )

        latest_eval = latest_epoch_result.eval_result if latest_epoch_result is not None else None
        last_meta = {
            "completed_epochs": int(completed_epochs),
            "total_planned_epochs": int(total_planned_epochs),
            "temperature": float(self.gate.temperature),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "best_eval_accuracy": (
                None if best_eval_accuracy in {None, float("-inf")} else best_eval_accuracy
            ),
            "best_eval_loss": (
                None if best_eval_loss in {None, float("inf")} else best_eval_loss
            ),
            "best_epoch": best_epoch,
            "latest_train_accuracy": (
                None if latest_epoch_result is None else latest_epoch_result.accuracy
            ),
            "latest_train_loss": (
                None if latest_epoch_result is None else latest_epoch_result.avg_loss
            ),
            "latest_eval_accuracy": (
                None if latest_eval is None else latest_eval.accuracy
            ),
            "latest_eval_loss": (
                None if latest_eval is None else latest_eval.avg_loss
            ),
            "gate_last_path": str(gate_last_path),
        }
        (checkpoint_dir / "last_meta.json").write_text(
            json.dumps(last_meta, indent=2)
        )

    def _save_eval_checkpoint(
        self,
        *,
        save_dir: str,
        epoch: int,
        global_step: int,
        eval_result: EvalResult,
        train_result: EpochResult | None = None,
    ) -> None:
        checkpoint_dir = Path(save_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"gate_step-{global_step}.pt"
        self.gate.save(str(checkpoint_path))

        config_path = checkpoint_dir / "config.json"
        config_path.write_text(
            json.dumps(self._export_checkpoint_config(), indent=2)
        )

        eval_response_dir = checkpoint_dir / "eval_responses"
        eval_response_path = eval_response_dir / f"{global_step}.jsonl"

        meta = {
            "epoch": epoch,
            "global_step": global_step,
            "eval_accuracy": eval_result.accuracy,
            "eval_loss": eval_result.avg_loss,
            "train_loss": None if train_result is None else train_result.avg_loss,
            "train_accuracy": None if train_result is None else train_result.accuracy,
            "temperature": self.gate.temperature,
            "mean_theta": eval_result.mean_theta,
            "schema_validity": eval_result.schema_validity,
            "richness_score_mean": eval_result.richness_score_mean,
            "schema_similarity_means": eval_result.schema_similarity_means,
            "checkpoint_path": str(checkpoint_path),
            "eval_responses_path": str(eval_response_path),
        }
        meta_path = checkpoint_dir / f"meta_step-{global_step}.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        eval_response_dir.mkdir(parents=True, exist_ok=True)
        with eval_response_path.open("w", encoding="utf-8") as f:
            for item in eval_result.per_item:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _load_last_checkpoint(
        self,
        *,
        resume_from_checkpoint: str,
        dataset: list[dict],
    ) -> dict[str, Any]:
        checkpoint_path = Path(resume_from_checkpoint)
        checkpoint_dir = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
        gate_last_path = (
            checkpoint_path
            if checkpoint_path.is_file() and checkpoint_path.name.endswith(".pt")
            else checkpoint_dir / "gate_last.pt"
        )
        trainer_state_path = checkpoint_dir / "trainer_last.pt"

        if not gate_last_path.exists():
            raise FileNotFoundError(f"Missing last gate checkpoint at {gate_last_path}")
        if not trainer_state_path.exists():
            raise FileNotFoundError(f"Missing trainer state at {trainer_state_path}")

        self.gate.load(str(gate_last_path))
        trainer_state = torch.load(trainer_state_path, map_location="cpu")
        self.optimizer.load_state_dict(trainer_state["optimizer"])
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.gate.device)
        self.rho = float(trainer_state.get("rho", self.rho))
        self._restore_reference_policy(
            trainer_state.get("reference_pooling_state"),
            trainer_state.get("reference_head_state"),
        )

        history = trainer_state.get("history", [])
        self.history = []
        for item in history:
            eval_result = item.get("eval_result")
            self.history.append(
                EpochResult(
                    epoch=item["epoch"],
                    accuracy=item["accuracy"],
                    avg_reward=item["avg_reward"],
                    avg_loss=item["avg_loss"],
                    temperature=item["temperature"],
                    strict_accuracy=item.get("strict_accuracy", 0.0),
                    avg_advantage=item.get("avg_advantage", 0.0),
                    avg_entropy=item.get("avg_entropy", 0.0),
                    informative_count=item.get("informative_count", 0),
                    informative_ratio=item.get("informative_ratio", 0.0),
                    single_accuracy=item.get("single_accuracy", 0.0),
                    entropy_reg_term_dominates_ratio=item.get(
                        "entropy_reg_term_dominates_ratio", 0.0
                    ),
                    per_group_rewards=item.get("per_group_rewards", {}),
                    mean_theta=item.get("mean_theta", []),
                    schema_match=item.get("schema_match", 0.0),
                    single_schema_match=item.get("single_schema_match", 0.0),
                    eval_result=(
                        None
                        if eval_result is None
                        else EvalResult(
                            accuracy=eval_result["accuracy"],
                            avg_loss=eval_result["avg_loss"],
                            n_total=eval_result["n_total"],
                            n_correct=eval_result["n_correct"],
                            mean_theta=eval_result["mean_theta"],
                            avg_theta_entropy=eval_result.get("avg_theta_entropy", 0.0),
                            schema_validity=eval_result.get("schema_validity", 0.0),
                            schema_match=eval_result.get("schema_match", 0.0),
                            avg_schema_sensitivity=eval_result.get("avg_schema_sensitivity", 0.0),
                            informative_count=eval_result.get("informative_count", 0),
                            informative_ratio=eval_result.get("informative_ratio", 0.0),
                            dominant_schema_ratio=eval_result.get("dominant_schema_ratio", {}),
                            richness_score_mean=eval_result.get("richness_score_mean"),
                            schema_similarity_means=eval_result.get(
                                "schema_similarity_means",
                                {},
                            ),
                            per_item=eval_result.get("per_item", []),
                        )
                    ),
                    richness_score_mean=item.get("richness_score_mean"),
                    richness_advantage_mean=item.get("richness_advantage_mean"),
                    schema_similarity_means=item.get("schema_similarity_means", {}),
                    step_results=[],
                )
            )

        split_state = trainer_state.get("split_state", {})
        warm_start_indices = split_state.get("warm_start_indices", [])
        grpo_indices = split_state.get("grpo_indices", [])
        if any(idx < 0 or idx >= len(dataset) for idx in warm_start_indices + grpo_indices):
            raise ValueError("Saved dataset split indices do not match the current training dataset.")

        if trainer_state.get("python_random_state") is not None:
            random.setstate(trainer_state["python_random_state"])
        if trainer_state.get("torch_rng_state") is not None:
            torch.set_rng_state(trainer_state["torch_rng_state"])
        if torch.cuda.is_available() and trainer_state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(trainer_state["cuda_rng_state_all"])
        if self.multi_llm and trainer_state.get("multi_llm_rng_state") is not None:
            self._multi_llm_rng.setstate(trainer_state["multi_llm_rng_state"])

        return {
            "checkpoint_dir": checkpoint_dir,
            "completed_epochs": int(trainer_state.get("completed_epochs", 0)),
            "total_planned_epochs": int(trainer_state.get("total_planned_epochs", 0)),
            "warm_start_indices": warm_start_indices,
            "grpo_indices": grpo_indices,
            "scheduler_state": trainer_state.get("scheduler"),
        }

    def _print_debug_numeric_summary(self) -> None:
        state = self.debug_last_train_batch_state
        if not state:
            print("[SIEVE DEBUG] No debug train batch state available.")
            return

        missing = "<missing>"

        def get_value(mapping: dict[str, Any], key: str) -> Any:
            return mapping.get(key, missing)

        def format_float(value: Any, digits: int = 6) -> str:
            if value == missing:
                return missing
            try:
                return f"{float(value):.{digits}f}"
            except (TypeError, ValueError):
                return str(value)

        print("[SIEVE DEBUG] Batch numeric summary")
        print(f"  batch_size: {get_value(state, 'batch_size')}")
        print(f"  sample_k: {get_value(state, 'sample_k')}")
        print(f"  policy_loss: {format_float(get_value(state, 'policy_loss'))}")
        print(f"  batch_loss: {format_float(get_value(state, 'batch_loss'))}")
        print(f"  entropy_bonus: {format_float(get_value(state, 'entropy_bonus'))}")
        print(f"  kl_term: {format_float(get_value(state, 'kl_term'))}")
        print(f"  entropy_reg_alpha: {format_float(get_value(state, 'entropy_reg_alpha'))}")
        print(f"  use_kl: {get_value(state, 'use_kl')}")
        print(f"  kl_weight: {format_float(get_value(state, 'kl_weight'))}")
        print(f"  use_clip: {get_value(state, 'use_clip')}")
        print(f"  clip_epsilon: {format_float(get_value(state, 'clip_epsilon'))}")
        print(
            f"  entropy_reg_active_scenarios: "
            f"{get_value(state, 'entropy_reg_active_scenarios')}/"
            f"{get_value(state, 'entropy_reg_total_scenarios')}"
        )
        print(f"  use_alignment_adv: {get_value(state, 'use_alignment_adv')}")
        print(f"  use_richness: {get_value(state, 'use_richness')}")
        print(f"  richness_alpha: {format_float(get_value(state, 'richness_alpha'))}")
        print(f"  richness_weight: {format_float(get_value(state, 'richness_weight'))}")
        print(f"  inference_add_eval: {get_value(state, 'inference_add_eval')}")
        print(
            f"  theta_entropy_penalty: "
            f"{format_float(get_value(state, 'theta_entropy_penalty'))}"
        )
        print(f"  use_entropy_loss: {get_value(state, 'use_entropy_loss')}")
        print(f"  entropy_loss_beta: {format_float(get_value(state, 'entropy_loss_beta'))}")
        print(f"  learning_rate: {format_float(get_value(state, 'learning_rate'), digits=8)}")
        print(f"  rho: {format_float(get_value(state, 'rho'))}")
        print(
            f"  informative_sigma_threshold: "
            f"{format_float(get_value(state, 'informative_sigma_threshold'))}"
        )

        print("[SIEVE DEBUG] Per-sample numeric summary")
        batch_items = state.get("batch_items")
        if not isinstance(batch_items, list):
            print(f"  batch_items: {missing}")
            return
        for idx, item in enumerate(batch_items):
            if not isinstance(item, dict):
                print(f"  sample[{idx}]: {item}")
                continue
            print(
                f"  sample[{idx}] "
                f"dataset_idx={get_value(item, 'dataset_idx')} "
                f"scenario_id={get_value(item, 'scenario_id')} "
                f"strict_accuracy={format_float(get_value(item, 'strict_accuracy'))} "
                f"avg_advantage={format_float(get_value(item, 'avg_advantage'))} "
                f"schema_sensitivity={format_float(get_value(item, 'schema_sensitivity'))} "
                f"lambda_weight={format_float(get_value(item, 'lambda_weight'))} "
                f"entropy_reg_applied={get_value(item, 'entropy_reg_applied')} "
                f"kl_term={format_float(get_value(item, 'kl_term'))} "
                f"scenario_loss={format_float(get_value(item, 'scenario_loss'))} "
                f"entropy={format_float(get_value(item, 'entropy'))} "
                f"theta_entropy_penalty={format_float(get_value(item, 'theta_entropy_penalty'))} "
                f"informative={get_value(item, 'informative')}"
            )
            print(f"    rewards: {get_value(item, 'rewards')}")
            print(f"    raw_responses: {get_value(item, 'raw_responses')}")
            print(f"    wildguard_rewards: {get_value(item, 'wildguard_rewards')}")
            print(f"    wildguard_outputs: {get_value(item, 'wildguard_outputs')}")
            print(f"    correctness_flags: {get_value(item, 'correctness_flags')}")
            print(f"    reasoning_lengths: {get_value(item, 'reasoning_lengths')}")
            print(f"    thetas: {get_value(item, 'thetas')}")
            print(f"    sample_logits: {get_value(item, 'sample_logits')}")
            print(f"    sample_log_probs: {get_value(item, 'sample_log_probs')}")
            print(f"    schema_advantages: {get_value(item, 'schema_advantages')}")
            print(f"    sample_advantages: {get_value(item, 'sample_advantages')}")
            print(f"    alignment_scores: {get_value(item, 'alignment_scores')}")
            print(f"    alignment_advantages: {get_value(item, 'alignment_advantages')}")
            print(f"    schema_similarity_means: {get_value(item, 'schema_similarity_means')}")
            print(
                f"    combined_sample_advantages: "
                f"{get_value(item, 'combined_sample_advantages')}"
            )
            print(f"    schema_item_counts: {get_value(item, 'schema_item_counts')}")
            print(f"    combined_advantages: {get_value(item, 'combined_advantages')}")
            print(f"    dominant_schemas: {get_value(item, 'dominant_schemas')}")
            print(f"    per_group_rewards: {get_value(item, 'per_group_rewards')}")

    def _init_metrics(self, num_epochs: int, train_size: int, eval_size: int) -> None:
        self._tracking_enabled = False

    def _log_metrics(self, payload: dict[str, Any], step: int | None = None) -> None:
        return

    def warm_start(
        self,
        dataset: list[dict],
        num_epochs: int = 1,
        learning_rate: float = 1e-3,
        batch_size: int | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Supervised warm-start using schema labels before GRPO.
        """
        schema_to_idx = {"PI": 0, "MN": 1, "PC": 2}
        ws_optimizer = torch.optim.Adam(
            self.gate.get_learnable_parameters(),
            lr=learning_rate,
        )
        warm_start_batch_size = max(1, batch_size or self.batch_size)
        min_loss_improvement = 0.05
        max_small_improvement_epochs = 2
        prev_epoch_loss: float | None = None
        small_improvement_streak = 0

        self.gate.train()

        if verbose and hasattr(self.gate, "_encode_batch"):
            schema_scenarios = {schema: [] for schema in SCHEMA_NAMES}
            for item in dataset:
                labels = self._extract_schema_label_set(item)
                label_str = sorted(labels)[0] if len(labels) == 1 else ""
                if label_str not in schema_scenarios or len(schema_scenarios[label_str]) >= 50:
                    continue
                schema_scenarios[label_str].append(
                    format_gate_input(item)
                )

            available_schemas = [
                schema for schema in SCHEMA_NAMES if len(schema_scenarios[schema]) >= 2
            ]
            if len(available_schemas) >= 2:
                embedding_by_schema = {}
                with torch.no_grad():
                    for schema in available_schemas:
                        pooled, _ = self.gate._encode_batch(schema_scenarios[schema])
                        embedding_by_schema[schema] = F.normalize(pooled.float(), dim=-1)

                def mean_within_cosine(embeddings: torch.Tensor) -> float:
                    if embeddings.size(0) < 2:
                        return float("nan")
                    sims = embeddings @ embeddings.T
                    mask = torch.triu(
                        torch.ones_like(sims, dtype=torch.bool),
                        diagonal=1,
                    )
                    selected = sims.masked_select(mask)
                    return float(selected.mean().item()) if selected.numel() > 0 else float("nan")

                def mean_between_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
                    sims = a @ b.T
                    return float(sims.mean().item()) if sims.numel() > 0 else float("nan")

                print("  [Warm-start diagnostic] Backbone cosine similarity check")
                for schema in available_schemas:
                    print(
                        f"    within_{schema}: "
                        f"{mean_within_cosine(embedding_by_schema[schema]):.4f} "
                        f"(n={embedding_by_schema[schema].size(0)})"
                    )
                for i, schema_a in enumerate(available_schemas):
                    for schema_b in available_schemas[i + 1:]:
                        print(
                            f"    between_{schema_a}_{schema_b}: "
                            f"{mean_between_cosine(embedding_by_schema[schema_a], embedding_by_schema[schema_b]):.4f}"
                        )
                print("    interpretation: within > between이면 separable, 비슷하면 backbone에서 schema 구분이 약함")
            else:
                print(
                    "  [Warm-start diagnostic] Not enough labeled samples per schema "
                    "for within/between cosine comparison."
                )

        for epoch in range(num_epochs):
            total_loss = 0.0
            correct = 0
            n_valid = 0

            for batch_start in range(0, len(dataset), warm_start_batch_size):
                batch_items = dataset[batch_start: batch_start + warm_start_batch_size]
                batch_scenarios = []
                batch_targets = []

                for item in batch_items:
                    labels = self._extract_schema_label_set(item)
                    label_str = sorted(labels)[0] if len(labels) == 1 else ""
                    if label_str not in schema_to_idx:
                        continue

                    batch_scenarios.append(
                        format_gate_input(item)
                    )
                    batch_targets.append(schema_to_idx[label_str])

                if not batch_scenarios:
                    continue

                _, logits_tensor = self.gate.forward_batch(batch_scenarios)
                target_tensor = torch.tensor(
                    batch_targets,
                    device=self.gate.device,
                )
                loss = F.cross_entropy(logits_tensor, target_tensor)

                ws_optimizer.zero_grad()
                loss.backward()
                ws_optimizer.step()

                effective_batch_size = len(batch_targets)
                preds = logits_tensor.argmax(dim=-1)
                total_loss += loss.item() * effective_batch_size
                correct += int((preds == target_tensor).sum().item())
                n_valid += effective_batch_size

            if n_valid > 0:
                epoch_loss = total_loss / n_valid
                if verbose:
                    print(
                        f"  Warm-start epoch {epoch + 1}/{num_epochs}: "
                        f"loss={epoch_loss:.4f}, acc={correct / n_valid:.3f}"
                    )

                if prev_epoch_loss is not None:
                    loss_improvement = prev_epoch_loss - epoch_loss
                    if verbose:
                        print(
                            "  [Warm-start early-stop] "
                            f"loss improvement vs prev epoch: {loss_improvement:.4f}"
                        )
                    if loss_improvement < min_loss_improvement:
                        small_improvement_streak += 1
                    else:
                        small_improvement_streak = 0

                    if small_improvement_streak >= max_small_improvement_epochs:
                        print(
                            "  [Warm-start early-stop] "
                            f"Stopping early at epoch {epoch + 1}/{num_epochs} because "
                            f"loss improvement was below {min_loss_improvement:.2f} for "
                            f"{max_small_improvement_epochs} consecutive epochs."
                        )
                        break

                prev_epoch_loss = epoch_loss

        self.optimizer = torch.optim.Adam(
            self.gate.get_learnable_parameters(),
            lr=self.optimizer.param_groups[0]["lr"],
        )
        self.scheduler = None
        if self.use_kl:
            self._capture_reference_policy()

    def _compute_balanced_warm_start_indices(
        self,
        dataset: list[dict],
        warm_start_size: int,
    ) -> tuple[list[int], list[int], dict[str, int]]:
        labeled_indices_by_schema = {schema: [] for schema in SCHEMA_NAMES}
        for idx, item in enumerate(dataset):
            labels = self._extract_schema_label_set(item)
            if len(labels) != 1:
                continue
            schema_label = sorted(labels)[0]
            labeled_indices_by_schema[schema_label].append(idx)

        target_per_schema = max(0, warm_start_size // len(SCHEMA_NAMES))
        balanced_per_schema = min(
            target_per_schema,
            *(len(indices) for indices in labeled_indices_by_schema.values()),
        ) if target_per_schema > 0 else 0

        selected_indices = []
        selected_counts = {}
        for schema in SCHEMA_NAMES:
            schema_indices = labeled_indices_by_schema[schema][:balanced_per_schema]
            selected_indices.extend(schema_indices)
            selected_counts[schema] = len(schema_indices)

        selected_index_set = set(selected_indices)
        grpo_indices = [
            idx for idx in range(len(dataset))
            if idx not in selected_index_set
        ]
        shuffle_rng = random.Random(self.seed)
        shuffled_selected_indices = list(selected_indices)
        shuffle_rng.shuffle(shuffled_selected_indices)
        return shuffled_selected_indices, grpo_indices, selected_counts

    def _select_balanced_warm_start_data(
        self,
        dataset: list[dict],
        warm_start_size: int,
    ) -> tuple[list[dict], list[dict], dict[str, int], list[int], list[int]]:
        warm_start_indices, grpo_indices, selected_counts = (
            self._compute_balanced_warm_start_indices(dataset, warm_start_size)
        )
        warm_start_data = [dataset[idx] for idx in warm_start_indices]
        grpo_data = [dataset[idx] for idx in grpo_indices]
        return warm_start_data, grpo_data, selected_counts, warm_start_indices, grpo_indices

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------

    def _scenario_matches_label(self, answer: str, gold_label: str) -> bool:
        label_result = "not_wrong" if "not" in gold_label.lower() else "wrong"
        answer_result = "not_wrong" if ("not_wrong" in answer.lower() or "not wrong" in answer.lower()) else "wrong"
        return label_result == answer_result

    def _uses_safety_reward(self, cache: ExtractionCache | None = None) -> bool:
        active_cache = cache or self.cache
        return bool(getattr(active_cache, "safety", False))

    def _get_safety_rewarder(self) -> WildGuardRewarder:
        with self._safety_rewarder_lock:
            if self._safety_rewarder is None:
                self._safety_rewarder = WildGuardRewarder()
            return self._safety_rewarder

    def _compute_safety_reward(
        self,
        *,
        scenario: str,
        response: str,
        gold_label: str,
    ) -> tuple[float, dict[str, Any]]:
        rewarder = self._get_safety_rewarder()
        rewards, parsed = rewarder.compute_rewards(
            [scenario],
            [response],
            [gold_label],
        )
        return rewards[0], parsed[0]

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()

    def _extract_gold_principles(self, item: dict) -> list[str]:
        for key in ("gold_principles", "principles", "gold_principle_list"):
            value = item.get(key)
            if isinstance(value, list):
                return [str(x).strip() for x in value if str(x).strip()]
        return []

    def _estimate_principle_coverage(
        self,
        answer: str,
        gold_principles: list[str],
    ) -> float | None:
        if not gold_principles:
            return None

        answer_norm = self._normalize_text(answer)
        answer_tokens = set(answer_norm.split())
        if not answer_tokens:
            return 0.0

        matched = 0
        for principle in gold_principles:
            principle_norm = self._normalize_text(principle)
            if not principle_norm:
                continue
            principle_tokens = {
                token for token in principle_norm.split() if len(token) >= 3
            }
            if not principle_tokens:
                continue
            overlap = len(principle_tokens & answer_tokens) / len(principle_tokens)
            if principle_norm in answer_norm or overlap >= 0.5:
                matched += 1
        return matched / max(1, len(gold_principles))

    def _compute_reward(
        self,
        *,
        is_correct: bool,
        principle_coverage: float | None = None,
    ) -> float:
        if principle_coverage is None:
            answer_reward = 1.0 if is_correct else 0.0
        else:
            answer_reward = (0.5 * float(is_correct)) + (0.5 * principle_coverage)

        return answer_reward

    def _schema_name_from_theta(self, theta_list: list[float]) -> str:
        dominant_idx = max(range(len(theta_list)), key=lambda idx: theta_list[idx])
        return SCHEMA_NAMES[dominant_idx]

    def _extract_schema_label_set(self, item: dict) -> set[str]:
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

    def _schema_match_from_theta(self, theta_list: list[float], item: dict) -> float | None:
        label_set = self._extract_schema_label_set(item)
        if not label_set:
            return None
        return float(self._schema_name_from_theta(theta_list) in label_set)

    def _schema_match_mean_from_thetas(
        self,
        theta_lists: list[list[float]],
        item: dict,
    ) -> float | None:
        label_set = self._extract_schema_label_set(item)
        if not label_set or not theta_lists:
            return None
        matches = [
            float(self._schema_name_from_theta(theta_list) in label_set)
            for theta_list in theta_lists
        ]
        return sum(matches) / len(matches)

    def _resolve_item_id(self, item: dict, fallback: str) -> str:
        return self.cache.resolve_item_id(item, fallback)

    def _set_epoch_temperature(self, epoch_num: int, num_epochs: int) -> float:
        if num_epochs <= 1:
            scheduled_temp = self.initial_temperature
        else:
            progress = epoch_num / max(1, num_epochs - 1)
            scheduled_temp = self.initial_temperature + (
                (self.final_temperature - self.initial_temperature) * progress
            )
        self.gate.temperature = float(scheduled_temp)
        return self.gate.temperature

    def _compute_single_theta_entropy_bonus(
        self,
        single_thetas: torch.Tensor | list,
    ) -> torch.Tensor:
        if not torch.is_tensor(single_thetas):
            single_thetas = torch.tensor(
                single_thetas,
                device=self.gate.device,
                dtype=torch.float32,
            )
        else:
            single_thetas = single_thetas.to(self.gate.device).float()
        if single_thetas.numel() == 0:
            return torch.tensor(0.0, device=self.gate.device)
        if single_thetas.dim() == 1:
            single_thetas = single_thetas.unsqueeze(0)

        theta = single_thetas.clamp_min(1e-8)
        entropy = -(theta * theta.log()).sum(dim=-1)
        return (entropy / math.log(len(SCHEMA_NAMES))).mean().clamp(0.0, 1.0)

    def _compute_avg_sample_theta_entropy(
        self,
        samples_batch: list[list],
        *,
        exclude_forced_min_coverage: bool = False,
    ) -> torch.Tensor:
        all_thetas = [
            sample.theta
            for samples in samples_batch
            for sample in samples
            if not (
                exclude_forced_min_coverage and getattr(sample, "forced_min_coverage", False)
            )
        ]
        if not all_thetas:
            return torch.tensor(0.0, device=self.gate.device)
        theta_tensor = torch.stack(all_thetas).clamp_min(1e-8)
        theta_entropy = -(theta_tensor * theta_tensor.log()).sum(dim=-1)
        return theta_entropy.mean()

    def _compute_theta_entropy_penalty(self, samples_batch: list[list]) -> torch.Tensor:
        return self._compute_avg_sample_theta_entropy(
            samples_batch,
            exclude_forced_min_coverage=True,
        )

    def _compute_correctness_conditioned_theta_metrics(
        self,
        theta_lists: list[list[float]],
        correctness_flags: list[bool],
    ) -> dict[str, float | int | None]:
        if not theta_lists:
            return {
                "correct_theta_count": 0,
                "wrong_theta_count": 0,
                "correct_theta_entropy_mean": None,
                "wrong_theta_entropy_mean": None,
                "correct_theta_max_mean": None,
                "wrong_theta_max_mean": None,
            }

        theta_tensor = torch.tensor(theta_lists, dtype=torch.float32).clamp_min(1e-8)
        theta_entropy = -(theta_tensor * theta_tensor.log()).sum(dim=-1)
        theta_max = theta_tensor.max(dim=-1).values
        correct_mask = torch.tensor(correctness_flags, dtype=torch.bool)
        wrong_mask = ~correct_mask

        def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float | None:
            if not bool(mask.any()):
                return None
            return float(values[mask].mean().item())

        return {
            "correct_theta_count": int(correct_mask.sum().item()),
            "wrong_theta_count": int(wrong_mask.sum().item()),
            "correct_theta_entropy_mean": masked_mean(theta_entropy, correct_mask),
            "wrong_theta_entropy_mean": masked_mean(theta_entropy, wrong_mask),
            "correct_theta_max_mean": masked_mean(theta_max, correct_mask),
            "wrong_theta_max_mean": masked_mean(theta_max, wrong_mask),
        }

    def _export_checkpoint_config(self) -> dict:
        head_hidden_dim = 256
        if hasattr(self.gate, "head"):
            try:
                head_hidden_dim = int(self.gate.head[0].out_features)
            except (TypeError, AttributeError, IndexError):
                head_hidden_dim = 256
        return {
            "gate_backbone_model": getattr(self.gate, "model_name", "unknown"),
            "hidden_dim": head_hidden_dim,
            "initial_temperature": self.initial_temperature,
            "gumbel_noise_scale": getattr(self.gate, "gumbel_noise_scale", 1.0),
            "max_length": getattr(self.gate, "max_length", 512),
            "extraction_N": self.cache.N,
            "budget_M": getattr(self.cache, "total_budget", 5),
            "use_persona": self.use_persona,
            "inst_regime": self.inst_regime,
            "inference_add_eval": self.inference_add_eval,
            "use_token_total_budget": self.use_token_total_budget,
            "extract_direction": getattr(self.cache, "extract_direction", True),
            "safety": getattr(self.cache, "safety", False),
            "use_alignment_adv": self.use_alignment_adv,
            "use_richness": self.use_richness,
            "richness_alpha": self.richness_alpha,
            "richness_weight": self.richness_weight,
            "use_kl": self.use_kl,
            "kl_weight": self.kl_weight,
            "use_clip": self.use_clip,
            "clip_epsilon": self.clip_epsilon,
            "continuous_group_reward": self.continuous_group_reward,
            "no_group_reward": self.no_group_reward,
            "lambda_comp": self.lambda_comp,
            "csa_mode": self.csa_mode,
            "filter_zero_influence": self.filter_zero_influence,
            "uniform_theta": getattr(self.gate, "uniform_theta", False),
            "dominant_schema": getattr(self.gate, "dominant_schema", None),
        }

    def _precompute_cache_for_model(
        self,
        model_name: str,
        dataset_for_cache: list[dict],
        prefix: str,
        verbose: bool,
    ) -> None:
        if verbose:
            print(f"  - {prefix} cache for {model_name}")
        self.cache_pool[model_name].precompute(
            dataset_for_cache,
            prefix=prefix,
            verbose=verbose,
        )

    def _precompute_multi_llm_caches(
        self,
        dataset_for_cache: list[dict],
        prefix: str,
        verbose: bool,
    ) -> None:
        max_workers = max(1, min(len(self._model_names), self.llm_max_concurrency))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(
                executor.map(
                    lambda model_name: self._precompute_cache_for_model(
                        model_name=model_name,
                        dataset_for_cache=dataset_for_cache,
                        prefix=prefix,
                        verbose=verbose,
                    ),
                    self._model_names,
                )
            )

    def _build_dataset_for_cache(self, dataset: list[dict]) -> list[dict]:
        dataset_for_cache = []
        for item in dataset:
            scenario = format_gate_input(item)
            dataset_for_cache.append({
                "id": self._resolve_item_id(item, str(len(dataset_for_cache))),
                "context": scenario,
            })
        return dataset_for_cache

    def _precompute_influence_vectors_for_cache(
        self,
        *,
        dataset: list[dict],
        llm: LLMClient,
        cache: ExtractionCache,
        model_name: str,
        prefix: str = "train",
        verbose: bool,
    ) -> None:
        if prefix != "train":
            return
        if not dataset:
            return

        from src.runs.sieve_preprocess_influence import (
            compute_influence_record,
            scenario_matches_label,
            summarize,
        )

        def process_item(task: tuple[int, dict]) -> dict:
            idx, item = task
            raw_sid = self._resolve_item_id(item, str(idx))
            scenario_id = cache.build_cache_key(raw_sid, prefix=prefix)
            extraction_scenario = format_gate_input(item)
            answer_scenario = format_full_input(item)
            phase2 = cache.get_phase2(extraction_scenario, scenario_id)
            influence = cache.get_influence_record(extraction_scenario, scenario_id)
            was_cached = influence is not None
            if influence is None:
                influence = compute_influence_record(
                    llm=llm,
                    phase2=phase2,
                    scenario=answer_scenario,
                    gold_label=item["label"],
                    max_gen_len=getattr(llm, "max_gen_len", None) or 2048,
                    safety=getattr(cache, "safety", False),
                )
                cache.save_influence_record(extraction_scenario, scenario_id, influence)

            vector = influence_vector_by_schema(influence.get("influence_vector"))
            if vector is None:
                raise ValueError(f"Invalid influence vector for {scenario_id}: {influence}")

            return {
                "id": raw_sid,
                "_idx": idx,
                "label": item.get("label"),
                "_was_cached": was_cached,
                **influence,
                "influence_vector": vector,
                "full_correct": scenario_matches_label(influence["a_full"], item["label"]),
                "no_pi_correct": scenario_matches_label(influence["a_no_pi"], item["label"]),
                "no_mn_correct": scenario_matches_label(influence["a_no_mn"], item["label"]),
                "no_pc_correct": scenario_matches_label(influence["a_no_pc"], item["label"]),
                "only_pi_correct": scenario_matches_label(influence["a_only_pi"], item["label"]),
                "only_mn_correct": scenario_matches_label(influence["a_only_mn"], item["label"]),
                "only_pc_correct": scenario_matches_label(influence["a_only_pc"], item["label"]),
            }

        max_workers = max(
            1,
            min(
                int(getattr(cache, "max_concurrency", self.llm_max_concurrency) or 1),
                len(dataset),
            ),
        )
        if verbose:
            print(
                f"Pre-computing missing influence vectors for {len(dataset)} "
                f"training scenarios ({model_name}, workers={max_workers})..."
            )

        records = []
        zero_vector_count = 0
        nonzero_vector_count = 0
        computed_count = 0
        tasks = list(enumerate(dataset))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            progress = tqdm(
                executor.map(process_item, tasks),
                total=len(tasks),
                desc=f"Precomputing Influence ({model_name})",
                disable=not verbose,
                dynamic_ncols=True,
            )
            for record in progress:
                records.append(record)
                if not record.pop("_was_cached", False):
                    computed_count += 1
                if record["influence_is_zero"]:
                    zero_vector_count += 1
                else:
                    nonzero_vector_count += 1
                progress.set_postfix(
                    zero_vectors=zero_vector_count,
                    nonzero_vectors=nonzero_vector_count,
                    computed=computed_count,
                )

        metadata = summarize(records)
        metadata["source"] = "trainer_precompute"
        metadata["model_name"] = model_name
        metadata["prefix"] = prefix
        metadata["computed_missing_influence_count"] = computed_count
        metadata["cache_dir"] = str(cache.cache_dir)
        cache.save_metadata(metadata)
        if verbose:
            print(
                f"Influence precompute complete for {model_name}: "
                f"computed_missing={computed_count}, "
                f"zero={metadata['zero_influence_count']}/{metadata['total_scenarios']} "
                f"({metadata['zero_influence_ratio']:.4f})"
            )

    def _precompute_continuous_csa_train_cache(
        self,
        dataset: list[dict],
        *,
        verbose: bool,
    ) -> None:
        if self.csa_mode != "continuous":
            return
        dataset_for_cache = self._build_dataset_for_cache(dataset)

        if verbose:
            print(
                f"Pre-computing extractions for {len(dataset)} continuous-CSA "
                "training scenarios..."
            )
        self.cache.precompute(dataset_for_cache, prefix="train", verbose=verbose)
        self._precompute_influence_vectors_for_cache(
            dataset=dataset,
            llm=self.llm,
            cache=self.cache,
            model_name=getattr(self.llm, "model_name", "single_llm"),
            prefix="train",
            verbose=verbose,
        )

        if not self.multi_llm:
            return
        self._precompute_multi_llm_caches(
            dataset_for_cache,
            prefix="train",
            verbose=verbose,
        )
        for model_name in self._model_names:
            llm_client, cache_client, resolved_model_name = self._resolve_llm_and_cache(model_name)
            self._precompute_influence_vectors_for_cache(
                dataset=dataset,
                llm=llm_client,
                cache=cache_client,
                model_name=resolved_model_name,
                prefix="train",
                verbose=verbose,
            )

    def _refill_model_pool(self) -> None:
        if not self.multi_llm:
            return
        refreshed = list(self._model_names)
        self._multi_llm_rng.shuffle(refreshed)
        self._model_pool.extend(refreshed)

    def _reset_model_pool(self) -> None:
        if not self.multi_llm:
            return
        self._model_pool = []
        self._refill_model_pool()

    def _assign_models_to_batch(self, batch_size: int) -> list[str | None]:
        if not self.multi_llm:
            return [None] * batch_size

        assignments: list[str] = []
        for _ in range(batch_size):
            if not self._model_pool:
                self._refill_model_pool()
            assignments.append(self._model_pool.pop())
        return assignments

    def _resolve_llm_and_cache(
        self,
        model_name: str | None,
    ) -> tuple[LLMClient, ExtractionCache, str]:
        if not self.multi_llm:
            return self.llm, self.cache, getattr(self.llm, "model_name", "single_llm")
        if model_name is None:
            raise ValueError("model_name must be provided when multi_llm=True.")
        return self.llm_pool[model_name], self.cache_pool[model_name], model_name

    def _sample_stratified(self, scenario: str) -> list:
        return self.gate.sample(scenario, k=self.k)

    def _sample_batch_stratified(self, scenarios: list[str]) -> list[list]:
        return self.gate.sample_batch(scenarios, k=self.k)

    def _safe_zscore(self, value: float, values: list[float]) -> float:
        if not values:
            return 0.0
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        std_val = math.sqrt(max(variance, 1e-8))
        return (value - mean_val) / (std_val)

    def _encode_alignment_texts(self, texts: list[str]) -> torch.Tensor:
        if self.alignment_encoder is None:
            raise RuntimeError("Alignment encoder is not initialized.")
        with self.alignment_lock:
            embeddings = self.alignment_encoder.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
        if embeddings.device != self.gate.device:
            embeddings = embeddings.to(self.gate.device)
        return embeddings.float()

    def _compute_cross_schema_similarities(
        self,
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
            left = torch.tensor(left_embedding, device=self.gate.device, dtype=torch.float32)
            right = torch.tensor(right_embedding, device=self.gate.device, dtype=torch.float32)
            denom = left.norm().clamp_min(1e-8) * right.norm().clamp_min(1e-8)
            similarities[metric_name] = float(((left @ right) / denom).item())
        return similarities

    @staticmethod
    def _average_metric_dicts(
        metric_dicts: list[dict[str, float | None]],
    ) -> dict[str, float]:
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

    def _compute_alignment_scores(
        self,
        responses: list[str],
        theta_lists: list[list[float]],
        schema_embeddings_batch: list[dict[str, list[float]]],
    ) -> list[float]:
        if not responses:
            return []

        response_embeddings = self._encode_alignment_texts([response or "" for response in responses])

        scores: list[float] = []
        per_schema_similarities = {schema: [] for schema in SCHEMA_NAMES}
        for response_embedding, theta_list, schema_embeddings in zip(
            response_embeddings,
            theta_lists,
            schema_embeddings_batch,
        ):
            schema_matrix = torch.tensor(
                [schema_embeddings[schema] for schema in SCHEMA_NAMES],
                device=self.gate.device,
                dtype=response_embedding.dtype,
            )
            response_norm = response_embedding.norm().clamp_min(1e-8)
            schema_norm = schema_matrix.norm(dim=-1).clamp_min(1e-8)
            cosine_sims = (schema_matrix @ response_embedding) / (schema_norm * response_norm)
            sims = (cosine_sims + 1.0) / 2.0
            for schema, sim_value in zip(SCHEMA_NAMES, sims.detach().cpu().tolist()):
                per_schema_similarities[schema].append(float(sim_value))
            sim_sum = float(sims.sum().item())
            if sim_sum < 1e-8:
                sim_dist = torch.full_like(sims, 1.0 / len(SCHEMA_NAMES))
            else:
                sim_dist = sims / sim_sum
            theta_tensor = torch.tensor(
                theta_list,
                device=sim_dist.device,
                dtype=sim_dist.dtype,
            )
            distance = torch.abs(sim_dist - theta_tensor).sum() / 2.0
            scores.append(float((1.0 - distance).item()))
        self._last_schema_similarity_means = {
            schema: (
                sum(values) / len(values)
                if values
                else None
            )
            for schema, values in per_schema_similarities.items()
        }
        return scores

    def _compute_alignment_advantages(
        self,
        responses: list[str],
        theta_lists: list[list[float]],
        schema_embeddings_batch: list[dict[str, list[float]]],
    ) -> tuple[list[float], list[float]]:
        alignment_scores = self._compute_alignment_scores(
            responses,
            theta_lists,
            schema_embeddings_batch,
        )
        if not alignment_scores:
            return [], []
        alignment_advantages = [
            self._safe_zscore(score, alignment_scores)
            for score in alignment_scores
        ]
        return alignment_scores, alignment_advantages

    def _compute_richness_scores(
        self,
        responses: list[str],
        schema_embeddings_batch: list[dict[str, list[float]]],
    ) -> list[float]:
        if not responses:
            return []

        response_embeddings = self._encode_alignment_texts([response or "" for response in responses])
        scores: list[float] = []
        per_schema_similarities = {schema: [] for schema in SCHEMA_NAMES}
        for response_embedding, schema_embeddings in zip(
            response_embeddings,
            schema_embeddings_batch,
        ):
            available_schemas = [
                schema for schema in SCHEMA_NAMES if schema_embeddings.get(schema)
            ]
            available_embeddings = [schema_embeddings[schema] for schema in available_schemas]
            if not available_embeddings:
                scores.append(0.0)
                continue
            schema_matrix = torch.tensor(
                available_embeddings,
                device=self.gate.device,
                dtype=response_embedding.dtype,
            )
            response_norm = response_embedding.norm().clamp_min(1e-8)
            schema_norm = schema_matrix.norm(dim=-1).clamp_min(1e-8)
            cosine_sims = (schema_matrix @ response_embedding) / (schema_norm * response_norm)
            sims = ((cosine_sims + 1.0) / 2.0).clamp(0.0, 1.0)
            for schema, sim_value in zip(available_schemas, sims.detach().cpu().tolist()):
                per_schema_similarities[schema].append(float(sim_value))
            scores.append(float(sims.mean().item()))
        self._last_schema_similarity_means = {
            schema: (
                sum(values) / len(values)
                if values
                else None
            )
            for schema, values in per_schema_similarities.items()
        }
        return scores

    def _compute_richness_advantages(
        self,
        responses: list[str],
        schema_embeddings_batch: list[dict[str, list[float]]],
    ) -> tuple[list[float], list[float]]:
        richness_scores = self._compute_richness_scores(
            responses,
            schema_embeddings_batch,
        )
        if not richness_scores:
            return [], []
        richness_advantages = [
            self._safe_zscore(score, richness_scores)
            for score in richness_scores
        ]
        return richness_scores, richness_advantages

    def _normalize_schema_sensitivity(self, sigma_sq: float) -> float:
        return min(max(sigma_sq / SCHEMA_SENSITIVITY_MAX, 0.0), 1.0)

    def _resolve_item_influence_vector(
        self,
        item: dict,
        *,
        scenario: str,
        scenario_id: str,
        cache: ExtractionCache | None = None,
    ) -> list[float]:
        direct_vector = coerce_influence_vector(item.get("influence_vector"))
        if direct_vector is not None:
            return direct_vector

        active_cache = cache or self.cache
        influence_record = active_cache.get_influence_record(
            scenario=scenario,
            scenario_id=scenario_id,
        )
        if influence_record is not None:
            vector = coerce_influence_vector(influence_record.get("influence_vector"))
            if vector is not None:
                return vector

        raise ValueError(
            "csa_mode='continuous' requires a current influence_vector for "
            f"scenario_id={scenario_id!r}. Run influence preprocessing for cache_dir="
            f"{active_cache.cache_dir}."
        )

    def _attach_influence_vectors(
        self,
        dataset: list[dict],
        *,
        prefix: str = "train",
    ) -> list[dict]:
        attached = []
        for idx, item in enumerate(dataset):
            scenario = format_gate_input(item)
            raw_sid = self._resolve_item_id(item, str(idx))
            scenario_id = self.cache.build_cache_key(raw_sid, prefix=prefix)
            vector = self._resolve_item_influence_vector(
                item,
                scenario=scenario,
                scenario_id=scenario_id,
                cache=self.cache,
            )
            enriched = dict(item)
            enriched["influence_vector"] = influence_vector_by_schema(vector)
            enriched["influence_is_zero"] = all(value == 0 for value in vector)
            attached.append(enriched)
        return attached

    def _filter_zero_influence_data(
        self,
        dataset: list[dict],
        *,
        indices: list[int] | None = None,
        verbose: bool,
    ) -> tuple[list[dict], list[int] | None]:
        if self.csa_mode != "continuous":
            return dataset, indices

        enriched = self._attach_influence_vectors(dataset, prefix="train")
        enriched_indices = list(indices) if indices is not None else None
        zero_count = sum(
            1 for item in enriched if bool(item.get("influence_is_zero", False))
        )
        kept = []
        kept_indices = [] if enriched_indices is not None else None
        for item_idx, item in enumerate(enriched):
            should_filter = (
                self.filter_zero_influence
                and bool(item.get("influence_is_zero", False))
            )
            if should_filter:
                continue
            kept.append(item)
            if kept_indices is not None and enriched_indices is not None:
                kept_indices.append(enriched_indices[item_idx])
        total_count = len(enriched)
        zero_ratio = zero_count / max(1, total_count)
        metadata = self.cache.load_metadata()
        if verbose:
            print("=" * 60)
            print("SIEVE INFLUENCE METADATA")
            print("=" * 60)
            print(f"cache_dir: {self.cache.cache_dir}")
            if metadata:
                print(
                    "preprocess zero_influence_ratio: "
                    f"{metadata.get('zero_influence_ratio', 'unknown')}"
                )
                print(
                    "preprocess zero_influence_count/total: "
                    f"{metadata.get('zero_influence_count', 'unknown')}/"
                    f"{metadata.get('total_scenarios', 'unknown')}"
                )
            else:
                print("preprocess metadata: missing meta_data.json")
            print(
                "training zero_influence_count/total: "
                f"{zero_count}/{total_count} ({zero_ratio:.4f})"
            )
            print(f"filter_zero_influence: {self.filter_zero_influence}")
            print(f"training data after influence filtering: {len(kept)}")
            print("=" * 60)
        return kept, kept_indices

    def _compute_continuous_csa_advantages(
        self,
        *,
        rewards_acc: list[float],
        theta_lists: list[list[float]],
        influence_vector: list[float],
    ) -> dict[str, Any]:
        theta_samples = torch.tensor(
            theta_lists,
            device=self.gate.device,
            dtype=torch.float32,
        )
        reward_acc = torch.tensor(
            rewards_acc,
            device=self.gate.device,
            dtype=torch.float32,
        )
        influence = torch.tensor(
            influence_vector,
            device=self.gate.device,
            dtype=torch.float32,
        )
        reward_comp = (theta_samples * influence.unsqueeze(0)).sum(dim=-1)
        reward_total = reward_acc + (self.lambda_comp * reward_comp)

        theta_detached = theta_samples.detach()
        numerator = (theta_detached * reward_total.unsqueeze(-1)).sum(dim=0)
        denominator = theta_detached.sum(dim=0).clamp_min(1e-8)
        baselines = numerator / denominator
        y_hat = (theta_detached * baselines.unsqueeze(0)).sum(dim=-1)
        advantages = (reward_total - y_hat).detach()

        return {
            "advantages": advantages.detach().cpu().tolist(),
            "reward_accs": reward_acc.detach().cpu().tolist(),
            "reward_comps": reward_comp.detach().cpu().tolist(),
            "reward_totals": reward_total.detach().cpu().tolist(),
            "baselines": baselines.detach().cpu().tolist(),
            "y_hat": y_hat.detach().cpu().tolist(),
            "influence_vector": [float(value) for value in influence_vector],
            "per_group_rewards": {
                schema: float(baselines[idx].detach().item())
                for idx, schema in enumerate(SCHEMA_NAMES)
            },
        }

    def _compute_dual_level_advantages(
        self,
        rewards: list[float],
        theta_lists: list[list[float]],
        responses: list[str] | None = None,
        schema_embeddings_batch: list[dict[str, list[float]]] | None = None,
    ) -> tuple[
        list[float],
        float,
        float,
        dict[str, float],
        list[str],
        list[float],
        list[float],
        list[float],
        list[float],
        list[float],
    ]:
        self._last_schema_similarity_means = {schema: None for schema in SCHEMA_NAMES}
        dominant_schemas = [
            self._schema_name_from_theta(theta_list) for theta_list in theta_lists
        ]

        if self.continuous_group_reward:
            schema_reward_values: list[float] = []
            per_group_rewards = {}
            for schema_idx, schema in enumerate(SCHEMA_NAMES):
                weighted_sum = 0.0
                weight_total = 0.0
                for theta_list, reward in zip(theta_lists, rewards):
                    if schema_idx >= len(theta_list):
                        continue
                    weight = max(0.0, float(theta_list[schema_idx]))
                    weighted_sum += weight * float(reward)
                    weight_total += weight
                schema_reward = weighted_sum / max(weight_total, 1e-8)
                per_group_rewards[schema] = schema_reward
                schema_reward_values.append(schema_reward)

            reward_mean = sum(schema_reward_values) / max(1, len(schema_reward_values))
            sigma_sq = sum(
                (value - reward_mean) ** 2 for value in schema_reward_values
            ) / max(1, len(schema_reward_values))

            theta_means = [
                sum(
                    float(theta_list[schema_idx]) if schema_idx < len(theta_list) else 0.0
                    for theta_list in theta_lists
                ) / max(1, len(theta_lists))
                for schema_idx in range(len(SCHEMA_NAMES))
            ]
            direction_advantages = []
            for theta_list in theta_lists:
                direction_advantages.append(
                    sum(
                        (
                            (float(theta_list[schema_idx]) if schema_idx < len(theta_list) else 0.0)
                            - theta_means[schema_idx]
                        )
                        * (schema_reward_values[schema_idx] - reward_mean)
                        for schema_idx in range(len(SCHEMA_NAMES))
                    )
                )
            schema_advantages = [
                self._safe_zscore(direction_adv, direction_advantages)
                for direction_adv in direction_advantages
            ]
        else:
            group_rewards = {schema: [] for schema in SCHEMA_NAMES}
            for schema, reward in zip(dominant_schemas, rewards):
                group_rewards[schema].append(reward)

            active_reward_means = [
                sum(vals) / len(vals)
                for vals in group_rewards.values()
                if vals
            ]
            per_group_rewards = {
                schema: (sum(vals) / len(vals) if vals else 0.0)
                for schema, vals in group_rewards.items()
            }

            if len(active_reward_means) <= 1:
                sigma_sq = 0.0
            else:
                reward_mean = sum(active_reward_means) / len(active_reward_means)
                sigma_sq = sum(
                    (value - reward_mean) ** 2 for value in active_reward_means
                ) / len(active_reward_means)

            stat_values = active_reward_means or [0.0]
            schema_advantages = [
                self._safe_zscore(per_group_rewards[schema], stat_values)
                for schema in dominant_schemas
            ]

        sample_advantages = [self._safe_zscore(reward, rewards) for reward in rewards]
        alignment_scores: list[float] = []
        alignment_advantages = [0.0 for _ in rewards]
        combined_sample_advantages = list(sample_advantages)
        if (
            self._use_richness_reward()
            and responses is not None
            and schema_embeddings_batch is not None
        ):
            alignment_scores, alignment_advantages = self._compute_richness_advantages(
                responses,
                schema_embeddings_batch,
            )
            if alignment_advantages:
                combined_sample_advantages = [
                    ((1.0 - self.richness_weight) * sample_adv)
                    + (self.richness_weight * richness_adv)
                    for sample_adv, richness_adv in zip(
                        sample_advantages,
                        alignment_advantages,
                    )
                ]
        elif (
            self.use_alignment_adv
            and responses is not None
            and schema_embeddings_batch is not None
        ):
            alignment_scores, alignment_advantages = self._compute_alignment_advantages(
                responses,
                theta_lists,
                schema_embeddings_batch,
            )
            if alignment_advantages:
                combined_sample_advantages = [
                    (0.5 * sample_adv) + (0.5 * alignment_adv)
                    for sample_adv, alignment_adv in zip(
                        sample_advantages,
                        alignment_advantages,
                    )
                ]
        if self.no_group_reward:
            normalized_sigma_sq = 0.0
            lambda_weight = 0.0
            schema_advantages = [0.0 for _ in rewards]
            advantages = list(combined_sample_advantages)
        else:
            normalized_sigma_sq = self._normalize_schema_sensitivity(sigma_sq)
            lambda_weight = normalized_sigma_sq
            advantages = [
                (lambda_weight * schema_adv) + ((1.0 - lambda_weight) * sample_adv)
                for schema_adv, sample_adv in zip(
                    schema_advantages,
                    combined_sample_advantages,
                )
            ]
        self.rho = (
            self.rho_momentum * self.rho
        ) + ((1.0 - self.rho_momentum) * normalized_sigma_sq)

        return (
            advantages,
            sigma_sq,
            lambda_weight,
            per_group_rewards,
            dominant_schemas,
            schema_advantages,
            sample_advantages,
            alignment_scores,
            alignment_advantages,
            combined_sample_advantages,
        )

    def _run_sample(
        self,
        scenario: str,
        scenario_id: str,
        gold_label: str,
        theta_list: list[float],
        gold_principles: list[str] | None = None,
        llm: LLMClient | None = None,
        cache: ExtractionCache | None = None,
        extraction_scenario: str | None = None,
    ) -> tuple[float, str, str, int, str, dict[str, int], dict[str, Any] | None]:
        active_llm = llm or self.llm
        active_cache = cache or self.cache
        cache_key = active_cache.build_cache_key(scenario_id, prefix="train")
        extraction_input = extraction_scenario if extraction_scenario is not None else scenario
        
        # for label, theta in [("PI", [0.9,0.05,0.05]), ("MN", [0.05, 0.9, 0.05]), ("PC", [0.05,0.05,0.9])]:
        #     phase3 = cache.get_phase3(scenario, cache_key, theta=theta)
        #     prompt = assemble_prompt(scenario, phase3)
        #     llm_output = active_llm.generate_with_metadata(
        #         prompt,
        #         max_tokens=1024,
        #         temperature=1.0,
        #     )
        #     response = llm_output.get("text", "")
        #     print(f"=== {label} dominant ===")
        #     print(prompt)
        #     print()
        #     print("[Response]")
        #     print(response)
        #     print("==="*50)
        #     print()
        # import pdb; pdb.set_trace()
        
        phase3 = active_cache.get_phase3(
            extraction_input,
            cache_key,
            theta=theta_list,
        )
        prompt = assemble_prompt(
            scenario,
            phase3,
            use_persona=self.use_persona,
            inst_regime=self.inst_regime,
            inference_add_eval=self.inference_add_eval,
            token_proportional=True,
            use_token_total_budget=getattr(self, "use_token_total_budget", False),
            tokenizer=self._resolve_prompt_tokenizer(active_llm),
            safety=getattr(active_cache, "safety", False),
        )
        llm_output = active_llm.generate_with_metadata(
            prompt,
            max_tokens=1024,
            temperature=0.0,
        )
        # print("[Prompt]")
        # print(prompt)
        # print("[Output]")
        # print(llm_output)
        # import pdb; pdb.set_trace()
        response = llm_output.get("text", "")
        answer = parse_answer(response)
        safety_judgment = None
        if self._uses_safety_reward(active_cache):
            reward, safety_judgment = self._compute_safety_reward(
                scenario=scenario,
                response=response,
                gold_label=gold_label,
            )
            is_correct = reward >= 1.0
        else:
            is_correct = self._scenario_matches_label(answer, gold_label)
            principle_coverage = self._estimate_principle_coverage(
                response,
                gold_principles or [],
            )
            reward = self._compute_reward(
                is_correct=is_correct,
                principle_coverage=principle_coverage,
            )
        reasoning_length = int(
            llm_output.get("output_tokens")
            or llm_output.get("completion_tokens")
            or max(1, len(response.split()))
        )
        schema_item_counts = {
            phase3.primary_schema: int(phase3.primary_budget),
            phase3.secondary_schema: int(phase3.secondary_budget),
            phase3.tertiary_schema: int(phase3.tertiary_budget),
        }
        return (
            reward,
            answer,
            response,
            reasoning_length,
            prompt,
            schema_item_counts,
            safety_judgment,
        )

    def _train_batch(self, items: list[tuple[int, dict]]) -> list[TrainStepResult]:
        gate_scenarios = [
            format_gate_input(item)
            for _, item in items
        ]
        extraction_scenarios = list(gate_scenarios)
        llm_scenarios = [
            format_full_input(item)
            for _, item in items
        ]
        scenario_ids = [self._resolve_item_id(item, str(idx)) for idx, item in items]
        gold_labels = [item["label"] for _, item in items]
        gold_principles_batch = [self._extract_gold_principles(item) for _, item in items]
        samples_batch = self._sample_batch_stratified(gate_scenarios)
        with torch.no_grad():
            single_theta_batch, _ = self.gate.forward_batch(gate_scenarios)
        assigned_models = self._assign_models_to_batch(len(items))
        llm_cache_batch = [
            self._resolve_llm_and_cache(model_name)
            for model_name in assigned_models
        ]

        tasks = []
        single_tasks = []
        for batch_idx, samples in enumerate(samples_batch):
            llm_client, cache_client, llm_model_name = llm_cache_batch[batch_idx]
            single_tasks.append(
                (
                    batch_idx,
                    llm_scenarios[batch_idx],
                    scenario_ids[batch_idx],
                    gold_labels[batch_idx],
                    single_theta_batch[batch_idx].detach().tolist(),
                    gold_principles_batch[batch_idx],
                    llm_client,
                    cache_client,
                    llm_model_name,
                )
            )
            for sample_idx, sample in enumerate(samples):
                tasks.append(
                    (
                        batch_idx,
                        sample_idx,
                        llm_scenarios[batch_idx],
                        scenario_ids[batch_idx],
                        gold_labels[batch_idx],
                        sample.theta.detach().tolist(),
                        gold_principles_batch[batch_idx],
                        llm_client,
                        cache_client,
                        llm_model_name,
                    )
            )

        all_tasks = [("sample", task) for task in tasks] + [
            ("single", task) for task in single_tasks
        ]
        max_workers = max(1, min(self.llm_max_concurrency, len(all_tasks)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            all_outputs = list(
                executor.map(
                    lambda typed_task: self._run_sample(
                        scenario=typed_task[1][2] if typed_task[0] == "sample" else typed_task[1][1],
                        scenario_id=typed_task[1][3] if typed_task[0] == "sample" else typed_task[1][2],
                        gold_label=typed_task[1][4] if typed_task[0] == "sample" else typed_task[1][3],
                        theta_list=typed_task[1][5] if typed_task[0] == "sample" else typed_task[1][4],
                        gold_principles=typed_task[1][6] if typed_task[0] == "sample" else typed_task[1][5],
                        llm=typed_task[1][7] if typed_task[0] == "sample" else typed_task[1][6],
                        cache=typed_task[1][8] if typed_task[0] == "sample" else typed_task[1][7],
                        extraction_scenario=extraction_scenarios[
                            typed_task[1][0]
                        ],
                    ),
                    all_tasks,
                )
            )

        grouped_outputs = [[] for _ in items]
        single_outputs = [None for _ in items]
        for typed_task, output in zip(all_tasks, all_outputs):
            kind, task = typed_task
            batch_idx = task[0]
            if kind == "sample":
                grouped_outputs[batch_idx].append((task[1], output))
            else:
                single_outputs[batch_idx] = output
        for batch_idx in range(len(grouped_outputs)):
            grouped_outputs[batch_idx].sort(key=lambda x: x[0])

        avg_sample_theta_entropy = self._compute_avg_sample_theta_entropy(samples_batch)
        entropy_value = float(avg_sample_theta_entropy.item())
        theta_entropy_penalty = self._compute_theta_entropy_penalty(samples_batch)
        theta_entropy_penalty_value = float(theta_entropy_penalty.item())
        kl_term = (
            self._compute_reference_kl(gate_scenarios)
            if self.use_kl
            else torch.tensor(0.0, device=self.gate.device)
        )
        kl_value = float(kl_term.detach().item())
        current_logits_batch = None
        if self.use_clip:
            _, current_logits_batch = self.gate.forward_batch(gate_scenarios)
        per_scenario_losses = []
        step_payloads = []
        schema_sensitivities = []
        for batch_idx, outputs_for_scenario in enumerate(grouped_outputs):
            _, source_item = items[batch_idx]
            single_output = single_outputs[batch_idx]
            single_answer = single_output[1] if single_output is not None else ""
            use_safety_reward = self._uses_safety_reward(llm_cache_batch[batch_idx][1])
            single_correct = (
                bool(single_output is not None and single_output[0] >= 1.0)
                if use_safety_reward
                else self._scenario_matches_label(
                    single_answer,
                    gold_labels[batch_idx],
                )
            )
            rewards = [reward for _, (reward, _, _, _, _, _, _) in outputs_for_scenario]
            answers = [answer for _, (_, answer, _, _, _, _, _) in outputs_for_scenario]
            raw_responses = [response for _, (_, _, response, _, _, _, _) in outputs_for_scenario]
            reasoning_lengths = [length for _, (_, _, _, length, _, _, _) in outputs_for_scenario]
            prompts = [prompt for _, (_, _, _, _, prompt, _, _) in outputs_for_scenario]
            schema_item_counts_batch = [
                schema_item_counts
                for _, (_, _, _, _, _, schema_item_counts, _) in outputs_for_scenario
            ]
            safety_judgments = [
                safety_judgment
                for _, (_, _, _, _, _, _, safety_judgment) in outputs_for_scenario
            ]
            full_schema_embeddings = llm_cache_batch[batch_idx][1].get_full_schema_embeddings(
                extraction_scenarios[batch_idx],
                llm_cache_batch[batch_idx][1].build_cache_key(
                    scenario_ids[batch_idx],
                    prefix="train",
                ),
            )
            cross_schema_similarities = self._compute_cross_schema_similarities(
                full_schema_embeddings
            )
            correctness_flags = (
                [float(reward) >= 1.0 for reward in rewards]
                if use_safety_reward
                else [
                    self._scenario_matches_label(answer, gold_labels[batch_idx])
                    for answer in answers
                ]
            )
            strict_accuracy = sum(correctness_flags) / max(1, len(correctness_flags))
            theta_lists = [
                sample.theta.detach().tolist()
                for sample in samples_batch[batch_idx]
            ]
            schema_match = self._schema_match_mean_from_thetas(
                theta_lists,
                source_item,
            )
            single_schema_match = self._schema_match_from_theta(
                single_theta_batch[batch_idx].detach().tolist(),
                source_item,
            )
            reward_accs = [float(flag) for flag in correctness_flags]
            reward_comps = [0.0 for _ in rewards]
            reward_totals = list(rewards)
            csa_baselines = [0.0 for _ in SCHEMA_NAMES]
            csa_y_hat = [0.0 for _ in rewards]
            influence_vector = None
            if self.csa_mode == "continuous":
                influence_vector = self._resolve_item_influence_vector(
                    source_item,
                    scenario=extraction_scenarios[batch_idx],
                    scenario_id=llm_cache_batch[batch_idx][1].build_cache_key(
                        scenario_ids[batch_idx],
                        prefix="train",
                    ),
                    cache=llm_cache_batch[batch_idx][1],
                )
                csa_payload = self._compute_continuous_csa_advantages(
                    rewards_acc=reward_accs,
                    theta_lists=theta_lists,
                    influence_vector=influence_vector,
                )
                advantages = csa_payload["advantages"]
                sigma_sq = 0.0
                lambda_weight = self.lambda_comp
                per_group_rewards = csa_payload["per_group_rewards"]
                dominant_schemas = [
                    self._schema_name_from_theta(theta_list) for theta_list in theta_lists
                ]
                schema_advantages = list(advantages)
                sample_advantages = list(advantages)
                alignment_scores = []
                alignment_advantages = [0.0 for _ in rewards]
                combined_sample_advantages = list(advantages)
                reward_accs = csa_payload["reward_accs"]
                reward_comps = csa_payload["reward_comps"]
                reward_totals = csa_payload["reward_totals"]
                csa_baselines = csa_payload["baselines"]
                csa_y_hat = csa_payload["y_hat"]
            else:
                (
                    advantages,
                    sigma_sq,
                    lambda_weight,
                    per_group_rewards,
                    dominant_schemas,
                    schema_advantages,
                    sample_advantages,
                    alignment_scores,
                    alignment_advantages,
                    combined_sample_advantages,
                ) = (
                    self._compute_dual_level_advantages(
                        rewards,
                        theta_lists,
                        responses=raw_responses,
                        schema_embeddings_batch=(
                            [
                                full_schema_embeddings
                                for _ in schema_item_counts_batch
                            ]
                            if self._use_richness_reward()
                            else [
                                llm_cache_batch[batch_idx][1].get_selected_schema_embeddings(
                                    extraction_scenarios[batch_idx],
                                    llm_cache_batch[batch_idx][1].build_cache_key(
                                        scenario_ids[batch_idx],
                                        prefix="train",
                                    ),
                                    schema_item_counts,
                                )
                                for schema_item_counts in schema_item_counts_batch
                            ]
                            if self.use_alignment_adv
                            else None
                        ),
                    )
                )
            avg_advantage = sum(advantages) / max(1, len(advantages))
            schema_sensitivities.append(sigma_sq)
            scenario_loss = torch.tensor(0.0, device=self.gate.device)
            clip_ratios = []
            clip_flags = []
            for sample_idx, _ in outputs_for_scenario:
                advantage = advantages[sample_idx]
                sample = samples_batch[batch_idx][sample_idx]
                if self.use_clip:
                    current_log_prob = self._theta_log_prob_from_logits(
                        current_logits_batch[batch_idx],
                        sample.theta,
                    )
                else:
                    current_log_prob = sample.log_prob
                sample_loss, ratio, clipped = self._policy_sample_loss(
                    current_log_prob=current_log_prob,
                    old_log_prob=sample.log_prob,
                    advantage=advantage,
                )
                scenario_loss = scenario_loss + sample_loss
                clip_ratios.append(ratio)
                clip_flags.append(clipped)
            scenario_loss = scenario_loss / self.k
            per_scenario_losses.append(scenario_loss)
            step_payloads.append(
                {
                    "scenario_id": scenario_ids[batch_idx],
                    "rewards": reward_totals if self.csa_mode == "continuous" else rewards,
                    "loss": scenario_loss.item(),
                    "thetas": theta_lists,
                    "prompts": prompts,
                    "answers": answers,
                    "raw_responses": raw_responses,
                    "gold_label": gold_labels[batch_idx],
                    "best_correct": any(correctness_flags),
                    "single_correct": bool(single_correct),
                    "single_theta": single_theta_batch[batch_idx].detach().tolist(),
                    "schema_match": schema_match,
                    "single_schema_match": single_schema_match,
                    "correctness_flags": correctness_flags,
                    "strict_accuracy": strict_accuracy,
                    "avg_advantage": avg_advantage,
                    "combined_advantages": advantages,
                    "schema_advantages": schema_advantages,
                    "sample_advantages": sample_advantages,
                    "alignment_scores": alignment_scores,
                    "alignment_advantages": alignment_advantages,
                    "schema_similarity_means": dict(self._last_schema_similarity_means),
                    "cross_schema_similarities": cross_schema_similarities,
                    "combined_sample_advantages": combined_sample_advantages,
                    "schema_item_counts": schema_item_counts_batch,
                    "safety_judgments": safety_judgments,
                    "schema_sensitivity": sigma_sq,
                    "lambda_weight": lambda_weight,
                    "dominant_schemas": dominant_schemas,
                    "per_group_rewards": per_group_rewards,
                    "reasoning_lengths": reasoning_lengths,
                    "llm_model_name": llm_cache_batch[batch_idx][2],
                    "entropy": entropy_value,
                    "theta_entropy_penalty": theta_entropy_penalty_value,
                    "kl_term": kl_value,
                    "clip_ratios": clip_ratios,
                    "clip_fraction": (
                        sum(1 for clipped in clip_flags if clipped) / max(1, len(clip_flags))
                    ),
                    "sample_log_probs": [
                        float(samples_batch[batch_idx][sample_idx].log_prob.detach().item())
                        for sample_idx, _ in outputs_for_scenario
                    ],
                    "reward_accs": reward_accs,
                    "reward_comps": reward_comps,
                    "reward_totals": reward_totals,
                    "csa_baselines": csa_baselines,
                    "csa_y_hat": csa_y_hat,
                    "influence_vector": influence_vector,
                    "sample_logits": [
                        samples_batch[batch_idx][sample_idx].logits.detach().cpu().tolist()
                        for sample_idx, _ in outputs_for_scenario
                    ],
                    **self._compute_correctness_conditioned_theta_metrics(
                        theta_lists,
                        correctness_flags,
                    ),
                    "informative": sigma_sq > self.informative_sigma_threshold,
                }
            )
        entropy_active_mask = [
            self.entropy_reg_alpha > 0.0
            for _ in samples_batch
        ]
        if self.entropy_reg_alpha > 0.0:
            entropy_theta_batch, _ = self.gate.forward_batch(gate_scenarios)
        else:
            entropy_theta_batch = single_theta_batch
        entropy_bonus = self._compute_single_theta_entropy_bonus(entropy_theta_batch)
        policy_loss = torch.stack(per_scenario_losses).mean()
        entropy_reg_term = self.entropy_reg_alpha * entropy_bonus
        batch_loss = policy_loss - (self.entropy_reg_alpha * entropy_bonus)
        if self.use_entropy_loss:
            batch_loss = batch_loss + (self.entropy_loss_beta * theta_entropy_penalty)
        if self.use_kl:
            batch_loss = batch_loss + (self.kl_weight * kl_term)
        all_clip_ratios = [
            ratio
            for payload in step_payloads
            for ratio in payload.get("clip_ratios", [])
        ]
        avg_clip_ratio = (
            sum(all_clip_ratios) / len(all_clip_ratios)
            if all_clip_ratios
            else 1.0
        )
        avg_clip_fraction = (
            sum(float(payload.get("clip_fraction", 0.0)) for payload in step_payloads)
            / max(1, len(step_payloads))
        )
        flat_reward_accs = [
            float(value)
            for payload in step_payloads
            for value in payload.get("reward_accs", [])
        ]
        flat_reward_comps = [
            float(value)
            for payload in step_payloads
            for value in payload.get("reward_comps", [])
        ]
        flat_reward_totals = [
            float(value)
            for payload in step_payloads
            for value in payload.get("reward_totals", [])
        ]
        flat_advantages = [
            float(value)
            for payload in step_payloads
            for value in payload.get("combined_advantages", [])
        ]
        flat_y_hat = [
            float(value)
            for payload in step_payloads
            for value in payload.get("csa_y_hat", [])
        ]
        csa_baseline_means = {}
        for schema_idx, schema in enumerate(SCHEMA_NAMES):
            values = [
                float(payload.get("csa_baselines", [0.0] * len(SCHEMA_NAMES))[schema_idx])
                for payload in step_payloads
                if payload.get("csa_baselines")
            ]
            csa_baseline_means[schema] = sum(values) / len(values) if values else None

        def mean_std(values: list[float]) -> tuple[float | None, float | None]:
            if not values:
                return None, None
            mean_value = sum(values) / len(values)
            variance = sum((value - mean_value) ** 2 for value in values) / len(values)
            return mean_value, math.sqrt(max(variance, 0.0))

        def pearson(left: list[float], right: list[float]) -> float | None:
            if len(left) != len(right) or len(left) < 2:
                return None
            left_mean, _ = mean_std(left)
            right_mean, _ = mean_std(right)
            if left_mean is None or right_mean is None:
                return None
            left_centered = [value - left_mean for value in left]
            right_centered = [value - right_mean for value in right]
            denom_left = math.sqrt(sum(value * value for value in left_centered))
            denom_right = math.sqrt(sum(value * value for value in right_centered))
            denom = denom_left * denom_right
            if denom <= 1e-8:
                return None
            return sum(lv * rv for lv, rv in zip(left_centered, right_centered)) / denom

        reward_acc_mean, reward_acc_std = mean_std(flat_reward_accs)
        reward_comp_mean, reward_comp_std = mean_std(flat_reward_comps)
        reward_total_mean, reward_total_std = mean_std(flat_reward_totals)
        y_hat_mean, y_hat_std = mean_std(flat_y_hat)
        advantage_mean, advantage_std = mean_std(flat_advantages)
        theta_base = single_theta_batch.detach().clamp_min(1e-8)
        theta_base_entropy = (-(theta_base * theta_base.log()).sum(dim=-1)).mean()
        theta_base_max = theta_base.max(dim=-1).values.mean()
        theta_base_dominant = theta_base.argmax(dim=-1)
        dominant_schema_ratios = {
            schema: float((theta_base_dominant == schema_idx).float().mean().item())
            for schema_idx, schema in enumerate(SCHEMA_NAMES)
        }
        influence_vectors = [
            payload.get("influence_vector")
            for payload in step_payloads
            if payload.get("influence_vector") is not None
        ]
        influence_metrics: dict[str, float | None] = {}
        if influence_vectors:
            influence_tensor = torch.tensor(
                influence_vectors,
                device=self.gate.device,
                dtype=torch.float32,
            )
            for schema_idx, schema in enumerate(SCHEMA_NAMES):
                nonzero_mask = influence_tensor[:, schema_idx] != 0
                positive_mask = influence_tensor[:, schema_idx] > 0
                negative_mask = influence_tensor[:, schema_idx] < 0
                influence_metrics[f"influence/coverage_{schema}"] = float(
                    nonzero_mask.float().mean().item()
                )
                influence_metrics[f"influence/alignment_{schema}"] = (
                    float(theta_base[positive_mask, schema_idx].mean().item())
                    if bool(positive_mask.any())
                    else None
                )
                influence_metrics[f"influence/misalignment_{schema}"] = (
                    float(theta_base[negative_mask, schema_idx].mean().item())
                    if bool(negative_mask.any())
                    else None
                )
        self.last_train_batch_monitor = {
            "policy_loss": float(policy_loss.detach().item()),
            "policy_loss_abs": float(policy_loss.detach().abs().item()),
            "entropy_reg_term": float(entropy_reg_term.detach().item()),
            "kl_term": kl_value,
            "clip_ratio": float(avg_clip_ratio),
            "clip_fraction": float(avg_clip_fraction),
            "entropy_reg_term_abs": float(entropy_reg_term.detach().abs().item()),
            "entropy_reg_term_dominates": bool(
                entropy_reg_term.detach().abs().item() > policy_loss.detach().abs().item()
            ),
            "entropy_reg_active_scenarios": int(sum(entropy_active_mask)),
            "entropy_reg_total_scenarios": int(len(entropy_active_mask)),
            "reward/R_acc_mean": reward_acc_mean,
            "reward/R_acc_std": reward_acc_std,
            "reward/R_comp_mean": reward_comp_mean,
            "reward/R_comp_std": reward_comp_std,
            "reward/R_total_mean": reward_total_mean,
            "reward/R_total_std": reward_total_std,
            "reward/correlation_acc_comp": pearson(flat_reward_accs, flat_reward_comps),
            "csa/b_PI": csa_baseline_means["PI"],
            "csa/b_MN": csa_baseline_means["MN"],
            "csa/b_PC": csa_baseline_means["PC"],
            "csa/y_hat_mean": y_hat_mean,
            "csa/y_hat_std": y_hat_std,
            "csa/advantage_mean": advantage_mean,
            "csa/advantage_std": advantage_std,
            "csa/advantage_max_abs": (
                max(abs(value) for value in flat_advantages)
                if flat_advantages
                else None
            ),
            "gate/theta_base_entropy": float(theta_base_entropy.item()),
            "gate/theta_base_max": float(theta_base_max.item()),
            **{
                f"gate/dominant_schema_ratio_{schema}": ratio
                for schema, ratio in dominant_schema_ratios.items()
            },
            **influence_metrics,
        }
        self.debug_last_train_batch_state = {
            "batch_items": [
                {
                    "dataset_idx": idx,
                    "scenario_id": scenario_ids[batch_idx],
                    "context": item.get("context"),
                    "question": item.get("question"),
                    "gold_label": item.get("label"),
                    "gold_principles": gold_principles_batch[batch_idx],
                    "assigned_llm_model": llm_cache_batch[batch_idx][2],
                    "thetas": step_payloads[batch_idx]["thetas"],
                    "prompts": step_payloads[batch_idx]["prompts"],
                    "sample_logits": step_payloads[batch_idx]["sample_logits"],
                    "sample_log_probs": step_payloads[batch_idx]["sample_log_probs"],
                    "rewards": step_payloads[batch_idx]["rewards"],
                    "reward_accs": step_payloads[batch_idx]["reward_accs"],
                    "reward_comps": step_payloads[batch_idx]["reward_comps"],
                    "reward_totals": step_payloads[batch_idx]["reward_totals"],
                    "csa_baselines": step_payloads[batch_idx]["csa_baselines"],
                    "csa_y_hat": step_payloads[batch_idx]["csa_y_hat"],
                    "influence_vector": step_payloads[batch_idx]["influence_vector"],
                    "answers": step_payloads[batch_idx]["answers"],
                    "raw_responses": step_payloads[batch_idx]["raw_responses"],
                    "wildguard_rewards": (
                        step_payloads[batch_idx]["rewards"]
                        if any(
                            item is not None
                            for item in step_payloads[batch_idx]["safety_judgments"]
                        )
                        else None
                    ),
                    "wildguard_outputs": step_payloads[batch_idx]["safety_judgments"],
                    "correctness_flags": step_payloads[batch_idx]["correctness_flags"],
                    "reasoning_lengths": step_payloads[batch_idx]["reasoning_lengths"],
                    "strict_accuracy": step_payloads[batch_idx]["strict_accuracy"],
                    "single_correct": step_payloads[batch_idx]["single_correct"],
                    "single_theta": step_payloads[batch_idx]["single_theta"],
                    "schema_match": step_payloads[batch_idx]["schema_match"],
                    "single_schema_match": step_payloads[batch_idx]["single_schema_match"],
                    "schema_advantages": step_payloads[batch_idx]["schema_advantages"],
                    "sample_advantages": step_payloads[batch_idx]["sample_advantages"],
                    "alignment_scores": step_payloads[batch_idx]["alignment_scores"],
                    "alignment_advantages": step_payloads[batch_idx]["alignment_advantages"],
                    "schema_similarity_means": step_payloads[batch_idx]["schema_similarity_means"],
                    "cross_schema_similarities": step_payloads[batch_idx]["cross_schema_similarities"],
                    "combined_sample_advantages": step_payloads[batch_idx]["combined_sample_advantages"],
                    "schema_item_counts": step_payloads[batch_idx]["schema_item_counts"],
                    "combined_advantages": step_payloads[batch_idx]["combined_advantages"],
                    "avg_advantage": step_payloads[batch_idx]["avg_advantage"],
                    "schema_sensitivity": step_payloads[batch_idx]["schema_sensitivity"],
                    "lambda_weight": step_payloads[batch_idx]["lambda_weight"],
                    "entropy_reg_applied": entropy_active_mask[batch_idx],
                    "kl_term": step_payloads[batch_idx]["kl_term"],
                    "clip_ratios": step_payloads[batch_idx]["clip_ratios"],
                    "clip_fraction": step_payloads[batch_idx]["clip_fraction"],
                    "dominant_schemas": step_payloads[batch_idx]["dominant_schemas"],
                    "per_group_rewards": step_payloads[batch_idx]["per_group_rewards"],
                    "scenario_loss": step_payloads[batch_idx]["loss"],
                    "entropy": step_payloads[batch_idx]["entropy"],
                    "theta_entropy_penalty": step_payloads[batch_idx]["theta_entropy_penalty"],
                    "informative": step_payloads[batch_idx]["informative"],
                }
                for batch_idx, (idx, item) in enumerate(items)
            ],
            "batch_size": len(items),
            "sample_k": self.k,
            "policy_loss": float(policy_loss.detach().item()),
            "entropy_reg_term": float(entropy_reg_term.detach().item()),
            "entropy_bonus": float(entropy_bonus.detach().item()),
            "entropy_reg_active_scenarios": int(sum(entropy_active_mask)),
            "entropy_reg_total_scenarios": int(len(entropy_active_mask)),
            "entropy_reg_alpha": float(self.entropy_reg_alpha),
            "use_alignment_adv": bool(self.use_alignment_adv),
            "use_richness": bool(self.use_richness),
            "richness_alpha": float(self.richness_alpha),
            "richness_weight": float(self.richness_weight),
            "inference_add_eval": bool(self.inference_add_eval),
            "theta_entropy_penalty": float(theta_entropy_penalty.detach().item()),
            "use_entropy_loss": bool(self.use_entropy_loss),
            "entropy_loss_beta": float(self.entropy_loss_beta),
            "use_kl": bool(self.use_kl),
            "kl_weight": float(self.kl_weight),
            "use_clip": bool(self.use_clip),
            "clip_epsilon": float(self.clip_epsilon),
            "continuous_group_reward": bool(self.continuous_group_reward),
            "lambda_comp": float(self.lambda_comp),
            "csa_mode": self.csa_mode,
            "filter_zero_influence": bool(self.filter_zero_influence),
            "no_group_reward": bool(self.no_group_reward),
            "clip_ratio": float(avg_clip_ratio),
            "clip_fraction": float(avg_clip_fraction),
            "batch_loss": float(batch_loss.detach().item()),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "rho": float(self.rho),
            "informative_sigma_threshold": float(self.informative_sigma_threshold),
        }
        if os.getenv("SIEVE_DEBUG_PDB", "0") == "1":
            self._print_debug_numeric_summary()
            import pdb; pdb.set_trace()
        if batch_loss.requires_grad:
            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.gate.get_learnable_parameters(),
                self.max_grad_norm,
            )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        return [
            TrainStepResult(
                scenario_id=payload["scenario_id"],
                rewards=payload["rewards"],
                loss=payload["loss"],
                thetas=payload["thetas"],
                answers=payload["answers"],
                gold_label=payload["gold_label"],
                best_correct=payload["best_correct"],
                single_correct=payload["single_correct"],
                single_theta=payload["single_theta"],
                schema_match=payload["schema_match"],
                single_schema_match=payload["single_schema_match"],
                strict_accuracy=payload["strict_accuracy"],
                avg_advantage=payload["avg_advantage"],
                schema_advantages=payload["schema_advantages"],
                sample_advantages=payload["sample_advantages"],
                alignment_scores=payload["alignment_scores"],
                alignment_advantages=payload["alignment_advantages"],
                schema_similarity_means=payload["schema_similarity_means"],
                cross_schema_similarities=payload["cross_schema_similarities"],
                combined_sample_advantages=payload["combined_sample_advantages"],
                schema_sensitivity=payload["schema_sensitivity"],
                lambda_weight=payload["lambda_weight"],
                dominant_schemas=payload["dominant_schemas"],
                per_group_rewards=payload["per_group_rewards"],
                reasoning_lengths=payload["reasoning_lengths"],
                llm_model_name=payload["llm_model_name"],
                entropy=payload["entropy"],
                theta_entropy_penalty=payload["theta_entropy_penalty"],
                informative=payload["informative"],
                correct_theta_count=payload["correct_theta_count"],
                wrong_theta_count=payload["wrong_theta_count"],
                correct_theta_entropy_mean=payload["correct_theta_entropy_mean"],
                wrong_theta_entropy_mean=payload["wrong_theta_entropy_mean"],
                correct_theta_max_mean=payload["correct_theta_max_mean"],
                wrong_theta_max_mean=payload["wrong_theta_max_mean"],
                reward_accs=payload["reward_accs"],
                reward_comps=payload["reward_comps"],
                reward_totals=payload["reward_totals"],
                influence_vector=payload["influence_vector"],
            )
            for payload in step_payloads
        ]

    def train_step(
        self,
        scenario: str,
        scenario_id: str,
        gold_label: str,
        gold_principles: list[str] | None = None,
        model_name: str | None = None,
    ) -> TrainStepResult:
        """One GRPO training step: sample k thetas, evaluate, update."""

        samples = self._sample_stratified(scenario)
        theta_lists = [s.theta.detach().tolist() for s in samples]
        with torch.no_grad():
            single_theta = self.gate.forward(scenario)
        single_theta_list = single_theta.detach().tolist()
        llm_client, cache_client, resolved_model_name = self._resolve_llm_and_cache(model_name)
        max_workers = max(1, min(self.llm_max_concurrency, len(theta_lists) + 1))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            sample_outputs = list(
                executor.map(
                    lambda theta_list: self._run_sample(
                        scenario=scenario,
                        scenario_id=scenario_id,
                        gold_label=gold_label,
                        theta_list=theta_list,
                        gold_principles=gold_principles,
                        llm=llm_client,
                        cache=cache_client,
                    ),
                    theta_lists,
                )
            )
            single_output = self._run_sample(
                scenario=scenario,
                scenario_id=scenario_id,
                gold_label=gold_label,
                theta_list=single_theta_list,
                gold_principles=gold_principles,
                llm=llm_client,
                cache=cache_client,
            )

        rewards = [reward for reward, _, _, _, _, _, _ in sample_outputs]
        answers = [answer for _, answer, _, _, _, _, _ in sample_outputs]
        raw_responses = [response for _, _, response, _, _, _, _ in sample_outputs]
        reasoning_lengths = [length for _, _, _, length, _, _, _ in sample_outputs]
        schema_item_counts_batch = [
            schema_item_counts for _, _, _, _, _, schema_item_counts, _ in sample_outputs
        ]
        safety_judgments = [
            safety_judgment for _, _, _, _, _, _, safety_judgment in sample_outputs
        ]
        full_schema_embeddings = cache_client.get_full_schema_embeddings(
            scenario,
            cache_client.build_cache_key(scenario_id, prefix="train"),
        )
        cross_schema_similarities = self._compute_cross_schema_similarities(
            full_schema_embeddings
        )
        use_safety_reward = self._uses_safety_reward(cache_client)
        correctness_flags = (
            [float(reward) >= 1.0 for reward in rewards]
            if use_safety_reward
            else [
                self._scenario_matches_label(answer, gold_label)
                for answer in answers
            ]
        )
        single_correct = (
            bool(single_output[0] >= 1.0)
            if use_safety_reward
            else self._scenario_matches_label(single_output[1], gold_label)
        )
        strict_accuracy = sum(correctness_flags) / max(1, len(correctness_flags))

        (
            advantages,
            sigma_sq,
            lambda_weight,
            per_group_rewards,
            dominant_schemas,
            schema_advantages,
            sample_advantages,
            alignment_scores,
            alignment_advantages,
            combined_sample_advantages,
        ) = (
            self._compute_dual_level_advantages(
                rewards,
                theta_lists,
                responses=raw_responses,
                schema_embeddings_batch=(
                    [
                        full_schema_embeddings
                        for _ in schema_item_counts_batch
                    ]
                    if self._use_richness_reward()
                    else [
                        cache_client.get_selected_schema_embeddings(
                            scenario,
                            cache_client.build_cache_key(scenario_id, prefix="train"),
                            schema_item_counts,
                        )
                        for schema_item_counts in schema_item_counts_batch
                    ]
                    if self.use_alignment_adv
                    else None
                ),
            )
        )
        avg_advantage = sum(advantages) / max(1, len(advantages))

        current_logits = None
        if self.use_clip:
            _, current_logits_batch = self.gate.forward_batch([scenario])
            current_logits = current_logits_batch.squeeze(0)
        loss = torch.tensor(0.0, device=self.gate.device, requires_grad=False)
        clip_ratios = []
        clip_flags = []
        for i in range(self.k):
            if self.use_clip:
                current_log_prob = self._theta_log_prob_from_logits(
                    current_logits,
                    samples[i].theta,
                )
            else:
                current_log_prob = samples[i].log_prob
            sample_loss, ratio, clipped = self._policy_sample_loss(
                current_log_prob=current_log_prob,
                old_log_prob=samples[i].log_prob,
                advantage=advantages[i],
            )
            loss = loss + sample_loss
            clip_ratios.append(ratio)
            clip_flags.append(clipped)
        loss = loss / self.k
        entropy_active_mask = [self.entropy_reg_alpha > 0.0]
        entropy_bonus = self._compute_single_theta_entropy_bonus(single_theta)
        avg_sample_theta_entropy = self._compute_avg_sample_theta_entropy([samples])
        entropy_value = float(avg_sample_theta_entropy.item())
        theta_entropy_penalty = self._compute_theta_entropy_penalty([samples])
        theta_entropy_penalty_value = float(theta_entropy_penalty.item())
        kl_term = (
            self._compute_reference_kl([scenario])
            if self.use_kl
            else torch.tensor(0.0, device=self.gate.device)
        )
        correctness_theta_metrics = self._compute_correctness_conditioned_theta_metrics(
            theta_lists,
            correctness_flags,
        )
        loss = loss - (self.entropy_reg_alpha * entropy_bonus)
        if self.use_entropy_loss:
            loss = loss + (self.entropy_loss_beta * theta_entropy_penalty)
        if self.use_kl:
            loss = loss + (self.kl_weight * kl_term)

        if loss.requires_grad:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.gate.get_learnable_parameters(),
                self.max_grad_norm,
            )
            self.optimizer.step()

        return TrainStepResult(
            scenario_id=scenario_id,
            rewards=rewards,
            loss=loss.item(),
            thetas=[s.theta.detach().tolist() for s in samples],
            answers=answers,
            gold_label=gold_label,
            best_correct=any(correctness_flags),
            single_correct=bool(single_correct),
            single_theta=single_theta_list,
            schema_match=None,
            single_schema_match=None,
            strict_accuracy=strict_accuracy,
            avg_advantage=avg_advantage,
            schema_advantages=schema_advantages,
            sample_advantages=sample_advantages,
            alignment_scores=alignment_scores,
            alignment_advantages=alignment_advantages,
            schema_similarity_means=dict(self._last_schema_similarity_means),
            cross_schema_similarities=cross_schema_similarities,
            combined_sample_advantages=combined_sample_advantages,
            schema_sensitivity=sigma_sq,
            lambda_weight=lambda_weight,
            dominant_schemas=dominant_schemas,
            per_group_rewards=per_group_rewards,
            reasoning_lengths=reasoning_lengths,
            llm_model_name=resolved_model_name,
            entropy=entropy_value,
            theta_entropy_penalty=theta_entropy_penalty_value,
            informative=sigma_sq > self.informative_sigma_threshold,
            correct_theta_count=correctness_theta_metrics["correct_theta_count"],
            wrong_theta_count=correctness_theta_metrics["wrong_theta_count"],
            correct_theta_entropy_mean=correctness_theta_metrics["correct_theta_entropy_mean"],
            wrong_theta_entropy_mean=correctness_theta_metrics["wrong_theta_entropy_mean"],
            correct_theta_max_mean=correctness_theta_metrics["correct_theta_max_mean"],
            wrong_theta_max_mean=correctness_theta_metrics["wrong_theta_max_mean"],
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        dataset: list[dict],
        verbose: bool = True,
        epoch_num: int | None = None,
        global_step: int | None = None,
    ) -> EvalResult:
        """
        Run evaluation on a dataset (no gradient, deterministic θ).

        Dataset format: [{"id": str, "context": str, "question": str,
                          "label": str, "reward": float}, ...]
        """
        self.gate.eval()
        if self.multi_llm:
            self._reset_model_pool()
        results = []
        eval_schema_embeddings_batch: list[dict[str, list[float]]] = []
        eval_responses: list[str] = []
        correct = 0
        total_loss = 0.0

        progress = tqdm(
            dataset,
            total=len(dataset),
            desc="Evaluation",
            leave=False,
            disable=not verbose,
        )

        for batch_start in range(0, len(dataset), self.batch_size):
            batch_items = dataset[batch_start: batch_start + self.batch_size]
            gate_scenarios = [
                format_gate_input(item)
                for item in batch_items
            ]
            extraction_scenarios = list(gate_scenarios)
            llm_scenarios = [format_full_input(item) for item in batch_items]
            sids = [
                self._resolve_item_id(item, str(batch_start + idx))
                for idx, item in enumerate(batch_items)
            ]
            gold_labels = [item["label"] for item in batch_items]
            assigned_models = self._assign_models_to_batch(len(batch_items))
            llm_cache_batch = [
                self._resolve_llm_and_cache(model_name)
                for model_name in assigned_models
            ]

            theta_batch, logits_batch = self.gate.forward_batch(
                gate_scenarios, inference_mode=True
            )
            theta_lists = theta_batch.tolist()
            phase3_batch = [
                cache_client.get_phase3(
                    extraction_scenario,
                    cache_client.build_cache_key(sid, prefix="eval"),
                    theta=theta_list,
                )
                for extraction_scenario, sid, theta_list, (_, cache_client, _) in zip(
                    extraction_scenarios, sids, theta_lists, llm_cache_batch
                )
            ]
            prompts = [
                assemble_prompt(
                    scenario,
                    phase3,
                    use_persona=self.use_persona,
                    inst_regime=self.inst_regime,
                    inference_add_eval=self.inference_add_eval,
                    token_proportional=True,
                    use_token_total_budget=getattr(self, "use_token_total_budget", False),
                    tokenizer=self._resolve_prompt_tokenizer(llm_client),
                    safety=getattr(cache_client, "safety", False),
                )
                for scenario, phase3, (llm_client, cache_client, _) in zip(
                    llm_scenarios,
                    phase3_batch,
                    llm_cache_batch,
                )
            ]

            max_workers = max(1, min(self.llm_max_concurrency, len(prompts)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                responses = list(
                    executor.map(
                        lambda payload: payload[0].generate(payload[1], max_tokens=1024),
                        [
                            (llm_client, prompt)
                            for (llm_client, _, _), prompt in zip(llm_cache_batch, prompts)
                        ],
                    )
                )

            for (
                item,
                sid,
                gold_label,
                gate_input,
                extraction_input,
                input_text,
                prompt,
                theta,
                logits,
                response,
                (_, cache_client, llm_model_name),
            ) in zip(
                batch_items,
                sids,
                gold_labels,
                gate_scenarios,
                extraction_scenarios,
                llm_scenarios,
                prompts,
                theta_batch,
                logits_batch,
                responses,
                llm_cache_batch,
            ):
                answer = parse_answer(response)
                if self._uses_safety_reward(cache_client):
                    reward, _ = self._compute_safety_reward(
                        scenario=input_text,
                        response=response,
                        gold_label=gold_label,
                    )
                    is_correct = reward >= 1.0
                else:
                    is_correct = self._scenario_matches_label(answer, gold_label)
                    reward = self._compute_reward(
                        is_correct=is_correct,
                    )
                correct += int(is_correct)

                log_probs = F.log_softmax(logits / self.gate.temperature, dim=-1)
                sample_loss = -(reward - 0.5) * (theta.detach() * log_probs).sum()
                total_loss += sample_loss.item()
                schema_match = self._schema_match_from_theta(theta.tolist(), item)
                full_schema_embeddings = cache_client.get_full_schema_embeddings(
                    extraction_input,
                    cache_client.build_cache_key(sid, prefix="eval"),
                )
                cross_schema_similarities = self._compute_cross_schema_similarities(
                    full_schema_embeddings
                )
                if self._use_richness_reward():
                    eval_schema_embeddings_batch.append(full_schema_embeddings)
                    eval_responses.append(response)

                results.append({
                    "id": sid,
                    "gold": gold_label,
                    "predicted": answer,
                    "correct": is_correct,
                    "theta": theta.tolist(),
                    "schema_label": item.get("schema_label"),
                    "schema_match": schema_match,
                    "cross_schema_similarities": cross_schema_similarities,
                    "gate_input": gate_input,
                    "input_text": input_text,
                    "prompt": prompt,
                    "response": response,
                    "llm_model_name": llm_model_name,
                })

            progress.update(len(batch_items))
            if verbose:
                running_acc = correct / len(results)
                progress.set_postfix(
                    acc=f"{running_acc:.3f}",
                    loss=f"{total_loss / len(results):.4f}",
                    refresh=False,
                )

        n = len(dataset)
        all_thetas = torch.tensor([r["theta"] for r in results])
        current_eval_thetas = [theta.detach().cpu() for theta in all_thetas]
        theta_stability: float | None = None
        if (
            self.prev_eval_thetas is not None
            and len(self.prev_eval_thetas) == len(current_eval_thetas)
        ):
            deltas = [
                torch.abs(curr - prev).sum().item()
                for curr, prev in zip(current_eval_thetas, self.prev_eval_thetas)
            ]
            theta_stability = sum(deltas) / len(deltas) if deltas else None
        self.prev_eval_thetas = current_eval_thetas
        mean_theta = all_thetas.mean(dim=0).tolist()
        theta_entropy = -(all_thetas.clamp_min(1e-8) * all_thetas.clamp_min(1e-8).log()).sum(dim=-1)
        theta_mean_scalar = all_thetas.mean(dim=-1, keepdim=True)
        theta_schema_sensitivity = ((all_thetas - theta_mean_scalar) ** 2).mean(dim=-1)
        informative_mask = theta_schema_sensitivity > self.informative_sigma_threshold
        avg_theta_entropy = float(theta_entropy.mean().item())
        mean_theta_tensor = all_thetas.mean(dim=0)
        uniform_theta = torch.full_like(mean_theta_tensor, 1.0 / len(SCHEMA_NAMES))
        entropy_term = 1.0 - avg_theta_entropy / math.log(len(SCHEMA_NAMES))
        balance_l1 = torch.abs(mean_theta_tensor - uniform_theta).sum().item()
        balance_term = 1.0 - balance_l1 / (2.0 * (len(SCHEMA_NAMES) - 1) / len(SCHEMA_NAMES))
        schema_validity = max(0.0, entropy_term) * max(0.0, balance_term)
        avg_schema_sensitivity = float(theta_schema_sensitivity.mean().item())
        informative_count = int(informative_mask.sum().item())
        informative_ratio = informative_count / max(1, n)
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
            dominant_schema = self._schema_name_from_theta(result["theta"])
            dominant_schema_counts[dominant_schema] += 1
        dominant_schema_ratio = {
            schema: dominant_schema_counts[schema] / max(1, n)
            for schema in SCHEMA_NAMES
        }
        cross_schema_similarities = self._average_metric_dicts(
            [result.get("cross_schema_similarities", {}) for result in results]
        )
        eval_richness_score_mean: float | None = None
        eval_schema_similarity_means = {schema: None for schema in SCHEMA_NAMES}
        if self._use_richness_reward() and eval_responses and eval_schema_embeddings_batch:
            eval_richness_scores = self._compute_richness_scores(
                eval_responses,
                eval_schema_embeddings_batch,
            )
            if eval_richness_scores:
                eval_richness_score_mean = sum(eval_richness_scores) / len(eval_richness_scores)
            eval_schema_similarity_means = dict(self._last_schema_similarity_means)
        eval_theta_metrics = {
            f"eval/mean_theta_{schema}": mean_theta[idx]
            for idx, schema in enumerate(SCHEMA_NAMES)
        }
        eval_dominant_ratio_metrics = {
            f"eval/dominant_ratio_{schema}": dominant_schema_ratio[schema]
            for schema in SCHEMA_NAMES
        }
        eval_schema_similarity_metrics = {
            f"eval/response_argument_similarity_{schema}": (
                eval_schema_similarity_means.get(schema)
            )
            for schema in SCHEMA_NAMES
        }

        self.gate.train()

        self._log_metrics(
            {
                "eval/accuracy": correct / n,
                "eval/loss": total_loss / n,
                "eval/avg_theta_entropy": avg_theta_entropy,
                "eval/schema_validity": schema_validity,
                "eval/schema_match": schema_match,
                "eval/avg_schema_sensitivity": avg_schema_sensitivity,
                "eval/informative_count": informative_count,
                "eval/informative_ratio": informative_ratio,
                "eval/richness_score_mean": (
                    eval_richness_score_mean if self._use_richness_reward() else None
                ),
                "eval/n_correct": correct,
                "eval/n_total": n,
                "eval/epoch": epoch_num,
                "eval/theta_stability": theta_stability,
                **eval_theta_metrics,
                **eval_dominant_ratio_metrics,
                **eval_schema_similarity_metrics,
                **{
                    f"eval/cross_sim_{pair_name}": sim_value
                    for pair_name, sim_value in cross_schema_similarities.items()
                },
            },
            step=global_step,
        )

        return EvalResult(
            accuracy=correct / n,
            avg_loss=total_loss / n,
            n_total=n,
            n_correct=correct,
            mean_theta=mean_theta,
            avg_theta_entropy=avg_theta_entropy,
            schema_validity=schema_validity,
            schema_match=schema_match,
            avg_schema_sensitivity=avg_schema_sensitivity,
            informative_count=informative_count,
            informative_ratio=informative_ratio,
            dominant_schema_ratio=dominant_schema_ratio,
            cross_schema_similarities=cross_schema_similarities,
            richness_score_mean=eval_richness_score_mean,
            schema_similarity_means=eval_schema_similarity_means,
            per_item=results,
        )

    # ------------------------------------------------------------------
    # Train epoch
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        dataset: list[dict],
        epoch_num: int = 0,
        verbose: bool = True,
        progress_bar: Any | None = None,
        global_step_start: int = 0,
        on_step_end: Callable[[int], EvalResult | None] | None = None,
    ) -> EpochResult:
        """
        One epoch: iterate over all scenarios.

        Dataset format: [{"id": str, "context": str, "question": str,
                          "label": str, "reward": float}, ...]
        """
        self.gate.train()
        step_results = []
        total_reward = 0.0
        total_loss = 0.0
        total_correct = 0
        total_single_correct = 0
        total_strict_accuracy = 0.0
        total_avg_advantage = 0.0
        total_avg_advantage_sq = 0.0
        total_entropy = 0.0
        total_schema_sensitivity = 0.0
        total_normalized_schema_sensitivity = 0.0
        informative_count = 0
        total_correct_theta_count = 0
        total_wrong_theta_count = 0
        total_correct_theta_entropy = 0.0
        total_wrong_theta_entropy = 0.0
        total_correct_theta_max = 0.0
        total_wrong_theta_max = 0.0
        total_schema_match = 0.0
        total_schema_match_count = 0
        total_single_schema_match = 0.0
        total_single_schema_match_count = 0
        epoch_group_reward_totals = {schema: 0.0 for schema in SCHEMA_NAMES}
        epoch_group_reward_counts = {schema: 0 for schema in SCHEMA_NAMES}
        epoch_theta_sum = torch.zeros(len(SCHEMA_NAMES), dtype=torch.float32)
        epoch_theta_count = 0
        epoch_cross_sim_totals = {
            f"{left}_{right}": 0.0
            for left, right in CROSS_SCHEMA_PAIRS
        }
        epoch_cross_sim_counts = {
            f"{left}_{right}": 0
            for left, right in CROSS_SCHEMA_PAIRS
        }

        active_items = [
            (i, item) for i, item in enumerate(dataset)
        ]
        processed_count = 0
        latest_eval_result: EvalResult | None = None
        processed_batch_count = 0
        entropy_reg_term_dominates_count = 0
        train_diagnostic_metric_keys = [
            "reward/R_acc_mean",
            "reward/R_acc_std",
            "reward/R_comp_mean",
            "reward/R_comp_std",
            "reward/R_total_mean",
            "reward/R_total_std",
            "reward/correlation_acc_comp",
            "csa/b_PI",
            "csa/b_MN",
            "csa/b_PC",
            "csa/y_hat_mean",
            "csa/y_hat_std",
            "csa/advantage_mean",
            "csa/advantage_std",
            "csa/advantage_max_abs",
            "gate/theta_base_entropy",
            "gate/theta_base_max",
            "gate/dominant_schema_ratio_PI",
            "gate/dominant_schema_ratio_MN",
            "gate/dominant_schema_ratio_PC",
            "influence/coverage_PI",
            "influence/coverage_MN",
            "influence/coverage_PC",
            "influence/alignment_PI",
            "influence/alignment_MN",
            "influence/alignment_PC",
            "influence/misalignment_PI",
            "influence/misalignment_MN",
            "influence/misalignment_PC",
        ]
        running_diagnostic_sums: dict[str, float] = {}
        running_diagnostic_counts: dict[str, int] = {}

        def _update_running_diagnostic_metrics(
            metrics: dict[str, Any],
        ) -> dict[str, float]:
            running_metrics: dict[str, float] = {}
            for key, value in metrics.items():
                if value is None:
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric_value):
                    continue

                running_diagnostic_sums[key] = (
                    running_diagnostic_sums.get(key, 0.0) + numeric_value
                )
                running_diagnostic_counts[key] = running_diagnostic_counts.get(key, 0) + 1
                prefix, metric_name = key.split("/", 1)
                running_metrics[f"{prefix}/running_{metric_name}"] = (
                    running_diagnostic_sums[key] / running_diagnostic_counts[key]
                )
            return running_metrics

        for batch_start in range(0, len(active_items), self.batch_size):
            batch_items = active_items[batch_start: batch_start + self.batch_size]
            batch_results = self._train_batch(batch_items)
            processed_batch_count += 1
            batch_monitor = self.last_train_batch_monitor or {}
            if batch_monitor.get("entropy_reg_term_dominates", False):
                entropy_reg_term_dominates_count += 1
            batch_diagnostic_metrics = {
                key: batch_monitor.get(key)
                for key in train_diagnostic_metric_keys
            }
            running_batch_diagnostic_metrics = _update_running_diagnostic_metrics(
                batch_diagnostic_metrics
            )

            for (dataset_idx, item), result in zip(batch_items, batch_results):
                sid = self._resolve_item_id(item, str(dataset_idx))
                step_results.append(result)
                total_reward += sum(result.rewards) / self.k
                total_loss += result.loss
                total_correct += int(result.best_correct)
                total_single_correct += int(result.single_correct)
                total_strict_accuracy += result.strict_accuracy
                total_avg_advantage += result.avg_advantage
                total_avg_advantage_sq += result.avg_advantage ** 2
                total_entropy += result.entropy
                total_schema_sensitivity += result.schema_sensitivity
                total_normalized_schema_sensitivity += self._normalize_schema_sensitivity(
                    result.schema_sensitivity
                )
                informative_count += int(result.informative)
                total_correct_theta_count += result.correct_theta_count
                total_wrong_theta_count += result.wrong_theta_count
                if result.schema_match is not None:
                    total_schema_match += result.schema_match
                    total_schema_match_count += 1
                if result.single_schema_match is not None:
                    total_single_schema_match += result.single_schema_match
                    total_single_schema_match_count += 1
                if result.correct_theta_entropy_mean is not None:
                    total_correct_theta_entropy += (
                        result.correct_theta_entropy_mean * result.correct_theta_count
                    )
                if result.wrong_theta_entropy_mean is not None:
                    total_wrong_theta_entropy += (
                        result.wrong_theta_entropy_mean * result.wrong_theta_count
                    )
                if result.correct_theta_max_mean is not None:
                    total_correct_theta_max += (
                        result.correct_theta_max_mean * result.correct_theta_count
                    )
                if result.wrong_theta_max_mean is not None:
                    total_wrong_theta_max += (
                        result.wrong_theta_max_mean * result.wrong_theta_count
                    )
                single_theta_tensor = torch.tensor(result.single_theta, dtype=torch.float32)
                epoch_theta_sum += single_theta_tensor
                epoch_theta_count += 1
                active_schemas = set(result.dominant_schemas)
                for schema in SCHEMA_NAMES:
                    if schema in active_schemas:
                        epoch_group_reward_totals[schema] += result.per_group_rewards.get(schema, 0.0)
                        epoch_group_reward_counts[schema] += 1
                for pair_name, sim_value in result.cross_schema_similarities.items():
                    if sim_value is not None:
                        epoch_cross_sim_totals[pair_name] += float(sim_value)
                        epoch_cross_sim_counts[pair_name] += 1
                processed_count += 1

                current_step = global_step_start + processed_count
                running_acc = total_correct / processed_count
                running_single_accuracy = total_single_correct / processed_count
                running_strict_accuracy = total_strict_accuracy / processed_count
                running_advantage = total_avg_advantage / processed_count
                running_advantage_var = max(
                    0.0,
                    (total_avg_advantage_sq / processed_count) - (running_advantage ** 2),
                )
                running_advantage_std = math.sqrt(running_advantage_var)
                running_reward = total_reward / processed_count
                running_entropy = total_entropy / processed_count
                running_schema_sensitivity = total_schema_sensitivity / processed_count
                running_normalized_schema_sensitivity = (
                    total_normalized_schema_sensitivity / processed_count
                )
                running_informative_ratio = informative_count / processed_count
                running_schema_match = (
                    total_schema_match / total_schema_match_count
                    if total_schema_match_count > 0
                    else None
                )
                running_single_schema_match = (
                    total_single_schema_match / total_single_schema_match_count
                    if total_single_schema_match_count > 0
                    else None
                )
                running_entropy_reg_term_dominates_ratio = (
                    entropy_reg_term_dominates_count / max(1, processed_batch_count)
                )
                running_per_group_reward_metrics = {
                    f"train/running_per_group_reward_{schema}": (
                        epoch_group_reward_totals[schema] / epoch_group_reward_counts[schema]
                        if epoch_group_reward_counts[schema] > 0
                        else 0.0
                    )
                    for schema in SCHEMA_NAMES
                }
                running_correct_theta_entropy = (
                    total_correct_theta_entropy / total_correct_theta_count
                    if total_correct_theta_count > 0
                    else None
                )
                running_wrong_theta_entropy = (
                    total_wrong_theta_entropy / total_wrong_theta_count
                    if total_wrong_theta_count > 0
                    else None
                )
                running_correct_theta_max = (
                    total_correct_theta_max / total_correct_theta_count
                    if total_correct_theta_count > 0
                    else None
                )
                running_wrong_theta_max = (
                    total_wrong_theta_max / total_wrong_theta_count
                    if total_wrong_theta_count > 0
                    else None
                )
                step_reward_mean = sum(result.rewards) / self.k
                step_alignment_score_mean = (
                    sum(result.alignment_scores) / max(1, len(result.alignment_scores))
                    if result.alignment_scores
                    else None
                )
                step_alignment_advantage_mean = (
                    sum(result.alignment_advantages) / max(1, len(result.alignment_advantages))
                    if result.alignment_advantages
                    else None
                )
                step_theta_metrics = {
                    f"train/step_theta_mean_{schema}": single_theta_tensor[idx].item()
                    for idx, schema in enumerate(SCHEMA_NAMES)
                }
                running_theta_mean = epoch_theta_sum / max(1, epoch_theta_count)
                running_theta_metrics = {
                    f"train/running_theta_mean_{schema}": running_theta_mean[idx].item()
                    for idx, schema in enumerate(SCHEMA_NAMES)
                }
                running_cross_sim_metrics = {
                    f"train/running_cross_sim_{pair_name}": (
                        epoch_cross_sim_totals[pair_name] / epoch_cross_sim_counts[pair_name]
                    )
                    for pair_name in epoch_cross_sim_totals
                    if epoch_cross_sim_counts[pair_name] > 0
                }
                schema_similarity_metrics = {
                    f"train/step_response_argument_similarity_{schema}": (
                        result.schema_similarity_means.get(schema)
                    )
                    for schema in SCHEMA_NAMES
                }

                self._log_metrics(
                    {
                        "train/step_loss": result.loss,
                        "train/step_accuracy": float(result.best_correct),
                        "train/step_single_accuracy": float(result.single_correct),
                        "train/step_strict_accuracy": result.strict_accuracy,
                        "train/step_avg_advantage": result.avg_advantage,
                        "train/step_reward_mean": step_reward_mean,
                        "train/step_alignment_score_mean": step_alignment_score_mean,
                        "train/step_alignment_advantage_mean": step_alignment_advantage_mean,
                        "train/step_richness_score_mean": (
                            step_alignment_score_mean if self._use_richness_reward() else None
                        ),
                        "train/step_richness_advantage_mean": (
                            step_alignment_advantage_mean if self._use_richness_reward() else None
                        ),
                        **schema_similarity_metrics,
                        **running_per_group_reward_metrics,
                        "train/step_entropy": result.entropy,
                        "train/step_theta_entropy_penalty": result.theta_entropy_penalty,
                        "train/running_entropy": running_entropy,
                        "train/step_correct_theta_count": result.correct_theta_count,
                        "train/step_wrong_theta_count": result.wrong_theta_count,
                        "train/step_correct_theta_entropy_mean": result.correct_theta_entropy_mean,
                        "train/step_wrong_theta_entropy_mean": result.wrong_theta_entropy_mean,
                        "train/step_correct_theta_max_mean": result.correct_theta_max_mean,
                        "train/step_wrong_theta_max_mean": result.wrong_theta_max_mean,
                        "train/running_correct_theta_entropy_mean": running_correct_theta_entropy,
                        "train/running_wrong_theta_entropy_mean": running_wrong_theta_entropy,
                        "train/running_correct_theta_max_mean": running_correct_theta_max,
                        "train/running_wrong_theta_max_mean": running_wrong_theta_max,
                        "train/step_informative": float(result.informative),
                        "train/step_schema_sensitivity": result.schema_sensitivity,
                        "train/step_normalized_schema_sensitivity": (
                            self._normalize_schema_sensitivity(result.schema_sensitivity)
                        ),
                        "train/running_schema_sensitivity": running_schema_sensitivity,
                        "train/running_normalized_schema_sensitivity": (
                            running_normalized_schema_sensitivity
                        ),
                        "train/informative_sigma_threshold": self.informative_sigma_threshold,
                        "train/lambda_weight": result.lambda_weight,
                        "train/rho": self.rho,
                        "train/mean_reasoning_length": (
                            sum(result.reasoning_lengths) / max(1, len(result.reasoning_lengths))
                        ),
                        "train/running_accuracy": running_acc,
                        "train/running_single_accuracy": running_single_accuracy,
                        "train/running_strict_accuracy": running_strict_accuracy,
                        "train/running_advantage": running_advantage,
                        "train/running_advantage_std": running_advantage_std,
                        "train/running_reward": running_reward,
                        "train/running_informative_ratio": running_informative_ratio,
                        "train/step_schema_match": result.schema_match,
                        "train/step_single_schema_match": result.single_schema_match,
                        "train/running_schema_match": running_schema_match,
                        "train/running_single_schema_match": running_single_schema_match,
                        "train/policy_loss_abs": batch_monitor.get("policy_loss_abs"),
                        "train/entropy_reg_term_abs": batch_monitor.get("entropy_reg_term_abs"),
                        "train/clip_ratio": batch_monitor.get("clip_ratio"),
                        "train/clip_fraction": batch_monitor.get("clip_fraction"),
                        "train/entropy_reg_term_dominates": float(
                            batch_monitor.get("entropy_reg_term_dominates", False)
                        ),
                        **batch_diagnostic_metrics,
                        **running_batch_diagnostic_metrics,
                        "train/running_entropy_reg_term_dominates_ratio": (
                            running_entropy_reg_term_dominates_ratio
                        ),
                        "train/temperature": self.gate.temperature,
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "train/entropy_reg_alpha": self.entropy_reg_alpha,
                        "train/use_entropy_loss": float(self.use_entropy_loss),
                        "train/entropy_loss_beta": self.entropy_loss_beta,
                        "train/use_clip": float(self.use_clip),
                        "train/clip_epsilon": self.clip_epsilon,
                        "train/epoch": epoch_num + 1,
                        "train/llm_model_name": result.llm_model_name,
                        **step_theta_metrics,
                        **running_theta_metrics,
                        **running_cross_sim_metrics,
                    },
                    step=current_step,
                )

                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        epoch=f"{epoch_num + 1}",
                        acc=f"{running_acc:.3f}",
                        single=f"{running_single_accuracy:.3f}",
                        strict=f"{running_strict_accuracy:.3f}",
                        adv=f"{running_advantage:.3f}",
                        adv_std=f"{running_advantage_std:.3f}",
                        ent=f"{running_entropy:.3f}",
                        loss=f"{result.loss:.4f}",
                        reward=f"{step_reward_mean:.3f}",
                        sigma=f"{running_schema_sensitivity:.4f}",
                        info=f"{int(result.informative)}",
                        tau=f"{self.gate.temperature:.3f}",
                        refresh=False,
                    )
                elif verbose:
                    print(
                        f"    [{processed_count}/{len(active_items)}] acc={running_acc:.3f} "
                        f"loss={result.loss:.4f}"
                    )

                if on_step_end is not None:
                    step_eval_result = on_step_end(current_step)
                    if step_eval_result is not None:
                        latest_eval_result = step_eval_result

        n = max(1, processed_count)
        epoch_per_group_rewards = {
            schema: (
                epoch_group_reward_totals[schema] / epoch_group_reward_counts[schema]
                if epoch_group_reward_counts[schema] > 0
                else 0.0
            )
            for schema in SCHEMA_NAMES
        }
        epoch_mean_theta = (epoch_theta_sum / max(1, epoch_theta_count)).tolist()
        epoch_cross_schema_similarities = {
            pair_name: (
                epoch_cross_sim_totals[pair_name] / epoch_cross_sim_counts[pair_name]
            )
            for pair_name in epoch_cross_sim_totals
            if epoch_cross_sim_counts[pair_name] > 0
        }
        epoch_schema_match = (
            total_schema_match / total_schema_match_count
            if total_schema_match_count > 0
            else 0.0
        )
        epoch_single_schema_match = (
            total_single_schema_match / total_single_schema_match_count
            if total_single_schema_match_count > 0
            else 0.0
        )
        epoch_richness_scores = [
            score
            for result in step_results
            for score in result.alignment_scores
        ] if self._use_richness_reward() else []
        epoch_richness_advantages = [
            advantage
            for result in step_results
            for advantage in result.alignment_advantages
        ] if self._use_richness_reward() else []
        epoch_schema_similarity_means = {
            schema: None for schema in SCHEMA_NAMES
        }
        if self._use_richness_reward():
            for schema in SCHEMA_NAMES:
                schema_values = [
                    result.schema_similarity_means.get(schema)
                    for result in step_results
                    if result.schema_similarity_means.get(schema) is not None
                ]
                if schema_values:
                    epoch_schema_similarity_means[schema] = (
                        sum(schema_values) / len(schema_values)
                    )
        epoch_result = EpochResult(
            epoch=epoch_num,
            accuracy=total_correct / n,
            single_accuracy=total_single_correct / n,
            strict_accuracy=total_strict_accuracy / n,
            avg_advantage=total_avg_advantage / n,
            avg_reward=total_reward / n,
            avg_loss=total_loss / n,
            temperature=self.gate.temperature,
            avg_entropy=total_entropy / n,
            informative_count=informative_count,
            informative_ratio=informative_count / n,
            per_group_rewards=epoch_per_group_rewards,
            mean_theta=epoch_mean_theta,
            cross_schema_similarities=epoch_cross_schema_similarities,
            richness_score_mean=(
                sum(epoch_richness_scores) / len(epoch_richness_scores)
                if epoch_richness_scores
                else None
            ),
            richness_advantage_mean=(
                sum(epoch_richness_advantages) / len(epoch_richness_advantages)
                if epoch_richness_advantages
                else None
            ),
            schema_similarity_means=epoch_schema_similarity_means,
            schema_match=epoch_schema_match,
            single_schema_match=epoch_single_schema_match,
            eval_result=latest_eval_result,
            step_results=step_results,
        )
        self.history.append(epoch_result)
        epoch_result.entropy_reg_term_dominates_ratio = (
            entropy_reg_term_dominates_count / max(1, processed_batch_count)
        )
        return epoch_result

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        dataset: list[dict],
        num_epochs: int = 10,
        eval_dataset: list[dict] = None,
        eval_step: int = 0,
        save_best: bool = True,
        save_last: bool = False,
        save_dir: str = "./checkpoints",
        verbose: bool = True,
        use_warmup: bool = False,
        warm_start_size: int = 600,
        warm_start_epochs: int = 1,
        warm_start_lr: float = 1e-3,
        warm_start_batch_size: int | None = None,
        warmup_only: bool = False,
        resume_from_checkpoint: str | None = None,
    ) -> list[EpochResult]:
        """
        Full training loop with optional step-based evaluation and best model saving.

        Args:
            dataset: training data
            num_epochs: number of epochs
            eval_dataset: if provided, run evaluation every `eval_step` steps
            eval_step: evaluation cadence in train steps; <=0 falls back to once per epoch
            save_best: if True (and eval_dataset provided), save model with highest eval_accuracy
            save_dir: directory for saving checkpoints
            verbose: print progress

        Returns:
            list of EpochResult (with eval_result attached if eval_dataset provided)
        """
        grpo_data = dataset
        warm_start_indices: list[int] = []
        grpo_indices = list(range(len(dataset)))
        completed_epochs = 0
        resume_state: dict[str, Any] | None = None
        effective_use_warmup = bool(use_warmup or warmup_only)
        if self.use_kl and not effective_use_warmup:
            raise ValueError("use_kl=True requires use_warmup=True.")
        if self.use_kl and warmup_only:
            raise ValueError("use_kl=True requires a GRPO phase, so warmup_only must be False.")

        if resume_from_checkpoint:
            resume_state = self._load_last_checkpoint(
                resume_from_checkpoint=resume_from_checkpoint,
                dataset=dataset,
            )
            completed_epochs = resume_state["completed_epochs"]
            warm_start_indices = list(resume_state["warm_start_indices"])
            grpo_indices = list(resume_state["grpo_indices"])
            grpo_data = [dataset[idx] for idx in grpo_indices]
            if verbose:
                print(
                    f"Resuming SIEVE training from {resume_state['checkpoint_dir']} "
                    f"after {completed_epochs} completed epoch(s)."
                )
        elif effective_use_warmup and warm_start_size > 0:
            ws_data_with_labels, grpo_data, warm_start_counts, warm_start_indices, grpo_indices = self._select_balanced_warm_start_data(
                dataset,
                warm_start_size,
            )

            if ws_data_with_labels:
                if verbose:
                    print(
                        "Phase 1: Warm-start with "
                        f"{len(ws_data_with_labels)} balanced samples "
                        f"(PI={warm_start_counts['PI']}, "
                        f"MN={warm_start_counts['MN']}, "
                        f"PC={warm_start_counts['PC']})..."
                    )
                self.warm_start(
                    ws_data_with_labels,
                    num_epochs=warm_start_epochs,
                    learning_rate=warm_start_lr,
                    batch_size=warm_start_batch_size,
                    verbose=verbose,
                )
                if self.use_kl:
                    self._capture_reference_policy()
            else:
                if verbose:
                    print("No schema labels found, skipping warm-start.")
                grpo_data = dataset
                warm_start_indices = []
                grpo_indices = list(range(len(dataset)))

        if self.use_kl and not self._has_reference_policy():
            raise ValueError(
                "use_kl=True requires a warm-start reference policy, but none was captured or restored."
            )

        continuous_train_cache_precomputed = False
        if self.csa_mode == "continuous" and not warmup_only:
            self._precompute_continuous_csa_train_cache(grpo_data, verbose=verbose)
            continuous_train_cache_precomputed = True
            grpo_data, filtered_grpo_indices = self._filter_zero_influence_data(
                grpo_data,
                indices=grpo_indices,
                verbose=verbose,
            )
            if filtered_grpo_indices is not None:
                grpo_indices = filtered_grpo_indices

        if len(grpo_data) == 0 and not warmup_only:
            raise ValueError("No GRPO data remaining after warm-start split.")

        split_state = self._build_split_state(
            dataset=dataset,
            warm_start_indices=warm_start_indices,
            grpo_indices=grpo_indices,
            eval_dataset=eval_dataset,
        )

        self._init_metrics(
            num_epochs=completed_epochs + num_epochs,
            train_size=0 if warmup_only else len(grpo_data),
            eval_size=len(eval_dataset) if eval_dataset is not None else 0,
        )
        if self._tracking_enabled:
            self._log_metrics(
                {
                    "warm_start/enabled": float(effective_use_warmup),
                    "warm_start/only": float(warmup_only),
                    "warm_start/size": len(dataset) - len(grpo_data) if effective_use_warmup else 0,
                    "warm_start/epochs": warm_start_epochs if effective_use_warmup else 0,
                    "warm_start/learning_rate": warm_start_lr if effective_use_warmup else 0.0,
                    "warm_start/batch_size": (
                        warm_start_batch_size if warm_start_batch_size is not None else self.batch_size
                    ) if effective_use_warmup else 0,
                    "warm_start/size_per_schema": (
                        (len(dataset) - len(grpo_data)) / len(SCHEMA_NAMES)
                        if effective_use_warmup and len(dataset) != len(grpo_data)
                        else 0.0
                    ),
                },
                step=0,
            )

        if warmup_only:
            if save_dir:
                checkpoint_dir = Path(save_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                warmup_gate_path = checkpoint_dir / "gate_warmup_only.pt"
                self.gate.save(str(warmup_gate_path))
                self._save_last_checkpoint(
                    save_dir=save_dir,
                    dataset=dataset,
                    eval_dataset=eval_dataset,
                    split_state=split_state,
                    completed_epochs=0,
                    total_planned_epochs=0,
                    best_eval_accuracy=None,
                    best_eval_loss=None,
                    best_epoch=None,
                    latest_epoch_result=None,
                )
                warmup_meta = {
                    "warmup_only": True,
                    "warm_start_size": int(warm_start_size),
                    "warm_start_epochs": int(warm_start_epochs),
                    "warm_start_learning_rate": float(warm_start_lr),
                    "warm_start_batch_size": (
                        warm_start_batch_size if warm_start_batch_size is not None else self.batch_size
                    ),
                    "warm_start_indices": warm_start_indices,
                    "grpo_indices": grpo_indices,
                    "gate_warmup_only_path": str(warmup_gate_path),
                    "gate_last_path": str(checkpoint_dir / "gate_last.pt"),
                }
                (checkpoint_dir / "warmup_only_meta.json").write_text(
                    json.dumps(warmup_meta, indent=2)
                )
                if verbose:
                    print(f"Warmup-only checkpoint saved to {warmup_gate_path}")
            if verbose:
                print("warmup_only=True: finished warm-start and skipped GRPO training.")
            return []

        # Pre-compute extractions for train
        if not continuous_train_cache_precomputed:
            if verbose:
                print(f"Pre-computing extractions for {len(grpo_data)} training scenarios...")
            dataset_for_cache = self._build_dataset_for_cache(grpo_data)
            if self.multi_llm:
                self._precompute_multi_llm_caches(
                    dataset_for_cache,
                    prefix="train",
                    verbose=verbose,
                )
            else:
                self.cache.precompute(dataset_for_cache, prefix="train")

        # Pre-compute extractions for eval
        if eval_dataset is not None:
            if verbose:
                print(f"Pre-computing extractions for {len(eval_dataset)} eval scenarios...")
            # Build eval scenarios and precompute
            eval_for_cache = []
            for item in eval_dataset:
                scenario = format_gate_input(item)
                eval_for_cache.append({
                    "id": self._resolve_item_id(item, str(len(eval_for_cache))),
                    "context": scenario,
                })
            if self.multi_llm:
                self._precompute_multi_llm_caches(
                    eval_for_cache,
                    prefix="eval",
                    verbose=verbose,
                )
            else:
                self.cache.precompute(eval_for_cache, prefix="eval")

        if verbose:
            print(f"\nStarting GRPO training: {num_epochs} epochs, k={self.k}")
            print(f"Batch size: {self.batch_size}")
            if self.multi_llm:
                print(f"Multi-LLM models: {', '.join(self._model_names)}")
            print(f"Learnable params: {sum(p.numel() for p in self.gate.get_learnable_parameters()):,}")
            print(f"LLM calls per epoch: {len(grpo_data) * self.k}")
            if eval_dataset:
                print(f"Eval dataset: {len(eval_dataset)} scenarios")
                print(f"Save best model: {save_best}")
                if save_best:
                    print(f"Save dir: {save_dir}")
            print()

        steps_per_epoch = max(1, math.ceil(len(grpo_data) / self.batch_size))
        resolved_eval_step = steps_per_epoch if int(eval_step) <= 0 else int(eval_step)
        total_scheduler_steps = max(1, (completed_epochs + num_epochs) * steps_per_epoch)
        if self.use_lr_scheduler:
            if verbose:
                print(
                    "LR scheduler: "
                    f"CosineAnnealingLR(T_max={total_scheduler_steps}, "
                    f"eta_min={self.scheduler_eta_min})"
                )

            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_scheduler_steps,
                eta_min=self.scheduler_eta_min,
            )
            if resume_state is not None and resume_state.get("scheduler_state") is not None:
                self.scheduler.load_state_dict(resume_state["scheduler_state"])
                self.scheduler.T_max = total_scheduler_steps
        else:
            if verbose:
                print("LR scheduler: disabled")
            self.scheduler = None

        # Best model tracking
        best_eval_accuracy = float("-inf")
        best_eval_loss = float("inf")
        best_epoch = -1
        best_global_step = -1
        best_eval_result: EvalResult | None = None

        if (save_best or save_last) and save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        if resume_state is not None and save_best and save_dir:
            best_meta_path = Path(save_dir) / "best_meta.json"
            if best_meta_path.exists():
                best_meta = json.loads(best_meta_path.read_text())
                best_eval_accuracy = best_meta.get("eval_accuracy", best_eval_accuracy)
                best_eval_loss = best_meta.get("eval_loss", best_eval_loss)
                best_epoch = best_meta.get("epoch", best_epoch)
                best_global_step = best_meta.get("global_step", best_global_step)

        results = []
        total_steps = num_epochs * len(grpo_data)
        progress_bar = tqdm(
            total=total_steps,
            desc="Training",
            disable=not verbose,
        )

        if verbose and eval_dataset is not None:
            print(f"Eval cadence: every {resolved_eval_step} train step(s)")

        def run_scheduled_eval(
            *,
            current_step: int,
            epoch: int,
            train_result: EpochResult | None = None,
            force: bool = False,
        ) -> EvalResult | None:
            nonlocal best_eval_accuracy, best_eval_loss, best_epoch, best_global_step, best_eval_result
            if eval_dataset is None:
                return None
            if not force and (current_step <= 0 or current_step % resolved_eval_step != 0):
                return None

            if verbose:
                print(f"  Running evaluation at global_step={current_step}...")

            eval_result = self.evaluate(
                eval_dataset,
                verbose=verbose,
                epoch_num=epoch + 1,
                global_step=current_step,
            )

            if verbose:
                richness_suffix = ""
                if eval_result.richness_score_mean is not None:
                    richness_suffix = f", richness={eval_result.richness_score_mean:.4f}"
                print(
                    f"  → eval:  acc={eval_result.accuracy:.3f}, "
                    f"avg_loss={eval_result.avg_loss:.4f}, "
                    f"mean_θ=[PI={eval_result.mean_theta[0]:.3f}, "
                    f"MN={eval_result.mean_theta[1]:.3f}, "
                    f"PC={eval_result.mean_theta[2]:.3f}]"
                    f"{richness_suffix}"
                )

            if save_dir:
                self._save_eval_checkpoint(
                    save_dir=save_dir,
                    epoch=epoch,
                    global_step=current_step,
                    eval_result=eval_result,
                    train_result=train_result,
                )
                if verbose:
                    print(
                        f"  saved eval checkpoint: "
                        f"{Path(save_dir) / f'gate_step-{current_step}.pt'}"
                    )

            is_better_accuracy = eval_result.accuracy > best_eval_accuracy
            is_tie_with_lower_loss = (
                eval_result.accuracy == best_eval_accuracy
                and eval_result.avg_loss < best_eval_loss
            )
            if save_best and (is_better_accuracy or is_tie_with_lower_loss):
                best_eval_accuracy = eval_result.accuracy
                best_eval_loss = eval_result.avg_loss
                best_epoch = epoch
                best_global_step = current_step
                best_eval_result = eval_result
                best_path = str(Path(save_dir) / "gate_best.pt")
                self.gate.save(best_path)
                config_path = Path(save_dir) / "config.json"
                config_path.write_text(
                    json.dumps(self._export_checkpoint_config(), indent=2)
                )
                meta = {
                    "epoch": epoch,
                    "global_step": current_step,
                    "eval_accuracy": eval_result.accuracy,
                    "eval_loss": eval_result.avg_loss,
                    "train_loss": None,
                    "train_accuracy": None,
                    "temperature": self.gate.temperature,
                    "mean_theta": eval_result.mean_theta,
                    "schema_validity": eval_result.schema_validity,
                    "richness_score_mean": eval_result.richness_score_mean,
                    "schema_similarity_means": eval_result.schema_similarity_means,
                }
                meta_path = Path(save_dir) / "best_meta.json"
                meta_path.write_text(json.dumps(meta, indent=2))

                if verbose:
                    print(
                        "  ★ New best model saved "
                        f"(eval_acc={best_eval_accuracy:.3f}, "
                        f"eval_loss={best_eval_loss:.4f}, "
                        f"step={best_global_step})"
                    )

            return eval_result

        skip_initial_eval_for_debug = os.getenv("SIEVE_DEBUG_PDB", "0") == "1"
        if eval_dataset is not None and resume_state is None and not skip_initial_eval_for_debug:
            if verbose:
                print("Running initial evaluation at step=0...")
            initial_eval_result = run_scheduled_eval(
                current_step=0,
                epoch=-1,
                force=True,
            )
            if verbose:
                richness_suffix = ""
                if initial_eval_result.richness_score_mean is not None:
                    richness_suffix = f", richness={initial_eval_result.richness_score_mean:.4f}"
                print(
                    f"  → initial eval: acc={initial_eval_result.accuracy:.3f}, "
                    f"avg_loss={initial_eval_result.avg_loss:.4f}, "
                    f"mean_θ=[PI={initial_eval_result.mean_theta[0]:.3f}, "
                    f"MN={initial_eval_result.mean_theta[1]:.3f}, "
                    f"PC={initial_eval_result.mean_theta[2]:.3f}]"
                    f"{richness_suffix}"
                )

            if save_best:
                best_eval_accuracy = initial_eval_result.accuracy
                best_eval_loss = initial_eval_result.avg_loss
                best_epoch = -1
                best_global_step = 0
                best_eval_result = initial_eval_result
                best_path = str(Path(save_dir) / "gate_best.pt")
                self.gate.save(best_path)
                config_path = Path(save_dir) / "config.json"
                config_path.write_text(
                    json.dumps(self._export_checkpoint_config(), indent=2)
                )
                meta = {
                    "epoch": -1,
                    "eval_accuracy": initial_eval_result.accuracy,
                    "eval_loss": initial_eval_result.avg_loss,
                    "train_loss": None,
                    "train_accuracy": None,
                    "temperature": self.gate.temperature,
                    "mean_theta": initial_eval_result.mean_theta,
                    "schema_validity": initial_eval_result.schema_validity,
                    "richness_score_mean": initial_eval_result.richness_score_mean,
                    "schema_similarity_means": initial_eval_result.schema_similarity_means,
                    "global_step": 0,
                }
                meta_path = Path(save_dir) / "best_meta.json"
                meta_path.write_text(json.dumps(meta, indent=2))
                if verbose:
                    print(
                        "  ★ Initial model saved as current best "
                        f"(eval_acc={best_eval_accuracy:.3f}, "
                        f"eval_loss={best_eval_loss:.4f})"
                    )

            if verbose:
                print()
        elif eval_dataset is not None and resume_state is None and skip_initial_eval_for_debug and verbose:
            print("Skipping initial evaluation because SIEVE_DEBUG_PDB=1")

        total_planned_epochs = completed_epochs + num_epochs
        for epoch in range(completed_epochs, total_planned_epochs):
            scheduled_temp = self._set_epoch_temperature(epoch, total_planned_epochs)
            if verbose:
                print(f"Epoch {epoch + 1}/{total_planned_epochs} (τ={scheduled_temp:.3f})")

            # --- Train ---
            epoch_result = self.train_epoch(
                grpo_data,
                epoch_num=epoch,
                verbose=verbose,
                progress_bar=progress_bar,
                global_step_start=epoch * len(grpo_data),
                on_step_end=lambda current_step, epoch_num=epoch: run_scheduled_eval(
                    current_step=current_step,
                    epoch=epoch_num,
                    train_result=None,
                ),
            )
            if verbose:
                richness_suffix = ""
                if epoch_result.richness_score_mean is not None:
                    richness_suffix = (
                        f", richness={epoch_result.richness_score_mean:.4f}, "
                        f"richness_adv={epoch_result.richness_advantage_mean:.4f}"
                    )
                print(
                    f"  → train: acc={epoch_result.accuracy:.3f}, "
                    f"loss={epoch_result.avg_loss:.4f}, "
                    f"reward={epoch_result.avg_reward:.4f}"
                    f"{richness_suffix}"
                )

            self._log_metrics(
                {
                    "train/epoch_accuracy": epoch_result.accuracy,
                    "train/epoch_single_accuracy": epoch_result.single_accuracy,
                    "train/epoch_strict_accuracy": epoch_result.strict_accuracy,
                    "train/epoch_avg_advantage": epoch_result.avg_advantage,
                    "train/epoch_avg_reward": epoch_result.avg_reward,
                    "train/epoch_avg_loss": epoch_result.avg_loss,
                    "train/epoch_avg_entropy": epoch_result.avg_entropy,
                    "train/epoch_informative_count": epoch_result.informative_count,
                    "train/epoch_informative_ratio": epoch_result.informative_ratio,
                    "train/epoch_richness_score_mean": epoch_result.richness_score_mean,
                    "train/epoch_richness_advantage_mean": epoch_result.richness_advantage_mean,
                    "train/epoch_schema_match": epoch_result.schema_match,
                    "train/epoch_single_schema_match": epoch_result.single_schema_match,
                    "train/epoch_entropy_reg_term_dominates_ratio": (
                        epoch_result.entropy_reg_term_dominates_ratio
                    ),
                    "train/epoch_temperature": epoch_result.temperature,
                    "train/epoch_learning_rate": self.optimizer.param_groups[0]["lr"],
                    **{
                        f"train/epoch_response_argument_similarity_{schema}": (
                            epoch_result.schema_similarity_means.get(schema)
                        )
                        for schema in SCHEMA_NAMES
                    },
                    **{
                        f"train/epoch_per_group_reward_{schema}": (
                            epoch_result.per_group_rewards.get(schema, 0.0)
                        )
                        for schema in SCHEMA_NAMES
                    },
                    **{
                        f"train/epoch_theta_mean_{schema}": epoch_result.mean_theta[idx]
                        for idx, schema in enumerate(SCHEMA_NAMES)
                    },
                    **{
                        f"train/epoch_cross_sim_{pair_name}": sim_value
                        for pair_name, sim_value in epoch_result.cross_schema_similarities.items()
                    },
                    "train/group_reward_PI": epoch_result.per_group_rewards.get("PI", 0.0),
                    "train/group_reward_MN": epoch_result.per_group_rewards.get("MN", 0.0),
                    "train/group_reward_PC": epoch_result.per_group_rewards.get("PC", 0.0),
                    "train/epoch": epoch + 1,
                },
                step=(epoch + 1) * len(grpo_data),
            )

            if verbose:
                print(
                    f"  → train: acc={epoch_result.accuracy:.3f}, "
                    f"strict_acc={epoch_result.strict_accuracy:.3f}, "
                    f"avg_adv={epoch_result.avg_advantage:.3f}, "
                    f"avg_loss={epoch_result.avg_loss:.4f}, "
                    f"informative={epoch_result.informative_count}/{len(grpo_data)} "
                    f"({epoch_result.informative_ratio:.3f}), "
                    f"τ={epoch_result.temperature:.3f}"
                )

            if save_last and save_dir:
                self._save_last_checkpoint(
                    save_dir=save_dir,
                    dataset=dataset,
                    eval_dataset=eval_dataset,
                    split_state=split_state,
                    completed_epochs=epoch + 1,
                    total_planned_epochs=total_planned_epochs,
                    best_eval_accuracy=best_eval_accuracy,
                    best_eval_loss=best_eval_loss,
                    best_epoch=best_epoch,
                    latest_epoch_result=epoch_result,
                )

            if verbose:
                print()

            results.append(epoch_result)

        progress_bar.close()

        # Summary
        if verbose and eval_dataset is not None and save_best:
            print(f"{'='*60}")
            print(f"Training complete.")
            print(
                f"Best model: epoch {best_epoch + 1}, "
                f"eval_acc={best_eval_accuracy:.3f}, "
                f"eval_loss={best_eval_loss:.4f}"
            )
            if best_global_step >= 0:
                print(f"  global_step={best_global_step}")
            if best_eval_result is not None:
                print(f"  eval_acc={best_eval_result.accuracy:.3f}")
                print(f"  eval_loss={best_eval_result.avg_loss:.4f}")
                print(f"  saved to: {Path(save_dir) / 'gate_best.pt'}")
            print(f"{'='*60}")

        return results
