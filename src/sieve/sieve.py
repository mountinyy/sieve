from __future__ import annotations

"""
PRISM: Main Manager.
Orchestrates the full pipeline — training, inference, evaluation, analysis.

Usage:
    from prism import PRISM

    # Initialize
    prism = PRISM(
        llm=your_llm_client,
        gate_model="Qwen/Qwen3-1.7B-Base",
        cache_dir="./cache/extractions",
    )

    # Train
    prism.train(train_data, num_epochs=10, k=4)

    # Evaluate
    results = prism.evaluate(test_data)

    # Single inference
    result = prism.predict("A camp has a no-cannonball rule...")

    # Save / Load
    prism.save("./checkpoints/prism_v1.pt")
    prism.load("./checkpoints/prism_v1.pt")

    # Analysis
    analysis = prism.analyze(test_data)
"""

from pathlib import Path
import json
from threading import Lock
import torch

from src.sieve.data_types import SCHEMA_NAMES, InferenceResult
from src.sieve.llm_client import LLMClient
from src.sieve.gate_module import GateModule
from src.sieve.extraction import (
    ExtractionCache,
    UNIFIED_ARGUMENT_BUDGET_M,
    UNIFIED_ARGUMENT_EXTRACTION_N,
)
from src.sieve.trainer import GRPOTrainer, EpochResult
from src.sieve.evaluation import (
    PreparedInference,
    complete_inference,
    evaluate,
    evaluate_transfer,
    inference,
    prepare_inference,
    prepare_inference_batch,
    prepare_theta_batch,
    record_theta_statistics,
    swap_high_low_theta,
)
import logging

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logging.getLogger("httpx").setLevel(logging.WARNING)


class GeneratorAdapter(LLMClient):
    """Wrap a generator object exposing `generate(prompt, max_tokens=...)`."""

    def __init__(self, generator):
        if not hasattr(generator, "generate"):
            raise TypeError("generator must expose a `generate(prompt, max_tokens=...)` method.")
        self.generator = generator
        self.tokenizer = getattr(generator, "tokenizer", None)

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        return self.generator.generate(prompt, max_tokens=max_tokens)

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int = 1024,
        temperature: float | None = None,
        save_dir: str | Path | None = None,
        batch_name: str = "batch",
    ) -> list[str]:
        if hasattr(self.generator, "generate_batch"):
            return self.generator.generate_batch(
                prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                save_dir=save_dir,
                batch_name=batch_name,
            )
        return super().generate_batch(
            prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            save_dir=save_dir,
            batch_name=batch_name,
        )


class SIEVE:
    """
    Principled Reasoning via Informed Schema Modulation.

    A framework for improving LLM moral reasoning through learned
    schema-based information gating.

    Components:
        - GateModule: scenario → θ (learnable)
        - ExtractionCache: schema-specific fact/principle extraction (cached)
        - InformationGate: θ → filtered info (deterministic)
        - Reasoner: filtered info → answer (frozen LLM)
    """

    def __init__(
        self,
        llm: LLMClient,
        gate_model: str = "Qwen/Qwen3-1.7B-Base",
        cache_dir: str = None,
        hidden_dim: int = 256,
        initial_temperature: float = 2.0,
        gumbel_noise_scale: float = 1.0,
        max_length: int = 512,
        device: str = None,
        N: int = 5,
        budget_M: int = 5,
        use_persona: bool = False,
        inst_regime: bool = False,
        uniform_theta: bool = False,
        dominant_schema: str | None = None,
        allowed_cache_ids: set[str] | None = None,
        use_alignment_adv: bool = False,
        use_richness: bool = False,
        richness_alpha: float = 10.0,
        enable_alignment_runtime: bool = True,
        inference_add_eval: bool = True,
        extract_direction: bool = True,
        use_token_total_budget: bool = False,
        unified_argument: bool = False,
        use_all: bool = False,
        use_top: bool = False,
        use_bottom: bool = False,
        random_theta: bool = False,
        schema_bias: str | None = "none",
        swap: bool = False,
        safety: bool = False,
        load_gate: bool = True,
        cache_max_concurrency: int | None = None,
    ):
        """
        Args:
            llm: LLM client for extraction and reasoning
            gate_model: HuggingFace model name for gate backbone
            cache_dir: directory to cache extractions on disk
            hidden_dim: MLP hidden dimension
            initial_temperature: softmax temperature for gate
            max_length: max token length for gate encoder
            device: "cuda" or "cpu"
        """
        self.llm = llm
        self.gate_model_name = gate_model
        self.hidden_dim = hidden_dim
        self.initial_temperature = initial_temperature
        self.gumbel_noise_scale = float(gumbel_noise_scale)
        self.max_length = max_length
        effective_extraction_n = (
            UNIFIED_ARGUMENT_EXTRACTION_N if unified_argument else N
        )
        effective_budget_m = (
            UNIFIED_ARGUMENT_BUDGET_M if unified_argument else budget_M
        )
        self.extraction_N = effective_extraction_n
        self.budget_M = effective_budget_m
        self.use_persona = use_persona
        self.inst_regime = inst_regime
        self.uniform_theta = uniform_theta
        self.dominant_schema = dominant_schema
        self.use_alignment_adv = bool(use_alignment_adv)
        self.use_richness = bool(use_richness)
        self.richness_alpha = float(richness_alpha)
        self.enable_alignment_runtime = bool(enable_alignment_runtime)
        self.inference_add_eval = bool(inference_add_eval)
        self.extract_direction = bool(extract_direction)
        self.use_token_total_budget = bool(use_token_total_budget)
        self.unified_argument = bool(unified_argument)
        self.use_all = bool(use_all)
        self.use_top = bool(use_top)
        self.use_bottom = bool(use_bottom)
        self.random_theta = bool(random_theta)
        self.schema_bias = self._normalize_schema_bias(schema_bias)
        self.swap = bool(swap)
        self.safety = bool(safety)
        self.alignment_encoder = None
        self.alignment_encoder_lock = Lock()

        needs_alignment_runtime = self.use_alignment_adv or self.use_richness
        if (
            self.enable_alignment_runtime
            and needs_alignment_runtime
            and SentenceTransformer is not None
        ):
            encoder_device = (
                str(device)
                if device is not None
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            self.alignment_encoder = SentenceTransformer(
                "all-mpnet-base-v2",
                device=encoder_device,
            )
        elif needs_alignment_runtime and self.enable_alignment_runtime:
            raise ImportError(
                "use_alignment_adv=True or use_richness=True requires the `sentence-transformers` package."
            )

        self.gate = None
        if load_gate:
            self.gate = GateModule(
                model_name=gate_model,
                hidden_dim=hidden_dim,
                initial_temperature=initial_temperature,
                gumbel_noise_scale=gumbel_noise_scale,
                max_length=max_length,
                device=device,
                uniform_theta=uniform_theta,
                dominant_schema=dominant_schema,
            )

        self.cache = ExtractionCache(
            llm,
            cache_dir=cache_dir,
            N=effective_extraction_n,
            total_budget=effective_budget_m,
            allowed_cache_ids=allowed_cache_ids,
            use_alignment_adv=self.use_alignment_adv,
            use_richness=self.use_richness,
            allow_variable_count=True,
            alignment_encoder=self.alignment_encoder,
            alignment_lock=self.alignment_encoder_lock,
            compute_alignment_embeddings=self.enable_alignment_runtime,
            extract_direction=self.extract_direction,
            unified_argument=self.unified_argument,
            safety=self.safety,
            max_concurrency=cache_max_concurrency,
        )

        # Trainer (initialized on first train() call)
        self._trainer = None

        print(f"===== SIEVE initialized =====")
        print(f"  Gate backbone: {gate_model}")
        if self.gate is not None:
            params = self.gate.count_parameters()
            print(f"  Learnable params: {params['total_learnable']:,}")
            print(f"  Backbone params: {params['backbone_frozen']:,} (frozen)")
        else:
            print("  Gate runtime: skipped (fixed/random inference path)")
        print(f"  Cache dir: {cache_dir or '(memory only)'}")
        print(f"  Cache max concurrency: {self.cache.max_concurrency}")
        resolved_device = (
            self.gate.device
            if self.gate is not None
            else (device or ("cuda" if torch.cuda.is_available() else "cpu"))
        )
        print(f"  Device: {resolved_device}")
        print(f"  Extraction N: {self.extraction_N}")
        print(f"  Budget M: {self.budget_M}")
        print(f"  use_alignment_adv: {self.use_alignment_adv}")
        print(f"  use_richness: {self.use_richness}")
        print(f"  use_token_total_budget: {self.use_token_total_budget}")
        print(f"  unified_argument: {self.unified_argument}")
        print(f"  use_all: {self.use_all}")
        print(f"  use_top: {self.use_top}")
        print(f"  use_bottom: {self.use_bottom}")
        print(f"  random_theta: {self.random_theta}")
        print(f"  schema_bias: {self.schema_bias or 'none'}")
        print(f"  swap: {self.swap}")
        print(f"  safety: {self.safety}")
        print(f"  richness_alpha: {self.richness_alpha}")
        print(f"  enable_alignment_runtime: {self.enable_alignment_runtime}")
        print(f"==============================")

    @staticmethod
    def _normalize_schema_bias(schema_bias: str | None) -> str | None:
        if schema_bias is None:
            return None
        normalized = str(schema_bias).strip().upper()
        if normalized in {"", "NONE", "NULL"}:
            return None
        if normalized not in SCHEMA_NAMES:
            raise ValueError(
                f"schema_bias must be one of none, {', '.join(SCHEMA_NAMES)}; got {schema_bias!r}"
            )
        return normalized

    def _apply_schema_bias(self, theta: list[float]) -> list[float]:
        theta_list = [float(value) for value in theta]
        if self.schema_bias is not None:
            theta_list = self._schema_bias_theta()
        return self._apply_swap(theta_list)

    def _apply_schema_bias_batch(self, theta_items: list[list[float]]) -> list[list[float]]:
        return [self._apply_schema_bias(theta) for theta in theta_items]

    def _apply_swap(self, theta: list[float]) -> list[float]:
        theta_list = [float(value) for value in theta]
        return swap_high_low_theta(theta_list) if self.swap else theta_list

    def _schema_bias_theta(self) -> list[float]:
        if self.schema_bias is None:
            raise RuntimeError("schema_bias theta requested without an active schema_bias.")
        return [
            0.8 if schema == self.schema_bias else 0.1
            for schema in SCHEMA_NAMES
        ]

    def _uses_multi_llm_trainer(self) -> bool:
        return self._trainer is not None and getattr(self._trainer, "multi_llm", False)

    def _trainer_eval_to_dict(self, eval_result) -> dict:
        return {
            "metrics": {
                "accuracy": eval_result.accuracy,
                "avg_loss": eval_result.avg_loss,
                "n_total": eval_result.n_total,
                "n_correct": eval_result.n_correct,
                "mean_theta": eval_result.mean_theta,
                "avg_theta_entropy": eval_result.avg_theta_entropy,
                "schema_validity": eval_result.schema_validity,
                "schema_match": eval_result.schema_match,
                "avg_schema_sensitivity": eval_result.avg_schema_sensitivity,
                "informative_count": eval_result.informative_count,
                "informative_ratio": eval_result.informative_ratio,
                "dominant_schema_ratio": eval_result.dominant_schema_ratio,
                "cross_schema_similarities": eval_result.cross_schema_similarities,
            },
            "results": eval_result.per_item,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, scenario: str, scenario_id: str = "0") -> InferenceResult:
        """
        Run PRISM inference on a single scenario.

        Returns InferenceResult with answer, reasoning, theta, gate_stats.
        """
        if self.schema_bias is not None:
            prepared = self.prepare_generation_with_theta(
                scenario,
                self._apply_swap(self._schema_bias_theta()),
                scenario_id=scenario_id,
            )
            raw = prepared.llm.generate(prepared.prompt, max_tokens=prepared.max_tokens)
            return self.complete_generation(prepared, raw)
        if self.gate is not None:
            self.gate.eval()
        return inference(
            scenario,
            scenario_id,
            self.gate,
            self.llm,
            self.cache,
            use_persona=self.use_persona,
            inst_regime=self.inst_regime,
            inference_add_eval=self.inference_add_eval,
            token_proportional=True,
            use_token_total_budget=self.use_token_total_budget,
            use_all=self.use_all,
            use_top=self.use_top,
            use_bottom=self.use_bottom,
            safety=self.safety,
            uniform_theta=self.uniform_theta,
            random_theta=self.random_theta,
            swap_theta=self.swap,
            use_cache=False,
        )

    def prepare_generation(
        self,
        scenario: str,
        scenario_id: str = "0",
        max_tokens: int = 1024,
    ) -> PreparedInference:
        """
        Prepare SIEVE inference through prompt assembly without generating.

        This is useful for parallel benchmark execution: gate/cache/prompt work can
        be serialized by the caller, while remote vLLM generation can run outside
        that lock.
        """
        if self.schema_bias is not None:
            return self.prepare_generation_with_theta(
                scenario,
                self._apply_swap(self._schema_bias_theta()),
                scenario_id=scenario_id,
                max_tokens=max_tokens,
            )
        if self.gate is not None:
            self.gate.eval()
        return prepare_inference(
            scenario,
            scenario_id,
            self.gate,
            self.llm,
            self.cache,
            use_persona=self.use_persona,
            inst_regime=self.inst_regime,
            inference_add_eval=self.inference_add_eval,
            token_proportional=True,
            use_token_total_budget=self.use_token_total_budget,
            use_all=self.use_all,
            use_top=self.use_top,
            use_bottom=self.use_bottom,
            safety=self.safety,
            uniform_theta=self.uniform_theta,
            random_theta=self.random_theta,
            swap_theta=self.swap,
            use_cache=False,
            max_tokens=max_tokens,
        )

    def prepare_generation_with_theta(
        self,
        scenario: str,
        theta: list[float],
        scenario_id: str = "0",
        max_tokens: int = 1024,
    ) -> PreparedInference:
        """Prepare SIEVE prompt state from a precomputed theta vector."""
        return prepare_inference(
            scenario,
            scenario_id,
            None,
            self.llm,
            self.cache,
            use_persona=self.use_persona,
            inst_regime=self.inst_regime,
            inference_add_eval=self.inference_add_eval,
            token_proportional=True,
            use_token_total_budget=self.use_token_total_budget,
            use_all=self.use_all,
            use_top=self.use_top,
            use_bottom=self.use_bottom,
            safety=self.safety,
            uniform_theta=self.uniform_theta,
            random_theta=self.random_theta,
            swap_theta=False,
            use_cache=True,
            max_tokens=max_tokens,
            theta_override=theta,
        )

    def prepare_theta_batch(
        self,
        scenarios: list[str],
        batch_size: int = 32,
        verbose: bool = True,
        desc: str = "SIEVE theta precompute",
    ) -> list[list[float]]:
        """
        Compute only gate theta values for benchmark inputs.

        This intentionally does not run argument extraction, prompt assembly, or
        cache writes; benchmark argument generation happens later with the target
        LLM runtime.
        """
        if self.gate is not None:
            self.gate.eval()
        if self.schema_bias is not None:
            theta = self._schema_bias_theta()
            theta_items = [self._apply_swap(theta) for _ in scenarios]
            record_theta_statistics(desc, theta_items)
            return theta_items
        if self.gate is None and self.dominant_schema in SCHEMA_NAMES:
            theta = [
                1.0 if schema == self.dominant_schema else 0.0
                for schema in SCHEMA_NAMES
            ]
            theta_items = self._apply_schema_bias_batch([list(theta) for _ in scenarios])
            record_theta_statistics(desc, theta_items)
            return theta_items
        theta_items = prepare_theta_batch(
            scenarios,
            self.gate,
            use_all=self.use_all,
            uniform_theta=self.uniform_theta,
            random_theta=self.random_theta,
            swap_theta=self.swap,
            batch_size=batch_size,
            verbose=verbose,
            desc=desc,
            record_stats=False,
        )
        if self.schema_bias is not None:
            theta_items = self._apply_schema_bias_batch(theta_items)
        record_theta_statistics(desc, theta_items)
        return theta_items

    def release_gate(self) -> None:
        """Unload the gate model so benchmark vLLM serving can reuse the GPU."""
        if self.gate is not None:
            try:
                self.gate.to("cpu")
            except Exception:
                pass
            self.gate = None
        if self.alignment_encoder is not None:
            try:
                self.alignment_encoder.to("cpu")
            except Exception:
                pass
            self.alignment_encoder = None
            self.cache.alignment_encoder = None
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        print("[INFO] Released SIEVE gate runtime and cleared CUDA cache.")

    def prepare_generation_batch(
        self,
        scenarios: list[str],
        scenario_ids: list[str],
        max_tokens: int = 1024,
        batch_size: int = 16,
        verbose: bool = True,
        desc: str = "SIEVE gate/prompt precompute",
    ) -> list[PreparedInference]:
        """
        Prepare SIEVE prompts for many inputs before generation.

        The gate forward pass is batched, while cache lookup and prompt assembly are
        completed up front so benchmark runners can send only remote vLLM requests
        in the parallel phase.
        """
        if self.schema_bias is not None:
            return [
                self.prepare_generation_with_theta(
                    scenario,
                    self._apply_swap(self._schema_bias_theta()),
                    scenario_id=scenario_id,
                    max_tokens=max_tokens,
                )
                for scenario, scenario_id in zip(scenarios, scenario_ids)
            ]
        if self.gate is not None:
            self.gate.eval()
        return prepare_inference_batch(
            scenarios,
            scenario_ids,
            self.gate,
            self.llm,
            self.cache,
            use_persona=self.use_persona,
            inst_regime=self.inst_regime,
            inference_add_eval=self.inference_add_eval,
            token_proportional=True,
            use_token_total_budget=self.use_token_total_budget,
            use_all=self.use_all,
            use_top=self.use_top,
            use_bottom=self.use_bottom,
            safety=self.safety,
            uniform_theta=self.uniform_theta,
            random_theta=self.random_theta,
            swap_theta=self.swap,
            use_cache=False,
            max_tokens=max_tokens,
            batch_size=batch_size,
            verbose=verbose,
            desc=desc,
        )

    def complete_generation(
        self,
        prepared: PreparedInference,
        raw_response: str,
    ) -> InferenceResult:
        """Complete a prepared SIEVE inference from a raw reasoner response."""
        return complete_inference(prepared, raw_response)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        dataset: list[dict],
        num_epochs: int = 10,
        eval_dataset: list[dict] = None,
        eval_step: int = 0,
        k: int = 4,
        learning_rate: float = 5e-4,
        temperature_decay: float = 0.95,
        min_temperature: float = 0.5,
        final_temperature: float | None = None,
        save_best: bool = True,
        save_last: bool = False,
        save_dir: str = "./checkpoints",
        verbose: bool = True,
        use_tracking: bool = False,
        tracking_project: str | None = None,
        tracking_run_name: str | None = None,
        tracking_entity: str | None = None,
        tracking_tags: list[str] | None = None,
        tracking_config: dict | None = None,
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
        use_warmup: bool = False,
        warm_start_size: int = 600,
        warm_start_epochs: int = 1,
        warm_start_lr: float = 1e-3,
        warm_start_batch_size: int | None = None,
        warmup_only: bool = False,
        use_persona: bool = False,
        inst_regime: bool = False,
        uniform_theta: bool = False,
        dominant_schema: str | None = None,
        multi_llm: bool = False,
        multi_llm_clients: dict[str, LLMClient] | None = None,
        multi_llm_caches: dict[str, ExtractionCache] | None = None,
        resume_from_checkpoint: str | None = None,
        use_alignment_adv: bool = False,
        use_richness: bool = False,
        use_token_total_budget: bool = False,
        unified_argument: bool = False,
        richness_alpha: float = 10.0,
        richness_weight: float = 0.5,
        inference_add_eval: bool | None = None,
        lambda_comp: float = 0.5,
        csa_mode: str = "discrete",
        filter_zero_influence: bool = True,
    ) -> list[EpochResult]:
        """
        Train the gate module via GRPO.

        Dataset format: [{"id": str, "scenario": str, "gold_label": str}, ...]

        Args:
            dataset: training data
            num_epochs: number of training epochs
            k: GRPO samples per scenario
            learning_rate: Adam learning rate
            temperature_decay: per-epoch temperature decay
            min_temperature: minimum temperature floor
            verbose: print progress

        Returns:
            list of EpochResult
        """
        self.use_persona = use_persona
        self.inst_regime = inst_regime
        if inference_add_eval is not None:
            self.inference_add_eval = bool(inference_add_eval)
        if bool(use_alignment_adv) != self.use_alignment_adv:
            raise ValueError(
                "SIEVE.train use_alignment_adv must match the SIEVE instance configuration."
            )
        if bool(use_richness) != self.use_richness:
            raise ValueError(
                "SIEVE.train use_richness must match the SIEVE instance configuration."
            )
        if bool(use_token_total_budget) != self.use_token_total_budget:
            raise ValueError(
                "SIEVE.train use_token_total_budget must match the SIEVE instance configuration."
            )
        if bool(unified_argument) != self.unified_argument:
            raise ValueError(
                "SIEVE.train unified_argument must match the SIEVE instance configuration."
            )
        if self.gate is None:
            raise RuntimeError("Training requires a loaded gate module.")
        self._trainer = GRPOTrainer(
            gate_module=self.gate,
            llm=self.llm,
            extraction_cache=self.cache,
            k=k,
            learning_rate=learning_rate,
            temperature_decay=temperature_decay,
            min_temperature=min_temperature,
            final_temperature=final_temperature,
            use_tracking=use_tracking,
            tracking_project=tracking_project,
            tracking_run_name=tracking_run_name,
            tracking_entity=tracking_entity,
            tracking_tags=tracking_tags,
            tracking_config=tracking_config,
            llm_max_concurrency=llm_max_concurrency,
            batch_size=batch_size,
            rho_init=rho_init,
            rho_momentum=rho_momentum,
            seed=seed,
            entropy_reg_alpha=entropy_reg_alpha,
            use_entropy_loss=use_entropy_loss,
            entropy_loss_beta=entropy_loss_beta,
            use_kl=use_kl,
            kl_weight=kl_weight,
            use_clip=use_clip,
            clip_epsilon=clip_epsilon,
            continuous_group_reward=continuous_group_reward,
            no_group_reward=no_group_reward,
            informative_sigma_threshold=informative_sigma_threshold,
            use_lr_scheduler=use_lr_scheduler,
            lr_scheduler_eta_min=lr_scheduler_eta_min,
            use_persona=use_persona,
            inst_regime=inst_regime,
            inference_add_eval=self.inference_add_eval,
            use_token_total_budget=self.use_token_total_budget,
            use_alignment_adv=use_alignment_adv,
            use_richness=use_richness,
            richness_alpha=richness_alpha,
            richness_weight=richness_weight,
            multi_llm=multi_llm,
            llm_pool=multi_llm_clients,
            cache_pool=multi_llm_caches,
            alignment_encoder=self.alignment_encoder,
            alignment_lock=self.alignment_encoder_lock,
            lambda_comp=lambda_comp,
            csa_mode=csa_mode,
            filter_zero_influence=filter_zero_influence,
        )

        return self._trainer.train(
            dataset, 
            num_epochs=num_epochs, 
            eval_dataset=eval_dataset, 
            eval_step=eval_step,
            save_best=save_best,
            save_last=save_last,
            save_dir=save_dir,
            verbose=verbose,
            use_warmup=use_warmup,
            warm_start_size=warm_start_size,
            warm_start_epochs=warm_start_epochs,
            warm_start_lr=warm_start_lr,
            warm_start_batch_size=warm_start_batch_size,
            warmup_only=warmup_only,
            resume_from_checkpoint=resume_from_checkpoint,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataset: list[dict],
        verbose: bool = True,
        batch_size: int = 1,
    ) -> dict:
        """
        Evaluate on a test dataset.

        Returns: {"metrics": {...}, "results": [{...}, ...]}
        """
        if self._uses_multi_llm_trainer():
            eval_result = self._trainer.evaluate(
                dataset,
                verbose=verbose,
            )
            return self._trainer_eval_to_dict(eval_result)

        return evaluate(
            dataset,
            self.gate,
            self.llm,
            self.cache,
            verbose,
            batch_size=batch_size,
            use_persona=self.use_persona,
            inst_regime=self.inst_regime,
            inference_add_eval=self.inference_add_eval,
            token_proportional=True,
            use_token_total_budget=self.use_token_total_budget,
            use_all=self.use_all,
            use_top=self.use_top,
            use_bottom=self.use_bottom,
            safety=self.safety,
            uniform_theta=self.uniform_theta,
            random_theta=self.random_theta,
            swap_theta=self.swap,
        )

    def evaluate_transfer(
        self,
        dataset: list[dict],
        target_llm: LLMClient,
        target_cache_dir: str = None,
        verbose: bool = True,
        batch_size: int = 1,
    ) -> dict:
        """
        Cross-model transfer: same gate, different LLM.

        Returns comparison of source vs target performance.
        """
        return evaluate_transfer(
            dataset,
            self.gate,
            source_llm=self.llm,
            target_llm=target_llm,
            target_cache_dir=target_cache_dir,
            verbose=verbose,
            batch_size=batch_size,
            use_persona=self.use_persona,
            inst_regime=self.inst_regime,
            inference_add_eval=self.inference_add_eval,
            token_proportional=True,
            use_token_total_budget=self.use_token_total_budget,
            use_all=self.use_all,
        )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        dataset: list[dict],
        verbose: bool = True,
        batch_size: int = 1,
        eval_result: dict | None = None,
    ) -> dict:
        """
        Run evaluation + detailed theta/gate analysis.

        Returns evaluation results plus:
            - theta distribution per scenario category (if "category" field exists)
            - gate openness analysis
            - error analysis
        """
        if eval_result is None:
            eval_result = self.evaluate(dataset, verbose=verbose, batch_size=batch_size)
        results = eval_result["results"]

        analysis = {"eval_metrics": eval_result["metrics"]}

        # Theta distribution by category
        if any("category" in item for item in dataset):
            category_thetas = {}
            for item, result in zip(dataset, results):
                cat = item.get("category", "unknown")
                if cat not in category_thetas:
                    category_thetas[cat] = {"thetas": [], "accuracy": []}
                category_thetas[cat]["thetas"].append(result["theta"])
                category_thetas[cat]["accuracy"].append(int(result["correct"]))

            import torch
            category_summary = {}
            for cat, data in category_thetas.items():
                thetas = torch.tensor(data["thetas"])
                category_summary[cat] = {
                    "n": len(data["thetas"]),
                    "accuracy": sum(data["accuracy"]) / len(data["accuracy"]),
                    "mean_theta": {
                        name: round(thetas[:, i].mean().item(), 4)
                        for i, name in enumerate(SCHEMA_NAMES)
                    },
                    "std_theta": {
                        name: round(thetas[:, i].std().item(), 4)
                        for i, name in enumerate(SCHEMA_NAMES)
                    },
                }
            analysis["category_analysis"] = category_summary

        # Error analysis
        errors = [r for r in results if not r["correct"]]
        if errors:
            import torch
            error_thetas = torch.tensor([e["theta"] for e in errors])
            analysis["error_analysis"] = {
                "n_errors": len(errors),
                "mean_theta_errors": {
                    name: round(error_thetas[:, i].mean().item(), 4)
                    for i, name in enumerate(SCHEMA_NAMES)
                },
                "error_ids": [e["id"] for e in errors],
            }

        # Gate openness (entropy of theta)
        import torch
        all_thetas = torch.tensor([r["theta"] for r in results])
        entropy = -(all_thetas * all_thetas.log()).sum(dim=-1)  # per scenario
        analysis["gate_openness"] = {
            "mean_entropy": round(entropy.mean().item(), 4),
            "std_entropy": round(entropy.std().item(), 4),
            "max_entropy": round(entropy.max().item(), 4),  # most uniform
            "min_entropy": round(entropy.min().item(), 4),  # most peaked
        }

        return analysis

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def interpret(self, scenario: str) -> dict:
        """
        Full interpretability output for a single scenario.
        Returns theta + token attention + gate stats.
        """
        if self.gate is None:
            raise RuntimeError("Interpretability requires a loaded gate module.")
        self.gate.eval()
        interp = self.gate.interpret(scenario)

        # Also run full pipeline to get gate stats
        result = self.predict(scenario)
        interp["answer"] = result.answer
        interp["gate_stats"] = result.gate_stats
        interp["reasoning"] = result.reasoning[:500]

        return interp

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save gate module weights (learnable params only)."""
        if self.gate is None:
            raise RuntimeError("Cannot save weights when no gate module is loaded.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.gate.save(path)
        (Path(path).parent / "config.json").write_text(
            json.dumps(self.export_config(), indent=2)
        )
        print(f"Gate module saved to {path}")

    def load(self, path: str):
        """Load gate module weights."""
        if self.gate is None:
            raise RuntimeError("Cannot load weights when no gate module is loaded.")
        self.gate.load(path)
        print(f"Gate module loaded from {path}")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def precompute_extractions(self, dataset: list[dict], verbose: bool = True):
        """Pre-compute and cache extractions for a dataset."""
        if verbose:
            print(f"Pre-computing extractions for {len(dataset)} scenarios...")
        self.cache.precompute(dataset, prefix="inference")
        if verbose:
            print(f"Done. {len(self.cache)} scenarios cached.")

    def export_config(self) -> dict:
        return {
            "gate_backbone_model": self.gate_model_name,
            "hidden_dim": self.hidden_dim,
            "initial_temperature": self.initial_temperature,
            "gumbel_noise_scale": self.gumbel_noise_scale,
            "max_length": self.max_length,
            "extraction_N": self.extraction_N,
            "budget_M": self.budget_M,
            "use_persona": self.use_persona,
            "inst_regime": self.inst_regime,
            "uniform_theta": self.uniform_theta,
            "dominant_schema": self.dominant_schema,
            "use_alignment_adv": self.use_alignment_adv,
            "use_richness": self.use_richness,
            "richness_alpha": self.richness_alpha,
            "use_token_total_budget": self.use_token_total_budget,
            "unified_argument": self.unified_argument,
            "use_all": self.use_all,
            "use_top": self.use_top,
            "use_bottom": self.use_bottom,
            "safety": self.safety,
        }


def _resolve_checkpoint_file(checkpoint_dir: Path) -> Path:
    preferred = checkpoint_dir / "gate_best.pt"
    if preferred.exists():
        return preferred

    last_checkpoint = checkpoint_dir / "gate_last.pt"
    if last_checkpoint.exists():
        return last_checkpoint

    candidates = sorted(checkpoint_dir.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoint found in {checkpoint_dir}")
    return candidates[0]


def _load_sieve_config(checkpoint_dir: Path) -> dict:
    config_path = checkpoint_dir / "config.json"
    default_config = {
        "hidden_dim": 256,
        "extraction_N": 5,
        "budget_M": 5,
        "initial_temperature": 1.0,
        "gumbel_noise_scale": 1.0,
        "max_length": 1024,
        "gate_backbone_model": "Qwen/Qwen3-1.7B-Base",
        "use_persona": False,
        "inst_regime": False,
        "uniform_theta": False,
        "dominant_schema": None,
        "use_alignment_adv": False,
        "use_richness": False,
        "richness_alpha": 10.0,
        "use_token_total_budget": False,
        "unified_argument": False,
        "use_top": False,
        "use_bottom": False,
        "safety": False,
    }
    if not config_path.exists():
        return default_config

    loaded = json.loads(config_path.read_text())
    return {
        "hidden_dim": loaded.get("hidden_dim", 256),
        "extraction_N": loaded.get("extraction_N", 5),
        "budget_M": loaded.get("budget_M", 5),
        "initial_temperature": loaded.get("initial_temperature", 1.0),
        "gumbel_noise_scale": loaded.get("gumbel_noise_scale", 1.0),
        "max_length": loaded.get("max_length", 1024),
        "gate_backbone_model": loaded.get("gate_backbone_model"),
        "use_persona": loaded.get("use_persona", False),
        "inst_regime": loaded.get("inst_regime", False),
        "uniform_theta": loaded.get("uniform_theta", False),
        "dominant_schema": loaded.get("dominant_schema"),
        "use_alignment_adv": loaded.get("use_alignment_adv", False),
        "use_richness": loaded.get("use_richness", False),
        "richness_alpha": loaded.get("richness_alpha", 10.0),
        "use_token_total_budget": loaded.get("use_token_total_budget", False),
        "unified_argument": loaded.get("unified_argument", False),
        "use_top": loaded.get("use_top", False),
        "use_bottom": loaded.get("use_bottom", False),
        "safety": loaded.get("safety", False),
    }


def load_sieve_for_inference(
    *,
    gate_checkpoint_path: str,
    generator,
    device: str | None = None,
    cache_dir: str | None = None,
    allowed_cache_ids: set[str] | None = None,
    inference_add_eval: bool = True,
    extract_direction: bool = True,
    gate_model: str | None = None,
    extraction_N: int | None = None,
    budget_M: int | None = None,
    use_persona: bool | None = None,
    inst_regime: bool | None = None,
    use_alignment_adv: bool | None = None,
    use_richness: bool | None = None,
    richness_alpha: float | None = None,
    use_token_total_budget: bool | None = None,
    unified_argument: bool | None = None,
    use_top: bool | None = None,
    use_top_oracle: bool = False,
    use_bottom: bool | None = None,
    use_bottom_oracle: bool = False,
    uniform_theta: bool | None = None,
    random_theta: bool = False,
    use_all: bool = False,
    schema_bias: str | None = "none",
    swap: bool = False,
    safety: bool | None = None,
) -> SIEVE:
    """
    Load a trained SIEVE gate and bind it to an inference-time generator.

    Args:
        gate_backbone_model: fallback backbone model name if config.json is absent
        gate_checkpoint_path: checkpoint directory containing `.pt` and optional config.json
        generator: object exposing `generate(prompt, max_tokens=...)`
        device: target device for the gate module
    """
    checkpoint_path = Path(gate_checkpoint_path)
    print(f"Checkpoint path: {checkpoint_path}")
    checkpoint_dir = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    config = _load_sieve_config(checkpoint_dir)
    resolved_backbone = gate_model or config.get("gate_backbone_model")
    runtime_uniform_theta = config["uniform_theta"] if uniform_theta is None else bool(uniform_theta)
    normalized_schema_bias = SIEVE._normalize_schema_bias(schema_bias)
    should_load_gate = not (
        bool(use_all)
        or bool(random_theta)
        or bool(use_top_oracle)
        or bool(use_bottom_oracle)
        or bool(runtime_uniform_theta)
        or normalized_schema_bias is not None
    )
    oracle_fixed_theta = bool(use_top_oracle) or bool(use_bottom_oracle)

    sieve = SIEVE(
        llm=GeneratorAdapter(generator),
        gate_model=resolved_backbone,
        cache_dir=cache_dir,
        hidden_dim=config["hidden_dim"],
        initial_temperature=config["initial_temperature"],
        gumbel_noise_scale=config["gumbel_noise_scale"],
        max_length=config["max_length"],
        device=device,
        N=config["extraction_N"] if extraction_N is None else int(extraction_N),
        budget_M=config["budget_M"] if budget_M is None else int(budget_M),
        use_persona=config["use_persona"] if use_persona is None else bool(use_persona),
        inst_regime=config["inst_regime"] if inst_regime is None else bool(inst_regime),
        uniform_theta=runtime_uniform_theta,
        dominant_schema=config["dominant_schema"],
        use_alignment_adv=(
            config["use_alignment_adv"]
            if use_alignment_adv is None
            else bool(use_alignment_adv)
        ),
        use_richness=config["use_richness"] if use_richness is None else bool(use_richness),
        richness_alpha=config["richness_alpha"] if richness_alpha is None else float(richness_alpha),
        use_token_total_budget=(
            config["use_token_total_budget"]
            if use_token_total_budget is None
            else bool(use_token_total_budget)
        ),
        unified_argument=(
            config["unified_argument"]
            if unified_argument is None
            else bool(unified_argument)
        ),
        use_all=False if oracle_fixed_theta else bool(use_all),
        use_top=oracle_fixed_theta or (config["use_top"] if use_top is None else bool(use_top)),
        use_bottom=False if oracle_fixed_theta else (config["use_bottom"] if use_bottom is None else bool(use_bottom)),
        enable_alignment_runtime=False,
        inference_add_eval=inference_add_eval,
        extract_direction=extract_direction,
        allowed_cache_ids=allowed_cache_ids,
        random_theta=bool(random_theta),
        schema_bias=normalized_schema_bias,
        swap=bool(swap),
        safety=config["safety"] if safety is None else bool(safety),
        load_gate=should_load_gate,
    )
    if should_load_gate:
        checkpoint_file = checkpoint_path if checkpoint_path.is_file() else _resolve_checkpoint_file(checkpoint_dir)
        sieve.load(str(checkpoint_file))
        sieve.gate.temperature = 1.0
    return sieve


def build_sieve_inference_runner(
    *,
    gate_checkpoint_path: str,
    generator,
    device: str | None = None,
    cache_dir: str | None = None,
    allowed_cache_ids: set[str] | None = None,
    inference_add_eval: bool = True,
    extract_direction: bool = True,
    gate_model: str | None = None,
    extraction_N: int | None = None,
    budget_M: int | None = None,
    use_persona: bool | None = None,
    inst_regime: bool | None = None,
    use_alignment_adv: bool | None = None,
    use_richness: bool | None = None,
    richness_alpha: float | None = None,
    use_token_total_budget: bool | None = None,
    unified_argument: bool | None = None,
    use_top: bool | None = None,
    use_top_oracle: bool = False,
    use_bottom: bool | None = None,
    use_bottom_oracle: bool = False,
    uniform_theta: bool | None = None,
    random_theta: bool = False,
    use_all: bool = False,
    schema_bias: str | None = "none",
    swap: bool = False,
    safety: bool | None = None,
) -> tuple[SIEVE, object]:
    """Load a SIEVE gate for inference and return it with the bound generator."""
    sieve = load_sieve_for_inference(
        gate_checkpoint_path=gate_checkpoint_path,
        generator=generator,
        device=device,
        cache_dir=cache_dir,
        allowed_cache_ids=allowed_cache_ids,
        inference_add_eval=inference_add_eval,
        extract_direction=extract_direction,
        gate_model=gate_model,
        extraction_N=extraction_N,
        budget_M=budget_M,
        use_persona=use_persona,
        inst_regime=inst_regime,
        use_alignment_adv=use_alignment_adv,
        use_richness=use_richness,
        richness_alpha=richness_alpha,
        use_token_total_budget=use_token_total_budget,
        unified_argument=unified_argument,
        use_top=use_top,
        use_top_oracle=use_top_oracle,
        use_bottom=use_bottom,
        use_bottom_oracle=use_bottom_oracle,
        uniform_theta=uniform_theta,
        random_theta=random_theta,
        use_all=use_all,
        schema_bias=schema_bias,
        swap=swap,
        safety=safety,
    )
    return sieve, generator


def set_sieve_dominant_schema(sieve: SIEVE, schema: str | None) -> None:
    """Update the dominant schema override for a loaded SIEVE instance."""
    normalized = None if schema is None else str(schema).upper()
    sieve.dominant_schema = normalized
    if sieve.gate is not None:
        sieve.gate.dominant_schema = normalized


def set_sieve_uniform_theta(sieve: SIEVE, enabled: bool) -> None:
    """Update the uniform-theta override for a loaded SIEVE instance."""
    uniform = bool(enabled)
    sieve.uniform_theta = uniform
    if sieve.gate is not None:
        sieve.gate.uniform_theta = uniform


def set_sieve_use_all(sieve: SIEVE, enabled: bool) -> None:
    """Update the use-all-arguments inference override for a loaded SIEVE instance."""
    sieve.use_all = bool(enabled)


def run_sieve_inference(
    sieve: SIEVE,
    input_text: str,
    input_id: str = "0",
    return_result=False,
) -> str:
    """
    Run cache-free SIEVE inference for one input and return the final response text.
    """
    if sieve.gate is not None:
        sieve.gate.temperature = 1.0
    result = sieve.predict(input_text, scenario_id=input_id)
    if return_result:
        return result
    return result.raw_response
