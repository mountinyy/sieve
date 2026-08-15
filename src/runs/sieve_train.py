import hydra
from src.utils import CONF_PATH
from src.sieve.sieve import SIEVE, LLMClient
from src.sieve.extraction import ExtractionCache
from src.sieve.llm_client import DEFAULT_REFUSAL
from src.inference import EquilibriumManager
import os
import json
import random
from src.inference import API_MODELS
from openai import OpenAI
from openai import BadRequestError
import torch


def parse_csv_values(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    normalized = str(raw_value).strip()
    if (
        len(normalized) >= 2 and
        normalized[0] == normalized[-1] and
        normalized[0] in {"'", '"'}
    ):
        normalized = normalized[1:-1]
    values = []
    for value in normalized.split(","):
        cleaned = value.strip().strip("'").strip('"')
        if cleaned:
            values.append(cleaned)
    return values


def sanitize_model_name(model_name: str) -> str:
    return model_name.strip().strip("'").strip('"').replace("/", "--")


def build_model_tag(model_names: list[str]) -> str:
    return "__".join(sanitize_model_name(name) for name in model_names)


def normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def supports_enable_thinking(model_name: str) -> bool:
    return "qwen3" in str(model_name).strip().lower()


def resolve_resume_from_checkpoint(
    raw_value: str | None,
    save_dir: str,
) -> str | None:
    normalized = str(raw_value or "").strip()
    if not normalized:
        return None

    if normalized.lower() != "auto":
        return normalized

    candidate_dir = os.path.abspath(save_dir)
    required_files = [
        os.path.join(candidate_dir, "gate_last.pt"),
        os.path.join(candidate_dir, "trainer_last.pt"),
        os.path.join(candidate_dir, "data_state.json"),
        os.path.join(candidate_dir, "last_meta.json"),
    ]
    if all(os.path.exists(path) for path in required_files):
        return candidate_dir

    print(
        "[INFO] resume_from_checkpoint=auto requested, but no complete last "
        f"checkpoint was found in {candidate_dir}. Starting a fresh run."
    )
    return None


def print_training_hyperparameter_summary(
    *,
    conf,
    use_multi_llm: bool,
    multi_model_names: list[str],
    multi_ports: list[int],
    save_path: str,
    raw_resume_from_checkpoint: str | None,
    resume_from_checkpoint: str | None,
) -> None:
    print("=" * 60)
    print("SIEVE TRAIN HYPERPARAMETERS")
    print("=" * 60)
    print(f"model_name: {conf.model_name}")
    print(f"gate_model: {conf.gate_model}")
    print(f"train_data_path: {conf.train_data_path}")
    print(f"test_data_path: {conf.test_data_path}")
    print(f"save_path: {save_path}")
    print(f"save_dir: {conf.save_dir}")
    print(f"cache_dir: {conf.cache_dir}")
    print(f"resume_from_checkpoint_raw: {raw_resume_from_checkpoint}")
    print(f"resume_from_checkpoint: {resume_from_checkpoint}")
    print(f"multi_llm: {use_multi_llm}")
    print(f"multi_llm_models: {multi_model_names}")
    print(f"multi_llm_ports: {multi_ports}")
    print(f"vllm_port: {conf.vllm_port}")
    print(f"max_model_len: {getattr(conf, 'max_model_len', None)}")
    print(f"max_gen_len: {getattr(conf, 'max_gen_len', None)}")
    print(f"num_epochs: {conf.num_epochs}")
    print(f"eval_step: {conf.eval_step}")
    print(f"sample_k: {conf.sample_k}")
    print(f"gumbel_temperature: {conf.gumbel_temperature}")
    print(f"gumbel_noise_scale: {conf.gumbel_noise_scale}")
    print(f"batch_size: {conf.batch_size}")
    print(f"llm_max_concurrency: {conf.llm_max_concurrency}")
    print(f"learning_rate: {conf.learning_rate}")
    print(f"use_lr_scheduler: {conf.use_lr_scheduler}")
    print(f"lr_scheduler_eta_min: {conf.lr_scheduler_eta_min}")
    print(f"use_warmup: {conf.use_warmup}")
    print(f"warmup_only: {conf.warmup_only}")
    print(f"warm_start_size: {conf.warm_start_size}")
    print(f"warm_start_epochs: {conf.warm_start_epochs}")
    print(f"warm_start_lr: {conf.warm_start_lr}")
    print(f"warm_start_batch_size: {conf.warm_start_batch_size}")
    print(f"seed: {conf.seed}")
    print(f"extraction_N: {conf.extraction_N}")
    print(f"budget_M: {conf.budget_M}")
    print(f"entropy_reg_alpha: {conf.entropy_reg_alpha}")
    print(f"use_entropy_loss: {conf.use_entropy_loss}")
    print(f"entropy_loss_beta: {conf.entropy_loss_beta}")
    print(f"use_kl: {conf.use_kl}")
    print(f"kl_weight: {conf.kl_weight}")
    print(f"use_clip: {conf.use_clip}")
    print(f"clip_epsilon: {conf.clip_epsilon}")
    print(f"continuous_group_reward: {conf.continuous_group_reward}")
    print(f"no_group_reward: {conf.no_group_reward}")
    print(f"lambda_comp: {getattr(conf, 'lambda_comp', 0.5)}")
    print(f"csa_mode: {getattr(conf, 'csa_mode', 'discrete')}")
    print(f"filter_zero_influence: {getattr(conf, 'filter_zero_influence', True)}")
    print(f"informative_sigma_threshold: {conf.informative_sigma_threshold}")
    print(f"use_alignment_adv: {conf.use_alignment_adv}")
    print(f"use_richness: {conf.use_richness}")
    print(f"use_token_total_budget: {conf.use_token_total_budget}")
    print(f"unified_argument: {getattr(conf, 'unified_argument', False)}")
    print(f"safety: {getattr(conf, 'safety', False)}")
    print(f"richness_alpha: {conf.richness_alpha}")
    print(f"richness_weight: {conf.richness_weight}")
    print(f"inference_add_eval: {conf.inference_add_eval}")
    print(f"extract_direction: {conf.extract_direction}")
    print(f"use_persona: {conf.use_persona}")
    print(f"inst_regime: {conf.inst_regime}")
    print(f"enable_thinking: {conf.enable_thinking}")
    print(f"uniform_theta (ignored during training): {conf.uniform_theta}")
    print(f"dominant_schema: {conf.dominant_schema}")
    print("=" * 60)

class Client(LLMClient):
    def __init__(
        self,
        model_name: str,
        device="cuda",
        port=8000,
        enable_thinking: bool = False,
        max_gen_len: int | None = None,
    ):
        self.model_name = model_name
        self.vllm_port = port
        self.enable_thinking = enable_thinking
        self.max_gen_len = max_gen_len

        if self.model_name in API_MODELS:
            self.manager = EquilibriumManager(
                self.model_name, 
                temperature=0.0
            )
        else:
            # if self.model_name in MODEL_DICT:
            #     self.model_name = MODEL_DICT[self.model_name]
            self.backend = "vllm"
            self.vllm_client = OpenAI(
                base_url=f"http://127.0.0.1:{self.vllm_port}/v1",
                api_key="EMPTY",   # vLLM server에서는 dummy 값이면 됨
            )
            # vllm serve --served-model-name 으로 지정한 이름과 맞춰야 함
            self.vllm_model_name = self.model_name

    def _log_default_refusal(self, reason: str) -> None:
        print(
            f"[WARN] Returning DEFAULT_REFUSAL for model {self.model_name}: {reason}"
        )

    def generate(self, prompt: str, max_tokens: int = 1024, temperature=0.0) -> str:
        resolved_max_tokens = min(
            int(max_tokens),
            int(self.max_gen_len) if self.max_gen_len is not None else int(max_tokens),
        )
        gen_kwargs = {}
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
        if self.model_name in API_MODELS:
            try:
                output = self.manager.prompt_llm(
                    prompt_kwargs={},
                    system_prompt="",
                    user_prompt=prompt,
                    max_tokens=resolved_max_tokens,
                    enable_thinking=self.enable_thinking,
                    **gen_kwargs,
                )
            except BadRequestError as exc:
                print(f"SIEVE train API BadRequestError for {self.model_name}: {exc}")
                self._log_default_refusal("API BadRequestError")
                return DEFAULT_REFUSAL
            output = (output or "").strip()
            if not output:
                self._log_default_refusal("empty API response")
                return DEFAULT_REFUSAL
            return output
        else:
            messages = [{"role": "user", "content": prompt}]
            request_kwargs = {}
            if supports_enable_thinking(self.model_name):
                request_kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
                }
            try:
                response = self.vllm_client.chat.completions.create(
                    model=self.vllm_model_name,
                    messages=messages,
                    max_tokens=resolved_max_tokens,
                    **gen_kwargs,
                    **request_kwargs,
                )
            except BadRequestError as exc:
                print(f"SIEVE train vLLM BadRequestError for {self.model_name}: {exc}")
                self._log_default_refusal("vLLM BadRequestError")
                return DEFAULT_REFUSAL
            content = response.choices[0].message.content
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            texts.append(item.get("text", ""))
                    else:
                        if getattr(item, "type", None) == "text":
                            texts.append(getattr(item, "text", ""))
                content = "".join(texts)

            content = (content or "").strip()
            if not content:
                self._log_default_refusal("empty vLLM response content")
                return DEFAULT_REFUSAL
            return content

@hydra.main(version_base=None, config_path=CONF_PATH, config_name="sieve_train")
def main(conf):
    gate_device = "cuda"
    llm_device = "cuda"
    random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(conf.seed)
    multi_model_names = parse_csv_values(conf.multi_llm_models)
    multi_ports = [int(port) for port in parse_csv_values(conf.multi_llm_ports)]
    use_multi_llm = normalize_bool(conf.multi_llm)
    effective_llm_max_concurrency = int(conf.llm_max_concurrency or conf.sample_k)
    effective_use_richness = bool(conf.use_richness and float(conf.richness_weight) > 0.0)
    if conf.use_richness and not effective_use_richness:
        print(
            "[INFO] use_richness=True but richness_weight=0.0; "
            "disabling richness similarity reward/advantage computation."
        )
    train_dataset = json.load(open(conf.train_data_path))
    eval_dataset = json.load(open(conf.test_data_path))
    if os.getenv("SIEVE_DEBUG_PDB", "0") == "1":
        train_dataset = train_dataset[:20]
        eval_dataset = eval_dataset[:20]
        print(
            "[INFO] SIEVE_DEBUG_PDB=1 detected. "
            "Using only the first 20 train/eval samples."
        )
    allowed_cache_ids = (
        ExtractionCache.build_allowed_cache_ids(train_dataset, prefix="train")
        | ExtractionCache.build_allowed_cache_ids(eval_dataset, prefix="eval")
    )

    if use_multi_llm:
        if not multi_model_names:
            raise ValueError("multi_llm=True requires multi_llm_models to be set.")
        if len(multi_model_names) != len(multi_ports):
            raise ValueError("multi_llm_models and multi_llm_ports must have the same length.")
        primary_model_name = multi_model_names[0]
        llm_pool = {
            model_name: Client(
                model_name,
                device=llm_device,
                port=port,
                enable_thinking=normalize_bool(conf.enable_thinking),
                max_gen_len=getattr(conf, "max_gen_len", None),
            )
            for model_name, port in zip(multi_model_names, multi_ports)
        }
        cache_pool = None
        llm = llm_pool[primary_model_name]
    else:
        primary_model_name = conf.model_name
        llm_pool = None
        cache_pool = None
        llm = Client(
            conf.model_name,
            device=llm_device,
            port=conf.vllm_port,
            enable_thinking=normalize_bool(conf.enable_thinking),
            max_gen_len=getattr(conf, "max_gen_len", None),
        )

    sieve = SIEVE(
        llm=llm,
        gate_model=conf.gate_model,
        cache_dir=conf.cache_dir,
        allowed_cache_ids=allowed_cache_ids,
        hidden_dim=256,
        # initial_temperature=2.0,
        initial_temperature=conf.gumbel_temperature,
        gumbel_noise_scale=conf.gumbel_noise_scale,
        device=gate_device,
        N=conf.extraction_N,
        budget_M=conf.budget_M,
        use_persona=conf.use_persona,
        inst_regime=conf.inst_regime,
        uniform_theta=False,
        dominant_schema=conf.dominant_schema,
        use_alignment_adv=conf.use_alignment_adv,
        use_richness=effective_use_richness,
        use_token_total_budget=conf.use_token_total_budget,
        unified_argument=bool(getattr(conf, "unified_argument", False)),
        safety=bool(getattr(conf, "safety", False)),
        richness_alpha=conf.richness_alpha,
        inference_add_eval=conf.inference_add_eval,
        extract_direction=conf.extract_direction,
        cache_max_concurrency=effective_llm_max_concurrency,
    )
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    if use_multi_llm:
        model_tag = build_model_tag(multi_model_names)
        cache_pool = {
            model_name: ExtractionCache(
                llm_pool[model_name],
                cache_dir=os.path.join(conf.cache_dir, sanitize_model_name(model_name)),
                N=conf.extraction_N,
                total_budget=conf.budget_M,
                allowed_cache_ids=allowed_cache_ids,
                use_alignment_adv=conf.use_alignment_adv,
                use_richness=effective_use_richness,
                allow_variable_count=True,
                alignment_encoder=sieve.alignment_encoder,
                alignment_lock=sieve.alignment_encoder_lock,
                extract_direction=conf.extract_direction,
                unified_argument=bool(getattr(conf, "unified_argument", False)),
                safety=bool(getattr(conf, "safety", False)),
                max_concurrency=effective_llm_max_concurrency,
            )
            for model_name in multi_model_names
        }
        print(f"[multi_llm] parsed models: {multi_model_names}")
        for model_name in multi_model_names:
            print(
                f"[multi_llm] cache dir for {model_name}: "
                f"{os.path.join(conf.cache_dir, sanitize_model_name(model_name))}"
            )
        save_filename = f"multi_llm__{model_tag}.pt"
    else:
        save_filename = f"{primary_model_name.split('/')[-1]}.pt"

    save_path = os.path.join(conf.save_dir, save_filename)
    raw_resume_from_checkpoint = str(
        getattr(conf, "resume_from_checkpoint", "") or ""
    ).strip()
    resume_from_checkpoint = resolve_resume_from_checkpoint(
        raw_resume_from_checkpoint,
        conf.save_dir,
    )
    print_training_hyperparameter_summary(
        conf=conf,
        use_multi_llm=use_multi_llm,
        multi_model_names=multi_model_names,
        multi_ports=multi_ports,
        save_path=save_path,
        raw_resume_from_checkpoint=raw_resume_from_checkpoint,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    epoch_results = sieve.train(
        dataset=train_dataset,
        num_epochs=conf.num_epochs,
        eval_dataset=eval_dataset,
        eval_step=conf.eval_step,
        save_dir=os.path.dirname(save_path),
        save_best=True,
        save_last=True,
        k=conf.sample_k,              # GRPO samples per scenario
        learning_rate=conf.learning_rate,
        temperature_decay=0.95,
        min_temperature=conf.gumbel_temperature,
        final_temperature=conf.gumbel_temperature,
        use_warmup=conf.use_warmup,
        warm_start_size=conf.warm_start_size,
        warm_start_epochs=conf.warm_start_epochs,
        warm_start_lr=conf.warm_start_lr,
        warm_start_batch_size=conf.warm_start_batch_size,
        warmup_only=conf.warmup_only,
        use_persona=conf.use_persona,
        inst_regime=conf.inst_regime,
        inference_add_eval=conf.inference_add_eval,
        llm_max_concurrency=effective_llm_max_concurrency,
        batch_size=conf.batch_size,
        seed=conf.seed,
        entropy_reg_alpha=conf.entropy_reg_alpha,
        use_entropy_loss=conf.use_entropy_loss,
        entropy_loss_beta=conf.entropy_loss_beta,
        use_kl=conf.use_kl,
        kl_weight=conf.kl_weight,
        use_clip=conf.use_clip,
        clip_epsilon=conf.clip_epsilon,
        continuous_group_reward=conf.continuous_group_reward,
        no_group_reward=conf.no_group_reward,
        lambda_comp=getattr(conf, "lambda_comp", 0.5),
        csa_mode=getattr(conf, "csa_mode", "discrete"),
        filter_zero_influence=getattr(conf, "filter_zero_influence", True),
        informative_sigma_threshold=conf.informative_sigma_threshold,
        use_alignment_adv=conf.use_alignment_adv,
        use_richness=effective_use_richness,
        use_token_total_budget=conf.use_token_total_budget,
        unified_argument=bool(getattr(conf, "unified_argument", False)),
        richness_alpha=conf.richness_alpha,
        richness_weight=conf.richness_weight,
        use_lr_scheduler=conf.use_lr_scheduler,
        lr_scheduler_eta_min=conf.lr_scheduler_eta_min,
        uniform_theta=False,
        dominant_schema=conf.dominant_schema,
        multi_llm=use_multi_llm,
        multi_llm_clients=llm_pool,
        multi_llm_caches=cache_pool,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    sieve.save(save_path)

    if conf.warmup_only:
        print("=" * 60)
        print("WARMUP ONLY COMPLETE")
        print("=" * 60)
        print(f"Warmup-only gate saved to: {save_path}")
        print("Skipping GRPO evaluation because warmup_only=True.")
        return
    
    
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    if use_multi_llm and sieve._trainer is not None:
        eval_result = sieve._trainer.evaluate(eval_dataset, verbose=True)
        print(f"\nTest accuracy: {eval_result.accuracy:.3f}")
        print(f"Mean θ: PI={eval_result.mean_theta[0]:.3f}, "
              f"MN={eval_result.mean_theta[1]:.3f}, "
              f"PC={eval_result.mean_theta[2]:.3f}")
        print("Skipping `analyze()` and `interpret()` in multi_llm mode because they still assume a single fixed LLM.")
        return

    eval_result = sieve.evaluate(eval_dataset, batch_size=conf.batch_size)
    print(f"\nTest accuracy: {eval_result['metrics']['accuracy']:.3f}")
    print(f"Mean θ: PI={eval_result['metrics']['mean_theta'][0]:.3f}, "
          f"MN={eval_result['metrics']['mean_theta'][1]:.3f}, "
          f"PC={eval_result['metrics']['mean_theta'][2]:.3f}")

    # ------------------------------------------------------------------
    # 6. Analysis
    # ------------------------------------------------------------------

    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    analysis = sieve.analyze(eval_dataset, batch_size=conf.batch_size)

    # Category-level analysis
    if "category_analysis" in analysis:
        print("\nPer-category results:")
        for cat, stats in analysis["category_analysis"].items():
            print(f"  {cat}: acc={stats['accuracy']:.3f}, "
                  f"θ={stats['mean_theta']}, n={stats['n']}")

    # Gate openness
    print(f"\nGate entropy: {analysis['gate_openness']['mean_entropy']:.4f} "
          f"(±{analysis['gate_openness']['std_entropy']:.4f})")

    # Error analysis
    if "error_analysis" in analysis:
        print(f"\nErrors: {analysis['error_analysis']['n_errors']}")
        print(f"Error θ: {analysis['error_analysis']['mean_theta_errors']}")

    # ------------------------------------------------------------------
    # 7. Interpret single scenario
    # ------------------------------------------------------------------

    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    interp = sieve.interpret(
        "At a summer camp, there is a pool. Right next to the pool is a tent "
        "where the kids have art class. The camp made a rule that there would "
        "be no cannonballing so that the art wouldn't get ruined. Today, there "
        "is no art class. This kid cannonballs into the pool."
    )

    print(f"\nθ: {interp['theta']}")
    print(f"Answer: {interp['answer']}")
    print(f"\nTop attended tokens:")
    for token, weight in interp["top_tokens"][:5]:
        bar = "█" * int(weight * 50)
        print(f"  {token:20s} {bar} ({weight:.4f})")
    
if __name__ == "__main__":
    main()
