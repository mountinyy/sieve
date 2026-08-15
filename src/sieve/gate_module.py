from __future__ import annotations

"""
PRISM: Gate Module.
Learnable schema activation predictor.
Qwen3-1.7B-Base (frozen) → Attention Pooling → MLP Head → θ
"""

import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.sieve.data_types import GateSample, SCHEMA_NAMES


class AttentionPooling(nn.Module):
    """
    Learnable attention pooling over token hidden states.
    Learns which tokens are important for predicting θ.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.query(hidden_states).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        pooled = (hidden_states * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights


class GateModule(nn.Module):
    """
    Maps scenario text → schema activation weights θ = [θ_PI, θ_MN, θ_PC].

    Architecture:
        Qwen3-1.7B-Base (frozen)
        → Attention Pooling (learnable, ~2K params)
        → MLP Head (learnable, ~400K params)
        → softmax with temperature → θ
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-1.7B-Base",
        hidden_dim: int = 256,
        num_schemas: int = 3,
        initial_temperature: float = 2.0,
        gumbel_noise_scale: float = 1.0,
        max_length: int = 512,
        device: str = None,
        uniform_theta: bool = False,
        dominant_schema: str | None = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_schemas = num_schemas
        self.max_length = max_length
        self.temperature = initial_temperature
        self.gumbel_noise_scale = float(gumbel_noise_scale)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.uniform_theta = uniform_theta
        self.dominant_schema = self._normalize_dominant_schema(dominant_schema)

        # Backbone (frozen)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._tokenizer_lock = threading.RLock()
        setattr(self.tokenizer, "_sieve_tokenizer_lock", self._tokenizer_lock)

        self.backbone = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        backbone_hidden_size = self.backbone.config.hidden_size

        # Learnable components
        self.pooling = AttentionPooling(backbone_hidden_size).to(self.device)
        self.head = nn.Sequential(
            nn.Linear(backbone_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_schemas),
        ).to(self.device)
        self._init_head_biases()

        # Cache
        self._last_logits = None
        self._last_attention_weights = None

    def _init_head_biases(self) -> None:
        for module in self.head:
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _normalize_dominant_schema(self, dominant_schema: str | None) -> str | None:
        if dominant_schema is None:
            return None
        normalized = str(dominant_schema).strip().upper()
        if normalized in {"", "NONE"}:
            return None
        if normalized not in SCHEMA_NAMES:
            raise ValueError(
                f"dominant_schema must be one of {SCHEMA_NAMES} or None, got {dominant_schema!r}."
            )
        return normalized

    def _fixed_theta(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dominant_schema is not None:
            dominant_idx = SCHEMA_NAMES.index(self.dominant_schema)
            theta = torch.full(
                (batch_size, self.num_schemas),
                0.05,
                device=self.device,
                dtype=torch.float32,
            )
            theta[:, dominant_idx] = 0.9
            logits = theta.log()
            return theta, logits

        theta = torch.full(
            (batch_size, self.num_schemas),
            1.0 / self.num_schemas,
            device=self.device,
            dtype=torch.float32,
        )
        logits = torch.zeros_like(theta)
        return theta, logits

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, scenario_text: str, inference_mode: bool = False) -> torch.Tensor:
        """Deterministic: scenario → θ (3,)."""
        if self.uniform_theta or self.dominant_schema is not None:
            theta, logits = self._fixed_theta(batch_size=1)
            self._last_logits = logits.squeeze(0)
            self._last_attention_weights = None
            return theta.squeeze(0)
        pooled, attn_weights = self._encode(scenario_text)
        logits = self.head(pooled.float()).squeeze(0)
        self._last_logits = logits
        self._last_attention_weights = attn_weights
        temp = 1.0 if inference_mode else self.temperature
        return F.softmax(logits / temp, dim=-1)

    def forward_batch(
        self,
        scenario_texts: list[str],
        inference_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic batched forward: scenarios -> theta batch, logits batch."""
        if self.uniform_theta or self.dominant_schema is not None:
            thetas, logits = self._fixed_theta(batch_size=len(scenario_texts))
            self._last_logits = logits
            self._last_attention_weights = None
            return thetas, logits
        pooled, attn_weights = self._encode_batch(scenario_texts)
        logits = self.head(pooled.float())
        self._last_logits = logits
        self._last_attention_weights = attn_weights
        temp = 1.0 if inference_mode else self.temperature
        thetas = F.softmax(logits / temp, dim=-1)
        return thetas, logits

    # ------------------------------------------------------------------
    # GRPO sampling
    # ------------------------------------------------------------------

    def sample(self, scenario_text: str, k: int = 4) -> list[GateSample]:
        """Sample k diverse thetas via Gumbel noise for GRPO training."""
        if self.uniform_theta or self.dominant_schema is not None:
            theta, logits = self._fixed_theta(batch_size=1)
            uniform_theta = theta.squeeze(0)
            zero_logits = logits.squeeze(0)
            return [
                GateSample(
                    theta=uniform_theta.clone(),
                    log_prob=torch.tensor(0.0, device=self.device),
                    logits=zero_logits.clone(),
                )
                for _ in range(k)
            ]
        pooled, attn_weights = self._encode(scenario_text)
        base_logits = self.head(pooled.float()).squeeze(0)
        self._last_attention_weights = attn_weights

        samples = []
        policy_logits = base_logits / self.temperature
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        
        for _ in range(k):
            gumbel = -torch.log(-torch.log(torch.rand_like(base_logits) + 1e-8) + 1e-8)
            noisy_logits = (base_logits + (self.gumbel_noise_scale * gumbel)) / self.temperature
            theta = F.softmax(noisy_logits, dim=-1)
            log_prob = (theta.detach() * policy_log_probs).sum()
            samples.append(GateSample(theta=theta, log_prob=log_prob, logits=noisy_logits))

        return samples

    def sample_batch(self, scenario_texts: list[str], k: int = 4) -> list[list[GateSample]]:
        """Sample k thetas for each scenario in a batch."""
        if self.uniform_theta or self.dominant_schema is not None:
            theta_batch, logits_batch = self._fixed_theta(batch_size=len(scenario_texts))
            batch_samples = [[] for _ in scenario_texts]
            for idx in range(len(scenario_texts)):
                for _ in range(k):
                    batch_samples[idx].append(
                        GateSample(
                            theta=theta_batch[idx].clone(),
                            log_prob=torch.tensor(0.0, device=self.device),
                            logits=logits_batch[idx].clone(),
                        )
                    )
            self._last_logits = logits_batch
            self._last_attention_weights = None
            return batch_samples
        pooled, attn_weights = self._encode_batch(scenario_texts)
        base_logits = self.head(pooled.float())
        self._last_logits = base_logits
        self._last_attention_weights = attn_weights

        batch_samples = [[] for _ in scenario_texts]
        policy_logits = base_logits / self.temperature
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        for _ in range(k):
            gumbel = -torch.log(-torch.log(torch.rand_like(base_logits) + 1e-8) + 1e-8)
            noisy_logits = (base_logits + (self.gumbel_noise_scale * gumbel)) / self.temperature
            theta = F.softmax(noisy_logits, dim=-1)
            log_prob = (theta.detach() * policy_log_probs).sum(dim=-1)
            for idx in range(len(scenario_texts)):
                batch_samples[idx].append(
                    GateSample(
                        theta=theta[idx],
                        log_prob=log_prob[idx],
                        logits=noisy_logits[idx],
                    )
                )

        return batch_samples

    # ------------------------------------------------------------------
    # Backbone encoding
    # ------------------------------------------------------------------

    def _encode(self, scenario_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Frozen backbone → learnable attention pooling."""
        return self._encode_batch([scenario_text])

    def _prepare_inputs(self, scenario_texts: list[str]):
        with self._tokenizer_lock:
            return self.tokenizer(
                scenario_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ).to(self.device)

    def _encode_hidden_batch(self, scenario_texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Frozen backbone -> hidden states and attention mask for a batch."""
        inputs = self._prepare_inputs(scenario_texts)
        with torch.no_grad():
            hidden_states = self.backbone(**inputs).last_hidden_state
        return hidden_states.float(), inputs["attention_mask"]

    def _compute_logits_from_hidden(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_module: nn.Module | None = None,
        head_module: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooling = pooling_module or self.pooling
        head = head_module or self.head
        pooled, attn_weights = pooling(hidden_states.float(), attention_mask)
        logits = head(pooled.float())
        return logits, attn_weights

    def _encode_batch(self, scenario_texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Frozen backbone -> learnable attention pooling for a batch."""
        hidden_states, attention_mask = self._encode_hidden_batch(scenario_texts)
        pooled, attn_weights = self.pooling(hidden_states.float(), attention_mask)
        return pooled, attn_weights

    # ------------------------------------------------------------------
    # Temperature
    # ------------------------------------------------------------------

    def anneal_temperature(self, decay: float = 0.95, min_temp: float = 0.5):
        self.temperature = max(min_temp, self.temperature * decay)

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def interpret(self, scenario_text: str) -> dict:
        """Return θ + token-level attention for analysis."""
        theta = self.forward(scenario_text)
        attn = self._last_attention_weights.squeeze(0)
        with self._tokenizer_lock:
            tokens = self.tokenizer.tokenize(scenario_text)
        weights = attn[: len(tokens)].detach().cpu().tolist()

        token_attention = list(zip(tokens, weights))
        top_tokens = sorted(token_attention, key=lambda x: x[1], reverse=True)[:10]

        return {
            "theta": {name: round(theta[i].item(), 4) for i, name in enumerate(SCHEMA_NAMES)},
            "token_attention": token_attention,
            "top_tokens": top_tokens,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_learnable_parameters(self) -> list[nn.Parameter]:
        params = list(self.pooling.parameters()) + list(self.head.parameters())
        return params

    def count_parameters(self) -> dict:
        p = sum(p.numel() for p in self.pooling.parameters())
        h = sum(p.numel() for p in self.head.parameters())
        b = sum(p.numel() for p in self.backbone.parameters())
        return {
            "pooling": p, "head": h,
            "backbone_frozen": b,
            "total_learnable": p + h,
        }

    def save(self, path: str):
        torch.save({
            "pooling": self.pooling.state_dict(),
            "head": self.head.state_dict(),
            "temperature": self.temperature,
            "gumbel_noise_scale": self.gumbel_noise_scale,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.pooling.load_state_dict(ckpt["pooling"])
        self.head.load_state_dict(ckpt["head"])
        self.temperature = ckpt["temperature"]
        self.gumbel_noise_scale = ckpt.get("gumbel_noise_scale", 1.0)
