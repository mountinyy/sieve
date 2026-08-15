from __future__ import annotations

"""
PRISM: Principled Reasoning via Informed Schema Modulation
Shared data structures.
"""

from dataclasses import dataclass, field


@dataclass
class Consideration:
    """One schema-sourced moral consideration extracted from a scenario."""
    index: int
    principle: str
    supporting_context: str
    direction: str = ""
    source_schema: str = ""


@dataclass
class Phase2Extraction:
    """Phase 2 output: schema-sourced considerations."""
    schema_considerations: dict[str, list[Consideration]] = field(default_factory=dict)
    schema_embeddings: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    influence: dict = field(default_factory=dict)
    influence_cache_version: int | None = None
    influence_source: dict = field(default_factory=dict)
    safety: bool = False


@dataclass
class Phase3Selection:
    """Output of deterministic Phase 3 budgeted selection."""
    theta: list[float]
    primary_schema: str
    secondary_schema: str
    tertiary_schema: str
    total_budget: int
    primary_budget: int
    secondary_budget: int
    tertiary_budget: int
    primary_arguments: list[Consideration] = field(default_factory=list)
    secondary_arguments: list[Consideration] = field(default_factory=list)
    tertiary_principles: list[Consideration] = field(default_factory=list)

# ==============================================================================
# Gate Sampling
# ==============================================================================

@dataclass
class GateSample:
    """One sampled theta with its log probability (for GRPO)."""
    theta: "torch.Tensor"       # (3,) — [θ_PI, θ_MN, θ_PC]
    log_prob: "torch.Tensor"    # scalar
    logits: "torch.Tensor"      # (3,)
    forced_min_coverage: bool = False


# ==============================================================================
# Inference Result
# ==============================================================================

@dataclass
class InferenceResult:
    """Full output from one PRISM inference run."""
    answer: str
    reasoning: str
    theta: list[float]
    gate_stats: dict
    raw_response: str = ""
    prompt: str = ""
    schema_arguments: dict = field(default_factory=dict)


# ==============================================================================
# Constants
# ==============================================================================

SCHEMA_NAMES = ["PI", "MN", "PC"]
SCHEMA_ORDER = {name: i for i, name in enumerate(SCHEMA_NAMES)}
SCHEMA_FULL_NAMES = {
    "PI": "Personal Interest (PI)",
    "MN": "Maintaining Norms (MN)",
    "PC": "Postconventional (PC)",
}
