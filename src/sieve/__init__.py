"""
PRISM: Principled Reasoning via Informed Schema Modulation

A framework for improving LLM moral reasoning through learned
schema-based information gating.

Quick start:
    from prism import PRISM
    
    prism = PRISM(llm=your_client, cache_dir="./cache")
    prism.train(train_data, num_epochs=10)
    results = prism.evaluate(test_data)
"""

from src.sieve.sieve import (
    SIEVE,
    GeneratorAdapter,
    build_sieve_inference_runner,
    load_sieve_for_inference,
    run_sieve_inference,
    set_sieve_dominant_schema,
    set_sieve_uniform_theta,
    set_sieve_use_all,
)
from src.sieve.data_types import (
    Consideration,
    Phase2Extraction,
    Phase3Selection,
    GateSample,
    InferenceResult,
    SCHEMA_NAMES,
)
from src.sieve.gate_module import GateModule
from src.sieve.extraction import (
    ExtractionCache,
    extract_for_schema,
    extract_phase2,
)
from src.sieve.info_gate import (
    allocate_phase3_budgets,
    select_phase3_arguments,
)
from src.sieve.prompt_assembly import assemble_prompt
from src.sieve.trainer import GRPOTrainer
from src.sieve.evaluation import inference, evaluate, evaluate_transfer
from src.sieve.llm_client import LLMClient

__all__ = [
    "SIEVE",
    "GeneratorAdapter",
    "build_sieve_inference_runner",
    "load_sieve_for_inference",
    "run_sieve_inference",
    "set_sieve_dominant_schema",
    "set_sieve_uniform_theta",
    "set_sieve_use_all",
    "GateModule",
    "ExtractionCache",
    "extract_for_schema",
    "extract_phase2",
    "GRPOTrainer",
    "LLMClient",
    "allocate_phase3_budgets",
    "select_phase3_arguments",
    "assemble_prompt",
    "inference",
    "evaluate",
    "evaluate_transfer",
    "Consideration",
    "Phase2Extraction",
    "Phase3Selection",
    "GateSample",
    "InferenceResult",
    "SCHEMA_NAMES",
]
