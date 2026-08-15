from __future__ import annotations

import ast
import json
from typing import Any

import numpy as np

from src.sieve.data_types import SCHEMA_NAMES


def coerce_theta_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            return None
        return [float(x) for x in value.tolist()]
    if isinstance(value, (list, tuple)):
        try:
            return [float(x) for x in value]
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        parsed = None
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(stripped)
                break
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue
        if parsed is None:
            return None
        return coerce_theta_vector(parsed)
    return None


def schema_activation_by_schema(theta: Any) -> dict[str, float] | None:
    theta_vector = coerce_theta_vector(theta)
    if theta_vector is None:
        return None
    return {
        schema: float(theta_vector[idx])
        for idx, schema in enumerate(SCHEMA_NAMES)
        if idx < len(theta_vector)
    }


def attach_schema_activation(record: dict, theta: Any) -> dict:
    theta_vector = coerce_theta_vector(theta)
    if theta_vector is None:
        return record
    record["theta"] = theta_vector
    record["schema_activation"] = theta_vector
    record["schema_activation_by_schema"] = schema_activation_by_schema(theta_vector)
    return record


def summarize_average_theta(records: list[dict], theta_key: str = "theta") -> dict[str, Any] | None:
    theta_rows: list[list[float]] = []
    for record in records:
        theta = coerce_theta_vector(record.get(theta_key))
        if theta is not None:
            theta_rows.append(theta)

    if not theta_rows:
        return None

    lengths = {len(row) for row in theta_rows}
    if len(lengths) != 1:
        return {
            "n_theta_rows": len(theta_rows),
            "average_theta": None,
            "note": "Inconsistent theta dimensions across records.",
        }

    theta_avg = np.mean(np.asarray(theta_rows, dtype=float), axis=0)
    return {
        "n_theta_rows": len(theta_rows),
        "average_theta": [round(float(x), 4) for x in theta_avg.tolist()],
    }


def coerce_influence_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        try:
            vector = [float(value[schema]) for schema in SCHEMA_NAMES]
        except (KeyError, TypeError, ValueError):
            return None
    elif isinstance(value, np.ndarray):
        if value.ndim != 1:
            return None
        try:
            vector = [float(x) for x in value.tolist()]
        except (TypeError, ValueError):
            return None
    elif isinstance(value, (list, tuple)):
        try:
            vector = [float(x) for x in value]
        except (TypeError, ValueError):
            return None
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        parsed = None
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(stripped)
                break
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue
        if parsed is None:
            return None
        return coerce_influence_vector(parsed)
    else:
        return None

    if len(vector) != len(SCHEMA_NAMES):
        return None
    allowed_values = {-0.5, 0.0, 0.5, 1.0}
    if any(item not in allowed_values for item in vector):
        return None
    return vector


def influence_vector_by_schema(value: Any) -> dict[str, float] | None:
    vector = coerce_influence_vector(value)
    if vector is None:
        return None
    return {
        schema: float(vector[idx])
        for idx, schema in enumerate(SCHEMA_NAMES)
    }


def is_zero_influence(value: Any) -> bool:
    vector = coerce_influence_vector(value)
    return vector is not None and all(item == 0 for item in vector)
