from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONF_PATH = str(PROJECT_ROOT / "conf")


def set_generation_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def sync_load(path: str | os.PathLike[str]) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_entropy(values) -> float:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.numel() == 0:
        return 0.0
    probs = tensor / tensor.sum().clamp_min(1e-12)
    return float(-(probs * probs.clamp_min(1e-12).log()).sum().item())
