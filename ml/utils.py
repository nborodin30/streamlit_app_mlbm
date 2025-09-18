"""Utility helpers: logging, determinism, and hashing.

Design principles (scientific code):
----------------------------------
* Fail fast: use ``assert`` for invariants so that any divergence stops execution.
* Determinism: explicit seeding and disabling of non‑deterministic backends.
* Reproducibility: log all configuration and metrics both to stdout and a file.
* Simplicity: minimal hidden state; explicit returns instead of globals where possible.
"""

from __future__ import annotations

import logging
import os
import random
import hashlib
from typing import Any, Dict, Callable
import json

import numpy as np
import torch


def set_determinism(seed: int) -> None:
    """Set seeds for python, numpy, and torch, enforce deterministic behavior.

    Raises
    ------
    AssertionError
        If seed is negative.
    """
    assert seed >= 0, "Seed must be non-negative"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def file_sha256(path: str) -> str:
    """Return SHA256 for a file path."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest().upper()


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""

    def default(self, obj):  # noqa: D102
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save a dictionary to a JSON file with support for NumPy types.

    Parameters
    ----------
    data : Dict[str, Any]
        The dictionary to save.
    path : str
        The output file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, cls=NpEncoder)


def log_factory(logger: logging.Logger) -> Callable[[Dict[str, Any]], None]:
    """Create a logging callable using standard logging."""

    def log_fn(obj: Dict[str, Any]):
        logger.info(
            " | ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in obj.items()
            )
        )

    return log_fn


