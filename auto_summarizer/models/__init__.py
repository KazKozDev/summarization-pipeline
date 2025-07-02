"""Model registry and factory utilities for Auto Summarizer.

This package exposes the function `get_summarizer` that instantiates a summarizer
class based on a model key defined in `models.yml`.

Example
-------
>>> from auto_summarizer.models import get_summarizer
>>> summarizer = get_summarizer("bart-large", device="cuda")
>>> summary = summarizer("Some long article text ...")
"""
from __future__ import annotations

from pathlib import Path
import importlib
import logging
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------
_PACKAGE_DIR: Path = Path(__file__).resolve().parent
_REGISTRY_PATH: Path = _PACKAGE_DIR / "models.yml"

if not _REGISTRY_PATH.exists():
    raise FileNotFoundError(f"Model registry not found: {_REGISTRY_PATH}")

with _REGISTRY_PATH.open("r", encoding="utf-8") as fh:
    _MODEL_REGISTRY: Dict[str, Dict[str, Any]] = yaml.safe_load(fh)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    "get_summarizer",
]


def get_summarizer(model_key: str, **overrides: Any):
    """Instantiate a summarizer by key.

    Parameters
    ----------
    model_key : str
        Key defined in `models.yml` (e.g. ``"bart-large"``).
    **overrides : Any
        Runtime overrides for parameters such as ``device`` or ``model_name``.

    Returns
    -------
    BaseTransformerSummarizer
        An instance of a subclass of :class:`~auto_summarizer.models.transformers.base_summarizer.BaseTransformerSummarizer`.
    """
    if model_key not in _MODEL_REGISTRY:
        valid = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(f"Unknown model_key '{model_key}'. Available: {valid}")

    cfg = _MODEL_REGISTRY[model_key].copy()

    # Extract non-model parameters
    non_model_params = {
        'class_path', 'description', 'default_device', 'model_name'
    }
    
    # Get model parameters (excluding non-model params)
    model_params = {k: v for k, v in cfg.items() if k not in non_model_params}
    
    # Merge overrides (device, etc.)
    model_params.update({k: v for k, v in overrides.items() 
                        if k != 'model_name' and v is not None})

    class_path = cfg["class_path"]
    model_name = overrides.get("model_name", cfg["model_name"])

    module_name, class_name = class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(f"Cannot import '{class_path}': {exc}") from exc

    logger.info("Instantiating summarizer %s with model %s", class_name, model_name)

    return cls(model_name=model_name, **model_params)
