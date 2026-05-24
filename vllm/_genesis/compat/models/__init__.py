# SPDX-License-Identifier: Apache-2.0
"""Genesis models — curated registry of supported models + download tools.

Re-exports the registry data and the pull tool for `from
vllm._genesis.compat.models import SUPPORTED_MODELS, pull_model`.
"""
from __future__ import annotations

from vllm._genesis.compat.models.registry import (
    SUPPORTED_MODELS,
    ModelEntry,
    get_model,
    list_models,
    list_recommended_for_hardware,
)

__all__ = [
    "SUPPORTED_MODELS",
    "ModelEntry",
    "get_model",
    "list_models",
    "list_recommended_for_hardware",
]
