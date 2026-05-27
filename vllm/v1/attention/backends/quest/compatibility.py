# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility checks for the Quest sparse offload attention backend.

This is the **only** place that contains model-family branching. Other files
in `backends/quest/` MUST stay model-agnostic.
"""
from __future__ import annotations

from typing import Any

# Known-good families for Phase A. Extend by appending the architecture name.
# Match against `model_config.architecture`, lowercased.
SUPPORTED_MODEL_FAMILIES: frozenset[str] = frozenset(
    {"llama", "mistral", "qwen2", "qwen2.5", "yi"}
)


def check_model_compat(model_config: Any) -> list[str]:
    """Return a list of human-readable reasons why this model is not
    compatible with the Quest backend. Empty list means compatible.
    """
    errors: list[str] = []

    architecture = getattr(model_config, "architecture", None)
    if architecture is None:
        errors.append(
            "Quest requires model_config.architecture to be set; got None"
        )
    else:
        normalized = str(architecture).lower()
        if normalized not in SUPPORTED_MODEL_FAMILIES:
            errors.append(
                f"Quest does not yet support architecture {architecture!r}. "
                f"Supported: {sorted(SUPPORTED_MODEL_FAMILIES)}"
            )

    if getattr(model_config, "is_mla", False):
        errors.append("Quest does not yet support MLA models")

    if getattr(model_config, "has_sliding_window", False):
        errors.append("Quest does not yet support sliding window attention")

    return errors
