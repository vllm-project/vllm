# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GLM-5.3-Flash specialized-decode switch, shared by every vendor. Reads only
the config, so it imports no vendor code."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

AUTO, SPECIALIZED = "auto", "latency_specialized"


def requested(vllm_config: VllmConfig) -> bool:
    """Whether the deployment asks for the specialized decode: ``--monokernel``, or
    ``additional_config={"decode_backend": "latency_specialized"}`` (RFC #60904)."""
    extra = vllm_config.additional_config
    backend = extra.get("decode_backend", AUTO) if isinstance(extra, dict) else AUTO
    if backend not in (AUTO, SPECIALIZED):
        raise ValueError(
            f"decode_backend must be '{AUTO}' or '{SPECIALIZED}', got {backend!r}"
        )
    return bool(vllm_config.kernel_config.monokernel) or backend == SPECIALIZED
