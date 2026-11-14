# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The switch for GLM-5.3-Flash's ROCm specialized decode (the backbone's per-layer
kernels). Reads only the config, so it imports no vendor code. The NVIDIA executor
(`nvidia/specialized_decode/runtime`) parses the same `decode_backend` key on its
own: a new value has to go to both."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

AUTO, SPECIALIZED = "auto", "latency_specialized"


def requested(vllm_config: VllmConfig) -> bool:
    """Whether the deployment asks for the specialized decode: `--monokernel`, or
    `additional_config={"decode_backend": "latency_specialized"}` (RFC #60904)."""
    extra = vllm_config.additional_config
    backend = extra.get("decode_backend", AUTO) if isinstance(extra, dict) else AUTO
    if backend not in (AUTO, SPECIALIZED):
        raise ValueError(
            f"decode_backend must be '{AUTO}' or '{SPECIALIZED}', got {backend!r}"
        )
    return bool(vllm_config.kernel_config.monokernel) or backend == SPECIALIZED
