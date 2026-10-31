# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The rank-uniform per-step decision.

It reads only what every rank builds alike from the scheduler output (the forward
context's metadata and graph mode), so all ranks in the kernel's collectives agree.
Under FULL graphs it is baked in at capture time."""

from __future__ import annotations

from typing import Any

import torch

from vllm.config.compilation import CUDAGraphMode

from .spec import DecodeConfig


def step_reason(
    fc: Any, tokens: int, cfg: DecodeConfig, metadata_names: list[str]
) -> str | None:
    """Why this forward runs vLLM's layers, or None for a kernel step."""
    if torch.compiler.is_compiling():
        return "torch.compile tracing"
    if fc.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
        return "piecewise CUDA graph"
    if tokens not in cfg.tokens:
        return f"{tokens} tokens"
    md = fc.attn_metadata
    if isinstance(md, list):
        return "micro-batched step"
    if not isinstance(md, dict) or any(name not in md for name in metadata_names):
        return "no attention metadata"
    for name in metadata_names:
        m = md[name]
        if getattr(m, "num_prefills", 0):
            return f"prefill rows ({type(m).__name__})"
        if getattr(m, "max_query_len", 1) > 1:
            return "query longer than one token"
    return None
