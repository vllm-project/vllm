# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility shim for the ROCm relative-attention implementation.

ROCm uses a Triton kernel which is compiled by vLLM's normal model warmup. The
NVIDIA path registers ahead-of-time CuTeDSL units; importing that provider on
ROCm would pull in CUDA-only tml-fa4 code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InklingFA4WarmupConfig:
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rel_extent: int
    window_size: tuple[int, int]
    is_local: bool
    max_kv_len: int
    dtype: torch.dtype
    kv_dtype: torch.dtype
    block_size: int
    max_num_reqs: int
    max_num_batched_tokens: int


def register_fa4_warmup(config: InklingFA4WarmupConfig) -> None:
    """Triton compilation is triggered by the ordinary vLLM warmup forward."""
    del config
