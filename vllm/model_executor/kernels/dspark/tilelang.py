# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang dispatch layer for the DSpark confidence head kernel.

Wraps the raw TileLang kernel in ``tilelang_kernels.py``:
  - caches compiled kernels by static (n_embed, markov_rank) pair
  - registers the kernel as a torch custom op so CUDA-Graph capture and
    ``torch.compile`` can see it

This deliberately does *not* subclass ``VllmJitKernel`` (introduced on vLLM
main after 0.19.x); on next rebase, converting ``_CACHE`` warmup/population
to a ``VllmJitKernel`` wrapper is trivial.
"""

from __future__ import annotations

import torch

from vllm.model_executor.kernels.dspark.tilelang_kernels import (
    confidence_head_fused_tilelang,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

# (n_embed, markov_rank) -> compiled kernel callable (TileLang JIT)
# Not used directly; TileLang caches by full arg specialization internally.
# We keep this dict to memoize *which* specializations we've already kicked.


def _confidence_head_run(
    hidden: torch.Tensor,  # (n, H) bf16, last dim contiguous
    markov_embed: torch.Tensor,  # (n, r) bf16 (or (n, 1) dummy when r=0)
    weight: torch.Tensor,  # (H + r,) fp32
    out: torch.Tensor,  # (n,) fp32, written in-place
    temp: float,
) -> None:
    n_embed = hidden.shape[-1]
    markov_rank = weight.shape[0] - n_embed
    confidence_head_fused_tilelang(
        hidden,
        markov_embed,
        weight,
        out,
        temp,
        n_embed=n_embed,
        markov_rank=markov_rank,
    )


def is_dspark_fused_available() -> bool:
    """True iff the fused TileLang path can run on this platform."""
    if not torch.cuda.is_available():
        return False
    if not current_platform.is_cuda():
        return False
    try:
        if torch.cuda.get_device_capability(0)[0] < 9:
            return False  # PDL assumed
    except Exception:
        return False
    return True


def _dspark_confidence_tilelang_fake(
    hidden: torch.Tensor,
    markov_embed: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    temp: float,
) -> None:
    """Fake impl for op-registration (used during torch.compile trace)."""
    return None


direct_register_custom_op(
    op_name="dspark_confidence_tilelang",
    op_func=_confidence_head_run,
    mutates_args=["out"],
    fake_impl=_dspark_confidence_tilelang_fake,
)
