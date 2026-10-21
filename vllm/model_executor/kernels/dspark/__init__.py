# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark confidence-head fused kernel.

Public API:
  - ``dspark_confidence_fused``: dispatch entry point; runs TileLang kernel
    when the input shapes/dtypes/platform allow, else PyTorch fallback.
  - ``dspark_confidence_torch``: pure-PyTorch reference (used for testing
    and as the fallback path).
"""

from __future__ import annotations

import torch

from vllm.model_executor.kernels.dspark.pytorch_fallback import dspark_confidence_torch

__all__ = [
    "dspark_confidence_fused",
    "dspark_confidence_torch",
    "is_dspark_fused_available",
]


def is_dspark_fused_available() -> bool:
    """True iff the fused TileLang path can run in this process.

    Thin wrapper that tolerates TileLang import failure (e.g. non-CUDA
    platform, or stale tilelang install).
    """
    if not torch.cuda.is_available():
        return False
    try:
        from vllm.model_executor.kernels.dspark.tilelang import (  # noqa: F401
            is_dspark_fused_available as _impl,
        )

        return _impl()
    except Exception:
        return False


def dspark_confidence_fused(
    hidden: torch.Tensor,  # (n, H) bf16
    markov_embed: torch.Tensor,  # (n, r) bf16
    weight: torch.Tensor,  # (1, H + r) fp32 (ReplicatedLinear stores output_dim=1)
    temp: float = 1.0,
    *,
    max_fused_batch: int = 2048,
) -> torch.Tensor | None:
    """Compute sigmoid((hidden ; markov_embed) @ weight / temp) with fused kernel.

    Returns ``None`` when the fused kernel isn't applicable (CPU tensors,
    wrong dtypes, batch > ``max_fused_batch``, wrong strides, etc.) so the
    caller can fall back to the PyTorch reference path.

    Returns: ``(n,) fp32`` post-sigmoid probabilities, or ``None``.
    """
    n = hidden.shape[0]
    # vLLM's ReplicatedLinear stores weight as (output_size=1, input_size=H+r).
    # The kernel treats GEMV as a 1D dot-product, so reshape (1, H+r) -> (H+r,).
    if weight.dim() == 2 and weight.shape[0] == 1:
        weight = weight.squeeze(0)
    elif weight.dim() != 1:
        return None
    r = weight.shape[0] - hidden.shape[-1]

    fused_ok = (
        is_dspark_fused_available()
        and hidden.is_cuda
        and hidden.dtype == torch.bfloat16
        and markov_embed.is_cuda
        and markov_embed.dtype == torch.bfloat16
        and weight.dtype == torch.float32
        and weight.is_cuda
        and hidden.stride(-1) == 1
        and (r == 0 or markov_embed.stride(-1) == 1)
        and markov_embed.shape[-1] >= r
        and n <= max_fused_batch
    )

    if not fused_ok:
        return None

    out = torch.empty(n, dtype=torch.float32, device=hidden.device)
    # Calls via custom op so CUDA-Graph capture and torch.compile work.
    torch.ops.vllm.dspark_confidence_tilelang(hidden, markov_embed, weight, out, temp)
    return out
