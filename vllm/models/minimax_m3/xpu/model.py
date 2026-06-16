# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 (text backbone) — Intel XPU implementation.

This reuses the entire NVIDIA model implementation and overrides only
``MiniMAXGemmaRMSNorm``: FlashInfer's Gemma RMSNorm kernels are CUDA-only, so
the XPU norm uses the portable Triton kernels in ``xpu/ops/gemma_rmsnorm.py``
(no dependency on vllm-xpu-kernels). Mirrors the ``amd`` per-platform override,
but reuses the NVIDIA classes verbatim instead of copying them (only the norm
differs).

The NVIDIA ``model`` and ``mtp`` modules bind ``MiniMAXGemmaRMSNorm`` in their
own namespaces, so the XPU norm is rebound in both before any layer is built.
"""

import torch
from torch import nn

from vllm.models.minimax_m3.xpu.ops import gemma_fused_add_rmsnorm, gemma_rmsnorm


class MiniMAXGemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm backed by the XPU Triton kernels.

    Numerically equivalent to the NVIDIA FlashInfer ``gemma_rmsnorm`` /
    ``gemma_fused_add_rmsnorm``: ``x * rsqrt(mean(x^2)+eps) * (1+weight)``.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return gemma_rmsnorm(x, self.weight, self.variance_epsilon)
        return gemma_fused_add_rmsnorm(x, residual, self.weight, self.variance_epsilon)


def _install_xpu_rmsnorm() -> None:
    from vllm.models.minimax_m3.nvidia import model as _nv_model
    from vllm.models.minimax_m3.nvidia import mtp as _nv_mtp

    _nv_model.MiniMAXGemmaRMSNorm = MiniMAXGemmaRMSNorm  # type: ignore[misc]
    _nv_mtp.MiniMAXGemmaRMSNorm = MiniMAXGemmaRMSNorm  # type: ignore[misc]


_install_xpu_rmsnorm()

from vllm.models.minimax_m3.nvidia.model import (  # noqa: E402
    MiniMaxM3SparseForCausalLM,
    MiniMaxM3SparseForConditionalGeneration,
)

__all__ = [
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
]
