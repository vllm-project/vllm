# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn
from torch import Tensor

from vllm.model_executor.hw_agnostic.custom_op import CustomOp


@CustomOp.register("rms_norm")
class RMSNorm(CustomOp):
    """``x -> w * x / sqrt(E[x^2] + eps)``. With ``residual``, fuses
    ``residual += x`` then RMSNorm and returns ``(normalized, residual)``."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        weight_dtype = dtype or torch.get_default_dtype()
        self.has_weight = has_weight
        weight = torch.ones(hidden_size, dtype=weight_dtype)
        if has_weight:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight, persistent=False)

    def _rms_norm(
        self,
        x: Tensor,
        weight: Tensor | None,
        epsilon: float,
        variance_size: int | None = None,
    ) -> Tensor:
        """Weighted root-mean-square layer normalization"""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        x_var = x if variance_size is None else x[..., :variance_size]
        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + epsilon)
        if weight is not None:
            x = x.to(weight.dtype) * weight
        return x.to(orig_dtype)

    def _fused_add_rms_norm(
        self,
        x: Tensor,
        x_residual: Tensor,
        weight: Tensor | None,
        epsilon: float,
        variance_size: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Fused add and weighted root-mean-square layer normalization"""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        x = x + x_residual.to(torch.float32)
        x_residual = x.to(orig_dtype)

        x_var = x if variance_size is None else x[..., :variance_size]
        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + epsilon)
        if weight is not None:
            x = x.to(weight.dtype) * weight
        return x.to(orig_dtype), x_residual

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        weight = self.weight if self.has_weight else None
        epsilon = self.variance_epsilon
        variance_size = self.variance_size_override

        if residual is None:
            return self._rms_norm(x, weight, epsilon, variance_size)
        else:
            return self._fused_add_rms_norm(x, residual, weight, epsilon, variance_size)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.variance_epsilon}"
