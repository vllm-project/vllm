# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer normalization — hw-agnostic native-only copies.

Vendored and pruned from ``vllm/model_executor/layers/layernorm.py``.
Only ``RMSNorm`` (with the optional fused-residual addition) and
``LayerNorm`` are kept — the variants used elsewhere in vLLM
(``GemmaRMSNorm``, ``RMSNormGated``, ``poly_norm``) are unused here.
The ``CustomOp`` dispatch layer is gone: each class subclasses
``nn.Module`` directly and computes the result in float32 PyTorch.
OOT plugins that want a CUDA / XPU / ROCm fast path subclass these and
override ``forward``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root mean square normalization.

    Computes ``x -> w * x / sqrt(E[x^2] + eps)`` where ``w`` is the
    learned weight (omitted when ``has_weight=False``). When called
    with a non-None ``residual``, performs the standard
    add-then-normalize pattern in place: ``residual += x``, then
    ``x = rms_norm(residual)``.
    """

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

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        if self.variance_size_override is None:
            variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        else:
            head = x_fp32[..., : self.variance_size_override]
            variance = head.pow(2).mean(dim=-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        out = x_fp32.to(orig_dtype)
        if self.has_weight:
            out = out * self.weight
        return out

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._rms_norm(x)
        # Fused add + RMSNorm: writes residual in place, returns
        # (normalized, residual) so callers can keep chaining.
        residual = residual + x.to(residual.dtype)
        return self._rms_norm(residual), residual

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.variance_epsilon}"


class LayerNorm(nn.Module):
    """Standard layer normalization with a learned weight + bias."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.float(), (self.dim,), self.weight, self.bias, self.eps
        ).type_as(x)
