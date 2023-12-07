"""Custom normalization layers."""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from vllm._C import ops


class RMSNorm(nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(self,
                 hidden_size: int,
                 eps: float = 1e-6,
                 use_quant: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.use_quant = use_quant

    def _forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out = torch.empty_like(x, dtype=torch.int8) if self.use_quant else torch.empty_like(x)
        if residual is not None:
            ops.fused_add_rms_norm(
                out,
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
                self.use_quant
            )
            return out, residual
        ops.rms_norm(
            out,
            x,
            self.weight.data,
            self.variance_epsilon,
            self.use_quant
        )
        return out
class DequantAddResidualI8RMSNormQuant(nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    # TODO(Zhang Ying): use_per_token_dequant
    def __init__(self,
                 hidden_size: int,
                 dequant_scale: float = 1.0,
                 use_per_token_dequant: bool = True,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.dequant_scale = Parameter(
            torch.tensor(dequant_scale, dtype=torch.float32),
            False
        )
        self.use_per_token_dequant = use_per_token_dequant

    def _apply(self, fn):
        super()._apply(fn)
        self.dequant_scale = self.dequant_scale.cpu()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.dequant_scale = self.dequant_scale.to(*args, **kwargs)
        self.dequant_scale = self.dequant_scale.to(torch.float32)
        return self

    def forward(self,
                x: torch.Tensor,
                residual: torch.Tensor,
                scale: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty_like(x, dtype=torch.int8)
        if self.use_per_token_dequant and scale is not None:
            scale = scale * self.dequant_scale.item()
            ops.dequant_add_residual_rms_norm_quant(
                out, x, residual, self.weight.data, scale,
                self.variance_epsilon)
        else:
            ops.dequant_add_residual_rms_norm_quant(
                out, x, residual, self.weight.data, self.dequant_scale.item(),
                self.variance_epsilon)
        return out, residual
