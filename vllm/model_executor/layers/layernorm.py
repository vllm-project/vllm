"""Custom normalization layers."""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from vllm._C import ops


class RMSNorm(nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

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
        if residual is not None:
            ops.fused_add_rms_norm(
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
            )
            return x, residual
        out = torch.empty_like(x)
        ops.rms_norm(
            out,
            x,
            self.weight.data,
            self.variance_epsilon,
        )
        return out


class RMSNormQuant(RMSNorm):
    """Root mean square normalization in SmoothQuant input_layernorm.
    It applies RMS normalization on x then quantizes outputs into int8.
    """

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out = torch.empty_like(x, dtype=torch.int8)
        if residual is not None:
            ops.add_residual_rms_norm_quant(out, x, residual, self.weight.data,
                                            self.variance_epsilon)
            return out, residual
        ops.rms_norm_quant(out, x, self.weight.data, self.variance_epsilon)
        return out


class DequantAddResidualI8RMSNormQuant(RMSNorm):
    """Root mean square normalization in SmoothQuant post_attn_layernorm.
    It first dequantizex x, then applies RMS normalization on the dequantized x, 
    finally quantizes outputs into int8.
    """

    # TODO(Zhang Ying): use_per_token_dequant
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        #  dequant_scale: float = 1.0,
        use_per_token_dequant: bool = True,
    ) -> None:
        super().__init__(hidden_size, eps)
        self.use_per_token_dequant = use_per_token_dequant

    def forward(
            self,
            x: torch.Tensor,
            residual: torch.Tensor,
            weight_dequant_scale: float,
            scale: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty_like(x, dtype=torch.int8)
        if self.use_per_token_dequant and scale is not None:
            ops.dequant_add_residual_rms_norm_quant(out, x, residual,
                                                    self.weight.data, scale,
                                                    self.variance_epsilon,
                                                    weight_dequant_scale)
        else:
            ops.dequant_add_residual_rms_norm_quant(out, x, residual,
                                                    self.weight.data,
                                                    weight_dequant_scale,
                                                    self.variance_epsilon)
        return out, residual
