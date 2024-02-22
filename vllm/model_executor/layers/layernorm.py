"""Custom normalization layers."""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from vllm._C import ops
from vllm.lowering_utils import vllm_lib, register_vllm_lowering


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
        torch.ops.vllm.rms_norm(
            out,
            x,
            self.weight.data,
            self.variance_epsilon,
        )
        return out

# needed for compile
vllm_lib.define(
    "rms_norm(Tensor(a!) out, Tensor input, Tensor weight, float epsilon) -> Tensor(a!)"
)

@torch.library.impl(vllm_lib, "rms_norm", "Meta")
def _rms_norm_meta(out, input, weight, epsilon):
    return out


@torch.library.impl(vllm_lib, "rms_norm", "CUDA")
def _rms_norm(out, input, weight, epsilon):
    ops.rms_norm(
        out,
        input,
        weight,
        epsilon,
    )

register_vllm_lowering(torch.ops.vllm.rms_norm, [0])
