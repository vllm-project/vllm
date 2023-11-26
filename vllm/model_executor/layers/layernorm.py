"""Custom normalization layers."""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch._custom_ops as torch_custom_ops

from vllm._C import ops


@torch_custom_ops.custom_op("vllm::rms")
def rms(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    raise NotImplementedError()


@torch_custom_ops.impl("vllm::rms", device_types="cuda")
def rms_impl(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    out = torch.empty_like(hidden_states)
    ops.rms_norm(
        out,
        hidden_states,
        weight,
        eps,
    )
    return out


@torch_custom_ops.impl_abstract("vllm::rms")
def rms_abstract(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


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

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            # FIXME: Used fused_add_rms_norm.
            x = x + residual
            out = torch.ops.vllm.rms(x, self.weight.data,
                                     self.variance_epsilon)
            return out, x
        out = torch.ops.vllm.rms(x, self.weight.data, self.variance_epsilon)
        return out
