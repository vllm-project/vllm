"""Custom normalization layers."""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from vllm.model_executor.custom_op import CustomOp

from vllm import _custom_ops as ops
import intel_extension_for_pytorch as ipex

@torch.library.impl("myops::_fused_add_rms_norm", "cpu")
def _fused_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, variance_epsilon: float):
    ops.fused_add_rms_norm(
        x,
        residual,
        weight.data,
        variance_epsilon,
    )
    return x, residual

torch.library.define(
    "myops::_fused_add_rms_norm",
    "(Tensor x,Tensor residual,Tensor weight,float variance_epsilon) -> (Tensor, Tensor)",
)


@torch.library.impl("myops::_rms_norm", "cpu")
def _rms_norm(
    x: torch.Tensor, weight: torch.Tensor, variance_epsilon: float):
    out = torch.empty_like(x)
    ops.rms_norm(
        out,
        x,
        weight.data,
        variance_epsilon,
    )
    return out

torch.library.define(
    "myops::_rms_norm",
    "(Tensor x,Tensor weight,float variance_epsilon) -> Tensor",
)

class RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        tp_size:int = 1,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.tp_size = tp_size
    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.tp_size > 1 :

            if residual is not None:
                x, residual = torch.ops.myops._fused_add_rms_norm(
                    x,
                    residual,
                    self.weight.data,
                    self.variance_epsilon,
                )
                return x, residual
            out = torch.ops.myops._rms_norm(
                x,
                self.weight.data,
                self.variance_epsilon,
            )
            return out
        else:
            if residual is not None:
                x = ipex.llm.functional.add_rms_norm(
                    residual,
                    x,
                    self.weight.data,
                    None,
                    self.variance_epsilon,
                    True
                )
                return x, residual
            out = ipex.llm.functional.rms_norm(
                x,
                self.weight.data,
                self.variance_epsilon,
            )
            return out

    def forward_native(
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


    def forward_cpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Note: the forward_native() with torch.compile has significant
        # performance regression.
        return self.forward_cuda(
            x,
            residual,
        )

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s
