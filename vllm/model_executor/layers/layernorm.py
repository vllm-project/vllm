"""Custom normalization layers."""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from vllm.model_executor.custom_op import CustomOp


class RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = (None if var_hidden_size == hidden_size
                                       else var_hidden_size)

        self.weight = nn.Parameter(torch.ones(hidden_size))

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

        hidden_size = x.shape[-1]
        if hidden_size != self.hidden_size:
            raise ValueError("Expected hidden_size to be "
                             f"{self.hidden_size}, but found: {hidden_size}")

        if self.variance_size_override is None:
            x_var = x
        else:
            if hidden_size < self.variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {hidden_size}")

            x_var = x[:, :, :self.variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)

        from vllm import _custom_ops as ops

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

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)

        from vllm._ipex_ops import ipex_ops as ops

        if residual is not None:
            ops.fused_add_rms_norm(
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
            )
            return x, residual
        return ops.rms_norm(
            x,
            self.weight.data,
            self.variance_epsilon,
        )

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s


class GemmaRMSNorm(CustomOp):
    """RMS normalization for Gemma.

    Two differences from the above RMSNorm:
        1. x * (1 + w) instead of x * w.
        2. (x * w).to(orig_dtype) instead of x.to(orig_dtype) * w.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    @staticmethod
    def forward_static(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        if residual is not None:
            x = x + residual
            residual = x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        x = x * (1.0 + weight.float())
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward()."""
        return self.forward_static(self.weight.data, self.variance_epsilon, x,
                                   residual)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if torch.compiler.is_compiling():
            return self.forward_native(x, residual)

        if not getattr(self, "_is_compiled", False):
            self.forward_static = torch.compile(  # type: ignore
                self.forward_static)
            self._is_compiled = True
        return self.forward_native(x, residual)
