# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom normalization layers."""
from typing import Optional, Union

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform


def is_rocm_aiter_rmsnorm_enabled() -> bool:
    return current_platform.is_rocm() \
        and envs.VLLM_ROCM_USE_AITER_RMSNORM \
        and envs.VLLM_ROCM_USE_AITER


def rms_norm(x: torch.Tensor, weight: torch.Tensor,
             variance_epsilon: float) -> torch.Tensor:
    from vllm import _custom_ops as ops
    out = torch.empty_like(x)
    ops.rms_norm(
        out,
        x,
        weight,
        variance_epsilon,
    )
    return out


def fused_add_rms_norm(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
        variance_epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm import _custom_ops as ops
    ops.fused_add_rms_norm(
        x,
        residual,
        weight,
        variance_epsilon,
    )
    return x, residual


def rocm_aiter_rms_norm(x: torch.Tensor, weight: torch.Tensor,
                        variance_epsilon: float) -> torch.Tensor:

    import aiter as rocm_aiter
    if x.dim() > 2:
        x_original_shape = x.shape
        x = x.reshape(-1, x_original_shape[-1])
        x = rocm_aiter.rms_norm(x, weight, variance_epsilon)
        return x.reshape(x_original_shape)

    return rocm_aiter.rms_norm(x, weight, variance_epsilon)


def rocm_aiter_fused_add_rms_norm(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
        variance_epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:

    import aiter as rocm_aiter

    residual_out = torch.empty_like(residual)
    output = torch.empty_like(x)
    rocm_aiter.rmsnorm2d_fwd_with_add(
        output,  # output
        x,  # input
        residual,  # residual input
        residual_out,  # residual output
        weight,
        variance_epsilon,
    )
    return output, residual_out


def dispatch_cuda_rmsnorm_func(add_residual: bool):
    if add_residual:
        if is_rocm_aiter_rmsnorm_enabled():
            return rocm_aiter_fused_add_rms_norm
        return fused_add_rms_norm

    if is_rocm_aiter_rmsnorm_enabled():
        return rocm_aiter_rms_norm
    return rms_norm


@CustomOp.register("rms_norm")
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
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = (None if var_hidden_size == hidden_size
                                       else var_hidden_size)
        self.has_weight = has_weight
        if dtype is not None:
            self.weight = torch.ones(hidden_size, dtype=dtype)
        else:
            self.weight = torch.ones(hidden_size)
        if self.has_weight:
            self.weight = nn.Parameter(self.weight)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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
        x = x.to(orig_dtype)
        if self.has_weight:
            x = x * self.weight
        if residual is None:
            return x
        else:
            return x, residual

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)

        add_residual = residual is not None
        norm_func = dispatch_cuda_rmsnorm_func(add_residual)

        if add_residual:
            return norm_func(x, residual, self.weight.data,
                             self.variance_epsilon)
        else:
            return norm_func(x, self.weight.data, self.variance_epsilon)

    def forward_hpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from vllm_hpu_extension.kernels import rms_norm
        HPUFusedRMSNorm = rms_norm()
        if HPUFusedRMSNorm is None:
            return self.forward_native(x, residual)
        if residual is not None:
            orig_shape = x.shape
            residual += x.view(residual.shape)
            # Note: HPUFusedRMSNorm requires 3D tensors as inputs
            x = HPUFusedRMSNorm.apply(residual, self.weight,
                                      self.variance_epsilon)
            return x.view(orig_shape), residual

        x = HPUFusedRMSNorm.apply(x, self.weight, self.variance_epsilon)
        return x

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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


@CustomOp.register("gemma_rms_norm")
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
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        if residual is not None:
            if orig_dtype == torch.float16:
                x = x + residual.float()
            else:
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
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward()."""
        return self.forward_static(self.weight.data, self.variance_epsilon, x,
                                   residual)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if torch.compiler.is_compiling():
            return self.forward_native(x, residual)

        if not getattr(self, "_is_compiled", False):
            self.forward_static = torch.compile(  # type: ignore
                self.forward_static)
            self._is_compiled = True
        return self.forward_native(x, residual)
