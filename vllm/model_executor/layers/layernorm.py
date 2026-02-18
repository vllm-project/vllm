# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom normalization layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import kernels
import vllm.kernels  # noqa: F401
from vllm import ir
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.batch_invariant import (
    rms_norm_batch_invariant,
    vllm_is_batch_invariant,
)
from vllm.platforms import current_platform


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, variance_epsilon: float
) -> torch.Tensor:
    from vllm import _custom_ops as ops

    if vllm_is_batch_invariant():
        return rms_norm_batch_invariant(x, weight, variance_epsilon)
    out = torch.empty_like(x)
    ops.rms_norm(
        out,
        x,
        weight,
        variance_epsilon,
    )
    return out


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm import _custom_ops as ops

    if vllm_is_batch_invariant():
        return rms_norm_batch_invariant(
            x + residual, weight, variance_epsilon
        ), x + residual
    ops.fused_add_rms_norm(
        x,
        residual,
        weight,
        variance_epsilon,
    )
    return x, residual


def poly_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, variance_epsilon: float
) -> torch.Tensor:
    from vllm import _custom_ops as ops

    out = torch.empty_like(x)
    ops.poly_norm(
        out,
        x,
        weight,
        bias,
        variance_epsilon,
    )
    return out


def dispatch_rocm_rmsnorm_func(
    with_fused_add: bool, dtype: torch.dtype, use_aiter: bool = False
):
    use_aiter = use_aiter and dtype in [
        torch.float16,
        torch.bfloat16,
    ]

    if use_aiter and with_fused_add:
        return rocm_aiter_ops.rms_norm2d_with_add
    if use_aiter:
        return rocm_aiter_ops.rms_norm

    # fall back to CUDA implementation
    if with_fused_add:
        return fused_add_rms_norm
    return rms_norm


# --8<-- [start:rms_norm]
@CustomOp.register("rms_norm")
class RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    # --8<-- [end:rms_norm]

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
        self.weight = torch.ones(hidden_size, dtype=weight_dtype)
        if self.has_weight:
            self.weight = nn.Parameter(self.weight)

        if current_platform.is_rocm():
            aiter_rmsnorm_enabled = rocm_aiter_ops.is_rmsnorm_enabled()
            self.rocm_norm_func = dispatch_rocm_rmsnorm_func(
                with_fused_add=False,
                dtype=weight_dtype,
                use_aiter=aiter_rmsnorm_enabled,
            )
            self.rocm_norm_func_with_add = dispatch_rocm_rmsnorm_func(
                with_fused_add=True, dtype=weight_dtype, use_aiter=aiter_rmsnorm_enabled
            )

    @staticmethod
    def forward_static(
        x: torch.Tensor,
        variance_epsilon: float,
        hidden_size: int,
        orig_dtype: torch.dtype,
        weight: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        variance_size_override: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        x = x.to(torch.float32)
        if residual is not None:
            # residual promoted f16->f32 automatically,
            # otherwise Inductor eliminates the casts to and from f16,
            # increasing memory usage (and complicating pattern matching)
            x = x + residual
            residual = x.to(orig_dtype)

        if x.shape[-1] != hidden_size:
            raise ValueError(
                f"Expected hidden_size to be {hidden_size}, but found: {x.shape[-1]}"
            )

        if variance_size_override is None:
            x_var = x
        else:
            if hidden_size < variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{variance_size_override}, but found: {hidden_size}"
                )

            x_var = x[:, :, :variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x.to(orig_dtype)
        if weight is not None:
            x = x * weight
        if residual is None:
            return x
        else:
            return x, residual

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        add_residual = residual is not None
        weight = self.weight.data if self.has_weight else None
        if not add_residual:
            return ir.ops.rms_norm(
                x, weight, self.variance_epsilon, self.variance_size_override
            )

        return self.forward_static(
            x,
            self.variance_epsilon,
            self.hidden_size,
            x.dtype,
            self.weight.data if self.has_weight else None,
            residual,
            self.variance_size_override,
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        add_residual = residual is not None
        weight = self.weight.data if self.has_weight else None
        if not add_residual and not vllm_is_batch_invariant():
            return ir.ops.rms_norm(
                x, weight, self.variance_epsilon, self.variance_size_override
            )

        if self.variance_size_override is not None:
            return self.forward_native(x, residual)

        if add_residual:
            return fused_add_rms_norm(
                x, residual, self.weight.data, self.variance_epsilon
            )
        else:
            return rms_norm(x, self.weight.data, self.variance_epsilon)

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        add_residual = residual is not None
        weight = self.weight.data if self.has_weight else None
        if not add_residual:
            return ir.ops.rms_norm(
                x, weight, self.variance_epsilon, self.variance_size_override
            )

        if self.variance_size_override is not None:
            return self.forward_native(x, residual)

        add_residual = residual is not None
        if add_residual:
            return self.rocm_norm_func_with_add(
                x, residual, self.weight.data, self.variance_epsilon
            )
        else:
            return self.rocm_norm_func(x, self.weight.data, self.variance_epsilon)

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_cuda(x, residual)

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s


# --8<-- [start:gemma_rms_norm]
@CustomOp.register("gemma_rms_norm")
class GemmaRMSNorm(CustomOp):
    """RMS normalization for Gemma.

    Two differences from the above RMSNorm:
        1. x * (1 + w) instead of x * w.
        2. (x * w).to(orig_dtype) instead of x.to(orig_dtype) * w.
    """

    # --8<-- [end:gemma_rms_norm]

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    @staticmethod
    def _forward_static_no_residual(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward() without residual."""
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x * (1.0 + weight.float())
        x = x.to(orig_dtype)
        return x

    @staticmethod
    def _forward_static_with_residual(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward() with residual."""
        orig_dtype = x.dtype
        x = (
            x.float() + residual.float()
            if orig_dtype == torch.float16
            else x + residual
        )
        residual = x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        x = x * (1.0 + weight.float())
        x = x.to(orig_dtype)
        return x, residual

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        if residual is None:
            return self._forward_static_no_residual(
                self.weight.data, self.variance_epsilon, x
            )
        else:
            return self._forward_static_with_residual(
                self.weight.data, self.variance_epsilon, x, residual
            )

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if torch.compiler.is_compiling():
            return self.forward_native(x, residual)

        if not getattr(self, "_is_compiled", False):
            self._forward_static_no_residual = torch.compile(  # type: ignore
                self._forward_static_no_residual
            )
            self._forward_static_with_residual = torch.compile(  # type: ignore
                self._forward_static_with_residual
            )
            self._is_compiled = True
        return self.forward_native(x, residual)


# --8<-- [start:rms_norm_gated]
@CustomOp.register("rms_norm_gated")
class RMSNormGated(CustomOp):
    """RMS Normalization with optional gating.

    This is a native PyTorch implementation that supports:
    - Standard RMS normalization
    - Group RMS normalization
    - Optional gating with SiLU activation
    """

    # --8<-- [end:rms_norm_gated]

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize RMSNormGated.

        Args:
            hidden_size: Size of the hidden dimension
            eps: Epsilon for numerical stability
            group_size: If not None, do GroupNorm with each group
                        having group_size elements.
                        group_size=None is equivalent to group_size=hidden_size
                        (i.e. there's only 1 group).
            norm_before_gate: If True and z is provided: out = norm(x) * silu(z)
                              If False and z is provided: out = norm(x * silu(z))
            device: Device to create parameters on
            dtype: Data type for parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward_native(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Native PyTorch implementation of RMS normalization with gating.

        Args:
            x: Input tensor
            z: Optional gating tensor

        Returns:
            Normalized (and optionally gated) tensor

        If z is not None:
            - norm_before_gate=True: out = norm(x) * silu(z)
            - norm_before_gate=False: out = norm(x * silu(z))
        """
        # Apply gating before normalization if needed
        if z is not None and not self.norm_before_gate:
            x = x * F.silu(z)

        # RMS Normalization
        if self.group_size is None:
            # Standard RMS norm across the last dimension
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + self.eps)
            out = x_normed * self.weight
        else:
            # Group RMS norm
            from einops import rearrange

            x_group = rearrange(x, "... (g d) -> ... g d", d=self.group_size)
            variance = x_group.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x_group * torch.rsqrt(variance + self.eps)
            out = rearrange(x_normed, "... g d -> ... (g d)") * self.weight

        # Apply gating after normalization if needed
        if z is not None and self.norm_before_gate:
            out = out * F.silu(z)

        return out

    def forward_cuda(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn

        return rmsnorm_fn(
            x,
            self.weight,
            self.bias,
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
        )


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(
            x.float(), (self.dim,), self.weight, self.bias, self.eps
        ).type_as(x)
