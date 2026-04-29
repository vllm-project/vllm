# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom normalization layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import kernels
import vllm.kernels  # noqa: F401
from vllm import ir
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp

logger = init_logger(__name__)


def poly_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, variance_epsilon: float
) -> torch.Tensor:
    from vllm import _custom_ops as ops

    out = torch.empty_like(x)
    ops.poly_norm(  # type: ignore[attr-defined]
        out,
        x,
        weight,
        bias,
        variance_epsilon,
    )
    return out


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
        if residual is None:
            # TODO(luka): address the weight=None passing issue more generally
            return ir.ops.rms_norm(
                x,
                self.weight.data if self.has_weight else None,
                self.variance_epsilon,
                self.variance_size_override,
            )
        else:
            return ir.ops.fused_add_rms_norm.maybe_inplace(
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
                self.variance_size_override,
            )

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None and envs.VLLM_BATCH_INVARIANT:
            assert self.variance_size_override is None
            return rms_norm_batch_invariant(x, self.weight.data, self.variance_epsilon)
        return self.forward_native(x, residual)

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

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        weight = self.weight.data.float() + 1.0
        if residual is not None:
            x = (
                x.float() + residual.float()
                if orig_dtype == torch.float16
                else x + residual
            )
            residual = x
        # ir.ops.rms_norm handles fp32 upcast internally
        out = ir.ops.rms_norm(x, weight, self.variance_epsilon)
        return (
            out.to(orig_dtype) if residual is None else (out.to(orig_dtype), residual)
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        activation: str = "swish",
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
            activation: Activation function name for gating
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.activation = activation
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
        orig_dtype = x.dtype
        x = x.float()
        weight = self.weight.float()
        z = z.float() if z is not None else None

        assert self.activation in ["silu", "sigmoid", "swish"]
        act_fn = F.sigmoid if self.activation == "sigmoid" else F.silu

        # Apply gating before normalization if needed
        if z is not None and not self.norm_before_gate:
            x = x * act_fn(z)

        # RMS Normalization
        if self.group_size is None:
            # Standard RMS norm across the last dimension
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + self.eps)
            out = x_normed * weight
        else:
            # Group RMS norm
            from einops import rearrange

            x_group = rearrange(x, "... (g d) -> ... g d", d=self.group_size)
            variance = x_group.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x_group * torch.rsqrt(variance + self.eps)
            out = rearrange(x_normed, "... g d -> ... (g d)") * weight

        # Apply gating after normalization if needed
        if z is not None and self.norm_before_gate:
            out = out * act_fn(z)

        return out.to(orig_dtype)

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
            activation=self.activation,
        )

    def forward_xpu(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.forward_cuda(x, z)


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
