# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom normalization layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm import _oink_ops, envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.batch_invariant import (
    rms_norm_batch_invariant,
    vllm_is_batch_invariant,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _can_view_as_2d(x: torch.Tensor) -> bool:
    """Return True if x.view(-1, x.shape[-1]) is viewable (no copy)."""
    if x.dim() < 2:
        return False
    if x.dim() == 2:
        return True
    for dim in range(x.dim() - 1):
        # Strides for size-1 dims are irrelevant and can be arbitrary.
        if x.size(dim + 1) != 1 and x.stride(dim) != x.stride(dim + 1) * x.size(
            dim + 1
        ):
            return False
    return True


def _is_oink_stride_compatible_2d(x_2d: torch.Tensor) -> bool:
    """Return True if x_2d meets Oink's pointer-path stride constraints."""
    if x_2d.dim() != 2:
        return False
    if x_2d.stride(1) != 1:
        return False
    # Match Oink's vectorization constraint: stride(0) divisible by 256b.
    if x_2d.dtype in (torch.float16, torch.bfloat16):
        divby = 16
    elif x_2d.dtype == torch.float32:
        divby = 8
    else:
        return False
    return (x_2d.stride(0) % divby) == 0


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

        # Optional: enable Oink Blackwell RMSNorm custom-op fast path on
        # compatible CUDA devices (e.g., SM100) when the external Oink
        # package is available. This is detected once at construction time
        # to avoid per-call device queries in the hot path.
        self._use_oink_rmsnorm = False
        self._use_oink_fused_add_rmsnorm = False
        if (
            not current_platform.is_rocm()
            and torch.cuda.is_available()
            and bool(getattr(envs, "VLLM_USE_OINK_RMSNORM", False))
        ):
            # NOTE: vLLM disables custom ops by default when using Inductor.
            # If this op is disabled, CustomOp will dispatch to forward_native,
            # and the Oink path in forward_cuda will never run.
            if getattr(self._forward_method, "__func__", None) is getattr(
                self.forward_native, "__func__", None
            ):
                try:
                    from vllm.config import get_cached_compilation_config

                    custom_ops = get_cached_compilation_config().custom_ops
                except Exception:
                    custom_ops = ["<unknown>"]
                logger.warning_once(
                    "VLLM_USE_OINK_RMSNORM=1 but the `rms_norm` custom op is "
                    "disabled (CompilationConfig.custom_ops=%s). Enable it via "
                    "`compilation_config={'custom_ops': ['none', '+rms_norm']}` "
                    "(or `['all']`) to let vLLM call into torch.ops.oink.*.",
                    custom_ops,
                )
                # Custom op disabled => forward_cuda won't run. Avoid doing any
                # external Oink initialization work in this case.
            else:
                try:
                    device_index = torch.cuda.current_device()
                    if _oink_ops.is_oink_available_for_device(device_index):
                        self._use_oink_rmsnorm = True
                        self._use_oink_fused_add_rmsnorm = (
                            _oink_ops.has_fused_add_rms_norm()
                        )
                except Exception as e:
                    # If anything goes wrong (no Oink install, CPU-only env, etc.),
                    # silently fall back to the built-in RMSNorm path.
                    logger.warning_once(
                        "VLLM_USE_OINK_RMSNORM=1 but failed to initialize Oink "
                        "RMSNorm; falling back to vLLM RMSNorm. Error: %s",
                        e,
                    )
                    self._use_oink_rmsnorm = False
                    self._use_oink_fused_add_rmsnorm = False

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
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)

        # Optional Oink SM100 fast path (no residual). This path is
        # torch.compile-friendly via torch.ops.oink.rmsnorm and preserves
        # 2D layouts (including padded rows) when using the Oink
        # pointer-based kernel.
        if (
            residual is None
            and getattr(self, "_use_oink_rmsnorm", False)
            and x.is_cuda
            and x.dim() >= 2
            and self.has_weight
            and not vllm_is_batch_invariant()
            and self.weight.data.dtype == x.dtype
            and self.weight.data.is_contiguous()
        ):
            orig_shape = x.shape
            hidden_size = orig_shape[-1]
            if _can_view_as_2d(x):
                x_2d = x.view(-1, hidden_size)
                if _is_oink_stride_compatible_2d(x_2d):
                    y_2d = _oink_ops.rmsnorm(
                        x_2d,
                        self.weight.data,
                        self.variance_epsilon,
                    )
                    return y_2d.view(orig_shape)

        # Optional Oink SM100 fast path (fused residual-add + RMSNorm, in-place).
        # This mirrors vLLM's fused_add_rms_norm semantics by mutating both
        # `x` (normalized output) and `residual` (residual-out buffer).
        if (
            residual is not None
            and getattr(self, "_use_oink_fused_add_rmsnorm", False)
            and x.is_cuda
            and residual.is_cuda
            and x.shape == residual.shape
            and x.dtype == residual.dtype
            and x.dim() >= 2
            and self.has_weight
            and not vllm_is_batch_invariant()
            and self.weight.data.dtype == x.dtype
            and self.weight.data.is_contiguous()
        ):
            orig_shape = x.shape
            hidden_size = orig_shape[-1]
            if _can_view_as_2d(x) and _can_view_as_2d(residual):
                x_2d = x.view(-1, hidden_size)
                res_2d = residual.view(-1, hidden_size)

                # The Oink in-place pointer path supports the common vLLM
                # layout where:
                # - `x` may be strided/padded row-major (stride(1) == 1), and
                # - `residual` is contiguous row-major ([M, N] with stride(0) == N).
                # If these conditions are not met, fall back to vLLM's built-in
                # fused kernel.
                if (
                    _is_oink_stride_compatible_2d(x_2d)
                    and _is_oink_stride_compatible_2d(res_2d)
                    and res_2d.is_contiguous()
                ):
                    _oink_ops.fused_add_rms_norm_(
                        x_2d,
                        res_2d,
                        self.weight.data,
                        self.variance_epsilon,
                    )
                    return x, residual

        add_residual = residual is not None
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
        residual: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        if residual is not None:
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
        return x if residual is None else (x, residual)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        return self.forward_static(self.weight.data, self.variance_epsilon, x, residual)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if torch.compiler.is_compiling():
            return self.forward_native(x, residual)

        if not getattr(self, "_is_compiled", False):
            self.forward_static = torch.compile(  # type: ignore
                self.forward_static
            )
            self._is_compiled = True
        return self.forward_native(x, residual)


@CustomOp.register("rms_norm_gated")
class RMSNormGated(CustomOp):
    """RMS Normalization with optional gating.

    This is a native PyTorch implementation that supports:
    - Standard RMS normalization
    - Group RMS normalization
    - Optional gating with SiLU activation
    """

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
