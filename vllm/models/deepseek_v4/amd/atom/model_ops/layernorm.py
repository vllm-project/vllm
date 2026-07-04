# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Tuple

import aiter
import torch
from aiter import (
    QuantType,
    layernorm2d_fwd,
    layernorm2d_fwd_with_add,
    rmsnorm2d_fwd,
    rmsnorm2d_fwd_with_add,
)
from vllm.models.deepseek_v4.amd.atom.distributed.parallel import (
    tensor_model_parallel_fused_allreduce_rmsnorm,
    tensor_model_parallel_fused_allreduce_rmsnorm_quant,
)
from vllm.models.deepseek_v4.amd.atom.distributed.parallel import get_tensor_model_parallel_world_size
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.gated_rmsnorm_fp8_group_quant import gated_rmsnorm_fp8_group_quant
from aiter.ops.triton.fused_add_rmsnorm_pad import fused_add_rmsnorm_pad
from vllm.models.deepseek_v4.amd.atom.config import QuantizationConfig
from vllm.models.deepseek_v4.amd.atom.model_ops.utils import atom_parameter
from vllm.models.deepseek_v4.amd.atom.quant_spec import LayerQuantConfig, should_skip_online_quant
from vllm.models.deepseek_v4.amd.atom.utils.decorators import mark_trace
from vllm.models.deepseek_v4.amd.atom.utils import envs
from torch import Tensor, nn
from torch.overrides import handle_torch_function, has_torch_function_unary


def silu(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Apply the Sigmoid Linear Unit (SiLU) function, element-wise.

    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    See :class:`~torch.nn.SiLU` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(silu, (input,), input, inplace=inplace)
    if inplace:
        return torch._C._nn.silu_(input)
    return torch._C._nn.silu(input)


@torch_compile_guard()
def rmsnorm2d_fwd_(
    x: torch.Tensor, weight: torch.Tensor, eps: float, dim: int
) -> torch.Tensor:
    ori_shape = x.shape
    x = x.reshape(-1, dim)
    return rmsnorm2d_fwd(x, weight, eps).view(ori_shape)


@torch_compile_guard()
def rmsnorm2d_fwd_with_add_(
    x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor, eps: float, dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    ori_shape = x.shape
    x = x.reshape(-1, dim)
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    rmsnorm2d_fwd_with_add(out, x, residual, residual_out, weight, eps)
    return out.view(ori_shape), residual_out.view(ori_shape)


def fused_rmsnorm_pad_fake_tensors(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    x_pad_to_multiple: int = 0,
) -> torch.Tensor:
    M, N = x.shape
    N_out = (N + x_pad_to_multiple - 1) // x_pad_to_multiple * x_pad_to_multiple
    out = torch.empty((M, N_out), dtype=x.dtype, device=x.device)
    return out


@torch_compile_guard(gen_fake=fused_rmsnorm_pad_fake_tensors)
def fused_rmsnorm_pad_(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    x_pad_to_multiple: int = 0,
) -> torch.Tensor:
    return fused_add_rmsnorm_pad(x, weight, epsilon, None, x_pad_to_multiple)


def fused_add_rmsnorm_pad_fake_tensors(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    res: torch.Tensor,
    x_pad_to_multiple: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    N_out = (N + x_pad_to_multiple - 1) // x_pad_to_multiple * x_pad_to_multiple
    out = torch.empty((M, N_out), dtype=x.dtype, device=x.device)
    res_out = torch.empty((M, N), dtype=res.dtype, device=res.device)
    return out, res_out


@torch_compile_guard(gen_fake=fused_add_rmsnorm_pad_fake_tensors)
def fused_add_rmsnorm_pad_(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    res: torch.Tensor,
    x_pad_to_multiple: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return fused_add_rmsnorm_pad(x, weight, epsilon, res, x_pad_to_multiple)


# ---------------------------------------------------------------------------
# Aiter dynamic RMSNorm + quant — single dispatch covering per_1x32 (MXFP4),
# per_1x128 (FP8 block), and per_Token (FP8). All three reach
# aiter.{rmsnorm_quant, add_rmsnorm_quant} (HIP) which both normalizes and
# emits a freshly-computed scale, so callers must have x_scale=None
# (static-scale FP8 stays on its own branch). Per-quant params (out_dtype,
# scale shape, group_size, shuffle_scale) are derived from quant_type_value
# inside the fake helper so torch.compile's schema infer sees a single,
# stable signature.
#
# `mutates_args=[]` keeps torch.compile from functionalizing the out-buffers
# — same pattern as the legacy mxfp4 fuse helper.
# ---------------------------------------------------------------------------

_QV_PER_1X32 = QuantType.per_1x32.value
_QV_PER_1X128 = QuantType.per_1x128.value
_QV_PER_TOKEN = QuantType.per_Token.value
_AITER_RMS_QUANT_TYPE_VALUES = frozenset({_QV_PER_1X32, _QV_PER_1X128, _QV_PER_TOKEN})


def _aiter_rms_quant_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    quant_type_value: int,
    transpose_scale: bool,
    res1: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter.utility.dtypes import fp8

    M, N = x.shape
    if quant_type_value == _QV_PER_1X32:
        # MXFP4: out=(M, N/2) fp4x2; scale=(⌈M/256⌉*256, ⌈⌈N/32⌉/8⌉*8) UE8M0
        # bytes (kernel writes one uint8 per group; passing fp8_e8m0fnu
        # directly yields the matching byte layout — no fp32 view).
        out = torch.empty((M, N // 2), dtype=torch.float4_e2m1fn_x2, device=x.device)
        scale_m = ((M + 255) // 256) * 256
        scale_n = ((((N + 31) // 32) + 7) // 8) * 8
        scale = torch.empty(
            (scale_m, scale_n), dtype=torch.float8_e8m0fnu, device=x.device
        )
    elif quant_type_value == _QV_PER_1X128:
        # FP8 per-block: scale=(M, ⌈N/128⌉) fp32. Preshuffle GEMM expects
        # column-major; allocate (num_groups, M) row-major then view as
        # (M, num_groups). Matches GemmaRMSNorm._forward_fused_fp8.
        out = torch.empty((M, N), dtype=fp8, device=x.device)
        num_groups = N // 128
        if transpose_scale:
            scale = torch.empty(
                (num_groups, M), dtype=torch.float32, device=x.device
            ).view(M, num_groups)
        else:
            scale = torch.empty((M, num_groups), dtype=torch.float32, device=x.device)
    else:  # _QV_PER_TOKEN
        out = torch.empty((M, N), dtype=fp8, device=x.device)
        scale = torch.empty((M, 1), dtype=torch.float32, device=x.device)
    out_res1 = torch.empty_like(res1) if res1 is not None else None
    return (out, scale, out_res1)


@torch_compile_guard(gen_fake=_aiter_rms_quant_fake, mutates_args=[])
def _aiter_rms_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    quant_type_value: int,
    transpose_scale: bool,
    res1: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter import add_rmsnorm_quant, rmsnorm_quant

    out, scale, out_res1 = _aiter_rms_quant_fake(
        x, weight, eps, quant_type_value, transpose_scale, res1
    )
    if quant_type_value == _QV_PER_1X32:
        group_size, shuffle = 32, True
    elif quant_type_value == _QV_PER_1X128:
        group_size, shuffle = 128, transpose_scale
    else:  # _QV_PER_TOKEN
        group_size, shuffle = 0, False
    if res1 is None:
        rmsnorm_quant(out, x, scale, weight, eps, group_size, shuffle)
    else:
        add_rmsnorm_quant(
            out, x, res1, out_res1, scale, weight, eps, group_size, shuffle
        )
    return out, scale, out_res1


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        x_pad_to_multiple: int = 0,
        fused_allreduce: bool = False,
        fused_quant: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = atom_parameter(torch.ones(dim))
        self.x_pad_to_multiple = x_pad_to_multiple
        self.fused_allreduce = fused_allreduce
        self.use_fused_quant = fused_quant
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.prefix = prefix

        layer_quant_config = (
            LayerQuantConfig()
            if quant_config is None
            else quant_config.get_layer_quant_config(prefix)
        )
        quant_type = layer_quant_config.quant_type
        params_dtype = layer_quant_config.quant_dtype
        self.quant_type = quant_type
        self.params_dtype = params_dtype
        # transpose_scale (column-major scale) only applies to per_1x128 with
        # the preshuffle GEMM consumer; resolve the env once at init time so
        # forward sees a hot static bool instead of an env lookup per call.
        self._aiter_transpose_scale = (
            fused_quant
            and quant_type.value == _QV_PER_1X128
            and envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE
        )

    def process_weights_after_loading(self):
        """Post-load hook invoked by the model loader for every module.

        RMSNorm has no weights to re-quantize, but when online quantization is
        enabled the fused RMSNorm+quant activation scheme may need to follow the
        downstream weight's online-quant target -- delegated to
        ``online_quantize_activation``.
        """
        # Only the fused-quant path's output depends on `quant_type`; a plain
        # RMSNorm emits BF16 regardless, so nothing to realign.
        if self.use_fused_quant:
            self.online_quantize_activation()

    def online_quantize_activation(self):
        """Realign the fused RMSNorm activation quant scheme with online quant.

        The fused RMSNorm+quant path emits a quantized activation that a
        downstream Linear consumes directly without re-quantizing (e.g.
        DeepSeek-V4 ``q_norm`` -> ``attn.wq_b`` / ``indexer.wq_b``). The GEMM
        interprets that activation's dtype/scale layout according to the
        *weight*'s quant scheme. So when online quantization re-quantizes the
        consumer's weight to a new scheme (e.g. mxfp4 = per_1x32 + fp4x2), the
        activation emitted here MUST switch to the same scheme; otherwise the
        per_1x32 GEMM bit-reinterprets a per_1x128 fp8 activation via
        ``view(fp4x2)`` / ``view(e8m0)`` and produces garbage.

        Mirrors ``LinearBase.online_quantize_weight``'s quant-state update, but
        there is no weight to re-quantize here -- only the emitted activation
        ``quant_type`` / ``params_dtype`` (and the derived scale layout).
        """
        if self.quant_config is None or not self.quant_config.online_quant:
            return

        online_cfg = self.quant_config.get_layer_quant_config(
            self.prefix, use_online_quant=True
        )
        online_quant_type = online_cfg.quant_type
        # Skip if excluded (No) or already emitting the target scheme.
        if should_skip_online_quant(self.quant_type, self.params_dtype, online_cfg):
            return
        # The fused RMSNorm+quant HIP kernel only emits these activation schemes.
        assert online_quant_type.value in _AITER_RMS_QUANT_TYPE_VALUES, (
            f"Unsupported online activation quant for fused RMSNorm: "
            f"type={online_quant_type}, dtype={online_cfg.quant_dtype} "
            f"(layer={self.prefix})"
        )

        self.quant_type = online_quant_type
        self.params_dtype = online_cfg.quant_dtype
        self._aiter_transpose_scale = (
            self.quant_type.value == _QV_PER_1X128
            and envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE
        )
        # Surfaced by the loader's online-quant report alongside Linear layers.
        self._online_quant_info = {
            "layer": self.prefix,
            "quant_type": online_quant_type.name,
            "quant_dtype": str(online_cfg.quant_dtype),
            "kind": "rmsnorm_activation",
        }

    @mark_trace(prefix="rmsnorm", torch_compile=True)
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        x_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.x_pad_to_multiple > 0:
            assert (
                not self.fused_allreduce
            ), "fused_allreduce_rmsnorm is not supported with rms_norm padding!"
            if residual is None:
                x = fused_rmsnorm_pad_(x, self.weight, self.eps, self.x_pad_to_multiple)
                return x
            else:
                x, residual = fused_add_rmsnorm_pad_(
                    x, self.weight, self.eps, residual, self.x_pad_to_multiple
                )
                return x, residual
        if self.fused_allreduce and self.tp_size > 1:
            assert (
                residual is not None
            ), "fused_allreduce_rmsnorm requires residual input!"
            # tensor_model_parallel_fused_allreduce_rmsnorm does not support non-contiguous input
            x, residual = tensor_model_parallel_fused_allreduce_rmsnorm(
                x.contiguous(),
                residual,
                self.weight,
                self.eps,
            )
            return x, residual
        else:
            if x_scale is not None and self.use_fused_quant:
                import aiter as rocm_aiter
                from aiter.ops.triton.fused_fp8_quant import (
                    fused_rms_fp8_per_tensor_static_quant,
                )

                rocm_aiter_fp8_dtype = rocm_aiter.dtypes.fp8

                # static FP8 quantization
                if residual is None:
                    x, _, _, _ = fused_rms_fp8_per_tensor_static_quant(
                        x,
                        self.weight,
                        self.eps,
                        x_scale,
                        None,
                        None,
                        self.eps,
                        dtype_quant=rocm_aiter_fp8_dtype,
                        res1=None,
                    )
                    return (x, x_scale)
                else:
                    x, _, _, residual = fused_rms_fp8_per_tensor_static_quant(
                        x,
                        self.weight,
                        self.eps,
                        x_scale,
                        None,
                        None,
                        self.eps,
                        dtype_quant=rocm_aiter_fp8_dtype,
                        res1=residual,
                    )
                    return (x, x_scale), residual
            elif (
                self.use_fused_quant
                and x_scale is None
                and self.quant_type.value in _AITER_RMS_QUANT_TYPE_VALUES
            ):
                # Dynamic-scale fused RMSNorm + quant via aiter HIP kernels.
                # Static FP8 (x_scale provided) stays on the branch above.
                x, x_scale, residual_out = _aiter_rms_quant(
                    x,
                    self.weight,
                    self.eps,
                    self.quant_type.value,
                    self._aiter_transpose_scale,
                    residual,
                )
                if residual is None:
                    return x, x_scale
                return (x, x_scale), residual_out
            else:
                if residual is None:
                    # return rmsnorm2d_fwd(x, self.weight, self.eps).view(ori_shape)
                    x = rmsnorm2d_fwd_(x, self.weight, self.eps, self.dim)
                    return x
                else:
                    # return self.add_rms_forward(x, residual)
                    x, residual = rmsnorm2d_fwd_with_add_(
                        x, self.weight, residual, self.eps, self.dim
                    )
                    return x, residual


class RMSNormGated(nn.Module):
    """RMS Normalization with optional gating.

    This is a native PyTorch implementation that supports:
    - Standard RMS normalization
    - Group RMS normalization
    - Optional gating with SiLU activation
    - Fused FP8 group quantization (when quant_config is provided)
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        quant_config=None,
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
            dtype: Data type for parameters
            quant_config: Quantization config (enables FP8 fusion if configured)
        """
        super().__init__()
        self.eps = eps
        self.weight = atom_parameter(torch.empty(hidden_size))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

        # Determine if we should use fused FP8 group quantization
        self.use_fused_fp8_quant = False
        self.group_size_quant = 128  # Default quantization group size
        self.transpose_scale = False  # Whether to transpose scale output
        self.quant_config = quant_config

        if quant_config is not None:
            from aiter import QuantType

            quant_type = quant_config.quant_type

            # Use fused kernel for per-block quantization (per_1x128, per_1x32)
            if quant_type in [QuantType.per_1x128, QuantType.per_1x32]:
                self.use_fused_fp8_quant = True
                # Extract group size from quant type
                if quant_type == QuantType.per_1x128:
                    self.group_size_quant = 128
                    # preshuffle GEMM expects column-major x_scale;
                    # non-preshuffle GEMM expects row-major x_scale
                    self.transpose_scale = envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE
                elif quant_type == QuantType.per_1x32:
                    self.group_size_quant = 32
                    self.transpose_scale = False

                # Import kernel when needed

                self.gated_rmsnorm_fp8_group_quant = gated_rmsnorm_fp8_group_quant

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward_native(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        """
        Native PyTorch implementation of RMS normalization with gating.

        Args:
            x: Input tensor [num_tokens, num_heads, head_dim]
            z: Gating tensor [num_tokens, num_heads, head_dim] (can be None)

        Returns:
            Tuple of (bf16_tensor, None)
            - bf16_tensor: BF16 output [num_tokens, num_heads*head_dim] (flattened)
            - None: No scale

        If z is not None:
            - norm_before_gate=True: out = norm(x) * silu(z)
            - norm_before_gate=False: out = norm(x * silu(z))
        """
        # Apply gating before normalization if needed
        if z is not None and not self.norm_before_gate:
            x = x * silu(z)

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
            out = out * silu(z)

        # Flatten to match fused kernel output: [num_tokens, num_heads, head_dim] -> [num_tokens, num_heads*head_dim]
        if len(out.shape) == 3:
            num_tokens = out.shape[0]
            out = out.reshape(num_tokens, -1)

        return (out, None)

    def forward_fused_fp8(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fused FP8 group quantization implementation.

        Args:
            x: Input tensor [num_tokens, num_heads, head_dim]
            z: Gating tensor [num_tokens, num_heads, head_dim]

        Returns:
            Tuple of (fp8_tensor, scale_tensor)
            - fp8_tensor: FP8 quantized output [num_tokens, num_heads*head_dim]
            - scale_tensor: Per-group scales [num_tokens, num_heads*num_groups]
                           In column-major layout if transpose_scale=True

        Performs: out = quantize(rms_norm(x, weight, eps) * silu(z), group_size)
        """
        num_tokens, num_heads, head_dim = x.shape
        # Check kernel constraints
        if (
            self.group_size is not None
            or not self.norm_before_gate
            or head_dim != self.group_size_quant
        ):
            # Grouped norm not supported by kernel, fallback
            return self.forward_native(x, z)

        out_fp8 = torch.empty(
            [num_tokens, num_heads * head_dim], dtype=aiter.dtypes.fp8, device=x.device
        )
        out_scales = torch.empty(
            [num_tokens, (num_heads * head_dim) // self.group_size_quant],
            dtype=torch.float,
            device=x.device,
        )
        self.gated_rmsnorm_fp8_group_quant(
            out_fp8,
            out_scales,
            x,
            z,
            self.weight,
            self.eps,
            self.group_size_quant,
            self.transpose_scale,
        )
        # Kernel already returns flattened outputs - no reshaping needed!
        # out_fp8: [num_tokens, num_heads*head_dim]
        # out_scales: [num_tokens, (num_heads*head_dim)//group_size]
        return (out_fp8, out_scales)

    def forward(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass with optional FP8 fusion.

        Args:
            x: Input tensor
            z: Gating tensor (required positional argument, can be None)

        Returns:
            Tuple of (output, scale)
            - FP8 case: (fp8_tensor, scale_tensor)
            - BF16 case: (bf16_tensor, None)
        """
        # Use fused FP8 kernel if enabled
        if self.use_fused_fp8_quant:
            return self.forward_fused_fp8(x, z)

        return self.forward_native(x, z)


class GemmaRMSNorm(nn.Module):
    """RMS normalization for Gemma.

    Two differences from the above RMSNorm:
        1. x * (1 + w) instead of x * w.
        2. (x * w).to(orig_dtype) instead of x.to(orig_dtype) * w.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        quant_config: LayerQuantConfig | None = None,
        write_bf16: bool = False,
    ) -> None:
        super().__init__()
        self.weight = atom_parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.use_fused_quant = False
        self.write_bf16 = write_bf16
        if quant_config is not None:
            from aiter import QuantType

            if quant_config.quant_type == QuantType.per_1x128:
                self.use_fused_quant = True

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
        # Use the aiter HIP fused_qk_rmsnorm_group_quant kernel in no-quant mode
        # (q_out_scale=None) to perform Gemma RMSNorm + optional residual add.
        # Same math as the Triton kernel: out = rmsnorm(x [+ residual]) * (1 + w),
        # but executed by the aiter kernel for higher achieved bandwidth.
        from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm_group_quant

        ori_shape = x.shape
        x_2d = x.view(-1, ori_shape[-1])

        out = torch.empty_like(x_2d)
        if residual is not None:
            residual_2d = residual.view(-1, ori_shape[-1])
            res_out = torch.empty_like(x_2d)
        else:
            residual_2d = None
            res_out = None

        fused_qk_rmsnorm_group_quant(
            q=x_2d,
            q_weight=self.weight.data,
            q_epsilon=self.variance_epsilon,
            q_out_unquantized=out,
            q_res_out=res_out,
            q_residual=residual_2d,
            gemma_norm=True,
        )

        out = out.view(ori_shape)
        if residual is not None:
            return out, res_out.view(ori_shape)
        return out

    def _forward_fused_fp8(self, x, residual=None):
        from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm_group_quant
        from aiter.utility.dtypes import fp8

        transpose_scale = envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE
        group_size = 128
        M = x.shape[0]
        N = x.shape[1]
        num_groups = N // group_size

        out_fp8 = torch.empty((M, N), dtype=fp8, device=x.device)
        if transpose_scale:
            # column-major: allocate (num_groups, M) then view as (M, num_groups)
            out_scale = torch.empty(
                (num_groups, M), dtype=torch.float32, device=x.device
            ).view(M, num_groups)
        else:
            # row-major: allocate (M, num_groups) directly
            out_scale = torch.empty(
                (M, num_groups), dtype=torch.float32, device=x.device
            )
        out_bf16 = (
            torch.empty((M, N), dtype=x.dtype, device=x.device)
            if self.write_bf16
            else None
        )
        res_out = torch.empty_like(x) if residual is not None else None

        fused_qk_rmsnorm_group_quant(
            out_fp8,
            out_scale,
            x,
            self.weight,
            self.variance_epsilon,
            q_out_unquantized=out_bf16,
            q_res_out=res_out,
            q_residual=residual,
            group_size=group_size,
            transpose_scale=transpose_scale,
            gemma_norm=True,
        )
        if residual is not None:
            return out_fp8, out_scale, out_bf16, res_out
        return out_fp8, out_scale, out_bf16

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.use_fused_quant:
            return self._forward_fused_fp8(x, residual)
        return self.forward_cuda(x, residual)


def fused_allreduce_gemma_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: GemmaRMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MiniMax-M3 helper for delayed TP all-reduce followed by Gemma RMSNorm."""
    if get_tensor_model_parallel_world_size() > 1:
        return tensor_model_parallel_fused_allreduce_rmsnorm(
            hidden_states.contiguous(),
            residual,
            norm.weight,
            norm.variance_epsilon,
            gemma_norm=True,
        )
    return norm(hidden_states, residual)


def fused_allreduce_gemma_rms_norm_quant(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: GemmaRMSNorm,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MiniMax-M3 helper for AR + Gemma RMSNorm + per-token FP8 quant."""
    if get_tensor_model_parallel_world_size() > 1:
        out_fp8, residual_out, scale_out = (
            tensor_model_parallel_fused_allreduce_rmsnorm_quant(
                hidden_states.contiguous(),
                residual,
                norm.weight,
                norm.variance_epsilon,
                quant_type="per_token",
                gemma_norm=True,
            )
        )
        return out_fp8, scale_out, residual_out

    from aiter import get_hip_quant
    from aiter.utility.dtypes import fp8

    normed, residual_out = norm(hidden_states, residual)
    out_fp8, scale_out = get_hip_quant(QuantType.per_Token)(
        normed,
        quant_dtype=fp8,
    )
    return out_fp8, scale_out, residual_out


# ---------------------------------------------------------------------------
# Fused Q/K RMSNorm Triton kernel
# ---------------------------------------------------------------------------
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _fused_qk_norm_single_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    eps,
    num_tokens,
    head_dim,
    q_in_stride0,
    k_in_stride0,
    q_out_stride0,
    k_out_stride0,
    num_q_heads,
    num_k_heads,
    ADD_UNIT_OFFSET: tl.constexpr,
    Q_HAS_WEIGHT: tl.constexpr,
    RBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    """Fused Q/K RMSNorm in a single kernel launch (out-of-place)."""
    num_q_rows = num_tokens * num_q_heads
    total_rows = num_tokens * (num_q_heads + num_k_heads)

    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < total_rows
    cols = tl.arange(0, RBLOCK)[None, :]
    col_mask = cols < head_dim

    is_q = xindex < num_q_rows
    row_in_section = tl.where(is_q, xindex, xindex - num_q_rows)
    cur_num_heads = tl.where(is_q, num_q_heads, num_k_heads)

    tokens = row_in_section // cur_num_heads
    heads = row_in_section % cur_num_heads

    in_stride = tl.where(is_q, q_in_stride0, k_in_stride0)
    in_bases = tokens * in_stride + heads * head_dim

    # Output: contiguous, stride(1) = head_dim
    out_stride0 = tl.where(is_q, q_out_stride0, k_out_stride0)
    out_bases = tokens * out_stride0 + heads * head_dim

    mask = xmask & col_mask

    # Weight: load both (or use ones for Q when Q_HAS_WEIGHT=False), select via is_q.
    # Q_HAS_WEIGHT=False is for callers whose Q-side norm has implicit identity
    # weight (e.g. V4's per-head Q normalization, equivalent to the prior
    # `_rmsnorm_nw` helper) — saves a load + register row.
    if Q_HAS_WEIGHT:
        qw = tl.load(
            q_weight_ptr + cols, mask=col_mask, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        if ADD_UNIT_OFFSET:
            qw = qw + 1.0
    else:
        qw = tl.full((RBLOCK,), 1.0, tl.float32)
    kw = tl.load(
        k_weight_ptr + cols, mask=col_mask, other=0.0, eviction_policy="evict_last"
    ).to(tl.float32)
    if ADD_UNIT_OFFSET:
        kw = kw + 1.0
    w = tl.where(is_q, qw, kw)

    # Use runtime branching for pointer selection (avoids tl.where on pointers)
    # Since all threads in a program have the same is_q value (XBLOCK rows are
    # consecutive and Q/K boundary is far apart), this branch is uniform.
    # For the rare program straddling Q/K boundary, both branches execute.
    x = tl.load(
        q_ptr + in_bases + cols,
        mask=mask & is_q,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    x = x + tl.load(
        k_ptr + in_bases + cols,
        mask=mask & ~is_q,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    var = tl.sum(x * x, 1)[:, None]
    rstd = tl.rsqrt(var / head_dim + eps)

    out = (x * rstd * w).to(q_out_ptr.dtype.element_ty)
    tl.store(
        q_out_ptr + out_bases + cols,
        out,
        mask=mask & is_q,
        eviction_policy="evict_first",
    )
    tl.store(
        k_out_ptr + out_bases + cols,
        out,
        mask=mask & ~is_q,
        eviction_policy="evict_first",
    )


def fused_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: Optional[torch.Tensor],
    k_weight: torch.Tensor,
    eps: float,
    add_unit_offset: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused Q/K RMSNorm in a single Triton kernel launch.

    Args:
        q: [num_tokens, num_heads, head_dim]
        k: [num_tokens, num_kv_heads, head_dim]
        q_weight: [head_dim] norm weight, or None for an implicit ones weight
                  (skips the Q-side weight load — for callers whose Q norm
                  is the identity, e.g. V4's per-head Q normalization).
        k_weight: [head_dim] norm weight (always required)
        eps: epsilon for numerical stability
        add_unit_offset: True for GemmaRMSNorm (w+1), False for standard
    """
    head_dim = k_weight.shape[0]
    if q_weight is not None:
        assert (
            q_weight.shape[0] == head_dim
        ), f"q_weight head_dim {q_weight.shape[0]} != k_weight {head_dim}"
    num_tokens = q.shape[0]
    num_q_heads = q.shape[1]
    num_k_heads = k.shape[1]
    total_rows = num_tokens * (num_q_heads + num_k_heads)
    RBLOCK = triton.next_power_of_2(head_dim)

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    # Adaptive XBLOCK based on batch size.
    # Small batch: XBLOCK=1 minimizes register pressure per program.
    # Large batch: XBLOCK=2 amortizes overhead, but XBLOCK>2 hurts due to
    # cross-token stride jumps in non-contiguous split views.
    # num_warps=1 is universally optimal for head_dim=256 workloads on MI355X.
    XBLOCK = 2 if total_rows > 8192 else 1
    NUM_WARPS = 1
    # When q_weight is None pass k_weight as a placeholder pointer (the
    # kernel won't load from it — Q_HAS_WEIGHT=False gates the load).
    q_weight_arg = q_weight if q_weight is not None else k_weight
    _fused_qk_norm_single_kernel[((total_rows + XBLOCK - 1) // XBLOCK,)](
        q,
        k,
        q_out,
        k_out,
        q_weight_arg,
        k_weight,
        eps,
        num_tokens,
        head_dim,
        q.stride(0),
        k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        num_q_heads,
        num_k_heads,
        ADD_UNIT_OFFSET=add_unit_offset,
        Q_HAS_WEIGHT=q_weight is not None,
        RBLOCK=RBLOCK,
        XBLOCK=XBLOCK,
        num_warps=NUM_WARPS,
    )
    return q_out, k_out


class DualRMSNorm:
    """Fused Q/K RMSNorm — single Triton kernel launch.

    Not an nn.Module. References existing q_norm/k_norm for weights.

    Q-side weightless mode: when `q_norm.weight is None`, the Q-side norm
    is treated as the identity (implicit ones weight). The kernel skips
    the q_weight load entirely (Q_HAS_WEIGHT=False). Use this when the
    checkpoint does not ship a Q weight (e.g. V4's per-head Q normalization,
    equivalent to the prior `_rmsnorm_nw` helper).
    """

    def __init__(
        self,
        q_norm: nn.Module,
        k_norm: nn.Module,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        prefix: str,
    ) -> None:
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # add_unit_offset only applies when q has a real weight; the kernel's
        # weightless path (q_norm.weight is None) emits qw=1.0 directly with
        # no offset, so the GemmaRMSNorm test is gated on weight existence.
        self.add_unit_offset = getattr(
            q_norm, "weight", None
        ) is not None and isinstance(q_norm, GemmaRMSNorm)
        # Resolve eps once. Different RMSNorm implementations name the
        # attribute differently — `variance_epsilon` (HF/Gemma style) or
        # `eps` (ATOM RMSNorm). Cache to avoid the lookup per forward call.
        self._eps = getattr(q_norm, "variance_epsilon", None) or getattr(
            q_norm, "eps", None
        )
        assert self._eps is not None, (
            f"q_norm {type(q_norm).__name__} must expose `eps` or "
            f"`variance_epsilon`"
        )
        self.prefix = prefix

    @mark_trace
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: [num_tokens, num_q_heads * head_dim]
            k: [num_tokens, num_kv_heads * head_dim]
        Returns:
            (q_normed, k_normed) same shapes as input
        """
        # `self.q_norm.weight` is None for weightless q_norm (e.g. V4
        # `q_norm2`); fused_qk_norm forwards that None to the kernel which
        # takes the Q_HAS_WEIGHT=False fast path.
        q, k = fused_qk_norm(
            q.view(-1, self.num_q_heads, self.head_dim),
            k.view(-1, self.num_kv_heads, self.head_dim),
            self.q_norm.weight,
            self.k_norm.weight,
            self._eps,
            add_unit_offset=self.add_unit_offset,
        )
        return (
            q.view(-1, self.num_q_heads * self.head_dim),
            k.view(-1, self.num_kv_heads * self.head_dim),
        )


@torch_compile_guard()
def layernorm2d_fwd_(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float, dim: int
) -> torch.Tensor:
    ori_shape = x.shape
    x = x.reshape(-1, dim)
    return layernorm2d_fwd(x, weight, bias, eps).view(ori_shape)


@torch_compile_guard()
def layernorm2d_fwd_with_add_(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ori_shape = x.shape
    x = x.reshape(-1, dim)
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    layernorm2d_fwd_with_add(out, x, residual, residual_out, weight, bias, eps)
    return out.view(ori_shape), residual_out.view(ori_shape)


class LayerNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = atom_parameter(torch.ones(dim))
        self.bias = atom_parameter(torch.zeros(dim))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return layernorm2d_fwd_(x, self.weight, self.bias, self.eps, self.dim)
        else:
            return layernorm2d_fwd_with_add_(
                x, self.weight, residual, self.bias, self.eps, self.dim
            )
