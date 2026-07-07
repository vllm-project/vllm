# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Tuple

import aiter
import torch
from aiter import (
    QuantType,
    rmsnorm2d_fwd,
    rmsnorm2d_fwd_with_add,
)
from vllm.models.deepseek_v4.amd.atom.distributed.parallel import (
    tensor_model_parallel_fused_allreduce_rmsnorm,
)
from vllm.models.deepseek_v4.amd.atom.distributed.parallel import get_tensor_model_parallel_world_size
from aiter.jit.utils.torch_guard import torch_compile_guard
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

