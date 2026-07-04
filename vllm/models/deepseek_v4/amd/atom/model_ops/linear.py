# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from functools import partial as functools_partial
from typing import Callable, Optional

import torch
from aiter import (
    QuantType,
    dtypes,
    gemm_a4w4,
    gemm_a8w8,
    gemm_a8w8_blockscale_bpreshuffle,
    gemm_a8w8_bpreshuffle,
    gemm_a8w8_blockscale,
    get_hip_quant,
)

# import torch.distributed as dist
from vllm.models.deepseek_v4.amd.atom.distributed.parallel import get_tp_group
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.tuned_gemm import tgemm
from aiter.utility import fp4_utils
from vllm.models.deepseek_v4.amd.atom.config import QuantizationConfig, get_current_atom_config
from vllm.models.deepseek_v4.amd.atom.quant_spec import LayerQuantConfig, should_skip_online_quant
from vllm.models.deepseek_v4.amd.atom.model_ops.utils import (
    atom_parameter,
    normalize_e4m3fn_to_e4m3fnuz,
    requantize_with_max_scale,
    shuffle_weights,
)
from vllm.models.deepseek_v4.amd.atom.utils import envs
from vllm.models.deepseek_v4.amd.atom.utils.decorators import mark_trace
from vllm.models.deepseek_v4.amd.atom.quantization.quark.utils import (
    quant_weight_online,
    weight_dequant_fp8,
    weight_dequant_mxfp8,
)
from torch import nn

logger = logging.getLogger("atom")


def use_triton_gemm() -> bool:
    return envs.ATOM_USE_TRITON_GEMM


def use_fp4_non_shuffle_triton_gemm() -> bool:
    return envs.ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM


if use_fp4_non_shuffle_triton_gemm():
    try:
        from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4  # noqa: E402
    except ImportError as e:
        logger.warning(f"Triton FP4 GEMM not available: {e}")
        gemm_afp4wfp4 = None
else:
    gemm_afp4wfp4 = None


if use_triton_gemm():
    try:
        # from aiter.ops.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale_preshuffle as gemm_a8w8_blockscale_bpreshuffle_triton
        from aiter.ops.triton.gemm_afp4wfp4 import (
            gemm_afp4wfp4_preshuffle,
        )  # noqa: E402
    except ImportError as e:
        logger.warning(f"Triton FP4 GEMM not available: {e}")
        gemm_afp4wfp4_preshuffle = None

    # For Triton FP8 Blockscale GEMM is mostly slower then AITER GEMM, we turn off Triton FP8 GEMM
    try:
        from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
            gemm_a8w8_blockscale_preshuffle as gemm_a8w8_blockscale_bpreshuffle_triton,
        )  # noqa: E402
    except ImportError as e:
        logger.warning(f"Triton w8a8 GEMM not available: {e}")
        gemm_a8w8_blockscale_bpreshuffle_triton = None
else:
    gemm_afp4wfp4_preshuffle = None
    gemm_a8w8_blockscale_bpreshuffle_triton = None

from vllm.models.deepseek_v4.amd.atom.model_ops.utils import MXFP4_QUANT_BLOCK_SIZE  # noqa


def divide(numerator, denominator):
    assert (
        numerator % denominator == 0
    ), f"numerator {numerator} denominator {denominator}"
    return numerator // denominator


def gemm_a4w4_quant_fake(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    otype: torch.dtype,
    weight_scale: torch.Tensor,
    params_dtype: torch.dtype,
    input_scale: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=otype, device=x.device)


# It's important to use mutates_args=[] to avoid functionized_v2 op generation
@torch_compile_guard(gen_fake=gemm_a4w4_quant_fake, mutates_args=[])
def gemm_a4w4_quant(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    otype: torch.dtype,
    weight_scale: torch.Tensor,
    params_dtype: torch.dtype,
    input_scale: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    # Non-shuffle FP4 Triton path: keep x/weight/scale in the original MXFP4
    # layout and call the non-preshuffled gemm_afp4wfp4 kernel.
    if params_dtype == dtypes.fp4x2 and use_fp4_non_shuffle_triton_gemm():
        if gemm_afp4wfp4 is None:
            raise RuntimeError(
                "ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM=1 requires aiter.ops.triton.gemm_afp4wfp4"
            )
        if x_scale is None:
            quant_func = get_hip_quant(QuantType.per_1x32)
            x, x_scale = quant_func(
                x,
                quant_dtype=params_dtype,
                scale=input_scale,
                shuffle=False,
            )
        else:
            x_scale = x_scale.view(torch.float8_e8m0fnu)
            x = x.view(torch.float4_e2m1fn_x2)

        m = x.view(-1, x.size(-1)).shape[0]
        y = torch.empty(
            (
                (m + MXFP4_QUANT_BLOCK_SIZE - 1)
                // MXFP4_QUANT_BLOCK_SIZE
                * MXFP4_QUANT_BLOCK_SIZE,
                output_size,
            ),
            dtype=otype,
            device=x.device,
        )
        y = gemm_afp4wfp4(
            x.view(torch.uint8),
            weight.view(torch.uint8),
            x_scale.view(torch.uint8),
            weight_scale.view(torch.uint8),
            otype,
            y,
        )
    # Preshuffle FP4 Triton path: used when ATOM_USE_TRITON_GEMM is enabled
    # and the non-shuffle path is disabled. This expects preshuffled weights.
    elif (
        params_dtype == dtypes.fp4x2
        and use_triton_gemm()
        and not use_fp4_non_shuffle_triton_gemm()
        and gemm_afp4wfp4_preshuffle is not None
    ):
        m, k = x.view(-1, x.size(-1)).shape

        y = torch.empty(
            (
                (m + MXFP4_QUANT_BLOCK_SIZE - 1)
                // MXFP4_QUANT_BLOCK_SIZE
                * MXFP4_QUANT_BLOCK_SIZE,
                output_size,
            ),
            dtype=otype,
            device=x.device,
        )
        if x_scale is None:
            quant_func = get_hip_quant(QuantType.per_1x32)
            x, x_scale = quant_func(
                x,
                quant_dtype=params_dtype,
                shuffle=(m >= MXFP4_QUANT_BLOCK_SIZE),
            )
        else:
            x_scale = x_scale.view(torch.float8_e8m0fnu)
            x = x.view(torch.float4_e2m1fn_x2)

        if m >= MXFP4_QUANT_BLOCK_SIZE:
            x_scale = x_scale.view(torch.uint8).view(
                x_scale.shape[0] // MXFP4_QUANT_BLOCK_SIZE, -1
            )
        else:
            x_scale = x_scale[:m, ...].view(torch.uint8)

        y = gemm_afp4wfp4_preshuffle(
            x.view(torch.uint8),
            weight.view(torch.uint8).view(weight.shape[0] // 16, -1),
            x_scale,
            weight_scale.view(torch.uint8).view(
                weight_scale.shape[0] // MXFP4_QUANT_BLOCK_SIZE, -1
            ),
            y=y,
        )
    # Default AITER path: quantize/shuffle into the layout expected by gemm_a4w4
    # and use the backend ASM implementation.
    else:
        if x_scale is None:
            quant_func = get_hip_quant(QuantType.per_1x32)
            x, x_scale = quant_func(
                x,
                quant_dtype=params_dtype,
                scale=input_scale,
                shuffle=True,
            )
        else:
            x_scale = x_scale.view(torch.float8_e8m0fnu)
            x = x.view(torch.float4_e2m1fn_x2)

        m = x.view(-1, x.size(-1)).shape[0]
        y = torch.empty(
            (
                (m + MXFP4_QUANT_BLOCK_SIZE - 1)
                // MXFP4_QUANT_BLOCK_SIZE
                * MXFP4_QUANT_BLOCK_SIZE,
                output_size,
            ),
            dtype=otype,
            device=x.device,
        )
        y = gemm_a4w4(
            x,
            weight,
            x_scale,
            weight_scale,
            y,
        )

    return y[:m, ...]


def gemm_a8w8_blockscale_preshuffle_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    prefix: str = "",
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=dtype, device=x.device)


@mark_trace(torch_compile=False)
@torch_compile_guard(gen_fake=gemm_a8w8_blockscale_preshuffle_fake, mutates_args=[])
def gemm_a8w8_blockscale_preshuffle_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    prefix: str = "",
) -> torch.Tensor:
    if gemm_a8w8_blockscale_bpreshuffle_triton is not None:
        weight_shuffled = weight.reshape(weight.shape[0] // 16, weight.shape[1] * 16)
        y = gemm_a8w8_blockscale_bpreshuffle_triton(
            x, weight_shuffled, x_scale, w_scale, dtype
        )
    else:
        y = gemm_a8w8_blockscale_bpreshuffle(x, weight, x_scale, w_scale, dtype)
    return y


class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int | list[int],
        tp_dim: int | None = None,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = False,
        source_quant_dtype: torch.dtype | None = None,
        prefix: str = "",
    ):
        self.prefix = prefix
        layer_quant_config = (
            quant_config.get_layer_quant_config(prefix)
            if quant_config is not None
            else LayerQuantConfig()
        )
        quant_type = layer_quant_config.quant_type
        params_dtype = layer_quant_config.quant_dtype
        self.source_quant_dtype = source_quant_dtype
        self.layer_quant_config = layer_quant_config
        self.quant_config = quant_config
        super().__init__()
        self.reduce_results = reduce_results
        self.input_size = input_size
        self.output_size = (
            output_size if isinstance(output_size, int) else sum(output_size)
        )
        self.tp_dim = tp_dim
        self.tp_rank = get_tp_group().rank_in_group
        self.tp_size = get_tp_group().world_size
        self.output_partition_sizes = (
            output_size if isinstance(output_size, list) else [output_size]
        )
        if tp_dim == 1:
            self.input_size = divide(input_size, self.tp_size)
        elif tp_dim == 0:
            self.output_size = divide(self.output_size, self.tp_size)
            self.output_partition_sizes = [
                divide(s, self.tp_size) for s in self.output_partition_sizes
            ]

        if self.source_quant_dtype is not None:
            weight_size = (self.output_size, self.input_size)
            self.weight = atom_parameter(
                torch.empty(weight_size, dtype=self.source_quant_dtype)
            )
        else:
            weight_size = (
                (self.output_size, self.input_size)
                if params_dtype not in [dtypes.fp4x2, dtypes.i4x2]
                else (self.output_size, self.input_size // 2)
            )
            self.weight = atom_parameter(torch.empty(weight_size, dtype=params_dtype))
        if bias:
            output_type = get_current_atom_config().torch_dtype
            self.bias = atom_parameter(torch.empty(self.output_size, dtype=output_type))
            self.bias.weight_loader_process = self.weight_loader_process
        else:
            self.register_parameter("bias", None)
        self.quant_type = quant_type
        self.params_dtype = params_dtype

        if quant_type != QuantType.No and self.source_quant_dtype is None:
            if quant_type == QuantType.per_Tensor:
                self.weight_scale = atom_parameter(
                    torch.empty(len(self.output_partition_sizes), 1, dtype=dtypes.fp32)
                )
                if not layer_quant_config.is_dynamic:
                    self.input_scale = atom_parameter(
                        torch.empty(
                            len(self.output_partition_sizes), 1, dtype=dtypes.fp32
                        )
                    )
                    self.input_scale.weight_loader_process = self.weight_loader_process
                    self.input_scale.weight_loader = self.weight_loader
            elif quant_type == QuantType.per_Token:
                self.weight_scale = atom_parameter(
                    torch.empty(self.output_size, 1, dtype=dtypes.fp32)
                )
            elif quant_type == QuantType.per_1x128:
                self.weight_scale = atom_parameter(
                    torch.empty(
                        (self.output_size + 127) // 128,
                        (self.input_size + 127) // 128,
                        dtype=dtypes.fp32,
                    )
                )
            elif quant_type == QuantType.per_1x32:
                self.weight_scale = atom_parameter(
                    torch.empty(
                        self.output_size,
                        (self.input_size + 31) // 32,
                        dtype=dtypes.fp8_e8m0,
                    )
                )
            self.weight.weight_loader_process = self.weight_loader_process
            self.weight_scale.weight_loader_process = self.weight_loader_process
        else:
            self.weight.weight_loader_process = self.weight_loader_process
            self.register_parameter("weight_scale", None)
        self.weight.weight_loader = self.weight_loader
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader
        if self.weight_scale is not None:
            self.weight_scale.weight_loader = self.weight_loader
        self.need_normalize_e4m3fn_to_e4m3fnuz = params_dtype == torch.float8_e4m3fnuz
        self.quant_func = get_hip_quant(self.quant_type)

    @staticmethod
    def weight_loader_process(
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        post_process_func: Callable = lambda a: a,
    ):
        if param.data.dtype != loaded_weight.dtype:
            if param.data.element_size() == loaded_weight.element_size():
                # Same byte-width: use view for raw-bit-compatible pairs
                # (e.g. fp8 variants) but convert for semantically different
                # formats (float16 ↔ bfloat16) where bit reinterpretation
                # would corrupt values.
                incompatible = {torch.float16, torch.bfloat16}
                if {param.data.dtype, loaded_weight.dtype} == incompatible:
                    loaded_weight = loaded_weight.to(param.data.dtype)
                else:
                    param.data = param.data.view(loaded_weight.dtype)
            else:
                loaded_weight = loaded_weight.to(param.data.dtype)
        loaded_weight = post_process_func(loaded_weight)
        if (
            loaded_weight.shape != param.data.shape
            and loaded_weight.numel() == param.data.numel()
        ):
            loaded_weight = loaded_weight.reshape(param.data.shape)
        if param.data.dtype != dtypes.fp4x2:
            param.data.copy_(loaded_weight)
        else:
            param.data.view(torch.uint8).copy_(loaded_weight.view(torch.uint8))

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        param.weight_loader_process(param_data, loaded_weight)

    def _gather_full_weight(self, weight):
        """Gather sharded weight from all TP ranks to reconstruct the full unpartitioned weight."""
        if self.tp_size <= 1 or self.tp_dim is None:
            return weight
        # NCCL cannot all_gather E8M0 scales (MXFP8 source); gather the raw
        # bytes as uint8 and reinterpret afterwards. The gather only moves
        # bytes, so this is bit-exact.
        if weight.dtype == dtypes.fp8_e8m0:
            gathered = get_tp_group().all_gather(
                weight.view(torch.uint8), dim=self.tp_dim
            )
            return gathered.view(dtypes.fp8_e8m0)
        return get_tp_group().all_gather(weight, dim=self.tp_dim)

    def _shard_quantized_weight(self, q_weight, weight_scale):
        """Split the quantized full weight and scale back to this rank's shard."""
        if self.tp_size <= 1 or self.tp_dim is None:
            return q_weight, weight_scale
        # fp4x2 packs 2 values per byte; view as uint8 so narrow() works correctly
        is_mxfp4 = q_weight.dtype == torch.float4_e2m1fn_x2
        if is_mxfp4:
            q_weight = q_weight.view(torch.uint8)
        w_shard = q_weight.shape[self.tp_dim] // self.tp_size
        w_start = self.tp_rank * w_shard
        q_weight_local = q_weight.narrow(self.tp_dim, w_start, w_shard).contiguous()
        if is_mxfp4:
            q_weight_local = q_weight_local.view(torch.float4_e2m1fn_x2)

        # for col linear qkv, w13, [m, n] -> [m // tp, n] -> [m // tp, 1], TP and non-TP are equivalent
        if weight_scale.dim() > self.tp_dim and weight_scale.shape[self.tp_dim] > 1:
            s_shard = weight_scale.shape[self.tp_dim] // self.tp_size
            s_start = self.tp_rank * s_shard
            weight_scale_local = weight_scale.narrow(
                self.tp_dim, s_start, s_shard
            ).contiguous()
        # for row linear o, w2, [m, n] -> [m, n // tp] -> [m, 1], TP and non-TP are not equivalent
        else:
            weight_scale_local = weight_scale

        return q_weight_local, weight_scale_local

    def online_quantize_weight(self):
        """Re-quantize this layer's weight at load time using the online quant config.

        Handles TP gather/shard when local quantization would produce
        different results than quantizing the full unpartitioned weight.
        """
        online_layer_quant_config = self.quant_config.get_layer_quant_config(
            self.prefix, use_online_quant=True
        )
        online_quant_type = online_layer_quant_config.quant_type
        online_quant_dtype = online_layer_quant_config.quant_dtype
        if should_skip_online_quant(
            self.quant_type, self.params_dtype, online_layer_quant_config
        ):
            return

        assert online_quant_dtype in [
            torch.float8_e4m3fn,
            torch.float4_e2m1fn_x2,
        ], (
            f"Unsupported online quant: "
            f"dtype={online_quant_dtype}, type={online_quant_type}"
        )
        assert self.quant_type in [
            QuantType.No,
            QuantType.per_1x128,
            QuantType.per_1x32,
        ], (
            f"Unsupported source quant_type for online quantization: "
            f"{self.quant_type} (layer={self.prefix})"
        )
        weight = self.weight.data
        weight_scale = getattr(self, "weight_scale", None)
        # Gather is required whenever local quantization would differ from
        # quantizing the full unpartitioned weight (bit-exact with offline).
        need_gather = False
        if self.tp_size > 1 and self.tp_dim is not None:
            if isinstance(self, ReplicatedLinear):
                # W and S of kv_a_proj_with_mqa don't match,
                # but it doesn't need to be split.
                return
            # col qkv w13, tp_dim=0, [m, n] -> [m // tp, n] -> [m // tp, 1], don't need gather
            # row o, w2, tp_dim=1, [m, n] -> [m, n //tp] -> [m, 1], need gather
            if online_quant_type == QuantType.per_Token:
                # per_Token: scale = max(|full_row|), tp_dim=1 has partial rows
                need_gather = self.tp_dim == 1
            elif online_quant_type == QuantType.per_1x128:
                # 128×128 blocks: misaligned if partition size % 128 != 0
                if self.tp_dim == 0:
                    need_gather = self.output_size % 128 != 0
                else:
                    need_gather = self.input_size % 128 != 0
            elif online_quant_type == QuantType.per_1x32:
                # 1×32 blocks: row dim is 1 (always aligned), only column matters
                # col qkv w13, tp_dim=0, [m, n] -> [m // tp, n] -> [m // tp, n // 32], don't need gather
                # row o, w2, tp_dim=1, [m, n] -> [m, n //tp] -> [m, n // tp // 32], need gather
                need_gather = self.tp_dim == 1 and self.input_size % 32 != 0
        if need_gather:
            weight = self._gather_full_weight(weight)
            if weight_scale is not None:
                weight_scale = self._gather_full_weight(weight_scale)

        if self.quant_type == QuantType.per_1x128:
            # dequant per block fp8
            weight = weight_dequant_fp8(weight, weight_scale)
        elif self.quant_type == QuantType.per_1x32:
            # dequant MXFP8 (FP8 elements + 1x32 E8M0 shared scale)
            weight = weight_dequant_mxfp8(weight, weight_scale)

        q_weight, weight_scale = quant_weight_online(
            weight, online_quant_type, online_quant_dtype
        )
        if need_gather:
            q_weight, weight_scale = self._shard_quantized_weight(
                q_weight, weight_scale
            )
        self.weight = nn.Parameter(q_weight, requires_grad=False)
        self.weight_scale = nn.Parameter(weight_scale, requires_grad=False)

        # Update quant state
        self.quant_type = online_quant_type
        self.params_dtype = online_quant_dtype
        self.quant_func = get_hip_quant(online_quant_type)
        self.need_normalize_e4m3fn_to_e4m3fnuz = (
            online_quant_dtype == torch.float8_e4m3fnuz
        )
        self._online_quant_info = {
            "layer": self.prefix,
            "quant_type": online_quant_type.name,
            "quant_dtype": str(online_quant_dtype),
        }

    def process_weights_after_loading(self):
        # Re-quantize before process_weights if online quantization is enabled
        if self.quant_config is not None and self.quant_config.online_quant:
            self.online_quantize_weight()
        if (
            self.quant_type == QuantType.per_Tensor
            and len(self.output_partition_sizes) > 1
        ):
            weight_scale, weight = requantize_with_max_scale(
                weight=self.weight.data,
                weight_scale=self.weight_scale.data,
                logical_widths=self.output_partition_sizes,
                normalize_e4m3fn_to_e4m3fnuz=self.need_normalize_e4m3fn_to_e4m3fnuz,
            )
            self.weight.data = weight
            self.weight_scale.data = weight_scale.view(-1)
            if hasattr(self, "input_scale"):
                self.input_scale.data = (
                    self.input_scale.data.max() * 2.0
                    if self.need_normalize_e4m3fn_to_e4m3fnuz
                    else self.input_scale.data.max()
                )
        elif self.need_normalize_e4m3fn_to_e4m3fnuz:
            self.weight.data, self.weight_scale.data, _ = normalize_e4m3fn_to_e4m3fnuz(
                self.weight.data, self.weight_scale.data
            )
        if (
            self.source_quant_dtype == torch.bfloat16
            and self.quant_type == QuantType.per_1x32
            and self.params_dtype == torch.float4_e2m1fn_x2
        ):
            w_q, w_s = self.quant_func(
                self.weight.data,
                quant_dtype=self.params_dtype,
                shuffle=False,
            )
            self.weight.data = w_q
            self.weight_scale = atom_parameter(w_s)
            # Only quantized 2D GEMM weights use aiter's preshuffle layout.
            # Qwen3-Next/Qwen3.5 GDN conv1d expands its weight to 3D, so FP8/blocked
            # quantized models must keep that tensor unshuffled here.
            if self.weight.dim() == 2 and not use_fp4_non_shuffle_triton_gemm():
                shuffle_weights(self.weight)
            # self.weight_scale.data = fp4_utils.e8m0_shuffle(self.weight_scale.data)
        else:
            is_fp4_blockscale = (
                self.quant_type == QuantType.per_1x32
                and self.params_dtype == dtypes.fp4x2
            )
            need_shuffle = (
                self.quant_type == QuantType.per_Token
                and self.params_dtype == dtypes.fp8
            ) or (
                self.quant_type == QuantType.per_1x32
                and (not is_fp4_blockscale or not use_fp4_non_shuffle_triton_gemm())
            )
            # per_1x128 only needs shuffle when using the preshuffle GEMM path
            if not need_shuffle and self.quant_type == QuantType.per_1x128:
                need_shuffle = envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE
            if need_shuffle:
                if self.weight.dim() == 2:
                    shuffle_weights(self.weight)
                # self.weight_scale.data = fp4_utils.e8m0_shuffle(self.weight_scale.data)
        # shuffle weight scale once so no reshuffling for every gemm
        if self.quant_type == QuantType.per_1x32 and (
            self.params_dtype != dtypes.fp4x2 or not use_fp4_non_shuffle_triton_gemm()
        ):
            self.weight_scale.data = fp4_utils.e8m0_shuffle(self.weight_scale.data)

    @mark_trace
    def forward(
        self, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None, otype=dtypes.bf16
    ) -> torch.Tensor:
        if self.quant_type.value == QuantType.No.value:
            y = tgemm.mm(
                x,
                self.weight,
                self.bias,
                otype=otype,
            )
        else:
            if x_scale is None:
                quant_func = self.quant_func
                if self.quant_type.value == QuantType.per_1x128.value:
                    # preshuffle GEMM expects column-major x_scale;
                    # non-preshuffle GEMM expects row-major x_scale
                    quant_func = functools_partial(
                        self.quant_func,
                        transpose_scale=envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE,
                    )
                if self.quant_type.value != QuantType.per_1x32.value:
                    x, x_scale = quant_func(
                        x,
                        quant_dtype=self.params_dtype,
                        scale=getattr(self, "input_scale", None),
                    )
            if self.quant_type.value == QuantType.per_Tensor.value:
                y = tgemm.mm(
                    x,
                    self.weight,
                    self.bias,
                    otype=otype,
                    scale_a=x_scale,
                    scale_b=self.weight_scale,
                )
            elif self.quant_type.value == QuantType.per_Token.value:
                if self.params_dtype == dtypes.i8:
                    y = gemm_a8w8(
                        x,
                        self.weight,
                        x_scale,
                        self.weight_scale,
                        self.bias,
                        dtype=otype,
                    )
                else:
                    y = gemm_a8w8_bpreshuffle(
                        x,
                        self.weight,
                        x_scale,
                        self.weight_scale,
                        dtype=otype,
                    )
                    if self.bias is not None:
                        y += self.bias
            elif self.quant_type.value == QuantType.per_1x128.value:
                if envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE:
                    y = gemm_a8w8_blockscale_preshuffle_impl(
                        x,
                        self.weight,
                        x_scale,
                        self.weight_scale,
                        dtype=otype,
                        prefix=self.prefix,
                    )
                else:
                    y = gemm_a8w8_blockscale(
                        x,
                        self.weight,
                        x_scale,
                        self.weight_scale,
                        dtype=otype,
                    )
                if self.bias is not None:
                    y += self.bias
            elif self.quant_type.value == QuantType.per_1x32.value:
                y = gemm_a4w4_quant(
                    x,
                    x_scale,
                    self.weight,
                    otype,
                    self.weight_scale.data,
                    self.params_dtype,
                    getattr(self, "input_scale", None),
                    self.output_size,
                )
                if self.bias is not None:
                    y += self.bias
        if self.tp_dim == 1 and self.tp_size > 1 and self.reduce_results:
            y = get_tp_group().all_reduce(y, ca_fp8_quant=False)
        return y


class ReplicatedLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__(
            input_size,
            output_size,
            tp_dim=None,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        param.weight_loader_process(param_data, loaded_weight)


class ColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        self.tp_dim = 0
        super().__init__(
            input_size,
            output_size,
            self.tp_dim,
            bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.weight_loader_process(param_data, loaded_weight)


class MergedColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        self.output_sizes = output_sizes
        super().__init__(
            input_size,
            output_sizes,
            tp_dim=0,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | tuple[int, ...] | None = None,
    ):
        # Support loading multiple consecutive shards in a single tensor.
        # This mirrors vLLM's behavior for packed modules like QKV.
        if isinstance(loaded_shard_id, tuple):
            if len(loaded_shard_id) == 0:
                raise ValueError("loaded_shard_id tuple cannot be empty")
            if any(idx < 0 or idx >= len(self.output_sizes) for idx in loaded_shard_id):
                raise ValueError(
                    f"Invalid shard id in {loaded_shard_id}; "
                    f"valid range is [0, {len(self.output_sizes) - 1}]"
                )
            if len(loaded_shard_id) > 1 and any(
                b - a != 1 for a, b in zip(loaded_shard_id[:-1], loaded_shard_id[1:])
            ):
                raise ValueError(
                    "Shard id with multiple indices should be consecutive. "
                    f"Got shard id {loaded_shard_id}."
                )

            # Split loaded_weight by the requested shard sizes (pre-TP),
            # then load each shard individually.
            shard_sizes = [self.output_sizes[i] for i in loaded_shard_id]
            current_offset = 0
            for shard_id, shard_size in zip(loaded_shard_id, shard_sizes):
                if param is getattr(self, "weight_scale", None) or param is getattr(
                    self, "input_scale", None
                ):
                    if self.quant_type != QuantType.per_1x32:
                        shard_size //= 128
                shard = loaded_weight.narrow(self.tp_dim, current_offset, shard_size)
                self.weight_loader(param, shard, shard_id)
                current_offset += shard_size
            return

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk
            # Split it and load each shard individually.
            param_data = param.data
            # Check if this is weight or weight_scale
            is_scale_param = param is getattr(
                self, "weight_scale", None
            ) or param is getattr(self, "input_scale", None)

            # For fused weight, need to match param shape
            if param_data.shape == loaded_weight.shape:
                # Shapes match - direct copy
                param.weight_loader_process(param_data, loaded_weight)
                return

            # Otherwise, split the fused weight and load each output shard
            current_offset = 0
            for shard_id, output_size in enumerate(self.output_sizes):
                shard_size = output_size
                if is_scale_param and self.quant_type == QuantType.per_1x128:
                    shard_size //= 128

                shard = loaded_weight.narrow(self.tp_dim, current_offset, shard_size)
                self.weight_loader(param, shard, shard_id)
                current_offset += shard_size
            return

        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        if param is getattr(self, "weight_scale", None) or param is getattr(
            self, "input_scale", None
        ):
            if self.quant_type == QuantType.per_1x128:
                shard_offset = (shard_offset + 127) // 128
                shard_size = (shard_size + 127) // 128
            elif self.quant_type == QuantType.per_Tensor:
                loaded_weight = loaded_weight.view(1, 1).repeat(self.tp_size, 1)
                shard_offset = loaded_shard_id
                shard_size = 1

        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param.weight_loader_process(param_data, loaded_weight)


class QKVZBAParallelLinear(ColumnParallelLinear):
    """Fused QKVZBA linear with deinterleaved output layout.

    Output layout is always ``[q | k | v | z | b | a]`` contiguous on dim-0,
    so the caller can use ``torch.split`` for zero-copy views.

    The weight_loader deinterleaves the Qwen3-Next interleaved per-k-head-group
    checkpoint layout during loading.  Qwen3.5 checkpoints (which have separate
    per-component weights) are loaded via individual shard_ids and work without
    deinterleaving.
    """

    def __init__(
        self,
        input_size: int,
        head_k_dim: int,
        head_v_dim: int,
        num_k_heads: int,
        num_v_heads: int,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        tp_size = get_tp_group().world_size
        self.num_k_heads = divide(self.num_k_heads, tp_size)
        self.num_v_heads = divide(self.num_v_heads, tp_size)
        # Output layout: [q_all | k_all | v_all | z_all | b_all | a_all]
        q_size = self.num_k_heads * head_k_dim
        k_size = self.num_k_heads * head_k_dim
        v_size = self.num_v_heads * head_v_dim
        z_size = self.num_v_heads * head_v_dim
        b_size = self.num_v_heads
        a_size = self.num_v_heads
        self._section_sizes = [q_size, k_size, v_size, z_size, b_size, a_size]
        output_sizes = [s * tp_size for s in self._section_sizes]
        super().__init__(
            input_size,
            output_sizes,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    # -- helpers for deinterleaving during weight loading --

    def _deinterleave_qkvz(self, param_data: torch.Tensor, loaded_weight: torch.Tensor):
        """Scatter interleaved qkvz checkpoint rows into [q|k|v|z] regions.

        Checkpoint layout per k-head group (``QKVZ_DIM_SIZE`` rows):
            [q(head_k_dim) | k(head_k_dim) | v0..vR(R*head_v_dim) | z0..zR(R*head_v_dim)]
        where R = num_v_heads / num_k_heads (KV_HEAD_RATIO).
        """
        nk = self.num_k_heads
        hk = self.head_k_dim
        hv = self.head_v_dim
        R = self.num_v_heads // nk  # KV_HEAD_RATIO
        group_size = 2 * hk + 2 * hv * R

        q_total = nk * hk
        k_total = nk * hk
        v_total = self.num_v_heads * hv

        # TP shard the source
        src = loaded_weight.narrow(
            self.tp_dim, self.tp_rank * nk * group_size, nk * group_size
        )

        for g in range(nk):
            base = g * group_size
            # q rows
            param_data[g * hk : (g + 1) * hk] = src[base : base + hk]
            # k rows
            param_data[q_total + g * hk : q_total + (g + 1) * hk] = src[
                base + hk : base + 2 * hk
            ]
            # v sub-heads
            for s in range(R):
                v_src_start = base + 2 * hk + s * hv
                v_dst_start = q_total + k_total + (g * R + s) * hv
                param_data[v_dst_start : v_dst_start + hv] = src[
                    v_src_start : v_src_start + hv
                ]
            # z sub-heads
            for s in range(R):
                z_src_start = base + 2 * hk + R * hv + s * hv
                z_dst_start = q_total + k_total + v_total + (g * R + s) * hv
                param_data[z_dst_start : z_dst_start + hv] = src[
                    z_src_start : z_src_start + hv
                ]

    def _deinterleave_ba(self, param_data: torch.Tensor, loaded_weight: torch.Tensor):
        """Scatter interleaved ba checkpoint rows into [b|a] regions.

        Checkpoint layout per k-head group (2*R elements):
            [b_sub0, b_sub1, ..., b_subR-1, a_sub0, a_sub1, ..., a_subR-1]
        where R = num_v_heads / num_k_heads.
        """
        nk = self.num_k_heads
        nv = self.num_v_heads
        R = nv // nk
        ba_total = 2 * nv

        # TP shard the source
        src = loaded_weight.narrow(self.tp_dim, self.tp_rank * ba_total, ba_total)

        qkvz_total = sum(self._section_sizes[:4])
        b_offset = qkvz_total
        a_offset = qkvz_total + nv

        for g in range(nk):
            group_base = g * 2 * R
            # b sub-heads
            for s in range(R):
                param_data[b_offset + g * R + s] = src[group_base + s]
            # a sub-heads
            for s in range(R):
                param_data[a_offset + g * R + s] = src[group_base + R + s]

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):
        param_data = param.data
        assert loaded_shard_id in ["qkvz", "ba", "qkv", "z", "b", "a"]

        is_scale = param is getattr(self, "weight_scale", None) or param is getattr(
            self, "input_scale", None
        )

        # For interleaved checkpoint shards ("qkvz", "ba"), deinterleave
        # weight rows so output is [q|k|v|z|b|a] contiguous.
        if loaded_shard_id == "qkvz" and not is_scale:
            self._deinterleave_qkvz(param_data, loaded_weight)
            return
        if loaded_shard_id == "ba" and not is_scale:
            self._deinterleave_ba(param_data, loaded_weight)
            return

        # For individual shard_ids or scale params, use offset-based loading.
        q_size, k_size, v_size, z_size, b_size, a_size = self._section_sizes
        if loaded_shard_id == "qkvz":
            shard_size = q_size + k_size + v_size + z_size
            shard_offset = 0
        elif loaded_shard_id == "qkv":
            shard_size = q_size + k_size + v_size
            shard_offset = 0
        elif loaded_shard_id == "z":
            shard_size = z_size
            shard_offset = q_size + k_size + v_size
        elif loaded_shard_id == "ba":
            shard_size = b_size + a_size
            shard_offset = q_size + k_size + v_size + z_size
        elif loaded_shard_id == "b":
            shard_size = b_size
            shard_offset = q_size + k_size + v_size + z_size
        elif loaded_shard_id == "a":
            shard_size = a_size
            shard_offset = q_size + k_size + v_size + z_size + b_size

        if is_scale:
            if self.quant_type == QuantType.per_1x128:
                shard_offset = (shard_offset + 127) // 128
                shard_size = (shard_size + 127) // 128
            elif self.quant_type == QuantType.per_Tensor:
                loaded_weight = loaded_weight.view(1, 1).repeat(self.tp_size, 1)
                shard_offset = ["qkvz", "ba", "qkv", "z", "b", "a"].index(
                    loaded_shard_id
                )
                shard_size = 1
        start_idx = self.tp_rank * shard_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.weight_loader_process(param_data, loaded_weight)


class QKVZParallelLinear(ColumnParallelLinear):
    """Deinterleaving linear for the separate ``in_proj_qkvz`` checkpoint.

    The Qwen3-Next checkpoint stores qkvz in per-k-head-group interleaved
    layout::

        [q(hk) | k(hk) | v0..vR(R*hv) | z0..zR(R*hv)]  per group

    This class deinterleaves both weight rows **and** per-1x128 block scales
    during loading so the output layout is ``[q_all | k_all | v_all | z_all]``
    contiguous, enabling zero-copy ``torch.split`` at runtime.
    """

    def __init__(
        self,
        input_size: int,
        head_k_dim: int,
        head_v_dim: int,
        num_k_heads: int,
        num_v_heads: int,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        tp_size = get_tp_group().world_size
        self.nk = divide(num_k_heads, tp_size)
        self.nv = divide(num_v_heads, tp_size)
        self.R = num_v_heads // num_k_heads  # v-heads per k-head group
        output_size = 2 * num_k_heads * head_k_dim + 2 * num_v_heads * head_v_dim
        super().__init__(
            input_size,
            output_size,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    @staticmethod
    def _deinterleave(param_data, src, nk, R, hk, hv):
        """Scatter interleaved rows into [q|k|v|z] regions."""
        group_size = 2 * hk + 2 * hv * R
        q_total = nk * hk
        k_total = nk * hk
        v_total = nk * R * hv

        for g in range(nk):
            base = g * group_size
            # q
            param_data[g * hk : (g + 1) * hk] = src[base : base + hk]
            # k
            param_data[q_total + g * hk : q_total + (g + 1) * hk] = src[
                base + hk : base + 2 * hk
            ]
            # v sub-heads
            for s in range(R):
                v_src = base + 2 * hk + s * hv
                v_dst = q_total + k_total + (g * R + s) * hv
                param_data[v_dst : v_dst + hv] = src[v_src : v_src + hv]
            # z sub-heads
            for s in range(R):
                z_src = base + 2 * hk + R * hv + s * hv
                z_dst = q_total + k_total + v_total + (g * R + s) * hv
                param_data[z_dst : z_dst + hv] = src[z_src : z_src + hv]

    @staticmethod
    def _match_dtype(param_data, loaded_weight):
        """View param_data as loaded_weight's dtype if they differ but share element size.

        This mirrors ``weight_loader_process`` behaviour for FP8 on ROCm where
        the param is ``float8_e4m3fnuz`` but the checkpoint stores
        ``float8_e4m3fn``.  The normalisation happens later in
        ``process_weights_after_loading``.
        """
        if (
            param_data.dtype != loaded_weight.dtype
            and param_data.element_size() == loaded_weight.element_size()
        ):
            return param_data.view(loaded_weight.dtype)
        return param_data

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # Load like ColumnParallelLinear (no deinterleave at load time).
        super().weight_loader(param, loaded_weight)

    def process_weights_after_loading(self):
        nk, R = self.nk, self.R
        hk, hv = self.head_k_dim, self.head_v_dim

        # Deinterleave weight rows: interleaved → [q|k|v|z]
        w = self.weight.data
        dw = torch.empty_like(w)
        self._deinterleave(dw, w, nk, R, hk, hv)
        self.weight.data = dw

        # Deinterleave weight_scale rows (per_1x128 block scale)
        ws = getattr(self, "weight_scale", None)
        if ws is not None:
            hk_s, hv_s = hk // 128, hv // 128
            s = ws.data
            ds = torch.empty_like(s)
            self._deinterleave(ds, s, nk, R, hk_s, hv_s)
            self.weight_scale.data = ds

        super().process_weights_after_loading()


class BAParallelLinear(ColumnParallelLinear):
    """Deinterleaving linear for the separate ``in_proj_ba`` checkpoint.

    The Qwen3-Next checkpoint stores ba in per-k-head-group layout::

        [b_s0..b_sR | a_s0..a_sR]  per group  (R = num_v_heads / num_k_heads)

    This class deinterleaves during loading so the output layout is
    ``[b_all | a_all]`` contiguous, enabling zero-copy ``torch.split``.

    ``in_proj_ba`` is always BF16 (listed in ``modules_to_not_convert``),
    so no weight-scale handling is needed.
    """

    def __init__(
        self,
        input_size: int,
        num_k_heads: int,
        num_v_heads: int,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        tp_size = get_tp_group().world_size
        self.nk = divide(num_k_heads, tp_size)
        self.R = num_v_heads // num_k_heads
        self.nv = self.nk * self.R
        output_size = 2 * num_v_heads
        super().__init__(
            input_size,
            output_size,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # Load like ColumnParallelLinear (no deinterleave at load time).
        super().weight_loader(param, loaded_weight)

    def process_weights_after_loading(self):
        nk, R, nv = self.nk, self.R, self.nv
        w = self.weight.data
        dw = torch.empty_like(w)
        group_size = 2 * R
        for g in range(nk):
            base = g * group_size
            dw[g * R : (g + 1) * R] = w[base : base + R]
            dw[nv + g * R : nv + (g + 1) * R] = w[base + R : base + 2 * R]
        self.weight.data = dw
        super().process_weights_after_loading()


class QKVGParallelLinear(ColumnParallelLinear):
    """QKV + output-Gate parallel linear.

    Rearranges interleaved Q+Gate weights from HF checkpoint into grouped
    layout [Gate, Q, K, V] during loading, so inference uses a single split().
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        source_quant_dtype: torch.dtype | None = None,
        prefix: str = "",
        **kwargs,
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        tp_size = get_tp_group().world_size
        self.num_heads = divide(self.total_num_heads, tp_size)
        if self.total_num_kv_heads >= tp_size:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        else:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)

        input_size = hidden_size
        output_sizes = [
            self.num_heads * self.head_size * tp_size,  # Gate
            self.num_heads * self.head_size * tp_size,  # Q
            self.num_kv_heads * self.head_size * tp_size,  # K
            self.num_kv_heads * self.head_size * tp_size,  # V
        ]

        super().__init__(
            input_size,
            output_sizes,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def _deinterleave(
        self, weight: torch.Tensor, head_stride: int | None = None
    ) -> torch.Tensor:
        """Rearrange Q+Gate from interleaved [q0,g0,q1,g1,...] to grouped [Q_all, Gate_all].

        Args:
            head_stride: number of elements per head along dim 0.
                         Defaults to self.head_size (weights); use head_size//128
                         for per-1x128 scales.
        """
        hs = head_stride if head_stride is not None else self.head_size
        return (
            weight.view(self.num_heads, 2, hs, -1)
            .transpose(0, 1)
            .reshape(self.num_heads * 2 * hs, -1)
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        # Layout: [Gate, Q, K, V]
        if loaded_shard_id == "q":
            # HF q_proj contains interleaved Q+Gate; deinterleave then
            # write Gate to offset 0 and Q to offset q_size
            shard_size = q_size * 2
            shard_offset = 0  # placeholder, handled below
            shard_rank = self.tp_rank
        elif loaded_shard_id == "k":
            shard_size = kv_size
            shard_offset = q_size * 2
            shard_rank = self.tp_rank // self.num_kv_head_replicas
        else:
            shard_size = kv_size
            shard_offset = q_size * 2 + kv_size
            shard_rank = self.tp_rank // self.num_kv_head_replicas

        is_scale = param is getattr(self, "weight_scale", None) or param is getattr(
            self, "input_scale", None
        )

        if loaded_shard_id == "q":
            # Q+Gate: deinterleave [q0,g0,...] -> [Q_all, Gate_all], then
            # write Gate to offset 0 and Q to offset q_size → layout [Gate, Q, ...]
            if is_scale and self.quant_type == QuantType.per_Tensor:
                loaded_weight = loaded_weight.view(1, 1).repeat(self.tp_size, 1)
                # Gate scale -> slot 0, Q scale -> slot 1
                q_scale = loaded_weight.narrow(self.tp_dim, shard_rank, 1)
                param.weight_loader_process(
                    param_data.narrow(self.tp_dim, 0, 1), q_scale.clone()
                )
                param.weight_loader_process(
                    param_data.narrow(self.tp_dim, 1, 1), q_scale
                )
                return

            scale_factor = (
                128 if (is_scale and self.quant_type == QuantType.per_1x128) else 1
            )
            half = q_size // scale_factor
            start_idx = shard_rank * shard_size // scale_factor
            loaded_weight = loaded_weight.narrow(
                self.tp_dim, start_idx, shard_size // scale_factor
            )
            stride = self.head_size // scale_factor
            loaded_weight = self._deinterleave(
                loaded_weight, head_stride=stride if scale_factor > 1 else None
            )
            q_part = loaded_weight.narrow(self.tp_dim, 0, half)
            gate_part = loaded_weight.narrow(self.tp_dim, half, half)
            q_offset = q_size // scale_factor
            # Gate at offset 0, Q at offset q_size
            param.weight_loader_process(
                param_data.narrow(self.tp_dim, 0, half), gate_part
            )
            param.weight_loader_process(
                param_data.narrow(self.tp_dim, q_offset, half), q_part
            )
        else:
            # K or V: straightforward load
            if is_scale:
                if self.quant_type == QuantType.per_1x128:
                    shard_offset //= 128
                    shard_size //= 128
                elif self.quant_type == QuantType.per_Tensor:
                    loaded_weight = loaded_weight.view(1, 1).repeat(self.tp_size, 1)
                    # [Gate, Q, K, V] -> K=2, V=3
                    shard_offset = {"k": 2, "v": 3}[loaded_shard_id]
                    shard_size = 1

            start_idx = shard_rank * shard_size
            param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
            loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
            param.weight_loader_process(param_data, loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        v_head_size: int | None = None,
        **kwargs,
    ):
        self.head_size = head_size
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        tp_size = get_tp_group().world_size
        self.num_heads = divide(self.total_num_heads, tp_size)
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)

        input_size = hidden_size
        output_sizes = [
            self.num_heads * self.head_size * tp_size,
            self.num_kv_heads * self.head_size * tp_size,
            self.num_kv_heads * self.v_head_size * tp_size,
        ]

        super().__init__(
            input_size,
            output_sizes,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
            shard_rank = self.tp_rank
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
            shard_rank = self.tp_rank // self.num_kv_head_replicas
        else:
            shard_size = self.num_kv_heads * self.v_head_size
            shard_offset = (
                self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            )
            shard_rank = self.tp_rank // self.num_kv_head_replicas
        if param is getattr(self, "weight_scale", None) or param is getattr(
            self, "input_scale", None
        ):
            if self.quant_type == QuantType.per_1x128:
                shard_offset = (shard_offset + 127) // 128
                shard_size = (shard_size + 127) // 128
            elif self.quant_type == QuantType.per_Tensor:
                loaded_weight = loaded_weight.view(1, 1).repeat(self.tp_size, 1)
                shard_offset = ["q", "k", "v"].index(loaded_shard_id)
                shard_size = 1

        start_idx = shard_rank * shard_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.weight_loader_process(param_data, loaded_weight)


class MinimaxM3QKVParallelLinearWithIndexer(QKVParallelLinear):
    """QKV projection fused with MiniMax-M3 lightning-indexer projections.

    The sparse attention layers emit ``[q | k | v | index_q | index_k]`` from a
    single column-parallel GEMM. ``index_q`` follows the KV-head sharding and
    replication rules, while ``index_k`` is a single replicated head.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        total_num_index_heads: int,
        index_head_size: int,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        if total_num_index_heads != total_num_kv_heads:
            raise ValueError(
                "MiniMax-M3 index_q must shard like KV heads: "
                "total_num_index_heads must equal total_num_kv_heads."
            )

        self.head_size = head_size
        self.v_head_size = head_size
        self.index_head_size = index_head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.total_num_index_heads = total_num_index_heads

        tp_size = get_tp_group().world_size
        self.num_heads = divide(self.total_num_heads, tp_size)
        if self.total_num_kv_heads >= tp_size:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        else:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        self.num_index_heads = self.num_kv_heads

        output_sizes = [
            self.num_heads * self.head_size * tp_size,
            self.num_kv_heads * self.head_size * tp_size,
            self.num_kv_heads * self.v_head_size * tp_size,
            self.num_index_heads * self.index_head_size * tp_size,
            self.index_head_size * tp_size,
        ]

        ColumnParallelLinear.__init__(
            self,
            hidden_size,
            output_sizes,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
            **kwargs,
        )

    def _shard_offset_size(self, loaded_shard_id: str) -> tuple[int, int]:
        h = self.head_size
        ih = self.index_head_size
        nq = self.num_heads
        nkv = self.num_kv_heads
        nidx = self.num_index_heads
        mapping = {
            "q": (0, nq * h),
            "k": (nq * h, nkv * h),
            "v": ((nq + nkv) * h, nkv * h),
            "index_q": ((nq + 2 * nkv) * h, nidx * ih),
            "index_k": ((nq + 2 * nkv) * h + nidx * ih, ih),
        }
        if loaded_shard_id not in mapping:
            raise ValueError(
                "MiniMax-M3 QKV/indexer shard id must be one of "
                "'q', 'k', 'v', 'index_q', 'index_k'; got "
                f"{loaded_shard_id!r}."
            )
        return mapping[loaded_shard_id]

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str,
    ):
        shard_offset, shard_size = self._shard_offset_size(loaded_shard_id)
        if param is getattr(self, "weight_scale", None) or param is getattr(
            self, "input_scale", None
        ):
            if self.quant_type == QuantType.per_1x128:
                shard_offset = (shard_offset + 127) // 128
                shard_size = (shard_size + 127) // 128
            elif self.quant_type == QuantType.per_Tensor:
                loaded_weight = loaded_weight.view(1, 1).repeat(self.tp_size, 1)
                shard_offset = ["q", "k", "v", "index_q", "index_k"].index(
                    loaded_shard_id
                )
                shard_size = 1

        if loaded_shard_id == "q":
            shard_rank = self.tp_rank
        elif loaded_shard_id == "index_k":
            shard_rank = 0
        else:
            shard_rank = self.tp_rank // self.num_kv_head_replicas

        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.narrow(
            self.tp_dim,
            shard_rank * shard_size,
            shard_size,
        )
        param.weight_loader_process(param_data, loaded_weight)


class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        self.tp_rank = get_tp_group().rank_in_group
        super().__init__(
            input_size,
            output_size,
            tp_dim=1,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param is not getattr(self, "bias", None):
            if len(loaded_weight.shape) == 0:
                loaded_weight = loaded_weight.view(1, 1)
            if loaded_weight.ndim <= self.tp_dim:
                # dims < tp_dim (1D per-channel scale with
                # tp_dim=1)
                param.weight_loader_process(param_data, loaded_weight)
                return
            shard_size = param_data.size(self.tp_dim)
            if loaded_weight.size(self.tp_dim) == 1 and self.tp_size > 1:
                loaded_weight = loaded_weight.repeat(1, self.tp_size)
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        else:
            if self.tp_size > 0 and self.tp_rank != 0:
                loaded_weight.zero_()
        param.weight_loader_process(param_data, loaded_weight)


class MergedReplicatedLinear(ReplicatedLinear):
    def __init__(
        self,
        input_size: int,
        output_size: list[int],
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        self.output_sizes = output_size
        super().__init__(
            input_size,
            sum(output_size),  # ？
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[int] = None,
    ):  # ？
        param_data = param.data
        assert loaded_shard_id is not None
        assert loaded_shard_id < len(self.output_sizes)
        if param is getattr(self, "weight_scale", None) or param is getattr(
            self, "input_scale", None
        ):
            if self.quant_type == QuantType.per_1x128:
                shard_offset = (
                    sum(self.output_sizes[:loaded_shard_id]) + 128 - 1
                ) // 128
                shard_size = (self.output_sizes[loaded_shard_id] + 128 - 1) // 128
            elif self.quant_type == QuantType.per_Tensor:
                shard_offset = loaded_shard_id
                shard_size = 1
            else:
                # Per-channel same layout as weights
                shard_offset = sum(self.output_sizes[:loaded_shard_id])
                shard_size = self.output_sizes[loaded_shard_id]
        else:
            shard_offset = sum(self.output_sizes[:loaded_shard_id])
            shard_size = self.output_sizes[loaded_shard_id]
        param_data = param_data.narrow(0, shard_offset, shard_size)
        param.weight_loader_process(param_data, loaded_weight)
