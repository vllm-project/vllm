# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum
from typing import TYPE_CHECKING

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int4_w4afp8_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8DynamicTokenSym,
    kInt4Static,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
        CutlassExpertsW4A8Fp8,
    )

logger = init_logger(__name__)


class W4A8MoeBackend(Enum):
    CUTLASS = "CUTLASS"


def backend_to_kernel_cls(
    backend: W4A8MoeBackend,
) -> list[type["CutlassExpertsW4A8Fp8"]]:
    if backend == W4A8MoeBackend.CUTLASS:
        from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
            CutlassExpertsW4A8Fp8,
        )

        return [CutlassExpertsW4A8Fp8]
    else:
        raise ValueError(f"Unknown W4A8 MoE backend: {backend.value}")


def select_w4a8_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None = kInt4Static,
    activation_key: QuantKey | None = kFp8DynamicTokenSym,
) -> tuple[W4A8MoeBackend, type["CutlassExpertsW4A8Fp8"]]:
    backend = W4A8MoeBackend.CUTLASS

    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    last_reason: str | None = None
    for kernel_cls in backend_to_kernel_cls(backend):
        supported, reason = kernel_cls.is_supported_config(
            kernel_cls,
            config,
            weight_key,
            activation_key,
            activation_format,
        )
        if supported:
            logger.info_once("Using %s W4A8 MoE backend.", backend.value)
            return backend, kernel_cls
        last_reason = reason

    raise NotImplementedError(
        f"W4A8 MoE backend {backend.value} does not support the "
        f"deployment configuration: {last_reason}."
    )


def convert_to_w4a8_moe_kernel_format(
    w13_weight_packed: torch.Tensor,
    w2_weight_packed: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
        convert_bf16_scales_to_fp8,
        convert_packed_uint4b8_to_signed_int4_inplace,
    )

    quant_fp8 = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)

    convert_packed_uint4b8_to_signed_int4_inplace(w13_weight_packed)
    # Mirror the sync in CutlassW4A8LinearKernel; required for TP>1 correctness.
    torch.accelerator.synchronize()
    w13_weight_shuffled, b_strides1 = ops.cutlass_encode_and_reorder_int4b_grouped(
        w13_weight_packed
    )

    convert_packed_uint4b8_to_signed_int4_inplace(w2_weight_packed)
    # Mirror the sync in CutlassW4A8LinearKernel; required for TP>1 correctness.
    torch.accelerator.synchronize()
    w2_weight_shuffled, b_strides2 = ops.cutlass_encode_and_reorder_int4b_grouped(
        w2_weight_packed
    )

    w13_weight_scale, w13_weight_chan_scale = convert_bf16_scales_to_fp8(
        quant_fp8, w13_weight_scale
    )
    w2_weight_scale, w2_weight_chan_scale = convert_bf16_scales_to_fp8(
        quant_fp8, w2_weight_scale
    )

    # Scales are stored as (E, N, K // 128), but the kernel expects
    # (E, K // 128, N) in row-major format.
    w13_weight_scale_packed = ops.cutlass_pack_scale_fp8(
        w13_weight_scale.permute(0, 2, 1).contiguous()
    )
    w2_weight_scale_packed = ops.cutlass_pack_scale_fp8(
        w2_weight_scale.permute(0, 2, 1).contiguous()
    )

    return (
        w13_weight_shuffled,
        w2_weight_shuffled,
        w13_weight_scale_packed,
        w2_weight_scale_packed,
        w13_weight_chan_scale,
        w2_weight_chan_scale,
        b_strides1,
        b_strides2,
    )


def make_w4a8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    g1_alphas: torch.Tensor,
    g2_alphas: torch.Tensor,
) -> FusedMoEQuantConfig:
    return int4_w4afp8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        per_act_token_quant=True,
        per_out_ch_quant=True,
    )


def make_w4a8_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type["CutlassExpertsW4A8Fp8"],
    b_strides1: torch.Tensor,
    b_strides2: torch.Tensor,
    group_size: int,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__)

    experts = experts_cls(
        moe_config=moe_config,
        quant_config=moe_quant_config,
        b_strides1=b_strides1,
        b_strides2=b_strides2,
        group_size=group_size,
    )

    return mk.FusedMoEKernel(
        prepare_finalize,
        experts,
        inplace=not moe_config.disable_inplace,
    )
