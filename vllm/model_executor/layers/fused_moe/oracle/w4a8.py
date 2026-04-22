# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum

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
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)

logger = init_logger(__name__)


class W4A8MoeBackend(Enum):
    CUTLASS = "CUTLASS"


def backend_to_kernel_cls(
    backend: W4A8MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == W4A8MoeBackend.CUTLASS:
        from vllm.model_executor.layers.fused_moe.cutlass_moe import (
            CutlassExpertsW4A8Fp8,
        )

        return [CutlassExpertsW4A8Fp8]
    else:
        raise ValueError(f"Unknown W4A8 MoE backend: {backend.value}")


def select_w4a8_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None = None,
    activation_key: QuantKey | None = None,
) -> tuple[W4A8MoeBackend, type[mk.FusedMoEExperts]]:
    """
    Select the W4A8 MoE backend.
    Currently only CUTLASS is supported.
    """
    backend = W4A8MoeBackend.CUTLASS

    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    for k_cls in backend_to_kernel_cls(backend):
        supported, reason = k_cls.is_supported_config(
            k_cls, config, weight_key, activation_key, activation_format
        )
        if supported:
            logger.info_once("Using %s W4A8 MoE backend.", backend.value, scope="local")
            return backend, k_cls

    raise NotImplementedError(
        f"W4A8 MoE backend {backend.value} does not support the "
        f"deployment configuration: {reason}."
    )


def convert_to_w4a8_moe_kernel_format(
    layer: torch.nn.Module,
    quant_fp8: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Process W4A8 weights after loading: convert packed weights,
    reorder for CUTLASS, and convert scales to FP8.

    Returns (b_strides1, b_strides2) from weight reordering.
    """
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        convert_bf16_scales_to_fp8,
        convert_packed_uint4b8_to_signed_int4_inplace,
    )
    from vllm.model_executor.utils import replace_parameter

    # Convert and reorder w13 weights.
    convert_packed_uint4b8_to_signed_int4_inplace(layer.w13_weight_packed)
    w13_weight_shuffled, b_strides1 = ops.cutlass_encode_and_reorder_int4b_grouped(
        layer.w13_weight_packed
    )
    replace_parameter(layer, "w13_weight_packed", w13_weight_shuffled)

    # Convert and reorder w2 weights.
    convert_packed_uint4b8_to_signed_int4_inplace(layer.w2_weight_packed)
    w2_weight_shuffled, b_strides2 = ops.cutlass_encode_and_reorder_int4b_grouped(
        layer.w2_weight_packed
    )
    replace_parameter(layer, "w2_weight_packed", w2_weight_shuffled)

    # Convert bf16 scales to (fp8_scales, channel_scales).
    w13_weight_scale, w13_weight_chan_scale = convert_bf16_scales_to_fp8(
        quant_fp8, layer.w13_weight_scale
    )
    w2_weight_scale, w2_weight_chan_scale = convert_bf16_scales_to_fp8(
        quant_fp8, layer.w2_weight_scale
    )

    # Replace channel scales on the layer (pre-registered in create_weights).
    replace_parameter(layer, "w13_weight_chan_scale", w13_weight_chan_scale)
    replace_parameter(layer, "w2_weight_chan_scale", w2_weight_chan_scale)

    # Permute and pack group-wise scales for the kernel.
    # Scales are stored as (E, N, K // 128) but the kernel expects
    # (E, K // 128, N) in row-major format.
    w13_weight_scale_packed = ops.cutlass_pack_scale_fp8(
        w13_weight_scale.permute(0, 2, 1).contiguous()
    )
    replace_parameter(layer, "w13_weight_scale", w13_weight_scale_packed)
    w2_weight_scale_packed = ops.cutlass_pack_scale_fp8(
        w2_weight_scale.permute(0, 2, 1).contiguous()
    )
    replace_parameter(layer, "w2_weight_scale", w2_weight_scale_packed)

    return b_strides1, b_strides2


def make_w4a8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    g1_alphas: torch.Tensor,
    g2_alphas: torch.Tensor,
) -> FusedMoEQuantConfig:
    """
    Create FusedMoEQuantConfig for W4A8 FP8 MoE.
    W4A8 always uses dynamic per-token activation quantization
    and per-channel weight scales.
    """
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
    experts_cls: type[mk.FusedMoEExperts],
    b_strides1: torch.Tensor,
    b_strides2: torch.Tensor,
    group_size: int,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    shared_experts: SharedExperts | None = None,
) -> mk.FusedMoEKernel:
    # Create Prepare/Finalize.
    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__, scope="local")

    # Create Experts.
    experts = experts_cls(
        moe_config=moe_config,
        quant_config=moe_quant_config,
        b_strides1=b_strides1,
        b_strides2=b_strides2,
        group_size=group_size,
    )

    kernel = mk.FusedMoEKernel(
        prepare_finalize,
        experts,
        shared_experts=shared_experts,
        inplace=not moe_config.disable_inplace,
    )

    return kernel
