# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Oracle for W4A8 (INT4, FP8) MoE
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
    from vllm.model_executor.layers.fused_moe import RoutedExperts

logger = init_logger(__name__)


class W4A8MoeBackend(Enum):
    CUTLASS = "CUTLASS"
    HUMMING = "HUMMING"


def backend_to_kernel_cls(
    backend: W4A8MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == W4A8MoeBackend.CUTLASS:
        from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
            CutlassExpertsW4A8Fp8,
        )

        return [CutlassExpertsW4A8Fp8]
    elif backend == W4A8MoeBackend.HUMMING:
        from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
            BatchedHummingGroupedExperts,
            HummingGroupedExperts,
            HummingIndexedExperts,
        )

        return [
            BatchedHummingGroupedExperts,
            HummingGroupedExperts,
            HummingIndexedExperts,
        ]
    else:
        raise ValueError(f"Unknown W4A8 MoE backend: {backend.value}")


def select_w4a8_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None = kInt4Static,
    activation_key: QuantKey | None = kFp8DynamicTokenSym,
) -> tuple[W4A8MoeBackend, type[mk.FusedMoEExperts]]:
    backends = [W4A8MoeBackend.CUTLASS, W4A8MoeBackend.HUMMING]
    if config.moe_backend != "auto":
        backend_map = {
            "cutlass": W4A8MoeBackend.CUTLASS,
            "humming": W4A8MoeBackend.HUMMING,
        }
        if config.moe_backend not in backend_map:
            raise ValueError(
                f"moe_backend='{config.moe_backend}' is not supported for W4A8 MoE. "
                f"Expected one of {list(backend_map)}."
            )
        backends = [backend_map[config.moe_backend]]

    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    reasons = []
    for backend in backends:
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
            reasons.append(f"{kernel_cls.__name__}: {reason}")

    raise NotImplementedError(
        "No W4A8 MoE backend supports the deployment configuration: "
        + "; ".join(reasons)
    )


def convert_to_w4a8_moe_kernel_format(
    backend: W4A8MoeBackend,
    layer: "RoutedExperts",
    group_size: int,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if backend == W4A8MoeBackend.HUMMING:
        from vllm.model_executor.layers.quantization.utils.humming import (
            convert_to_humming_moe_kernel_format,
        )
        from vllm.utils.humming import (
            HummingInputSchema,
            HummingWeightSchema,
            dtypes,
        )

        convert_to_humming_moe_kernel_format(
            layer,
            weight_schema=HummingWeightSchema(
                b_dtype=dtypes.uint4, weight_scale_group_size=group_size
            ),
            input_schema=HummingInputSchema(a_dtype=dtypes.float8e4m3),
            allow_input_schema_fallback=False,
        )
        return (
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            None,
            None,
            None,
            None,
        )

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
        convert_bf16_scales_to_fp8,
        convert_packed_uint4b8_to_signed_int4_inplace,
    )

    quant_fp8 = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)

    convert_packed_uint4b8_to_signed_int4_inplace(w13_weight)
    # Mirror the sync in CutlassW4A8LinearKernel; required for TP>1 correctness.
    torch.accelerator.synchronize()
    w13_weight_shuffled, b_strides1 = ops.cutlass_encode_and_reorder_int4b_grouped(
        w13_weight
    )

    convert_packed_uint4b8_to_signed_int4_inplace(w2_weight)
    # Mirror the sync in CutlassW4A8LinearKernel; required for TP>1 correctness.
    torch.accelerator.synchronize()
    w2_weight_shuffled, b_strides2 = ops.cutlass_encode_and_reorder_int4b_grouped(
        w2_weight
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
    backend: W4A8MoeBackend,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    g1_alphas: torch.Tensor | None,
    g2_alphas: torch.Tensor | None,
    layer: torch.nn.Module | None = None,
) -> FusedMoEQuantConfig:
    if backend == W4A8MoeBackend.HUMMING:
        from vllm.model_executor.layers.fused_moe import RoutedExperts
        from vllm.model_executor.layers.quantization.utils.humming import (
            get_humming_moe_quant_config,
        )

        assert isinstance(layer, RoutedExperts)
        return get_humming_moe_quant_config(layer)

    assert g1_alphas is not None and g2_alphas is not None
    return int4_w4afp8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        per_act_token_quant=True,
        per_out_ch_quant=True,
    )


def make_w4a8_moe_kernel(
    backend: W4A8MoeBackend,
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    b_strides1: torch.Tensor | None,
    b_strides2: torch.Tensor | None,
    group_size: int,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    if backend == W4A8MoeBackend.HUMMING:
        from vllm.model_executor.layers.quantization.utils.humming import (
            make_humming_moe_kernel,
        )

        return make_humming_moe_kernel(
            moe_quant_config, moe_config, experts_cls, routing_tables=routing_tables
        )

    from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
        CutlassExpertsW4A8Fp8,
    )

    assert issubclass(experts_cls, CutlassExpertsW4A8Fp8)
    assert b_strides1 is not None and b_strides2 is not None
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
    )
