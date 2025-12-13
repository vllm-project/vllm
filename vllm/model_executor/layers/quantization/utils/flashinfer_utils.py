# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_prepare_finalize import (  # noqa: E501
    create_flashinfer_prepare_finalize,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class FlashinferMoeBackend(Enum):
    TENSORRT_LLM = "TensorRT-LLM"
    CUTLASS = "CUTLASS"
    CUTEDSL = "CUTEDSL"


def calculate_tile_tokens_dim(num_tokens, top_k, num_experts):
    from flashinfer import next_positive_power_of_2

    # FlashInfer 0.2.10 has issues with larger tile sizes. Set to 8 for now.
    # TODO: Revert this to dynamic calculation once a new version of FlashInfer
    # with the necessary kernels is released.
    tile_tokens_dim = 8

    # A factor considering tokens are not perfectly balanced among experts.
    imbalance_factor = 1.3
    # Calculate the number of tokens per expert
    # assuming perfect distribution.
    num_tokens_per_expert = (num_tokens * top_k) // num_experts
    # Apply the imbalance factor.
    num_tokens_per_expert = int(num_tokens_per_expert * imbalance_factor)
    # And pad the number to the next power of 2.
    tile_tokens_dim = next_positive_power_of_2(num_tokens_per_expert)
    # Cap to 8-max_tile_tokens_dim tokens per CTA tile
    # as it's the range supported by the kernel.
    tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)

    return tile_tokens_dim


def swap_w13_to_w31(x: torch.Tensor) -> torch.Tensor:
    return (
        x.reshape(-1, 2, x.shape[-2] // 2, x.shape[-1]).flip(dims=[1]).reshape(x.shape)
    )


def rotate_flashinfer_fp8_moe_weights(
    gemm1_weights: torch.Tensor, gemm2_weights: torch.Tensor
):
    from flashinfer import reorder_rows_for_gated_act_gemm, shuffle_matrix_a

    epilogue_tile_m = 128
    num_experts = gemm1_weights.shape[0]
    hidden_size = gemm1_weights.shape[-1]
    intermediate_size = gemm1_weights.shape[1] // 2

    # Reorder rows of W1 for fused gated activation
    gemm1_weights_fp8_interleaved = []
    for i in range(num_experts):
        gemm1_weights_fp8_interleaved.append(
            reorder_rows_for_gated_act_gemm(gemm1_weights[i])
        )

    # Stack weights and scales for all experts
    gemm1_weights_fp8_interleaved = torch.stack(gemm1_weights_fp8_interleaved).reshape(
        num_experts, 2 * intermediate_size, hidden_size
    )

    # Shuffle weights and scaling factors for transposed mma output
    gemm1_weights_fp8_shuffled = []
    gemm2_weights_fp8_shuffled = []
    for i in range(num_experts):
        gemm1_weights_fp8_shuffled.append(
            shuffle_matrix_a(
                gemm1_weights_fp8_interleaved[i].view(torch.uint8), epilogue_tile_m
            )
        )

        gemm2_weights_fp8_shuffled.append(
            shuffle_matrix_a(gemm2_weights[i].view(torch.uint8), epilogue_tile_m)
        )

    # Stack weights for all experts
    gemm1_weights.data = torch.stack(gemm1_weights_fp8_shuffled).view(
        torch.float8_e4m3fn
    )
    gemm2_weights.data = torch.stack(gemm2_weights_fp8_shuffled).view(
        torch.float8_e4m3fn
    )


def apply_flashinfer_per_tensor_scale_fp8(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    routing_bias: torch.Tensor | None,
    top_k: int,
    num_expert_group: int | None,
    topk_group: int | None,
    global_num_experts: int,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    from flashinfer.fused_moe import RoutingMethodType

    import vllm.model_executor.layers.fused_moe.flashinfer_trtllm_moe  # noqa: E501, F401

    assert layer.output1_scales_scalar is not None, (
        "Expected output1_scales_scalar to be initialized"
    )
    assert layer.output1_scales_scalar is not None, (
        "Expected output1_scales_gate_scalar to be initialized"
    )
    assert layer.output1_scales_scalar is not None, (
        "Expected output2_scales_scalar to be initialized"
    )

    from vllm.model_executor.models.llama4 import Llama4MoE

    assert layer.custom_routing_function == Llama4MoE.custom_routing_function, (
        "FusedMoE flashinfer kernels are only supported for Llama4"
    )
    return torch.ops.vllm.flashinfer_fused_moe_per_tensor_scale_fp8(
        routing_logits=router_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        input_scale=layer.w13_input_scale,
        gemm1_weights=layer.w13_weight,
        gemm2_weights=layer.w2_weight,
        output1_scales_scalar=layer.output1_scales_scalar,
        output1_scales_gate_scalar=layer.output1_scales_gate_scalar,
        output2_scales_scalar=layer.output2_scales_scalar,
        num_experts=global_num_experts,
        top_k=top_k,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        intermediate_size=layer.intermediate_size_per_partition,
        local_expert_offset=layer.ep_rank * layer.local_num_experts,
        local_num_experts=layer.local_num_experts,
        use_routing_scales_on_input=apply_router_weight_on_input,
        routing_method_type=RoutingMethodType.Llama4,
    )


def get_moe_scaling_factors(
    input_scale: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    activation_scale: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output1_scales_scalar = gemm1_weights_scale * input_scale * (1.0 / activation_scale)
    output1_scales_gate_scalar = gemm1_weights_scale * input_scale
    output2_scales_scalar = activation_scale * gemm2_weights_scale

    return output1_scales_scalar, output1_scales_gate_scalar, output2_scales_scalar


def register_moe_scaling_factors(layer: torch.nn.Module) -> None:
    output1_scales, output1_gate_scales, output2_scales = get_moe_scaling_factors(
        layer.w13_input_scale,
        layer.w13_weight_scale,
        layer.w2_input_scale,
        layer.w2_weight_scale,
    )
    layer.register_parameter(
        "output1_scales_scalar", torch.nn.Parameter(output1_scales, requires_grad=False)
    )
    layer.register_parameter(
        "output1_scales_gate_scalar",
        torch.nn.Parameter(output1_gate_scales, requires_grad=False),
    )
    layer.register_parameter(
        "output2_scales_scalar", torch.nn.Parameter(output2_scales, requires_grad=False)
    )
    layer.register_parameter(
        "w2_input_scale_inv",
        torch.nn.Parameter(1.0 / layer.w2_input_scale, requires_grad=False),
    )


def build_flashinfer_fp8_cutlass_moe_prepare_finalize(
    moe: FusedMoEConfig | None, use_deepseek_fp8_block_scale: bool = False
) -> mk.FusedMoEPrepareAndFinalize:
    """Create a FlashInfer CUTLASS fused-MoE prepare finalize kernel"""
    use_dp = moe.moe_parallel_config.dp_size > 1 if moe is not None else False
    # Propagate block-scale flag so prepare/finalize can skip act quantization
    # and inform the kernel to consume per-block weight scales.
    return create_flashinfer_prepare_finalize(
        use_dp, use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale
    )


def select_cutlass_fp8_gemm_impl(
    moe: FusedMoEConfig | None,
    quant_config: FusedMoEQuantConfig,
    out_dtype: torch.dtype | None = None,
    use_deepseek_fp8_block_scale: bool = False,
) -> mk.FusedMoEPermuteExpertsUnpermute:
    """Return a GEMM *experts* implementation for fused-MoE layers"""

    if moe is not None:
        return FlashInferExperts(
            out_dtype=moe.in_dtype,
            quant_config=quant_config,
            ep_rank=moe.moe_parallel_config.ep_rank,
            ep_size=moe.moe_parallel_config.ep_size,
            tp_rank=moe.moe_parallel_config.tp_rank,
            tp_size=moe.moe_parallel_config.tp_size,
            use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
        )

    assert out_dtype is not None, "If moe config is None, out_dtype must be passed"
    return FlashInferExperts(
        out_dtype=out_dtype,
        quant_config=quant_config,
        use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
    )


def flashinfer_cutlass_moe_fp8(
    hidden_states: torch.Tensor,
    layer: torch.nn.Module,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    use_deepseek_fp8_block_scale: bool = False,
    moe: FusedMoEConfig | None = None,
) -> torch.Tensor:
    quant_config = layer.quant_method.get_fused_moe_quant_config(layer)
    assert quant_config is not None

    # Construct modular kernel with block-scale support when requested.
    parallel_config = getattr(
        getattr(layer, "vllm_config", None),
        "parallel_config",
        None,
    )
    fused_experts = mk.FusedMoEModularKernel(
        build_flashinfer_fp8_cutlass_moe_prepare_finalize(
            moe=moe, use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale
        ),
        select_cutlass_fp8_gemm_impl(
            moe=moe,
            quant_config=quant_config,
            out_dtype=hidden_states.dtype,
            use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
        ),
        parallel_config=parallel_config,
    )

    return fused_experts(
        hidden_states,
        layer.w13_weight,
        layer.w2_weight,
        topk_weights,
        topk_ids,
        inplace=inplace,
        activation=activation,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )


def get_flashinfer_moe_backend() -> FlashinferMoeBackend:
    backend_map = {
        "throughput": FlashinferMoeBackend.CUTLASS,
        "latency": FlashinferMoeBackend.TENSORRT_LLM,
        "masked_gemm": FlashinferMoeBackend.CUTEDSL,
    }

    flashinfer_moe_backend = envs.VLLM_FLASHINFER_MOE_BACKEND
    if flashinfer_moe_backend in backend_map:
        if (
            flashinfer_moe_backend == "latency"
            and not current_platform.has_device_capability(100)
        ):
            logger.info_once(
                "Flashinfer TRTLLM MOE backend is only supported on "
                "SM100 and later, using CUTLASS backend instead",
                scope="local",
            )
            return FlashinferMoeBackend.CUTLASS
        return backend_map[flashinfer_moe_backend]
    elif current_platform.is_device_capability(90):
        return FlashinferMoeBackend.CUTLASS

    raise ValueError(
        f"Unknown flashinfer moe backend: {flashinfer_moe_backend!r}. "
        f"Expected one of {list(backend_map.keys())}."
    )


def is_flashinfer_supporting_global_sf(backend: FlashinferMoeBackend | None) -> bool:
    # TODO(shuw@nvidia): Update when new backends are added.
    backends_supporting_global_sf = (
        FlashinferMoeBackend.CUTLASS,
        FlashinferMoeBackend.TENSORRT_LLM,
    )
    return backend in backends_supporting_global_sf


def convert_moe_weights_to_flashinfer_trtllm_block_layout(
    cache_permute_indices: dict[torch.Size, torch.Tensor],
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert expert weights to FlashInfer's block layout.

    This reorders W13 and W2 into the expected epilogue-tiled block layout and
    returns the shuffled weight tensors.
    """
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        convert_to_block_layout,
        get_w2_permute_indices_with_cache,
    )

    epilogue_tile_m = 128
    block_k = 128

    # Reorder rows of W13 and W2 for fused gated activation and convert to the
    # block layout expected by the FlashInfer kernel.
    num_experts = w13_weight.shape[0]
    device_w13 = w13_weight.device
    device_w2 = w2_weight.device

    w13_weights_shuffled: list[torch.Tensor] = []
    w2_weights_shuffled: list[torch.Tensor] = []

    for i in range(num_experts):
        permute_indices = _maybe_get_cached_w3_w1_permute_indices(
            cache_permute_indices,
            w13_weight[i].view(torch.uint8),
            epilogue_tile_m,
        )
        tmp_weights1 = (
            w13_weight[i]
            .clone()
            .view(torch.uint8)[permute_indices.to(device_w13)]
            .contiguous()
        )

        permute_indices = get_w2_permute_indices_with_cache(
            cache_permute_indices,
            w2_weight[i].view(torch.uint8),
            epilogue_tile_m,
        )
        tmp_weights2 = (
            w2_weight[i]
            .clone()
            .view(torch.uint8)[permute_indices.to(device_w2)]
            .contiguous()
        )

        tmp_weights1 = convert_to_block_layout(tmp_weights1.view(torch.uint8), block_k)
        tmp_weights2 = convert_to_block_layout(tmp_weights2.view(torch.uint8), block_k)

        w13_weights_shuffled.append(tmp_weights1.view(torch.bfloat16))
        w2_weights_shuffled.append(tmp_weights2.view(torch.bfloat16))

    # Stack weights for all experts and return as BF16 tensors.
    w13_weights_shuffled_tensor = (
        torch.stack(w13_weights_shuffled).view(torch.bfloat16).contiguous()
    )
    w2_weights_shuffled_tensor = (
        torch.stack(w2_weights_shuffled).view(torch.bfloat16).contiguous()
    )

    return w13_weights_shuffled_tensor, w2_weights_shuffled_tensor
