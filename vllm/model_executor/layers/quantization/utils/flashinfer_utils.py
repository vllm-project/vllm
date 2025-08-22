# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum
from typing import Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
    FlashInferExperts)
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_prepare_finalize import (  # noqa: E501
    FlashInferCutlassMoEPrepareAndFinalize)

logger = init_logger(__name__)


class FlashinferMoeBackend(Enum):
    TENSORRT_LLM = "TensorRT-LLM"
    CUTLASS = "CUTLASS"


def calculate_tile_tokens_dim(num_tokens, top_k, num_experts):

    # FlashInfer 0.2.10 has issues with larger tile sizes. Set to 8 for now.
    # TODO: Revert this to dynamic calculation once a new version of FlashInfer
    # with the necessary kernels is released.
    tile_tokens_dim = 8

    # from flashinfer import next_positive_power_of_2

    # # Guess tokens per expert assuming perfect expert distribution first.
    # num_tokens_per_expert = (num_tokens * top_k) // num_experts
    # # And pad the number to the next power of 2.
    # tile_tokens_dim = next_positive_power_of_2(num_tokens_per_expert)
    # # Cap to 8-64 tokens per CTA tile as it's the range supported by the
    # # kernel.
    # tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)

    return tile_tokens_dim


def swap_w13_to_w31(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, 2, x.shape[-2] // 2,
                     x.shape[-1]).flip(dims=[1]).reshape(x.shape)


def rotate_flashinfer_fp8_moe_weights(gemm1_weights: torch.Tensor,
                                      gemm2_weights: torch.Tensor):
    from flashinfer import reorder_rows_for_gated_act_gemm, shuffle_matrix_a
    epilogue_tile_m = 128
    num_experts = gemm1_weights.shape[0]
    hidden_size = gemm1_weights.shape[-1]
    intermediate_size = gemm1_weights.shape[1] // 2

    # Reorder rows of W1 for fused gated activation
    gemm1_weights_fp8_interleaved = []
    for i in range(num_experts):
        gemm1_weights_fp8_interleaved.append(
            reorder_rows_for_gated_act_gemm(gemm1_weights[i]))

    # Stack weights and scales for all experts
    gemm1_weights_fp8_interleaved = torch.stack(
        gemm1_weights_fp8_interleaved).reshape(num_experts,
                                               2 * intermediate_size,
                                               hidden_size)

    # Shuffle weights and scaling factors for transposed mma output
    gemm1_weights_fp8_shuffled = []
    gemm2_weights_fp8_shuffled = []
    for i in range(num_experts):
        gemm1_weights_fp8_shuffled.append(
            shuffle_matrix_a(
                gemm1_weights_fp8_interleaved[i].view(torch.uint8),
                epilogue_tile_m))

        gemm2_weights_fp8_shuffled.append(
            shuffle_matrix_a(gemm2_weights[i].view(torch.uint8),
                             epilogue_tile_m))

    # Stack weights for all experts
    gemm1_weights.data = torch.stack(gemm1_weights_fp8_shuffled).view(
        torch.float8_e4m3fn)
    gemm2_weights.data = torch.stack(gemm2_weights_fp8_shuffled).view(
        torch.float8_e4m3fn)


def apply_flashinfer_per_tensor_scale_fp8(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    top_k: int,
    num_expert_group: Optional[int],
    topk_group: Optional[int],
    global_num_experts: int,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    from flashinfer.fused_moe import RoutingMethodType
    assert layer.output1_scales_scalar is not None, (
        "Expected output1_scales_scalar to be initialized")
    assert layer.output1_scales_scalar is not None, (
        "Expected output1_scales_gate_scalar to be initialized")
    assert layer.output1_scales_scalar is not None, (
        "Expected output2_scales_scalar to be initialized")

    from vllm.model_executor.models.llama4 import Llama4MoE
    assert layer.custom_routing_function == Llama4MoE.custom_routing_function, \
        "FusedMoE flashinfer kernels are only supported for Llama4"
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
    output1_scales_scalar = gemm1_weights_scale * input_scale * (
        1.0 / activation_scale)
    output1_scales_gate_scalar = gemm1_weights_scale * input_scale
    output2_scales_scalar = activation_scale * gemm2_weights_scale

    return output1_scales_scalar, output1_scales_gate_scalar, \
        output2_scales_scalar


def register_moe_scaling_factors(layer: torch.nn.Module) -> None:
    output1_scales, output1_gate_scales, output2_scales = \
        get_moe_scaling_factors(
            layer.w13_input_scale, layer.w13_weight_scale,
            layer.w2_input_scale, layer.w2_weight_scale
        )
    layer.register_parameter(
        'output1_scales_scalar',
        torch.nn.Parameter(output1_scales, requires_grad=False))
    layer.register_parameter(
        'output1_scales_gate_scalar',
        torch.nn.Parameter(output1_gate_scales, requires_grad=False))
    layer.register_parameter(
        'output2_scales_scalar',
        torch.nn.Parameter(output2_scales, requires_grad=False))
    layer.register_parameter(
        'w2_input_scale_inv',
        torch.nn.Parameter(1.0 / layer.w2_input_scale, requires_grad=False))


def build_flashinfer_fp8_cutlass_moe_prepare_finalize(
    moe: Optional[FusedMoEConfig],
    layer: torch.nn.Module,
) -> mk.FusedMoEPrepareAndFinalize:
    """Create a FlashInfer CUTLASS fused-MoE prepare finalize kernel"""
    use_dp = moe.moe_parallel_config.dp_size > 1 if moe is not None else False
    return FlashInferCutlassMoEPrepareAndFinalize(
        use_dp, a1_gscale=layer.w13_input_scale)


def select_cutlass_fp8_gemm_impl(
    moe: Optional[FusedMoEConfig],
    layer: torch.nn.Module,
    out_dtype: Optional[torch.dtype] = None,
) -> mk.FusedMoEPermuteExpertsUnpermute:
    """Return a GEMM *experts* implementation for fused-MoE layers"""

    from vllm.model_executor.models.llama4 import Llama4MoE
    assert layer.custom_routing_function == Llama4MoE.custom_routing_function, \
        "FusedMoE flashinfer kernels are only supported for Llama4"

    if moe is not None:
        return FlashInferExperts(
            g1_alphas=layer.output1_scales_gate_scalar,
            g2_alphas=layer.output2_scales_scalar,
            a1_gscale=layer.w13_input_scale,
            a2_gscale=layer.w2_input_scale_inv,
            out_dtype=moe.in_dtype,
            quant_dtype=torch.float8_e4m3fn,
            ep_rank=moe.moe_parallel_config.ep_rank,
            ep_size=moe.moe_parallel_config.ep_size,
            tp_rank=moe.moe_parallel_config.tp_rank,
            tp_size=moe.moe_parallel_config.tp_size,
        )

    assert out_dtype is not None, (
        "If moe config is None, out_dtype must be passed")
    return FlashInferExperts(
        g1_alphas=layer.output1_scales_gate_scalar,
        g2_alphas=layer.output2_scales_scalar,
        a1_gscale=layer.w13_input_scale,
        a2_gscale=layer.w2_input_scale_inv,
        out_dtype=out_dtype,
        quant_dtype=torch.float8_e4m3fn,
    )


def flashinfer_cutlass_moe_fp8(
    hidden_states: torch.Tensor,
    layer: torch.nn.Module,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    fused_experts = mk.FusedMoEModularKernel(
        build_flashinfer_fp8_cutlass_moe_prepare_finalize(moe=None,
                                                          layer=layer),
        select_cutlass_fp8_gemm_impl(moe=None,
                                     layer=layer,
                                     out_dtype=hidden_states.dtype))

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
    flashinfer_moe_backend = envs.VLLM_FLASHINFER_MOE_BACKEND
    if flashinfer_moe_backend == "throughput":
        return FlashinferMoeBackend.CUTLASS
    elif flashinfer_moe_backend == "latency":
        return FlashinferMoeBackend.TENSORRT_LLM

    allowed_backends = ["throughput", "latency"]
    raise ValueError(
        f"Unknown flashinfer moe backend: {flashinfer_moe_backend}"
        f" expected one of {allowed_backends}")
