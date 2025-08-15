# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum
from typing import Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
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

    from vllm.model_executor.models.llama4 import Llama4MoE
    assert layer.custom_routing_function == Llama4MoE.custom_routing_function, \
        "FusedMoE flashinfer kernels are only supported for Llama4"
    return torch.ops.vllm.flashinfer_fused_moe_per_tensor_scale_fp8(
        routing_logits=router_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        input_scale=layer.w13_input_scale,
        gemm1_weights=layer.w13_weight,
        gemm1_weights_scale=layer.w13_weight_scale,
        gemm2_weights=layer.w2_weight,
        gemm2_weights_scale=layer.w2_weight_scale,
        activation_scale=layer.w2_input_scale,
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


def build_flashinfer_fp8_cutlass_moe_kernel(
    moe_parallel_config: FusedMoEParallelConfig, ) -> mk.FusedMoEModularKernel:
    """Create *and return* a FlashInfer CUTLASS fused-MoE modular kernel"""
    experts = FlashInferExperts(
        use_fp8_w8a8=True,
        use_dp=moe_parallel_config.dp_size > 1,
        ep_rank=moe_parallel_config.ep_rank,
        ep_size=moe_parallel_config.ep_size,
        tp_rank=moe_parallel_config.tp_rank,
        tp_size=moe_parallel_config.tp_size,
    )
    return mk.FusedMoEModularKernel(
        FlashInferCutlassMoEPrepareAndFinalize(
            quant_dtype=torch.float8_e4m3fn),
        experts,
    )


def flashinfer_fp8_cutlass_moe_forward(
    fused_experts: mk.FusedMoEModularKernel,
    layer: torch.nn.Module,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    global_num_experts: int,
    expert_map: Optional[torch.Tensor],
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    """Common forward wrapper for FlashInfer NV-FP4 fused-MoE"""

    from vllm.model_executor.models.llama4 import Llama4MoE
    assert layer.custom_routing_function == Llama4MoE.custom_routing_function, \
        "FusedMoE flashinfer kernels are only supported for Llama4"

    extra_expert_args = {
        "out_dtype": x.dtype,
    }
    extra_prepare_args = {
        "use_dp": layer.dp_size > 1,
        "local_tokens": x.shape[0],
    }
    extra_finalize_args = {
        "use_dp": layer.dp_size > 1,
        "local_tokens": x.shape[0],
    }

    return fused_experts(
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,  # TODO(shuw): fix later, now output is high prec
        activation=activation,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        a1_scale=layer.w13_input_scale,
        a2_scale=layer.w2_input_scale,
        apply_router_weight_on_input=apply_router_weight_on_input,
        extra_expert_args=extra_expert_args,
        extra_prepare_args=extra_prepare_args,
        extra_finalize_args=extra_finalize_args,
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
