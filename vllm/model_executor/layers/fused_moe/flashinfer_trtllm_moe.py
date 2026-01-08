# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    calculate_tile_tokens_dim,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.utils.torch_utils import direct_register_custom_op


def flashinfer_fused_moe_blockscale_fp8(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    x: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale_inv: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale_inv: torch.Tensor,
    global_num_experts: int,
    top_k: int,
    num_expert_group: int | None,
    topk_group: int | None,
    intermediate_size: int,
    expert_offset: int,
    local_num_experts: int,
    block_shape: list[int],
    routing_method_type: int = RoutingMethodType.DeepSeekV3,
    routed_scaling: float | None = 1.0,
) -> torch.Tensor:
    from vllm.utils.flashinfer import flashinfer_trtllm_fp8_block_scale_moe

    topk_group = topk_group if topk_group is not None else 0
    assert top_k <= global_num_experts
    assert top_k <= 10
    assert global_num_experts % 4 == 0
    assert block_shape == [128, 128]
    # Routing kernel expects #experts <= #threads 512
    assert global_num_experts <= 512

    a_q, a_sf = per_token_group_quant_fp8(x, block_shape[1])
    # NOTE: scales of hidden states have to be transposed!
    a_sf_t = a_sf.t().contiguous()
    return flashinfer_trtllm_fp8_block_scale_moe(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=a_q,
        hidden_states_scale=a_sf_t,
        gemm1_weights=w13_weight,
        gemm1_weights_scale=w13_weight_scale_inv,
        gemm2_weights=w2_weight,
        gemm2_weights_scale=w2_weight_scale_inv,
        num_experts=global_num_experts,
        top_k=top_k,
        n_group=num_expert_group,
        topk_group=topk_group,
        intermediate_size=intermediate_size,
        local_expert_offset=expert_offset,
        local_num_experts=local_num_experts,
        routed_scaling_factor=routed_scaling,
        tile_tokens_dim=None,
        routing_method_type=routing_method_type,
        use_shuffled_weight=False,
    )


def flashinfer_fused_moe_blockscale_fp8_fake(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    x: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale_inv: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale_inv: torch.Tensor,
    global_num_experts: int,
    top_k: int,
    num_expert_group: int,
    topk_group: int,
    intermediate_size: int,
    expert_offset: int,
    local_num_experts: int,
    block_shape: list[int],
    routing_method_type: int,
    routed_scaling: float = 1.0,
) -> torch.Tensor:
    return torch.empty_like(x)


# TODO(bnell): Does this really need to be a torch.op?
direct_register_custom_op(
    op_name="flashinfer_fused_moe_blockscale_fp8",
    op_func=flashinfer_fused_moe_blockscale_fp8,
    fake_impl=flashinfer_fused_moe_blockscale_fp8_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def fi_trtllm_fp8_per_tensor_moe(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor | None,
    hidden_states: torch.Tensor,
    input_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    output1_scales_scalar: torch.Tensor,
    output1_scales_gate_scalar: torch.Tensor,
    output2_scales_scalar: torch.Tensor,
    num_experts: int,
    top_k: int,
    num_expert_group: int | None,
    topk_group: int | None,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    use_routing_scales_on_input: bool,
    routing_method_type: int,
    routed_scaling_factor: float = 1.0,
) -> torch.Tensor:
    num_expert_group = num_expert_group if num_expert_group is not None else 0
    topk_group = topk_group if topk_group is not None else 0

    quant_hidden_states, _ = moe_kernel_quantize_input(
        hidden_states,
        input_scale,
        quant_dtype=torch.float8_e4m3fn,
        per_act_token_quant=False,
    )

    from vllm.utils.flashinfer import flashinfer_trtllm_fp8_per_tensor_scale_moe

    return flashinfer_trtllm_fp8_per_tensor_scale_moe(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=quant_hidden_states,
        gemm1_weights=gemm1_weights,
        output1_scales_scalar=output1_scales_scalar,
        output1_scales_gate_scalar=output1_scales_gate_scalar,
        gemm2_weights=gemm2_weights,
        output2_scales_scalar=output2_scales_scalar,
        num_experts=num_experts,
        top_k=top_k,
        n_group=num_expert_group,
        topk_group=topk_group,
        intermediate_size=intermediate_size,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routed_scaling_factor=routed_scaling_factor,
        use_routing_scales_on_input=use_routing_scales_on_input,
        tile_tokens_dim=calculate_tile_tokens_dim(
            hidden_states.shape[0], top_k, num_experts
        ),
        routing_method_type=routing_method_type,
    )


def fi_trtllm_fp8_per_tensor_moe_fake(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor | None,
    hidden_states: torch.Tensor,
    input_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    output1_scales_scalar: torch.Tensor,
    output1_scales_gate_scalar: torch.Tensor,
    output2_scales_scalar: torch.Tensor,
    num_experts: int,
    top_k: int,
    num_expert_group: int | None,
    topk_group: int | None,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    use_routing_scales_on_input: bool,
    routing_method_type: int,
    routed_scaling_factor: float = 1.0,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


# TODO(bnell): Does this really need to be a torch.op?
direct_register_custom_op(
    op_name="fi_trtllm_fp8_per_tensor_moe",
    op_func=fi_trtllm_fp8_per_tensor_moe,
    mutates_args=["hidden_states"],
    fake_impl=fi_trtllm_fp8_per_tensor_moe_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)
