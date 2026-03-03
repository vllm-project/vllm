# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-coupled FlashInfer MoE helpers.

These functions depend on ``torch.nn.Module`` layer objects and live here
(close to the fused-MoE implementation) rather than in the generic
``vllm.utils.flashinfer`` module, which contains only pure tensor helpers.
"""

import torch

from vllm.utils.flashinfer import (
    activation_to_flashinfer_int,
    align_fp8_moe_weights_for_fi,
    make_fp8_moe_alpha_scales_for_fi,
    rotate_weights_for_fi_trtllm_fp8_per_tensor_moe,
    swap_w13_to_w31,
)


def register_scales_for_trtllm_fp8_per_tensor_moe(
    layer: torch.nn.Module,
    w13_scale: torch.Tensor,
    w13_input_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_input_scale: torch.Tensor,
) -> None:
    """Register necessary scales for FlashInfer TRTLLM FP8 MoE kernel."""
    g1_alphas, g2_alphas = make_fp8_moe_alpha_scales_for_fi(
        w13_scale=w13_scale,
        w13_input_scale=w13_input_scale,
        w2_scale=w2_scale,
        w2_input_scale=w2_input_scale,
    )
    layer.w2_input_scale_inv = 1.0 / w2_input_scale
    layer.output1_scales_gate_scalar = g1_alphas

    if layer.activation.is_gated:
        layer.output1_scales_scalar = g1_alphas * layer.w2_input_scale_inv
    else:
        layer.output1_scales_scalar = (
            torch.ones_like(g1_alphas) * layer.w2_input_scale_inv
        )
    layer.output2_scales_scalar = g2_alphas


def apply_fi_trtllm_fp8_per_tensor_moe(
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
    from vllm.model_executor.models.llama4 import Llama4MoE

    # Added to the layer by: register_scales_for_trtllm_fp8_per_tensor_moe
    assert (
        hasattr(layer, "output1_scales_scalar")
        and hasattr(layer, "output1_scales_gate_scalar")
        and hasattr(layer, "output2_scales_scalar")
    )

    if layer.routing_method_type == RoutingMethodType.Llama4:
        assert (
            not layer.renormalize
            and layer.custom_routing_function == Llama4MoE.custom_routing_function
        ), (
            "FusedMoE flashinfer kernels with Llama4 routing method are only "
            "supported for Llama4"
        )
    else:
        assert layer.custom_routing_function is None, (
            "Custom routing function is only supported for Llama4"
        )
    activation_type = activation_to_flashinfer_int(layer.activation)

    return torch.ops.vllm.fi_trtllm_fp8_per_tensor_moe(
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
        routing_method_type=layer.routing_method_type,
        activation_type=activation_type,
    )


def prepare_fp8_moe_layer_for_fi(
    layer: torch.nn.Module,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_input_scale: torch.Tensor | None,
    w2_scale: torch.Tensor,
    w2_input_scale: torch.Tensor | None,
    is_trtllm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert FP8 MoE weights to FlashInfer kernel format.

    Note that for trtllm we update the model state dict
    with the scale format needed for these kernels.

    Note that for per-tensor, we update the layer's
    intermediate size if the weights needed padding.
    """

    assert hasattr(layer.moe_config, "is_act_and_mul")
    block_quant = (
        hasattr(layer, "weight_block_size") and layer.weight_block_size is not None
    )

    # Some FI MoE kernels require internal alignment of 16
    # for the gate-up proj. Pad the weights to respect this.
    is_gated = layer.activation.is_gated
    if not block_quant:
        min_alignment = 16 if is_gated else 128
        w13, w2, new_intermediate = align_fp8_moe_weights_for_fi(
            w13,
            w2,
            layer.moe_config.is_act_and_mul,
            min_alignment,
        )
        layer.intermediate_size_per_partition = new_intermediate

    # FI kernels require W31 layout rather than W13.
    if layer.moe_config.is_act_and_mul:
        w13 = swap_w13_to_w31(w13)
        if block_quant:
            w13_scale = swap_w13_to_w31(w13_scale)

    # FI TRT-LLM FP8 per-tensor MoE kernel requires weight shuffle
    # and registration of alpha scales. Note that we do not register
    # as nn.Parameters since they are not needed for weight-reloading.
    if is_trtllm and not block_quant:
        assert w13_input_scale is not None
        assert w2_input_scale is not None

        rotate_weights_for_fi_trtllm_fp8_per_tensor_moe(w13, w2, is_gated)
        register_scales_for_trtllm_fp8_per_tensor_moe(
            layer,
            w13_scale=w13_scale,
            w13_input_scale=w13_input_scale,
            w2_scale=w2_scale,
            w2_input_scale=w2_input_scale,
        )

    # Clamp block scales to avoid NaN from the FlashInfer CUTLASS kernel.
    # Some FP8 models have near-zero block scales (~1e-23) for dead/unused
    # experts. The CUTLASS kernel doesn't handle these correctly on Hopper
    # (SM 9.0), producing NaN instead of near-zero output. Clamping to a
    # small minimum prevents this without affecting model accuracy since
    # these experts' effective weights are already zero.
    if block_quant:
        _FI_CUTLASS_MIN_BLOCK_SCALE = 1e-10
        w13_scale.clamp_(min=_FI_CUTLASS_MIN_BLOCK_SCALE)
        w2_scale.clamp_(min=_FI_CUTLASS_MIN_BLOCK_SCALE)

    return w13, w2, w13_scale
