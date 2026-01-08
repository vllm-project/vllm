# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer MoE utilities.

This module contains MoE-specific utilities that depend on fused_moe layers.
Pure utility functions have been unified into vllm.utils.flashinfer.

For backwards compatibility, commonly used symbols are re-exported from
vllm.utils.flashinfer.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_prepare_finalize import (
    create_flashinfer_prepare_finalize,
)

# Re-export unified utilities from vllm.utils.flashinfer for backwards
# compatibility
from vllm.utils.flashinfer import (
    FlashinferMoeBackend,
    calculate_tile_tokens_dim,
    get_flashinfer_moe_backend,
    get_moe_scaling_factors,
    is_flashinfer_supporting_global_sf,
    rotate_flashinfer_fp8_moe_weights,
    swap_w13_to_w31,
)
from vllm.utils.math_utils import round_up

# Backwards compatibility alias for renamed function
rotate_weights_for_fi_trtllm_fp8_per_tensor_moe = rotate_flashinfer_fp8_moe_weights

logger = init_logger(__name__)


def register_scales_for_trtllm_fp8_per_tensor_moe(
    layer: torch.nn.Module,
    w13_scale: torch.Tensor,
    w13_input_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_input_scale: torch.Tensor,
) -> None:
    """Register necessary scales for FlashInfer TRTLLM FP8 MoE kernel"""
    g1_alphas, g2_alphas = make_fp8_moe_alpha_scales_for_fi(
        w13_scale=w13_scale,
        w13_input_scale=w13_input_scale,
        w2_scale=w2_scale,
        w2_input_scale=w2_input_scale,
    )
    layer.w2_input_scale_inv = 1.0 / w2_input_scale
    layer.output1_scales_gate_scalar = g1_alphas
    layer.output1_scales_scalar = g1_alphas * layer.w2_input_scale_inv
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

    # Added to the layer by: register_scales_for_trtllm_fp8_per_tensor_moe
    assert (
        hasattr(layer, "output1_scales_scalar")
        and hasattr(layer, "output1_scales_gate_scalar")
        and hasattr(layer, "output2_scales_scalar")
    )

    is_llama4 = layer.custom_routing_function == Llama4MoE.custom_routing_function
    assert is_llama4, "FusedMoE flashinfer kernels are only supported for Llama4"
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
        routing_method_type=RoutingMethodType.Llama4,
    )


def make_fp8_moe_alpha_scales_for_fi(
    w13_scale: torch.Tensor,
    w13_input_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_input_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    g1_alphas = (w13_scale * w13_input_scale).squeeze()
    g2_alphas = (w2_scale * w2_input_scale).squeeze()

    return g1_alphas, g2_alphas


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



def align_fp8_moe_weights_for_fi(
    w13: torch.Tensor, w2: torch.Tensor, is_act_and_mul: bool
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad intermediate size so FlashInfer kernels' alignment constraints hold.

    Some FlashInfer FP8 MoE kernels require the (gated) intermediate size
    used for GEMM to be divisible by a small alignment value. When this is
    not satisfied (e.g. with certain tensor-parallel sizes), we pad the
    gate/up and down projection weights along the intermediate dim.
    """

    # Current local intermediate size (per partition) is the K dimension of
    # the down projection.
    num_experts, hidden_size, intermediate = w2.shape

    min_alignment = 16
    padded_intermediate = round_up(intermediate, min_alignment)

    if padded_intermediate == intermediate:
        return w13, w2, intermediate

    logger.info_once(
        "Padding intermediate size from %d to %d for up/down projection weights.",
        intermediate,
        padded_intermediate,
        scope="local",
    )

    up_mult = 2 if is_act_and_mul else 1
    padded_gate_up_dim = up_mult * padded_intermediate

    # Pad w13 and w2 along its intermediate dimension.
    padded_w13 = w13.new_zeros((num_experts, padded_gate_up_dim, hidden_size))
    padded_w13[:, : w13.shape[1], :] = w13

    padded_w2 = w2.new_zeros((num_experts, hidden_size, padded_intermediate))
    padded_w2[:, :, :intermediate] = w2

    return padded_w13, padded_w2, padded_intermediate


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
    """
    Convert Fp8 MoE weights to flashinfer kernel format

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
    if not block_quant:
        w13, w2, new_intermediate = align_fp8_moe_weights_for_fi(
            w13,
            w2,
            layer.moe_config.is_act_and_mul,
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

        rotate_flashinfer_fp8_moe_weights(w13, w2)
        register_scales_for_trtllm_fp8_per_tensor_moe(
            layer,
            w13_scale=w13_scale,
            w13_input_scale=w13_input_scale,
            w2_scale=w2_scale,
            w2_input_scale=w2_input_scale,
        )

    return w13, w2, w13_scale


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
        moe_parallel_config=layer.moe_parallel_config,
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


__all__ = [
    # Re-exported from vllm.utils.flashinfer
    "FlashinferMoeBackend",
    "calculate_tile_tokens_dim",
    "get_flashinfer_moe_backend",
    "get_moe_scaling_factors",
    "is_flashinfer_supporting_global_sf",
    "rotate_flashinfer_fp8_moe_weights",
    "rotate_weights_for_fi_trtllm_fp8_per_tensor_moe",  # backwards compat alias
    "swap_w13_to_w31",
    # MoE-specific functions (depend on fused_moe)
    "apply_fi_trtllm_fp8_per_tensor_moe",
    "build_flashinfer_fp8_cutlass_moe_prepare_finalize",
    "flashinfer_cutlass_moe_fp8",
    "register_moe_scaling_factors",
    "select_cutlass_fp8_gemm_impl",
    "register_scales_for_trtllm_fp8_per_tensor_moe",
    "prepare_fp8_moe_layer_for_fi",
    "align_fp8_moe_weights_for_fi",
    "make_fp8_moe_alpha_scales_for_fi",
]
