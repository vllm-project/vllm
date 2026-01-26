# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)


class FlashinferMoeBackend(Enum):
    TENSORRT_LLM = "TensorRT-LLM"
    CUTLASS = "CUTLASS"
    CUTEDSL = "CUTEDSL"


def swap_w13_to_w31(x: torch.Tensor) -> torch.Tensor:
    return (
        x.reshape(-1, 2, x.shape[-2] // 2, x.shape[-1]).flip(dims=[1]).reshape(x.shape)
    )


def rotate_weights_for_fi_trtllm_fp8_per_tensor_moe(
    gemm1_weights: torch.Tensor, gemm2_weights: torch.Tensor
):
    """Shuffle weights for for FI TRT-LLM Format"""
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

    if layer.routing_method_type == RoutingMethodType.Llama4:
        assert (
            not layer.renormalize
            and layer.custom_routing_function == Llama4MoE.custom_routing_function
        ), (
            "FusedMoE flashinfer kernels with Llama4 routing method are only "
            "supported for Llama4"
        )

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
            and not current_platform.is_device_capability_family(100)
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
        FlashinferMoeBackend.CUTEDSL,
    )
    return backend in backends_supporting_global_sf


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

        rotate_weights_for_fi_trtllm_fp8_per_tensor_moe(w13, w2)
        register_scales_for_trtllm_fp8_per_tensor_moe(
            layer,
            w13_scale=w13_scale,
            w13_input_scale=w13_input_scale,
            w2_scale=w2_scale,
            w2_input_scale=w2_input_scale,
        )

    return w13, w2, w13_scale
