# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import RoutingMethodType
from vllm.utils.torch_utils import direct_register_custom_op

#
# Methods used by the oracle for kernel selection.
#


def _supports_current_device() -> bool:
    """Supports only Blackwell-family GPUs."""
    p = current_platform
    # Add check flashinfer trtllm is available
    return p.is_cuda() and p.is_device_capability_family(100)


def _supports_no_act_and_mul() -> bool:
    """Does not support non-gated MoE (i.e. Nanotron-Mini)."""
    return False


def _supports_quant_scheme(
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
) -> bool:
    """Supports Fp8 per-tensor and Fp8 block."""
    SUPPORTED_W_A = [
        (kFp8Static128BlockSym, kFp8Dynamic128Sym),
        (kFp8StaticTensorSym, kFp8StaticTensorSym),
    ]
    return (weight_key, activation_key) in SUPPORTED_W_A


def _supports_activation(activation: str) -> bool:
    """Supports silu activation only."""
    return activation in ["silu"]


def _supports_routing_method(
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
    routing_method: RoutingMethodType,
) -> bool:
    """Monolithic kernels need to express router support."""
    if (weight_key, activation_key) == (kFp8Static128BlockSym, kFp8Dynamic128Sym):
        # NOTE(rob): potentially allow others here. This is a conservative list.
        return routing_method in [
            RoutingMethodType.DeepSeekV3,
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        ]
    elif (weight_key, activation_key) == (kFp8StaticTensorSym, kFp8StaticTensorSym):
        # NOTE(rob): kernel requires Llama4.
        return routing_method == RoutingMethodType.Llama4

    else:
        raise ValueError("Unsupported quantization scheme.")


def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
    """Supports TRTLLM Kernel does not support EPLB."""
    return not moe_parallel_config.enable_eplb


def is_supported_config_trtllm(
    moe_config: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
    activation_format: mk.FusedMoEActivationFormat,
) -> tuple[bool, str | None]:
    """
    This method mirrors mk.FusedMoEPermuteExpertsUnpermute.is_supported_config
    """

    def _make_reason(reason: str) -> str:
        return f"kernel does not support {reason}"

    if not _supports_current_device():
        return False, _make_reason("current device")
    elif not (moe_config.is_act_and_mul or _supports_no_act_and_mul()):
        return False, _make_reason("no act_and_mul MLP layer")
    elif not _supports_activation(moe_config.activation):
        return False, _make_reason(f"{moe_config.activation} activation")
    elif not _supports_quant_scheme(weight_key, activation_key):
        return False, _make_reason("quantization scheme")
    elif not _supports_parallel_config(moe_config.moe_parallel_config):
        return False, _make_reason("parallel config")
    elif not _supports_routing_method(
        weight_key, activation_key, moe_config.routing_method
    ):
        return False, _make_reason("routing method")
    elif activation_format != mk.FusedMoEActivationFormat.Standard:
        return False, _make_reason("activation format")

    return True, None


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
    routing_method_type: int = int(RoutingMethodType.DeepSeekV3),
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
