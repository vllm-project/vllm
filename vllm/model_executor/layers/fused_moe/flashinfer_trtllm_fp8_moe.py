# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import flashinfer
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
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


class FlashInferTrtLlmFp8Experts(mk.FusedMoEPermuteExpertsUnpermute):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(quant_config)

        self.moe_config = moe_config
        # TODO: set this via the constructor
        self.routing_method_type = flashinfer.RoutingMethodType.Renormalize
        # self.routing_method_type = flashinfer.RoutingMethodType.Llama4
        # self.routing_method_type = flashinfer.RoutingMethodType.DeepSeekV3

        self.routing_bias = None
        # TODO: to: in_dtype.shape
        self.e_score_correction_bias = None
        self.topk_group = None
        self.num_expert_group = None
        self.routing_scaling_factor = None

        self.topk = moe_config.experts_per_token
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.hidden_dim = moe_config.hidden_dim
        self.local_num_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (
            mk.FusedMoEActivationFormat.Standard,
            mk.FusedMoEActivationFormat.Standard,
        )

    def supports_chunking(self) -> bool:
        return False

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        raise NotImplementedError(
            f"{self.__class__.__name__} only supports the apply_monolithic interface."
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        raise NotImplementedError(
            f"{self.__class__.__name__} only supports the apply_monolithic interface."
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        raise NotImplementedError(
            f"{self.__class__.__name__} only supports the apply_monolithic interface."
        )

    def apply_monolithic(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        assert activation == "silu"
        assert (
            self.e_score_correction_bias is None
            or self.e_score_correction_bias.dtype == hidden_states.dtype
        )

        if self.routing_method_type == RoutingMethodType.DeepSeekV3:
            router_logits = router_logits.to(torch.float32)

        topk_group = self.topk_group if self.topk_group is not None else 0

        assert self.topk <= global_num_experts
        assert self.topk <= 10
        assert global_num_experts % 4 == 0
        assert self.quant_config.block_shape == [128, 128]
        # Routing kernel expects #experts <= #threads 512
        assert global_num_experts <= 512

        a1_q, a1q_scale = per_token_group_quant_fp8(
            hidden_states, self.quant_config.block_shape[1]
        )
        # Kernel requires transposed hidden state scales
        # TODO: can we avoid this transpose in the kernel?
        a1q_scale_t = a1q_scale.t().contiguous()

        return flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
            routing_logits=router_logits,
            routing_bias=self.e_score_correction_bias,
            hidden_states=a1_q,
            hidden_states_scale=a1q_scale_t,
            gemm1_weights=w1,
            gemm1_weights_scale=self.quant_config.w1_scale,
            gemm2_weights=w2,
            gemm2_weights_scale=self.quant_config.w2_scale,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=self.num_expert_group,
            topk_group=topk_group,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=self.routing_scaling_factor,
            tile_tokens_dim=None,
            routing_method_type=self.routing_method_type,
            use_shuffled_weight=False,
        )
