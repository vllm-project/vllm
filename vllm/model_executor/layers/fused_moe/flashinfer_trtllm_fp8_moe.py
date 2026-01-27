# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import flashinfer
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
)
from vllm.v1.engine.utils import current_platform


class FlashInferTrtLlmFp8Experts(mk.FusedMoEPermuteExpertsUnpermute):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)

        self.routing_method_type = moe_config.routing_method
        self.topk = moe_config.experts_per_token
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.hidden_dim = moe_config.hidden_dim
        self.local_num_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank

        # Make additional scales for per-tensor interface.
        if self.quant_config.is_per_tensor:
            w1_scale = self.quant_config.w1_scale
            assert w1_scale is not None
            a1_scale = self.quant_config.a1_scale
            assert a1_scale is not None
            w2_scale = self.quant_config.w2_scale
            assert w2_scale is not None
            a2_scale = self.quant_config.a2_scale
            assert a2_scale is not None

            self._g1_alphas = (w1_scale * a1_scale).squeeze()
            self._g2_alphas = (w2_scale * a2_scale).squeeze()
            self._g1_scale_c = self._g1_alphas / self.quant_config.a2_scale

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        """Supports only Blackwell-family GPUs."""
        p = current_platform
        # Add check flashinfer trtllm is available
        return p.is_cuda() and p.is_device_capability_family(100)

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        """Does not support non-gated MoE (i.e. Nanotron-Mini)."""
        return False

    @staticmethod
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

    @staticmethod
    def _supports_activation(activation: str) -> bool:
        """Supports silu activation only."""
        return activation in ["silu"]

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Monolithic kernels need to express router support."""
        # NOTE(rob): potentially allow others here. This is a conservative list.
        if (weight_key, activation_key) == (kFp8Static128BlockSym, kFp8Dynamic128Sym):
            return routing_method in [
                RoutingMethodType.DeepSeekV3,
                RoutingMethodType.Renormalize,
                RoutingMethodType.RenormalizeNaive,
            ]
        elif (weight_key, activation_key) == (kFp8StaticTensorSym, kFp8StaticTensorSym):
            return routing_method == RoutingMethodType.Llama4

        else:
            raise ValueError("Unsupported quantization scheme.")

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        """TRTLLMGenKernel is monolithic, so it only supports TP or naive DP/EP."""
        return not moe_parallel_config.use_all2all_kernels or (
            moe_parallel_config.use_naive_all2all_kernels
            and not moe_parallel_config.enable_eplb
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

    def _apply_per_block_monolithic(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        assert not apply_router_weight_on_input
        assert activation == "silu"
        assert (
            e_score_correction_bias is None
            or e_score_correction_bias.dtype == hidden_states.dtype
        )

        if self.routing_method_type == RoutingMethodType.DeepSeekV3:
            router_logits = router_logits.to(torch.float32)

        assert self.topk <= global_num_experts
        assert self.topk <= 10
        assert global_num_experts % 4 == 0
        assert self.quant_config.block_shape == [128, 128]
        # Routing kernel expects #experts <= #threads 512
        assert global_num_experts <= 512

        # Kernel requires transposed hidden state scales
        # TODO: fuse into the quant kernel.
        assert a1q_scale is not None
        a1q_scale_t = a1q_scale.t().contiguous()

        return flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
            routing_logits=router_logits,
            routing_bias=e_score_correction_bias,
            hidden_states=hidden_states,
            hidden_states_scale=a1q_scale_t,
            gemm1_weights=w1,
            gemm1_weights_scale=self.quant_config.w1_scale,
            gemm2_weights=w2,
            gemm2_weights_scale=self.quant_config.w2_scale,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=num_expert_group,
            topk_group=(topk_group or 0),
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=routed_scaling_factor,
            routing_method_type=self.routing_method_type,
        )

    def _apply_per_tensor_monolithic(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        assert self.routing_method_type == RoutingMethodType.Llama4
        assert apply_router_weight_on_input

        # Should only have Llama4 routing here.
        assert routed_scaling_factor is None
        assert e_score_correction_bias is None
        assert num_expert_group is None

        return flashinfer.fused_moe.trtllm_fp8_per_tensor_scale_moe(
            routing_logits=router_logits,
            routing_bias=e_score_correction_bias,
            hidden_states=hidden_states,
            gemm1_weights=w1,
            output1_scales_scalar=self._g1_scale_c,
            output1_scales_gate_scalar=self._g1_alphas,
            gemm2_weights=w2,
            output2_scales_scalar=self._g2_alphas,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=num_expert_group,
            topk_group=topk_group or 0,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=routed_scaling_factor,
            use_routing_scales_on_input=apply_router_weight_on_input,
            routing_method_type=self.routing_method_type,
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
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        if self.quant_config.block_shape is not None:
            return self._apply_per_block_monolithic(
                hidden_states,
                w1,
                w2,
                router_logits,
                activation,
                global_num_experts,
                expert_map,
                a1q_scale,
                apply_router_weight_on_input,
                num_expert_group=num_expert_group,
                e_score_correction_bias=e_score_correction_bias,
                routed_scaling_factor=routed_scaling_factor,
            )
        elif self.quant_config.is_per_tensor:
            return self._apply_per_tensor_monolithic(
                hidden_states,
                w1,
                w2,
                router_logits,
                activation,
                global_num_experts,
                expert_map,
                a1q_scale,
                apply_router_weight_on_input,
                num_expert_group=num_expert_group,
                e_score_correction_bias=e_score_correction_bias,
                routed_scaling_factor=routed_scaling_factor,
            )
        else:
            raise NotImplementedError(
                "Only per-block and per-tensor quantization are supported in "
                f"{self.__class__.__name__}."
            )
