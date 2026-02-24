# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import flashinfer
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform


class TrtLlmFp8Experts(mk.FusedMoEExpertsMonolithic):
    """
    Fp8 TRTLLM-Gen MoE kernels. Supports monolithic interface.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)

        if moe_config.moe_parallel_config.use_ep and quant_config.is_per_tensor:
            raise NotImplementedError(
                "EP parallelism is not supported with TRTLLM"
                "per-tensor FP8 quantization."
            )

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
            self._g1_scale_c = (
                self._g1_alphas / self.quant_config.a2_scale
                if moe_config.is_act_and_mul
                else torch.ones_like(self._g1_alphas) / self.quant_config.a2_scale
            )

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
        """Does not support non-gated MoE (i.e. Nanotron-3-Nano)."""
        return True

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
    def _supports_activation(activation: MoEActivation) -> bool:
        """Supports only SiLU and RELU^2 non-gated activation."""
        return activation in [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL]

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Monolithic kernels need to express router support."""
        # NOTE(dbari): TopK routing could also be enabled, but need to validate models
        # NOTE(dbari): Default is not implemented and should not be enabled until it is
        if (weight_key, activation_key) == (kFp8Static128BlockSym, kFp8Dynamic128Sym):
            # NOTE(rob): potentially allow others here. This is a conservative list.
            return routing_method in [
                RoutingMethodType.DeepSeekV3,
                RoutingMethodType.Renormalize,
                RoutingMethodType.RenormalizeNaive,
            ]
        elif (weight_key, activation_key) == (kFp8StaticTensorSym, kFp8StaticTensorSym):
            # NOTE(dbari): as above, potentially allow others here.
            return routing_method in [
                RoutingMethodType.DeepSeekV3,
                RoutingMethodType.Llama4,
                RoutingMethodType.Renormalize,
                RoutingMethodType.RenormalizeNaive,
            ]
        else:
            raise ValueError("Unsupported quantization scheme.")

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        """Monolithic kernel so only use with naive DP/EP and TP."""
        return (
            not moe_parallel_config.use_all2all_kernels
            or moe_parallel_config.use_naive_all2all_kernels
        ) and not moe_parallel_config.enable_eplb

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        """
        The FlashInfer TRTLLM FP8 kernel expects bfloat16 router_logits by default.
        Only DeepSeekV3 routing supports float32 router_logits (which is converted
        internally in the kernel).
        """
        if router_logits_dtype == torch.float32:
            # Only DeepSeekV3 routing handles float32 logits
            # https://github.com/flashinfer-ai/flashinfer/issues/2469
            return routing_method == RoutingMethodType.DeepSeekV3
        return True

    def supports_chunking(self) -> bool:
        return False

    def supports_expert_map(self) -> bool:
        return False

    def _apply_per_block(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
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
        assert activation == MoEActivation.SILU

        if e_score_correction_bias is not None:
            e_score_correction_bias = e_score_correction_bias.to(hidden_states.dtype)

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
            n_group=(num_expert_group or 0),
            topk_group=(topk_group or 0),
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=routed_scaling_factor,
            routing_method_type=self.routing_method_type,
            use_shuffled_weight=False,
        )

    def _apply_per_tensor(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
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
        # Confirm supported activation function.
        assert activation in [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL]
        from flashinfer.fused_moe.core import ActivationType

        activation_type = ActivationType(activation_to_flashinfer_int(activation))

        # Confirm Llama-4 routing is proper.
        if self.routing_method_type == RoutingMethodType.Llama4:
            assert apply_router_weight_on_input
        else:
            assert not apply_router_weight_on_input

        # The DeepSeekV3 routing method requires float32 router logits.
        if self.routing_method_type == RoutingMethodType.DeepSeekV3:
            router_logits = router_logits.to(torch.float32)

        out = flashinfer.fused_moe.trtllm_fp8_per_tensor_scale_moe(
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
            n_group=num_expert_group or 0,
            topk_group=topk_group or 0,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=routed_scaling_factor,
            use_routing_scales_on_input=apply_router_weight_on_input,
            routing_method_type=self.routing_method_type,
            activation_type=activation_type,
        )
        return out

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
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
            return self._apply_per_block(
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
                topk_group=topk_group,
            )
        elif self.quant_config.is_per_tensor:
            return self._apply_per_tensor(
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
