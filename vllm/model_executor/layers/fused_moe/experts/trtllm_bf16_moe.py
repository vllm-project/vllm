# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    fi_moe_largest_bucket,
    trtllm_moe_pack_topk_ids_weights,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe


class TrtLlmBf16ExpertsBase:
    """
    BF16 unquantized TRTLLM-Gen MoE kernels. Shared base for modular and
    monolithic interfaces.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        self.routing_method_type = moe_config.routing_method
        self.topk = moe_config.experts_per_token
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.hidden_dim = moe_config.hidden_dim
        self.local_num_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank

        self.moe_config = moe_config
        self.quant_config = quant_config

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        """Supports only Blackwell-family GPUs."""
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(100)
            and has_flashinfer_trtllm_fused_moe()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        """BF16 kernels support non-gated MoE via RELU2_NO_MUL."""
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Supports only unquantized inputs."""
        return weight_key is None and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        """Supports SiLU (gated) and RELU^2 (non-gated) activations."""
        return activation in [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL]

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return True

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True


class TrtLlmBf16ExpertsModular(TrtLlmBf16ExpertsBase, mk.FusedMoEExpertsModular):
    """
    BF16 unquantized TRTLLM-Gen MoE kernels. Supports modular interface.
    """

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.use_ag_rs_all2all_kernels
            and not moe_parallel_config.enable_eplb
        )

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        """Override to handle 4D BlockMajorK weights (E, K/bk, Mn, bk)."""
        if w1.dim() == 4:
            E = w1.shape[0]
            N = w1.shape[2]
            K = a1.size(-1)
            M = a1.size(0) if a1.dim() == 2 else a1.size(1)
            topk = topk_ids.size(1)
            return E, M, N, K, topk
        return super().moe_problem_size(a1, w1, w2, topk_ids)

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # The workspaces for this implementation are managed by FlashInfer.
        workspace1 = (0,)
        workspace2 = (0,)
        output = (M, K)

        return workspace1, workspace2, output

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        import flashinfer
        from flashinfer.fused_moe import WeightLayout

        # Pack topk ids and weights into format expected by the TRTLLM kernel.
        packed_topk_ids = trtllm_moe_pack_topk_ids_weights(topk_ids, topk_weights)

        result = flashinfer.fused_moe.trtllm_bf16_routed_moe(
            topk_ids=packed_topk_ids,
            hidden_states=hidden_states,
            gemm1_weights=w1,
            gemm2_weights=w2,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=None,
            topk_group=None,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=None,
            routing_method_type=1,  # not used
            use_shuffled_weight=True,
            weight_layout=WeightLayout.BlockMajorK,
            do_finalize=True,
            activation_type=activation_to_flashinfer_int(activation),
        )
        # FlashInfer's BF16 routed wrapper does not expose an output= argument.
        output.copy_(result[0] if isinstance(result, list) else result)


class TrtLlmBf16ExpertsMonolithic(TrtLlmBf16ExpertsBase, mk.FusedMoEExpertsMonolithic):
    """
    BF16 unquantized TRTLLM-Gen MoE kernels. Supports monolithic interface.
    """

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        """Monolithic kernel supports no-all2all and AG/RS paths."""
        return (
            not moe_parallel_config.use_all2all_kernels
            or moe_parallel_config.use_ag_rs_all2all_kernels
        ) and not moe_parallel_config.enable_eplb

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in [
            RoutingMethodType.DeepSeekV3,
            RoutingMethodType.Llama4,
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
            RoutingMethodType.SigmoidRenorm,
            RoutingMethodType.Sigmoid,
        ]

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
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        import flashinfer

        assert activation in [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL]

        return flashinfer.fused_moe.trtllm_bf16_moe(
            routing_logits=router_logits,
            routing_bias=e_score_correction_bias,
            hidden_states=hidden_states,
            gemm1_weights=w1,
            gemm2_weights=w2,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=num_expert_group,
            topk_group=topk_group,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=routed_scaling_factor,
            routing_method_type=self.routing_method_type,
            activation_type=activation_to_flashinfer_int(activation),
            tune_max_num_tokens=fi_moe_largest_bucket(self.moe_config),
        )
