# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer CuTe-DSL fused MoE experts for Hopper (SM90), unquantized.

`cute_dsl_fused_moe_bf16` runs the MoE layer as routing index maps, a
gather-fused GEMM1 with the activation in its epilogue and a GEMM2 with the
router-scaled scatter-reduce finalize; the token permutation is never
materialized. Tactics come from the FlashInfer autotuner, which the kernel
warmup runs before CUDA graph capture. Weights are repacked once at load time
into the kernel's 32-row up/gate interleave (the `FLASHINFER_CUTEDSL` branch
of `convert_to_unquantized_kernel_format`).
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_cute_dsl_fused_moe_bf16,
    has_flashinfer_cutedsl_moe_bf16,
)

# FlashInfer `ActivationType` member per vLLM activation; SITU is `Swiglu`
# with `situ_beta`.
_KERNEL_ACTIVATION: dict[MoEActivation, str] = {
    MoEActivation.SILU: "Swiglu",
    MoEActivation.SWIGLUOAI: "Swiglu",
    MoEActivation.SWIGLUOAI_UNINTERLEAVE: "Swiglu",
    MoEActivation.SITU: "Swiglu",
    MoEActivation.GELU_TANH: "GegluTanh",
    MoEActivation.RELU2_NO_MUL: "Relu2",
}


def flashinfer_cutedsl_sm90_local_expert_offset(
    moe_parallel_config: FusedMoEParallelConfig, num_experts: int
) -> int:
    """Global id of this rank's first expert under the linear expert placement
    (the first `num_experts % ep_size` ranks hold one expert more)."""
    base, remainder = divmod(num_experts, moe_parallel_config.ep_size)
    rank = moe_parallel_config.ep_rank
    return rank * base + min(rank, remainder)


class FlashInferCuteDSLSm90Experts(mk.FusedMoEExpertsModular):
    """bf16/fp16 MoE experts on FlashInfer's SM90 CuTe-DSL kernels.

    The kernel applies the top-k weights and reduces internally. Expert
    parallelism uses the kernel's contiguous local shard
    (`local_expert_offset`), i.e. the linear expert placement. Expert biases
    are not supported.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.quant_dtype is None, (
            "FlashInferCuteDSLSm90Experts supports unquantized MoE only"
        )
        assert quant_config.weight_quant_dtype is None, (
            "FlashInferCuteDSLSm90Experts supports unquantized MoE only"
        )
        self.topk = moe_config.experts_per_token
        self.local_num_experts = moe_config.num_local_experts
        self.local_expert_offset = flashinfer_cutedsl_sm90_local_expert_offset(
            moe_config.moe_parallel_config, moe_config.num_experts
        )

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not supported:
            return False, reason
        if moe_config.has_bias:
            return False, "kernel does not support expert biases"
        if (
            moe_config.activation == MoEActivation.SITU
            and moe_config.activation_situ_beta is None
        ):
            return False, "kernel does not support SITU without activation_situ_beta"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        expert_map = getattr(layer, "expert_map", None)
        if expert_map is None:
            return
        expected = torch.full_like(expert_map, -1)
        expected[
            self.local_expert_offset : self.local_expert_offset + self.local_num_experts
        ] = torch.arange(
            self.local_num_experts, dtype=expert_map.dtype, device=expert_map.device
        )
        if not torch.equal(expert_map, expected):
            raise ValueError(
                "FlashInferCuteDSLSm90Experts requires the linear expert "
                "placement; use --expert-placement-strategy linear or another "
                "MoE backend."
            )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(90)
            and has_flashinfer_cutedsl_moe_bf16()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (None, None)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in _KERNEL_ACTIVATION

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        # The kernel reads global top-k ids: TP, or EP without a dispatching
        # all2all (except all-gather/reduce-scatter).
        return (
            not moe_parallel_config.use_all2all_kernels
            or moe_parallel_config.use_ag_rs_all2all_kernels
        )

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        # 16-byte token rows.
        return hidden_dim % 8 == 0

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

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
        # FlashInfer allocates the intermediate buffers.
        return (0,), (0,), (M, K)

    def _activation_kwargs(self, activation: MoEActivation) -> dict[str, float]:
        """Kernel constants matching `apply_moe_activation`; unset ones are
        omitted so the kernel keeps its defaults."""
        config = self.activation_config
        params: dict[str, float | None] = {}
        if activation == MoEActivation.SILU:
            params = {"swiglu_limit": config.clamp_limit}
        elif activation == MoEActivation.SWIGLUOAI:
            # The fixed constants of `swigluoai_and_mul`.
            params = {"swiglu_alpha": 1.702, "swiglu_beta": 1.0, "swiglu_limit": 7.0}
        elif activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
            params = {
                "swiglu_alpha": config.alpha,
                "swiglu_beta": config.beta,
                "swiglu_limit": config.clamp_limit,
            }
        elif activation == MoEActivation.SITU:
            params = {
                "situ_beta": config.activation_situ_beta,
                "situ_linear_beta": config.activation_situ_linear_beta,
            }
        return {k: v for k, v in params.items() if v is not None}

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
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        from flashinfer.fused_moe.core import ActivationType

        assert a1q_scale is None and a2_scale is None
        if apply_router_weight_on_input:
            # The weights are already applied to the input.
            token_final_scales = torch.ones_like(topk_weights, dtype=torch.float32)
        else:
            token_final_scales = topk_weights.to(torch.float32)

        flashinfer_cute_dsl_fused_moe_bf16(
            x=hidden_states,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=token_final_scales,
            w1_weight=w1,
            w2_weight=w2,
            num_experts=global_num_experts,
            top_k=self.topk,
            num_local_experts=self.local_num_experts,
            local_expert_offset=self.local_expert_offset,
            moe_output=output,
            activation_type=int(
                getattr(ActivationType, _KERNEL_ACTIVATION[activation])
            ),
            **self._activation_kwargs(activation),
        )
