# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE,
    activation_to_flashinfer_int,
    quantize_nvfp4_per_token_input,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4DynamicToken,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_cute_dsl_fused_moe_nvfp4,
    has_flashinfer_cutedsl_moe_nvfp4,
    has_flashinfer_cutedsl_moe_nvfp4_per_token,
)


class FlashInferCuteDSLExperts(mk.FusedMoEExpertsModular):
    """
    CuteDSL NvFP4 MoE experts using the FlashInfer functional API.

    Uses Standard activation format (non-batched). The kernel handles
    routing, expert computation, and reduction internally.
    Supports expert parallelism natively.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        per_token_activation: bool = False,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
        )
        assert quant_config.quant_dtype == "nvfp4", (
            "Only nvfp4 quantization is currently supported."
        )
        self.out_dtype = moe_config.in_dtype
        self.hidden_dim = moe_config.hidden_dim
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.topk = moe_config.experts_per_token
        self.local_num_experts = moe_config.num_local_experts
        self.global_num_experts = moe_config.num_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        self.local_expert_offset = self.ep_rank * self.local_num_experts
        self.per_token_activation = per_token_activation
        self.per_token_global_scale = None
        if per_token_activation:
            assert quant_config.a2_gscale is not None
            self.per_token_global_scale = quant_config.a2_gscale.new_full(
                (1,),
                NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE,
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight_scale_2.data.mul_(layer.w13_input_scale)
        layer.w2_weight_scale_2.data.mul_(layer.w2_input_scale)

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(100)
            and has_flashinfer_cutedsl_moe_nvfp4()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kNvfp4Static, kNvfp4Dynamic),
            (kNvfp4Static, kNvfp4DynamicToken),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        if (weight_key, activation_key) == (
            kNvfp4Static,
            kNvfp4DynamicToken,
        ) and not has_flashinfer_cutedsl_moe_nvfp4_per_token():
            return False, "installed FlashInfer lacks per-token CuTe-DSL NVFP4 MoE"
        return mk.FusedMoEExperts.is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.RELU2_NO_MUL)

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @property
    def expects_unquantized_inputs(self) -> bool:
        return self.per_token_activation

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
        workspace1 = (0,)
        workspace2 = (0,)
        expected_hidden_dim = K if self.per_token_activation else K * 2
        assert self.hidden_dim == expected_hidden_dim
        output = (M, self.hidden_dim)
        return (workspace1, workspace2, output)

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
        assert self.quant_dtype == "nvfp4"
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        if self.per_token_activation:
            hidden_states, block_scale, per_token_scale = (
                quantize_nvfp4_per_token_input(hidden_states)
            )
            fc2_input_scale = self.per_token_global_scale
        else:
            assert a1q_scale is not None
            block_scale = a1q_scale
            per_token_scale = None
            fc2_input_scale = self.a2_gscale

        assert block_scale is not None
        assert fc2_input_scale is not None

        # a1q_scale is (M, K//16) float8_e4m3fn from fp4_quantize.
        # The functional API expects x_sf with trailing dim: (M, K//16, 1).
        x_sf = block_scale.unsqueeze(-1)

        per_token_kwargs = (
            {"per_token_scale": per_token_scale} if self.per_token_activation else {}
        )
        flashinfer_cute_dsl_fused_moe_nvfp4(
            x=hidden_states,
            x_sf=x_sf,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights.float(),
            w1_weight=w1,
            w1_weight_sf=self.w1_scale,
            w1_alpha=self.g1_alphas,
            fc2_input_scale=fc2_input_scale,
            w2_weight=w2,
            w2_weight_sf=self.w2_scale,
            w2_alpha=self.g2_alphas,
            num_experts=self.global_num_experts,
            top_k=self.topk,
            num_local_experts=self.local_num_experts,
            local_expert_offset=self.local_expert_offset,
            moe_output=output,
            activation_type=activation_to_flashinfer_int(activation),
            **per_token_kwargs,
        )
