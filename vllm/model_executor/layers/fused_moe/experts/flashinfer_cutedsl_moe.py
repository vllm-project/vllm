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
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    flashinfer_cutedsl_weight_interleave,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kMxfp8Dynamic,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_cute_dsl_fused_moe,
    has_flashinfer_cutedsl_moe,
)


class FlashInferCuteDSLExpertsBase(mk.FusedMoEExperts):
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
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
        )
        assert quant_config.quant_dtype in ("nvfp4", "mxfp4", "mxfp8"), (
            "Only nvfp4, mxfp4, and mxfp8 quantization are currently supported."
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
        self.gemm1_alpha = quant_config.gemm1_alpha
        self.gemm1_beta = quant_config.gemm1_beta
        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit
        self.situ_beta = moe_config.activation_situ_beta
        self.situ_linear_beta = moe_config.activation_situ_linear_beta
        self._weight_interleave = flashinfer_cutedsl_weight_interleave()
        half_layout = quant_config.quant_dtype in ("mxfp4", "mxfp8")
        self._w1_bias, self._w2_bias = self._prepare_biases(
            quant_config.w1_bias,
            quant_config.w2_bias,
            half_layout=half_layout,
            activation=moe_config.activation,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.quant_dtype == "nvfp4":
            layer.w13_weight_scale_2.data.mul_(layer.w13_input_scale)
            layer.w2_weight_scale_2.data.mul_(layer.w2_input_scale)
        half_layout = self.quant_dtype in ("mxfp4", "mxfp8")
        self._w1_bias, self._w2_bias = self._prepare_biases(
            self.w1_bias,
            self.w2_bias,
            half_layout=half_layout,
            activation=self.moe_config.activation,
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(100)
            and has_flashinfer_cutedsl_moe()
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
            (kMxfp4Static, kMxfp8Dynamic),
        ]
        cap = current_platform.get_device_capability()
        if cap is not None and cap.to_int() == 107:
            # SM107 only has the W4A4 kernel.
            SUPPORTED_W_A = SUPPORTED_W_A[:1]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            MoEActivation.RELU2_NO_MUL,
            MoEActivation.SITU,
        )

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

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
        workspace1 = (0,)
        workspace2 = (0,)
        if self.quant_dtype == "nvfp4":
            # K is packed (K//2 for uint8), so output uses hidden_dim in nvfp4.
            assert self.hidden_dim == K * 2
        output = (M, self.hidden_dim)
        return (workspace1, workspace2, output)

    @staticmethod
    def _prepare_biases(
        w1_bias: torch.Tensor | None,
        w2_bias: torch.Tensor | None,
        *,
        half_layout: bool,
        activation: MoEActivation,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """The kernel splits w1_bias into [up | gate]; MXFP4 convert already
        emits that layout, else reorder into it."""
        from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
            reorder_w13_to_w31_for_flashinfer_cutedsl,
        )

        if w1_bias is not None:
            w1_bias = w1_bias.to(torch.float32)
            if not half_layout:
                w1_bias = reorder_w13_to_w31_for_flashinfer_cutedsl(
                    activation, w1_bias, w1_bias
                )[0]
            w1_bias = w1_bias.contiguous()
        if w2_bias is not None:
            w2_bias = w2_bias.to(torch.float32).contiguous()
        return w1_bias, w2_bias

    def _fused_moe(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        activation: MoEActivation,
        a1q_scale: torch.Tensor | None,
        token_selected_experts: torch.Tensor | None = None,
        token_final_scales: torch.Tensor | None = None,
        router_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.quant_dtype in ("nvfp4", "mxfp4", "mxfp8")
        assert a1q_scale is not None
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        # a1q_scale is (M, K//16) float8_e4m3fn from fp4_quantize.
        # The functional API expects x_sf with trailing dim: (M, K//16, 1).
        # fp8 has no trailing dim, so we don't need to unsqueeze.
        x_sf = a1q_scale.unsqueeze(-1) if self.quant_dtype == "nvfp4" else a1q_scale

        # The kernel defaults swiglu_{alpha,beta,limit} to the plain-SwiGLU
        # values, so only forward the ones the model actually sets.
        swiglu_params: dict[str, float | None] = {}
        if activation == MoEActivation.SILU:
            swiglu_params = {"swiglu_limit": self.gemm1_clamp_limit}
        elif activation in (
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        ):
            swiglu_params = {
                "swiglu_alpha": self.gemm1_alpha,
                "swiglu_beta": self.gemm1_beta,
                "swiglu_limit": self.gemm1_clamp_limit,
            }
        elif activation == MoEActivation.SITU:
            if self.situ_beta is None:
                raise ValueError(
                    "SITU activation requires moe_config.activation_situ_beta"
                )
            swiglu_params = {
                "situ_beta": self.situ_beta,
                "situ_linear_beta": self.situ_linear_beta,
            }
        swiglu_kwargs = {k: v for k, v in swiglu_params.items() if v is not None}

        # w2's hidden dim may be mx-alignment padded wider than output.
        kernel_dim = w2.size(1)
        if output.shape[-1] == kernel_dim:
            kernel_output = output
        else:
            kernel_output = torch.empty(
                *output.shape[:-1],
                kernel_dim,
                dtype=output.dtype,
                device=output.device,
            )

        flashinfer_cute_dsl_fused_moe(
            x=hidden_states,
            x_sf=x_sf,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            w1_weight=w1,
            w1_weight_sf=self.w1_scale,
            w1_alpha=self.g1_alphas,
            fc2_input_scale=self.a2_gscale,
            w2_weight=w2,
            w2_weight_sf=self.w2_scale,
            w2_alpha=self.g2_alphas,
            num_experts=self.global_num_experts,
            top_k=self.topk,
            num_local_experts=self.local_num_experts,
            local_expert_offset=self.local_expert_offset,
            output_dtype=self.out_dtype,
            moe_output=kernel_output,
            activation_type=activation_to_flashinfer_int(
                MoEActivation.SILU if activation == MoEActivation.SITU else activation
            ),
            w1_bias=self._w1_bias,
            w2_bias=self._w2_bias,
            quant_mode="w4a4" if self.quant_dtype == "nvfp4" else "w4a8",
            weight_interleave=self._weight_interleave,
            router_logits=router_logits,
            **swiglu_kwargs,
        )
        if kernel_output is not output:
            output.copy_(kernel_output[..., : output.shape[-1]])
        return output


class FlashInferCuteDSLExperts(FlashInferCuteDSLExpertsBase, mk.FusedMoEExpertsModular):
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
        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(torch.int32)
        if topk_weights.dtype != torch.float32:
            topk_weights = topk_weights.float()
        self._fused_moe(
            output=output,
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            activation=activation,
            a1q_scale=a1q_scale,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
        )


class FlashInferCuteDSLExpertsMonolithic(
    FlashInferCuteDSLExpertsBase, mk.FusedMoEExpertsMonolithic
):
    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return not moe_parallel_config.use_ep

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # Fused kernel does TopK -> softmax over selected; RenormalizeNaive matches.
        return routing_method in (
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        )

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return router_logits_dtype in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        )

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
        output = torch.empty(
            *hidden_states.shape[:-1],
            self.hidden_dim,
            dtype=self.out_dtype,
            device=hidden_states.device,
        )
        return self._fused_moe(
            output=output,
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            activation=activation,
            a1q_scale=a1q_scale,
            router_logits=router_logits.contiguous(),
        )
