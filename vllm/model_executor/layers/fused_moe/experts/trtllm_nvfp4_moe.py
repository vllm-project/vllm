# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
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
from vllm.model_executor.layers.fused_moe.utils import trtllm_moe_pack_topk_ids_weights
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe

logger = init_logger(__name__)


class TrtLlmNvFp4ExpertsBase:
    """
    NvFp4 TRTLLM-Gen MoE kernels. Supports modular and monolithic interface.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        self.moe_config = moe_config
        self.quant_config = quant_config

        self.routing_method_type = self.moe_config.routing_method
        self.topk = moe_config.experts_per_token
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.hidden_dim = moe_config.hidden_dim
        self.hidden_dim_unpadded = (
            moe_config.hidden_dim_unpadded or moe_config.hidden_dim
        )
        self.local_num_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank

        assert self.quant_config.g1_alphas is not None
        assert self.quant_config.a2_gscale is not None
        if moe_config.is_act_and_mul:
            # g1_alpha_s = a13_scale * w13_scale_2
            # a2_gscale = (1 / a2_scale)
            # g1_scale_c = a13_scale * w13_scale_2 / a2_scale
            self.g1_scale_c = self.quant_config.g1_alphas * self.quant_config.a2_gscale
        else:
            self.g1_scale_c = self.quant_config.a2_gscale.clone()

        # Fall back to moe_config.swiglu_* when quant_config doesn't carry them
        # (ModelOpt NVFP4 checkpoints store these on moe_config, not quant_config).
        device = torch.accelerator.current_device_index()

        def _per_expert(val: float | None) -> torch.Tensor | None:
            if val is None:
                return None
            return torch.full(
                (self.local_num_experts,),
                float(val),
                dtype=torch.float32,
                device=device,
            )

        clamp = quant_config.gemm1_clamp_limit
        if clamp is None:
            clamp = getattr(moe_config, "swiglu_limit", None)
        alpha = quant_config.gemm1_alpha
        if alpha is None:
            alpha = getattr(moe_config, "swiglu_alpha", None)
        beta = quant_config.gemm1_beta
        if beta is None:
            beta = getattr(moe_config, "swiglu_beta", None)

        if moe_config.is_act_and_mul:
            self.gemm1_clamp_limit = _per_expert(clamp)
            self.gemm1_alpha = _per_expert(alpha)
            self.gemm1_beta = _per_expert(beta)
        else:
            self.gemm1_clamp_limit = None
            self.gemm1_alpha = None
            self.gemm1_beta = None

        logger.debug_once(
            "activation=%s, gemm1_alpha=%s, gemm1_beta=%s, gemm1_clamp_limit=%s",
            moe_config.activation,
            alpha,
            beta,
            clamp,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight_scale_2.data.mul_(layer.w13_input_scale)
        layer.w2_weight_scale_2.data.mul_(layer.w2_input_scale)
        # Recompute g1_scale_c since g1_alphas was just fused in-place.
        # Register as a layer parameter so EPLB rearranges it alongside
        # other expert weights.
        assert self.quant_config.g1_alphas is not None
        assert self.quant_config.a2_gscale is not None
        if self.moe_config.is_act_and_mul:
            g1_scale_c = self.quant_config.g1_alphas * self.quant_config.a2_gscale
        else:
            g1_scale_c = self.quant_config.a2_gscale.clone()
        layer.register_parameter(
            "g1_scale_c",
            torch.nn.Parameter(g1_scale_c, requires_grad=False),
        )
        self.g1_scale_c = layer.g1_scale_c

        # Pre-fold the per-expert g1_alphas (= output1_scale_gate_scalar)
        # division so the TRTLLM kernel receives the raw-GEMM-space clamp
        # directly. g1_alphas is set once here in process_weights_after_loading
        # (via the in-place mul above) and never changes again, so this is a
        # static, per-expert constant. Register on the layer so EPLB
        # rearranges it alongside the other expert tensors.
        if self.gemm1_clamp_limit is not None:
            gemm1_clamp_limit = self.gemm1_clamp_limit / self.quant_config.g1_alphas
            layer.register_parameter(
                "gemm1_clamp_limit",
                torch.nn.Parameter(gemm1_clamp_limit, requires_grad=False),
            )
            self.gemm1_clamp_limit = layer.gemm1_clamp_limit

        # beta shifts the raw GEMM1 accumulator, so fold by g1_alphas like the
        # clamp limit. alpha is applied to the dequantized gate, so it stays
        # raw. Register both on the layer so EPLB rearranges them with the
        # other per-expert tensors.
        if self.gemm1_beta is not None:
            gemm1_beta = self.gemm1_beta / self.quant_config.g1_alphas
            layer.register_parameter(
                "gemm1_beta",
                torch.nn.Parameter(gemm1_beta, requires_grad=False),
            )
            self.gemm1_beta = layer.gemm1_beta

        if self.gemm1_alpha is not None:
            layer.register_parameter(
                "gemm1_alpha",
                torch.nn.Parameter(self.gemm1_alpha, requires_grad=False),
            )
            self.gemm1_alpha = layer.gemm1_alpha

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
        """Supports non-gated MoE (i.e. Nemotron-Nano)."""
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Supports Nvfp4 quantization."""
        SUPPORTED_W_A = [
            (kNvfp4Static, kNvfp4Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        """Supports SiLU, RELU^2 non-gated, GELU, and clamped SwiGLU-OAI."""
        return activation in [
            MoEActivation.SILU,
            MoEActivation.RELU2_NO_MUL,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        ]

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        # Weights are zero-padded to 256-alignment at load time and the MoE
        # runner pads activations via _maybe_pad_hidden_states, so any
        # hidden_dim is accepted.
        # NOTE: non-256-aligned dims will trigger a warning log and may
        # cause performance degradation due to activation slicing.
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def _get_chunk_size(self) -> int:
        MAX_GRID_Y = 65535
        MAX_TILE_TOKENS_DIM = 128

        def _calc_max_supported_tokens(top_k: int, num_experts: int) -> int:
            """Calculates the max number of supported tokens, so the CUDA grid.Y limit
            won't be reached.
            Based on getMaxNumCtasInBatchDim function in flashinfer's TRTLLM MoE runner:
            https://github.com/flashinfer-ai/flashinfer/blob/719ee23fd82cb220d51ad118ca60198718f6c9d1/include/flashinfer/trtllm/fused_moe/runner.h#L97
            Which given numTokens, topK, numExperts, tileTokensDim calculates maxNumCtas
            which is used as the CUDA grid.Y dimension, which we want to
            be <= MAX_GRID_Y. Solving for numTokens gives the formula below.
            """
            return (
                num_experts + (MAX_GRID_Y - num_experts + 1) * MAX_TILE_TOKENS_DIM - 1
            ) // top_k

        # Using 305k or more causes IMA error in the kernel, so limit to 300k.
        return min(
            300000, _calc_max_supported_tokens(self.topk, self.moe_config.num_experts)
        )


class TrtLlmNvFp4ExpertsModular(TrtLlmNvFp4ExpertsBase, mk.FusedMoEExpertsModular):
    """
    Modular version of the implementation (just the experts).
    """

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        """The modular implementation supports all parallel configs."""
        return True

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
        # The workspaces for this implementation are managed by flashinfer.
        workspace1 = (0,)
        workspace2 = (0,)

        # Hidden states are Nvfp4, packed into int8 dtype, so we
        # need to multiply K by 2 to get the output shape right.
        assert self.hidden_dim == K * 2
        output = (M, self.hidden_dim)

        return (workspace1, workspace2, output)

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def _invoke_kernel(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        a1q_scale: torch.Tensor,
    ):
        import flashinfer

        assert self.quant_config.w1_scale is not None
        assert self.quant_config.w2_scale is not None

        # Pack topk ids and weights into format expected by the kernel.
        packed_tensor = trtllm_moe_pack_topk_ids_weights(topk_ids, topk_weights)
        output1_scale_gate_scalar = self.quant_config.g1_alphas

        # Invoke kernel.
        flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(
            topk_ids=packed_tensor,
            routing_bias=None,
            hidden_states=hidden_states,
            hidden_states_scale=a1q_scale.view(torch.float8_e4m3fn).reshape(
                *hidden_states.shape[:-1], -1
            ),
            gemm1_weights=w1,
            gemm1_weights_scale=self.quant_config.w1_scale.view(torch.float8_e4m3fn),
            gemm1_bias=None,
            gemm1_alpha=self.gemm1_alpha,
            gemm1_beta=self.gemm1_beta,
            gemm1_clamp_limit=self.gemm1_clamp_limit,
            gemm2_weights=w2,
            gemm2_weights_scale=self.quant_config.w2_scale.view(torch.float8_e4m3fn),
            gemm2_bias=None,
            output1_scale_scalar=self.g1_scale_c,
            output1_scale_gate_scalar=output1_scale_gate_scalar,
            output2_scale_scalar=self.quant_config.g2_alphas,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=0,
            topk_group=0,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=None,
            routing_method_type=1,  # not used
            do_finalize=True,
            activation_type=activation_to_flashinfer_int(activation),
            output=output,
            tune_max_num_tokens=min(
                self.moe_config.tune_max_num_tokens, self._get_chunk_size()
            ),
        )

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
        assert self._supports_activation(activation)
        assert a1q_scale is not None

        M = hidden_states.shape[0]
        chunk_size = self._get_chunk_size()

        if chunk_size >= M:
            self._invoke_kernel(
                output,
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation,
                global_num_experts,
                a1q_scale,
            )
        else:
            for start in range(0, M, chunk_size):
                end = min(start + chunk_size, M)
                self._invoke_kernel(
                    output[start:end],
                    hidden_states[start:end],
                    w1,
                    w2,
                    topk_weights[start:end],
                    topk_ids[start:end],
                    activation,
                    global_num_experts,
                    a1q_scale[start:end],
                )


class TrtLlmNvFp4ExpertsMonolithic(
    TrtLlmNvFp4ExpertsBase, mk.FusedMoEExpertsMonolithic
):
    """
    Monolithic version of the kernel (router + experts).
    """

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        """The modular implementation should be used for the Dp/Ep or EPLB case."""
        return (
            not moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.enable_eplb
        )

    @staticmethod
    def _supports_routing_method(
        routing_method_type: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # NOTE(rob): this is a conservative list.
        return routing_method_type in [
            RoutingMethodType.DeepSeekV3,
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
            RoutingMethodType.Llama4,
            RoutingMethodType.SigmoidRenorm,
            RoutingMethodType.Sigmoid,
            RoutingMethodType.MiniMax2,
            RoutingMethodType.Simulated,
        ]

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return router_logits_dtype in [torch.bfloat16, torch.float32]

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
        import flashinfer

        assert self._supports_activation(activation)
        assert a1q_scale is not None
        assert self.quant_config.w1_scale is not None
        assert self.quant_config.w2_scale is not None
        assert (
            apply_router_weight_on_input
            and self.routing_method_type == RoutingMethodType.Llama4
        ) or (
            not apply_router_weight_on_input
            and self.routing_method_type != RoutingMethodType.Llama4
        )

        output1_scale_gate_scalar = self.quant_config.g1_alphas

        # Invoke kernel.
        # NOTE: Activation padding and output
        # truncation are handled by the MoE runner's
        return flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
            routing_logits=router_logits,
            routing_bias=e_score_correction_bias,
            hidden_states=hidden_states,
            hidden_states_scale=a1q_scale.view(torch.float8_e4m3fn).reshape(
                *hidden_states.shape[:-1], -1
            ),
            gemm1_weights=w1,
            gemm1_weights_scale=self.quant_config.w1_scale.view(torch.float8_e4m3fn),
            gemm1_bias=None,
            gemm1_alpha=self.gemm1_alpha,
            gemm1_beta=self.gemm1_beta,
            gemm1_clamp_limit=self.gemm1_clamp_limit,
            gemm2_weights=w2,
            gemm2_weights_scale=self.quant_config.w2_scale.view(torch.float8_e4m3fn),
            gemm2_bias=None,
            output1_scale_scalar=self.g1_scale_c,
            output1_scale_gate_scalar=output1_scale_gate_scalar,
            output2_scale_scalar=self.quant_config.g2_alphas,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=(num_expert_group or 0),
            topk_group=(topk_group or 0),
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=routed_scaling_factor,
            routing_method_type=self.routing_method_type,
            do_finalize=True,
            activation_type=activation_to_flashinfer_int(activation),
            tune_max_num_tokens=self.moe_config.tune_max_num_tokens,
        )[0]
