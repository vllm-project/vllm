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
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kMxfp8Dynamic,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer


class TrtLlmMxfp4ExpertsBase:
    """
    MXFP4 TRTLLM-Gen MoE kernels. Shared base for modular and monolithic.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        # NOTE: FusedMoEExperts.__init__ is called by the concrete subclass
        # (Monolithic/Modular) via MRO, not here, to avoid mypy issues with
        # multiple inheritance. This matches the NvFP4 expert pattern.
        self.moe_config = moe_config
        self.quant_config = quant_config

        self.routing_method_type = moe_config.routing_method
        self.topk = moe_config.experts_per_token
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.hidden_dim = moe_config.hidden_dim
        self.local_num_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank

        # MXFP4-specific TRTLLM parameters
        device = torch.accelerator.current_device_index()
        self.gemm1_alpha = torch.tensor(
            [1.702] * self.local_num_experts,
            dtype=torch.float32,
            device=device,
        )
        self.gemm1_beta = torch.tensor(
            [1.0] * self.local_num_experts,
            dtype=torch.float32,
            device=device,
        )
        self.gemm1_clamp_limit = torch.tensor(
            [7.0] * self.local_num_experts,
            dtype=torch.float32,
            device=device,
        )

        from vllm.config import get_current_vllm_config

        self.max_capture_size = (
            get_current_vllm_config().compilation_config.max_cudagraph_capture_size
        )

        # P1-5 fix: use public quant_dtype property instead of private _a1
        self.use_mxfp8_input = quant_config.quant_dtype == "mxfp8"

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return p.is_cuda() and p.is_device_capability_family(100) and has_flashinfer()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, None),
            (kMxfp4Static, kMxfp8Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SWIGLUOAI

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def supports_chunking(self) -> bool:
        return False

    def supports_expert_map(self) -> bool:
        return False

    @property
    def expects_unquantized_inputs(self) -> bool:
        # Expert handles MXFP8 quantization internally if needed
        return True


class TrtLlmMxfp4ExpertsMonolithic(
    TrtLlmMxfp4ExpertsBase, mk.FusedMoEExpertsMonolithic
):
    """
    Monolithic version of the MXFP4 TRTLLM kernel (router + experts).
    Wraps flashinfer.trtllm_fp4_block_scale_moe().
    """

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.enable_eplb
            and moe_parallel_config.dp_size <= 1
        )

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in [
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        ]

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        # Kernel converts to bfloat16 internally
        return True

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
        from flashinfer import trtllm_fp4_block_scale_moe

        # Handle input quantization
        if self.use_mxfp8_input:
            from flashinfer import mxfp8_quantize

            x_quant, x_scale = mxfp8_quantize(
                hidden_states,
                is_sf_swizzled_layout=False,
                alignment=256,
            )
            x_scale = x_scale.view(torch.float8_e4m3fn).reshape(
                *hidden_states.shape[:-1], -1
            )
        else:
            assert hidden_states.dtype == torch.bfloat16
            x_quant = hidden_states
            x_scale = None

        return trtllm_fp4_block_scale_moe(
            routing_logits=router_logits.to(torch.bfloat16),
            routing_bias=None,
            hidden_states=x_quant,
            hidden_states_scale=x_scale,
            gemm1_weights=w1,
            gemm1_weights_scale=self.w1_scale,
            gemm1_bias=self.w1_bias,
            gemm1_alpha=self.gemm1_alpha,
            gemm1_beta=self.gemm1_beta,
            gemm1_clamp_limit=self.gemm1_clamp_limit,
            gemm2_weights=w2,
            gemm2_weights_scale=self.w2_scale,
            gemm2_bias=self.w2_bias,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=None,
            topk_group=None,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=None,
            routing_method_type=self.routing_method_type,
            do_finalize=True,
            tune_max_num_tokens=max(self.max_capture_size, 1),
        )[0]


class TrtLlmMxfp4ExpertsModular(TrtLlmMxfp4ExpertsBase, mk.FusedMoEExpertsModular):
    """
    Modular version of the MXFP4 TRTLLM kernel (just the experts).
    Wraps flashinfer.trtllm_fp4_block_scale_routed_moe().
    Moved from trtllm_moe.py.
    """

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    def supports_expert_map(self) -> bool:
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
        # The workspaces for this implementation are managed by flashinfer.
        workspace1 = (0,)
        workspace2 = (0,)
        output = (M, K)
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
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        topk = topk_ids.size(-1)
        local_num_experts = w1.size(0)
        intermediate_size = w2.size(1)
        local_expert_offset = self.moe_config.ep_rank * local_num_experts

        # Handle input quantization
        if self.use_mxfp8_input:
            from flashinfer import mxfp8_quantize

            x_quant, x_scale = mxfp8_quantize(
                hidden_states,
                is_sf_swizzled_layout=False,
                alignment=256,
            )
            x_scale = x_scale.view(torch.float8_e4m3fn).reshape(
                *hidden_states.shape[:-1], -1
            )
        else:
            assert hidden_states.dtype == torch.bfloat16
            x_quant = hidden_states
            x_scale = None

        packed_tensor = (topk_ids.to(torch.int32) << 16) | topk_weights.to(
            torch.bfloat16
        ).view(torch.int16)

        assert self.w1_scale is not None
        assert self.w2_scale is not None
        kwargs = {
            "topk_ids": packed_tensor,
            "routing_bias": None,
            "hidden_states": x_quant,
            "hidden_states_scale": x_scale,
            "gemm1_weights": w1,
            "gemm1_weights_scale": self.w1_scale,
            "gemm1_bias": self.w1_bias,
            "gemm1_alpha": self.gemm1_alpha,
            "gemm1_beta": self.gemm1_beta,
            "gemm1_clamp_limit": self.gemm1_clamp_limit,
            "gemm2_weights": w2,
            "gemm2_weights_scale": self.w2_scale,
            "gemm2_bias": self.w2_bias,
            "output1_scale_scalar": None,
            "output1_scale_gate_scalar": None,
            "output2_scale_scalar": None,
            "num_experts": global_num_experts,
            "top_k": topk,
            "n_group": None,
            "topk_group": None,
            "intermediate_size": intermediate_size,
            "local_expert_offset": local_expert_offset,
            "local_num_experts": local_num_experts,
            "routed_scaling_factor": None,
            "routing_method_type": self.routing_method_type,
            "do_finalize": True,
            "output": output,
            "tune_max_num_tokens": max(self.max_capture_size, 1),
        }

        from flashinfer import trtllm_fp4_block_scale_routed_moe

        from vllm.utils.flashinfer import autotune

        with autotune(False):
            # Enable autotune when,
            # https://github.com/flashinfer-ai/flashinfer/issues/2023 is
            # resolved.
            trtllm_fp4_block_scale_routed_moe(**kwargs)

        return output
