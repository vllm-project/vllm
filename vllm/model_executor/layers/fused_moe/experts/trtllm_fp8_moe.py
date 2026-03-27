# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import requests
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
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    FLASHINFER_CUBINS_REPOSITORY,
    has_flashinfer_trtllm_fused_moe,
)

logger = init_logger(__name__)


class TrtLlmFp8ExpertsBase:
    """
    Fp8 TRTLLM-Gen MoE kernels. Shared base for modular and monolithic
    interfaces.
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
            and TrtLlmFp8ExpertsBase._can_load_batched_gemm_enums()
        )

    @staticmethod
    def _can_load_batched_gemm_enums() -> bool:
        try:
            from flashinfer.artifacts import ArtifactPath
            from flashinfer.jit import env as jit_env
        except Exception:
            return False

        header_path = (
            jit_env.FLASHINFER_CUBIN_DIR
            / "flashinfer"
            / "trtllm"
            / "batched_gemm"
            / "trtllmGen_bmm_export"
            / "BatchedGemmEnums.h"
        )
        if header_path.is_file():
            return True

        probe_url = (
            f"{FLASHINFER_CUBINS_REPOSITORY.rstrip('/')}/"
            f"{ArtifactPath.TRTLLM_GEN_BMM}/include/trtllmGen_bmm_export/"
            "BatchedGemmEnums.h"
        )
        try:
            return requests.get(probe_url, timeout=60).status_code == 200
        except Exception:
            return False

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        """Does not support non-gated MoE (i.e. Nanotron-3-Nano)."""
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        """Supports only SiLU and RELU^2 non-gated activation."""
        return activation in [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        """Monolithic kernel so only use with naive DP/EP and TP."""
        return (
            not moe_parallel_config.use_all2all_kernels
            or moe_parallel_config.use_ag_rs_all2all_kernels
        ) and not moe_parallel_config.enable_eplb

    def supports_chunking(self) -> bool:
        return False

    def supports_expert_map(self) -> bool:
        return False


class TrtLlmFp8ExpertsModular(TrtLlmFp8ExpertsBase, mk.FusedMoEExpertsModular):
    """
    Fp8 TRTLLM-Gen MoE kernels. Supports modular interface.
    """

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Supports Fp8 block and MXFP8."""
        SUPPORTED_W_A = [
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
            (kMxfp8Static, kMxfp8Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

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
        from flashinfer.fused_moe import Fp8QuantizationType

        # Pack topk_ids and topk_weights into single tensor
        # Format: (expert_id << 16) | (weight_bf16.view(int16))
        packed_topk_ids = (topk_ids << 16) | topk_weights.to(torch.bfloat16).view(
            torch.int16
        )

        # trtllm_fp8_block_scale_routed_moe does not support autotuning
        # so skip this kernel during dummy run for autotuning.
        import vllm.utils.flashinfer as fi_utils

        if fi_utils._is_fi_autotuning:
            return

        assert a1q_scale is not None

        is_mxfp8 = self.quant_config.block_shape == [1, 32]
        if is_mxfp8:
            fp8_quant_type = Fp8QuantizationType.MxFp8
            use_shuffled_weight = True
            hidden_states_scale = a1q_scale
        else:
            fp8_quant_type = Fp8QuantizationType.DeepSeekFp8
            use_shuffled_weight = False
            hidden_states_scale = a1q_scale.t().contiguous()

        # `trtllm_fp8_block_scale_routed_moe` has a bug and does not write to the
        # output tensor in-place so we need to manually copy the result to the
        # output tensor
        # https://github.com/flashinfer-ai/flashinfer/issues/2703
        result = flashinfer.fused_moe.trtllm_fp8_block_scale_routed_moe(
            topk_ids=packed_topk_ids,
            routing_bias=None,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            gemm1_weights=w1,
            gemm1_weights_scale=self.quant_config.w1_scale,
            gemm2_weights=w2,
            gemm2_weights_scale=self.quant_config.w2_scale,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=None,
            topk_group=None,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=None,
            routing_method_type=1,
            use_shuffled_weight=use_shuffled_weight,
            weight_layout=0,
            fp8_quantization_type=fp8_quant_type,
            # output=output,
        )
        output.copy_(result)


class TrtLlmFp8ExpertsMonolithic(TrtLlmFp8ExpertsBase, mk.FusedMoEExpertsMonolithic):
    """
    Fp8 TRTLLM-Gen MoE kernels. Supports monolithic interface.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)

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
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Supports Fp8 per-tensor, Fp8 block, and MXFP8."""
        SUPPORTED_W_A = [
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
            (kFp8StaticTensorSym, kFp8StaticTensorSym),
            (kMxfp8Static, kMxfp8Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

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

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Monolithic kernels need to express router support.
        Renormalize/RenormalizeNaive are excluded: the monolithic kernel's
        internal routing for these methods produces output uncorrelated
        with the modular kernel's output and with Triton kernel's output
        for Qwen3.5-35B-A3B-FP8.
        See: https://github.com/vllm-project/vllm/issues/37591
        """
        # NOTE(dbari): TopK routing could also be enabled, but need to validate models
        # NOTE(dbari): Default is not implemented and should not be enabled until it is

        if (weight_key, activation_key) in [
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
            (kMxfp8Static, kMxfp8Dynamic),
        ]:
            # NOTE(rob): potentially allow others here. This is a conservative list.
            return routing_method in [
                RoutingMethodType.DeepSeekV3,
            ]
        elif (weight_key, activation_key) == (kFp8StaticTensorSym, kFp8StaticTensorSym):
            # NOTE(dbari): as above, potentially allow others here.
            return routing_method in [
                RoutingMethodType.DeepSeekV3,
                RoutingMethodType.Llama4,
            ]
        else:
            raise ValueError("Unsupported quantization scheme.")

    def _apply_block_scale(
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
        from flashinfer.fused_moe import Fp8QuantizationType

        assert not apply_router_weight_on_input
        assert activation == MoEActivation.SILU
        assert self.topk <= global_num_experts
        assert self.topk <= 10
        assert global_num_experts % 4 == 0
        assert self.quant_config.block_shape in [[128, 128], [1, 32]]
        # Kernel expects #experts <= #threads 512
        assert global_num_experts <= 512
        # TODO: fuse into the quant kernel.
        assert a1q_scale is not None

        if self.routing_method_type == RoutingMethodType.DeepSeekV3:
            router_logits = router_logits.to(torch.float32)

        is_mxfp8 = self.quant_config.block_shape == [1, 32]
        if is_mxfp8:
            fp8_quant_type = Fp8QuantizationType.MxFp8
            use_shuffled_weight = True
            hidden_states_scale = a1q_scale
        else:
            fp8_quant_type = Fp8QuantizationType.DeepSeekFp8
            use_shuffled_weight = False
            hidden_states_scale = a1q_scale.t().contiguous()

        return flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
            routing_logits=router_logits,
            routing_bias=e_score_correction_bias,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
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
            use_shuffled_weight=use_shuffled_weight,
            fp8_quantization_type=fp8_quant_type,
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
        # Delay import for non-CUDA.
        import flashinfer

        # Confirm supported activation function.
        assert activation in [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL]

        activation_type = activation_to_flashinfer_int(activation)

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
            return self._apply_block_scale(
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
                "Only per-block, per-tensor, and MXFP8 quantization are "
                f"supported in {self.__class__.__name__}."
            )
