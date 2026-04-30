# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import (
    get_dp_group,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
)
from vllm.utils.deep_gemm import (
    is_deep_gemm_mega_moe_supported,
)


class DeepGemmMegaExperts(mk.FusedMoEExpertsModular):
    """DeepGemm-based fused MoE expert implementation."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        import deep_gemm

        super().__init__(moe_config=moe_config, quant_config=quant_config)
        # activation quantization is handled by apply
        assert quant_config.quant_dtype is None
        assert quant_config.weight_quant_dtype == "mxfp4"
        assert not quant_config.per_act_token_quant
        assert not quant_config.per_out_ch_quant

        # print(f"DG MOE_CONFIG {moe_config}")

        # Allocate symmetric memory buffer
        # NOTES: requires PyTorch >= 2.9
        self.buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            get_dp_group().device_group,
            moe_config.num_experts,
            moe_config.max_num_tokens,
            moe_config.experts_per_token,
            moe_config.hidden_dim,
            moe_config.intermediate_size_per_partition,
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return is_deep_gemm_mega_moe_supported()

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
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.SWIGLUSTEP,
            # TODO(bnell): hack temporarily add swigluoai for testing
            MoEActivation.SWIGLUOAI,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # Mega_moe handles all-to-all internally via symmetric memory.
        # Reject any config that would use an external all-to-all backend.
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
            or moe_parallel_config.use_deepep_ht_kernels
            or moe_parallel_config.use_deepep_ll_kernels
            or moe_parallel_config.use_mori_kernels
            or moe_parallel_config.use_nixl_ep_kernels
            or moe_parallel_config.use_batched_activation_format
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
        # Activation FP8 quantization (UE8M0 packed scales, gran_k=32) is
        # done inside apply(). Setting this to True causes the prepare step
        # to pass defer_input_quant=True, skipping standard quantization.
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
        workspace1 = (0,)
        workspace2 = (0,)
        output = (M, K)
        return (workspace1, workspace2, output)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # noqa: B027
        pass

    @staticmethod
    def convert_weights_for_mega_moe(
        w13_weight: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert gpt_oss checkpoint weights into mega_moe kernel format.

        Handles three transformations:
        1. De-interleave w13 gate/up rows from checkpoint interleaved layout
        2. Convert uint8 UE8M0 scales to packed int32 via
           transform_sf_into_required_layout
        3. Apply transform_weights_for_mega_moe for final kernel layout

        Returns (w13_weight, w13_scale, w2_weight, w2_scale) in mega_moe
        format.
        """
        import deep_gemm

        def xform_layout(w: torch.Tensor, w_sf: torch.Tensor) -> torch.Tensor:
            num_experts, w_mn, w_packed_k = w.shape
            w_sf = _upcast_e8m0_to_fp32(w_sf)
            return deep_gemm.transform_sf_into_required_layout(
                w_sf, w_mn, w_packed_k * 2, (1, 32), num_experts
            )

        # 0. Cast uint8 packed FP4 weights to int8 (kPackedFP4 = torch::kInt8).
        w13_weight = w13_weight.view(torch.int8)
        w2_weight = w2_weight.view(torch.int8)

        # 1. De-interleave w13 gate/up rows.
        #    Checkpoint: [gate[0], up[0], gate[1], up[1], ...]
        #    mega_moe expects: [gate[0..N-1], up[0..N-1]]
        gate_w, up_w = w13_weight[:, ::2, :], w13_weight[:, 1::2, :]
        w13_weight = torch.cat([gate_w, up_w], dim=1).contiguous()

        gate_s, up_s = w13_scale[:, ::2, :], w13_scale[:, 1::2, :]
        w13_scale = torch.cat([gate_s, up_s], dim=1).contiguous()

        # 2. Convert uint8 UE8M0 scales to float32, then pack via
        #    transform_sf_into_required_layout (TMA alignment + int32 packing).
        #    k parameter must be the unpacked dimension (K, not K//2).
        w13_scale = xform_layout(w13_weight, w13_scale)
        w2_scale = xform_layout(w2_weight, w2_scale)

        # 3. Transform weights for mega_moe layout:
        #    L1: re-interleave gate/up in groups of 8, transpose SF for UTCCP
        #    L2: transpose SF for UTCCP
        (
            (w13_weight, w13_scale),
            (w2_weight, w2_scale),
        ) = deep_gemm.transform_weights_for_mega_moe(
            (w13_weight, w13_scale),
            (w2_weight, w2_scale),
        )

        return w13_weight, w13_scale, w2_weight, w2_scale

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
    ) -> None:
        import deep_gemm

        num_tokens = hidden_states.shape[0]

        # Quantize activations to FP8 with UE8M0 packed scale factors.
        # expects_unquantized_inputs=True so hidden_states is BF16.
        assert a1q_scale is None
        tokens_fp8, tokens_sf = deep_gemm.utils.per_token_cast_to_fp8(
            hidden_states, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
        )

        # Copy into the symmetric buffer.
        self.buffer.x[:num_tokens].copy_(tokens_fp8)
        self.buffer.x_sf[:num_tokens].copy_(tokens_sf)
        self.buffer.topk_idx[:num_tokens].copy_(topk_ids)
        self.buffer.topk_weights[:num_tokens].copy_(topk_weights)

        # Run the fused mega MoE kernel
        deep_gemm.fp8_fp4_mega_moe(
            output,
            (w1, self.w1_scale),
            (w2, self.w2_scale),
            self.buffer,
        )
