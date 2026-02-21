# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exllama-based MoE experts for 4-bit GPTQ weight format."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up


class ExllamaExperts(mk.FusedMoEPermuteExpertsUnpermute):
    """
    Modular MoE experts using the exllama 4-bit GPTQ kernel.

    Weights are in exllama format: [E, K/8, N] int32 (packed 4-bit).
    Scales are [E, K/G, N] fp16.
    Zero-points are [E, K/G, N/8] int32 (packed 4-bit).
    """

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_rocm() or current_platform.is_cuda()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        raise NotImplementedError(
            "ExllamaExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [MoEActivation.SILU, MoEActivation.GELU]

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return not moe_parallel_config.use_fi_all2allv_kernels

    def supports_chunking(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @staticmethod
    def _select_block_size_m(num_tokens: int, topk: int, E: int) -> int:
        """Select block size in the M dimension for the exllama kernel.

        Prefill (num_tokens > 1): pick a larger block to amortize weight
        dequantization across rows.  This requires pre-permuting activations
        so each expert's tokens are contiguous.
        Decode (num_tokens <= 1): block_size_m=1, kernel handles
        scatter/gather via sorted_token_ids directly.
        """
        if num_tokens > 1:
            avg = num_tokens * topk / E
            return 4 if avg >= 4 else 2 if avg >= 2 else 1
        return 1

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        # Exllama format: w1 is [E, K/8, N], w2 is [E, K_inter/8, hidden]
        assert w1.dim() == 3 and w2.dim() == 3
        E = w1.size(0)
        N = w1.size(2)  # intermediate_size * 2 (gate + up)
        K = a1.size(-1)  # hidden_size
        assert a1.dim() == 2
        M = a1.size(0)
        topk = topk_ids.size(1)
        return E, M, N, K, topk

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
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        # Account for alignment padding from moe_align_block_size:
        # each expert may need up to (block_size_m - 1) padding rows.
        block_size_m = self._select_block_size_m(M, topk, global_num_experts)
        num_slots = round_up(
            M * topk + global_num_experts * (block_size_m - 1),
            block_size_m,
        )
        workspace1 = (num_slots, max(activation_out_dim, K))
        workspace2 = (num_slots, max(N, K))
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
        from vllm._custom_ops import fused_moe_exllama_gemm

        assert w1.dtype == torch.int32, "ExllamaExperts requires int32 weights"
        assert hidden_states.dim() == 2
        assert hidden_states.is_contiguous()

        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        if global_num_experts == -1:
            global_num_experts = E

        activation_out_dim = self.adjust_N_for_activation(N, activation)

        block_size_m = self._select_block_size_m(num_tokens, top_k_num, E)

        sorted_token_ids, expert_ids, _ = moe_align_block_size(
            topk_ids,
            block_size_m,
            global_num_experts,
            expert_map,
            ignore_invalid_experts=True,
        )

        tw = topk_weights if topk_weights is not None else hidden_states.new_empty(0)

        num_slots = sorted_token_ids.size(0)

        if block_size_m > 1:
            source_rows = (sorted_token_ids // top_k_num).clamp(max=num_tokens - 1)
            # Permute activations into workspace13 so each expert's
            # tokens are contiguous.  workspace13 is reused for act_out
            # after GEMM1 has consumed gemm1_in.
            gemm1_in = _resize_cache(workspace13, (num_slots, K))
            torch.index_select(hidden_states, 0, source_rows.long(), out=gemm1_in)
        else:
            gemm1_in = hidden_states

        gemm1_out = _resize_cache(workspace2, (num_slots, N))
        act_out = _resize_cache(workspace13, (num_slots, activation_out_dim))
        gemm2_out = _resize_cache(workspace2, (num_slots, K))

        # GEMM 1
        fused_moe_exllama_gemm(
            gemm1_in,
            w1,
            self.quant_config.w1_zp,
            self.w1_scale,
            gemm1_out,
            sorted_token_ids,
            expert_ids,
            tw,
            top_k_num,
            False,
            block_size_m,
        )

        # Activation (elementwise, order-independent)
        apply_moe_activation(activation, act_out, gemm1_out)

        # GEMM 2 -- always apply router weights here.
        # Decode (scattered kernel): weights applied inside the kernel.
        # Prefill (contiguous kernel): kernel uses router_weight=1.0
        # (single scalar, can't vary per row), so we apply weights
        # explicitly right after the kernel call.
        fused_moe_exllama_gemm(
            act_out,
            w2,
            self.quant_config.w2_zp,
            self.w2_scale,
            gemm2_out,
            sorted_token_ids,
            expert_ids,
            tw,
            1,
            not apply_router_weight_on_input,
            block_size_m,
        )

        # Reduce
        if block_size_m > 1:
            num_valid = num_tokens * top_k_num
            clamped_tids = sorted_token_ids.clamp(max=num_valid - 1)

            # Apply per-slot router weights (contiguous kernel can only
            # apply a single scalar, so we do it here).
            if not apply_router_weight_on_input and topk_weights is not None:
                flat_w = topk_weights.view(-1)
                gemm2_out *= flat_w[clamped_tids].unsqueeze(1)

            # Scatter-back + sum.  Use mask arithmetic (not nonzero())
            # to stay compatible with CUDA graph capture.
            # TODO: replace with moe_unpermute once supported on ROCm.
            orig_tokens = clamped_tids // top_k_num
            valid_mask = (sorted_token_ids < num_valid).unsqueeze(1)

            output.zero_()
            output.scatter_add_(
                0,
                orig_tokens.unsqueeze(1).expand(-1, K),
                gemm2_out * valid_mask,
            )
        else:
            ops.moe_sum(gemm2_out.view(num_tokens, top_k_num, K), output)
