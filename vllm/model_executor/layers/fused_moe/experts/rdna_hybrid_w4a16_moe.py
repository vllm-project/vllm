# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid W4A16 MoE experts for RDNA 3.5 (gfx115x).

Both GEMMs read one copy of the weights in the skinny layout shared with
``RDNAHybridW4A16LinearKernel``:
  - Packed int32 ``[E, N, K // 8]`` with ExLlama shuffle
  - Scales ``[E, N, K // group_size]`` in activation dtype
  - Symmetric (uint4b8) only

Decode (M <= MAX_SKINNY_BATCH_SIZE) runs the ``fused_moe_wvSplitK_int4_gemm``
HIP kernel; prefill runs the Triton ``fused_moe_kernel_gptq_awq`` with
``use_shuffle_w4a16``.
"""

import torch

import vllm._custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt4Static,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl
from vllm.utils.math_utils import round_up
from vllm.utils.platform_utils import num_compute_units

SUPPORTED_GROUP_SIZES = (128,)


def rdna_hybrid_w4a16_moe_kernel_available() -> bool:
    """Whether the hybrid W4A16 MoE HIP kernel is built and enabled here.

    Enabled only on gfx115x, the only targets it has been validated on.
    """
    if not current_platform.is_rocm():
        return False

    from vllm.platforms.rocm import on_gfx115x

    return (
        on_gfx115x()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "fused_moe_wvSplitK_int4_gemm")
    )


class RDNAHybridW4A16Experts(mk.FusedMoEExpertsModular):
    """W4A16 experts: HIP skinny GEMM for decode, Triton for prefill."""

    # Above this many tokens the Triton prefill kernel is used.
    MAX_SKINNY_BATCH_SIZE = 5

    # Alignment block size for moe_align_block_size and the Triton BLOCK_M;
    # the two must match. Padding adds up to E * (BLOCK_M - 1) slots, so for
    # large-E models a small BLOCK_M shrinks the workspace and skips padding
    # blocks.
    TRITON_BLOCK_SIZE_M = 32

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        assert quant_config.use_int4_w4a16, "Supports only int4_w4a16"
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        block_shape = self.quant_config.block_shape
        assert block_shape is not None
        self._group_size = block_shape[1]
        assert self._group_size in SUPPORTED_GROUP_SIZES
        self._cu_count = num_compute_units()
        self._cached_arange: torch.Tensor | None = None
        self._cached_inv_perm_buf: torch.Tensor | None = None
        self._cached_expert_ids_buf: torch.Tensor | None = None

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @staticmethod
    def _supports_current_device() -> bool:
        return rdna_hybrid_w4a16_moe_kernel_available()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return activation_key is None and weight_key == kInt4Static

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.GELU)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # With EP, padding blocks for non-local experts leave their output
        # slots unwritten, and the top-k reduction would read them.
        return not (
            moe_parallel_config.use_ep
            or moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    @classmethod
    def _select_block_size_m(cls, num_tokens: int, topk: int, E: int) -> int:
        """Rows per expert block: 1-4 for the HIP kernel, BLOCK_M for Triton."""
        if num_tokens > cls.MAX_SKINNY_BATCH_SIZE:
            return cls.TRITON_BLOCK_SIZE_M
        if num_tokens > 1:
            avg = num_tokens * topk / E
            return 4 if avg >= 4 else 2 if avg >= 2 else 1
        return 1

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
        block_size_m = self._select_block_size_m(M, topk, global_num_experts)
        num_slots = round_up(
            M * topk + global_num_experts * (block_size_m - 1), block_size_m
        )
        workspace2_rows = num_slots
        if block_size_m > 1 and M <= self.MAX_SKINNY_BATCH_SIZE:
            # Pre-permuted HIP decode gathers GEMM2's output back into slot
            # order after it in workspace2.
            workspace2_rows += M * topk
        workspace1 = (num_slots, max(activation_out_dim, K))
        workspace2 = (workspace2_rows, max(N, K))
        return (workspace1, workspace2, (M, K))

    def _triton_config(self, K: int, M_routed: int) -> dict:
        """Triton tile config, tuned on gfx1151 for the Qwen3.5-A3B shapes
        (gemm1 K=2048, gemm2 K=512)."""
        if K < 1024:
            # Short contraction: wider BN/BK amortize the short K loop.
            return {
                "BLOCK_SIZE_M": self.TRITON_BLOCK_SIZE_M,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "num_warps": 2,
                "num_stages": 1,
            }
        # Long contraction: wider N tiles only pay off once enough rows are
        # routed.
        return {
            "BLOCK_SIZE_M": self.TRITON_BLOCK_SIZE_M,
            "BLOCK_SIZE_N": 64 if M_routed >= 4096 else 16,
            "BLOCK_SIZE_K": self._group_size,
            "GROUP_SIZE_M": 4,
            "num_warps": 2,
            "num_stages": 1,
        }

    def _arange(self, n: int, device: torch.device) -> torch.Tensor:
        if self._cached_arange is None or self._cached_arange.size(0) < n:
            self._cached_arange = torch.arange(n, dtype=torch.int32, device=device)
        return self._cached_arange[:n]

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
        assert self.w1_scale is not None and self.w2_scale is not None
        assert w1.dtype == torch.int32 and w2.dtype == torch.int32
        assert hidden_states.dim() == 2 and hidden_states.is_contiguous()

        E, num_tokens, N, K, top_k = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )
        if global_num_experts == -1:
            global_num_experts = E

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        block_size_m = self._select_block_size_m(num_tokens, top_k, global_num_experts)
        P = num_tokens * top_k
        use_triton = num_tokens > self.MAX_SKINNY_BATCH_SIZE
        scattered = block_size_m == 1
        device = hidden_states.device

        if apply_router_weight_on_input and not use_triton:
            # top_k == 1 here, so weighting the input is exact. The Triton
            # path applies the weight inside the kernel instead.
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)

        num_tokens_post_padded: torch.Tensor | None = None
        if scattered and expert_map is None:
            # Decode: one slot per (token, expert), no alignment needed.
            if (
                self._cached_expert_ids_buf is None
                or self._cached_expert_ids_buf.size(0) < P
            ):
                self._cached_expert_ids_buf = torch.empty(
                    P, dtype=torch.int32, device=device
                )
            expert_ids = self._cached_expert_ids_buf[:P]
            expert_ids.copy_(topk_ids.view(-1))
            sorted_token_ids = self._arange(P, device)
            num_slots = P
            gemm1_in = hidden_states
        else:
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                topk_ids,
                block_size_m,
                global_num_experts,
                expert_map,
                ignore_invalid_experts=True,
            )
            num_slots = sorted_token_ids.size(0)
            if scattered or use_triton:
                # The Triton kernel gathers rows via sorted_token_ids // top_k.
                gemm1_in = hidden_states
            else:
                source_rows = (sorted_token_ids // top_k).clamp(max=num_tokens - 1)
                gemm1_in = _resize_cache(workspace13, (num_slots, K))
                torch.index_select(hidden_states, 0, source_rows.long(), out=gemm1_in)

        gemm1_out = _resize_cache(workspace2, (num_slots, N))
        act_out = _resize_cache(workspace13, (num_slots, activation_out_dim))
        gemm2_out = _resize_cache(workspace2, (num_slots, K))

        if use_triton:
            from vllm.model_executor.layers.fused_moe.fused_moe import (
                invoke_fused_moe_kernel_hybrid_triton,
            )

            assert num_tokens_post_padded is not None
            compute_type = (
                tl.float16 if hidden_states.dtype == torch.float16 else tl.bfloat16
            )

            # GEMM1 scatters its output by slot id, so rows [0, P) are always
            # written and padding rows are never read: no zero-init needed.
            invoke_fused_moe_kernel_hybrid_triton(
                A=gemm1_in,
                B=w1,
                C=gemm1_out,
                B_scale=self.w1_scale,
                topk_weights=topk_weights if apply_router_weight_on_input else None,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=apply_router_weight_on_input,
                top_k=top_k,
                config=self._triton_config(K, P),
                compute_type=compute_type,
                group_size=self._group_size,
            )

            self.activation(activation, act_out[:P], gemm1_out[:P])

            # act_out is in slot space, so top_k=1 reads act_out[slot].
            invoke_fused_moe_kernel_hybrid_triton(
                A=act_out,
                B=w2,
                C=gemm2_out,
                B_scale=self.w2_scale,
                topk_weights=None if apply_router_weight_on_input else topk_weights,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=not apply_router_weight_on_input,
                top_k=1,
                config=self._triton_config(activation_out_dim, P),
                compute_type=compute_type,
                group_size=self._group_size,
            )
        else:
            slot_ids = sorted_token_ids if scattered else None
            ops.fused_moe_wvSplitK_int4_gemm(
                gemm1_in,
                w1,
                self.w1_scale,
                gemm1_out,
                expert_ids,
                block_size_m,
                self._cu_count,
                self._group_size,
                slot_ids,
                top_k,
            )

            self.activation(activation, act_out, gemm1_out)

            ops.fused_moe_wvSplitK_int4_gemm(
                act_out,
                w2,
                self.w2_scale,
                gemm2_out,
                expert_ids,
                block_size_m,
                self._cu_count,
                self._group_size,
                slot_ids,
                1,
            )

        if use_triton:
            # GEMM2 applied the router weights; outputs are in slot order.
            ops.moe_sum(gemm2_out[:P].view(num_tokens, top_k, K), output)
            return

        if scattered:
            # Outputs are already in (token, expert) slot order.
            expert_out = gemm2_out[:P]
        else:
            # Pre-permuted blocks: invert sorted_token_ids, then gather.
            if (
                self._cached_inv_perm_buf is None
                or self._cached_inv_perm_buf.size(0) < P + 1
            ):
                self._cached_inv_perm_buf = torch.empty(
                    P + 1, dtype=torch.int32, device=device
                )
            inv_perm_buf = self._cached_inv_perm_buf[: P + 1]
            inv_perm_buf.scatter_(
                0,
                sorted_token_ids.clamp(max=P).long(),
                self._arange(num_slots, device),
            )
            # Not workspace13: the output may alias it.
            expert_out = _resize_cache(workspace2, (num_slots + P, K))[num_slots:]
            torch.index_select(gemm2_out, 0, inv_perm_buf[:P], out=expert_out)

        TopKWeightAndReduceContiguous().apply(
            output=output,
            fused_expert_output=expert_out,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
