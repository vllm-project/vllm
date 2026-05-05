# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid W4A16 MoE experts: HIP wvSplitK for decode, Triton for prefill.

Weights are stored ONCE in skinny layout [E, N, K//8] int32 (ExLlama
shuffle packed).  Both kernels read from the same weight tensors:
  - Decode (M<=5): HIP wvSplitK_int4_g kernel — optimized for skinny GEMV
  - Prefill (M>5):  Triton fused_moe kernel — better for large batch GEMMs

CUDA-graph compatible.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_unpermute,
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
from vllm.v1.utils import record_function_or_nullcontext


class HybridW4A16MoEExperts(mk.FusedMoEExpertsModular):
    """MoE experts using fused wvSplitK_int4_g kernel.

    Weights are stored in skinny layout per expert:
      w1: [E, N, K//8] int32 (ExLlama shuffle packed)
      w2: [E, K_hidden, K_inter//8] int32 (ExLlama shuffle packed)
      scales: [E, N, K//G] fp16/bf16
      zero_points: [E, N, K//G] fp16/bf16 (raw, optional) or None
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)

        block_shape = self.quant_config.block_shape
        self._group_size = block_shape[1] if block_shape else 128

        from vllm.utils.platform_utils import num_compute_units

        self._cu_count = num_compute_units()
        # Cached tensors to avoid repeated allocation in hot path
        self._cached_arange: torch.Tensor | None = None
        self._cached_inv_perm_buf: torch.Tensor | None = None
        self._cached_expert_ids_buf: torch.Tensor | None = None

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_rocm()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        raise NotImplementedError(
            "HybridW4A16MoEExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [MoEActivation.SILU, MoEActivation.GELU]

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return not moe_parallel_config.use_fi_nvl_two_sided_kernels

    def supports_chunking(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    # Maximum batch size for the HIP wvSplitK kernel path.  Above this
    # threshold the Triton prefill kernel is used instead.
    MAX_SKINNY_BATCH_SIZE = 5

    # Default Triton BLOCK_SIZE_M for prefill.
    TRITON_BLOCK_SIZE_M = 64

    @staticmethod
    def _select_block_size_m(num_tokens: int, topk: int, E: int) -> int:
        """Select block size in the M dimension.

        Decode (num_tokens <= MAX_SKINNY_BATCH_SIZE): use small block sizes
        compatible with the wvSplitK_int4 HIP kernel (N=1..5).
        Prefill (num_tokens > MAX_SKINNY_BATCH_SIZE): use the Triton kernel's
        BLOCK_SIZE_M for efficient batched GEMM.
        """
        if num_tokens > HybridW4A16MoEExperts.MAX_SKINNY_BATCH_SIZE:
            return HybridW4A16MoEExperts.TRITON_BLOCK_SIZE_M
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
        # Skinny format: w1 is [E, N, K//8]
        assert w1.dim() == 3 and w2.dim() == 3
        E = w1.size(0)
        N = w1.size(1)
        K = a1.size(-1)
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
        block_size_m = self._select_block_size_m(M, topk, global_num_experts)
        num_slots = round_up(
            M * topk + global_num_experts * (block_size_m - 1),
            block_size_m,
        )
        workspace1 = (num_slots, max(activation_out_dim, K))
        workspace2 = (num_slots, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def _triton_config(self, K: int) -> dict:
        """Return Triton kernel config for shuffle-packed MoE prefill."""
        BLOCK_SIZE_K = min(self._group_size, 32)
        # Ensure BLOCK_K is a multiple of 8 for shuffle interleave
        assert BLOCK_SIZE_K % 8 == 0
        return {
            "BLOCK_SIZE_M": self.TRITON_BLOCK_SIZE_M,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": BLOCK_SIZE_K,
            "GROUP_SIZE_M": 8,
        }

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
        assert w1.dtype == torch.int32
        assert hidden_states.dim() == 2
        assert hidden_states.is_contiguous()

        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        if global_num_experts == -1:
            global_num_experts = E

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        block_size_m = self._select_block_size_m(
            num_tokens, top_k_num, global_num_experts
        )

        P = num_tokens * top_k_num
        use_triton = num_tokens > self.MAX_SKINNY_BATCH_SIZE

        # ---- Route tokens to experts ----
        scattered = block_size_m == 1
        if scattered and expert_map is None:
            # Decode without EP: skip moe_align_block_size entirely.
            if (
                self._cached_expert_ids_buf is None
                or self._cached_expert_ids_buf.size(0) < P
            ):
                self._cached_expert_ids_buf = torch.empty(
                    P, dtype=torch.int32, device=hidden_states.device
                )
            expert_ids = self._cached_expert_ids_buf[:P]
            expert_ids.copy_(topk_ids.view(-1))
            if self._cached_arange is None or self._cached_arange.size(0) < P:
                self._cached_arange = torch.arange(
                    P, dtype=torch.int32, device=hidden_states.device
                )
            sorted_token_ids = self._cached_arange[:P]
            num_tokens_post_padded = None
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
            if scattered:
                gemm1_in = hidden_states
            elif use_triton:
                # Triton kernel gathers via sorted_token_ids // top_k
                # internally, so pass original hidden_states directly.
                gemm1_in = hidden_states
            else:
                source_rows = (sorted_token_ids // top_k_num).clamp(max=num_tokens - 1)
                gemm1_in = _resize_cache(workspace13, (num_slots, K))
                torch.index_select(hidden_states, 0, source_rows.long(), out=gemm1_in)

        gemm1_out = _resize_cache(workspace2, (num_slots, N))
        act_out = _resize_cache(workspace13, (num_slots, activation_out_dim))
        gemm2_out = _resize_cache(workspace2, (num_slots, K))

        if use_triton:
            # The Triton kernel skips padding blocks (expert_ids == -1),
            # leaving those slots unwritten.  Zero gemm1_out so the
            # activation doesn't see garbage/NaN in padding rows.
            # Note: gemm2_out aliases workspace2 (same as gemm1_out),
            # so only zero gemm1_out here; gemm2_out is zeroed after
            # activation completes.
            gemm1_out.zero_()
            from vllm.model_executor.layers.fused_moe.fused_moe import (
                invoke_fused_moe_kernel_hybrid_triton,
            )
            from vllm.triton_utils import tl

            compute_type = (
                tl.float16
                if w1.dtype == torch.int32 and hidden_states.dtype == torch.float16
                else tl.bfloat16
            )
            config = self._triton_config(K)

            # GEMM 1 (Triton prefill path)
            # The kernel gathers from hidden_states via
            # sorted_token_ids // top_k, so pass the original top_k.
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
                top_k=top_k_num,
                config=config,
                compute_type=compute_type,
                group_size=self._group_size,
            )

            # Activation
            apply_moe_activation(activation, act_out, gemm1_out)

            # Zero padding slots in gemm2_out (aliases workspace2, same
            # storage as gemm1_out which is no longer needed).
            gemm2_out.zero_()

            # GEMM 2 (Triton prefill path)
            # act_out is in slot-space (GEMM1 wrote at sorted_token_ids
            # positions). Pass top_k=1 so the kernel reads act_out[slot]
            # directly.
            invoke_fused_moe_kernel_hybrid_triton(
                A=act_out,
                B=w2,
                C=gemm2_out,
                B_scale=self.w2_scale,
                topk_weights=None,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=False,
                top_k=1,
                config=config,
                compute_type=compute_type,
                group_size=self._group_size,
            )
        else:
            from vllm._custom_ops import fused_moe_wvSplitK_int4_gemm

            # GEMM 1 (HIP wvSplitK decode path)
            with record_function_or_nullcontext(
                f"fused_moe_wvsplitk_int4 {num_tokens}x{N}x{K} "
                f"E={global_num_experts} top_k={top_k_num}"
            ):
                fused_moe_wvSplitK_int4_gemm(
                    gemm1_in,
                    w1,
                    self.w1_scale,
                    gemm1_out,
                    expert_ids,
                    block_size_m,
                    self._cu_count,
                    self._group_size,
                    self.quant_config.w1_zp,
                    sorted_token_ids if scattered else None,
                    top_k_num,
                )

            # Activation
            apply_moe_activation(activation, act_out, gemm1_out)

            # GEMM 2 (HIP wvSplitK decode path)
            with record_function_or_nullcontext(
                f"fused_moe_wvsplitk_int4 {num_tokens}x{K}x{activation_out_dim} "
                f"E={global_num_experts} top_k={top_k_num}"
            ):
                fused_moe_wvSplitK_int4_gemm(
                    act_out,
                    w2,
                    self.w2_scale,
                    gemm2_out,
                    expert_ids,
                    block_size_m,
                    self._cu_count,
                    self._group_size,
                    self.quant_config.w2_zp,
                    sorted_token_ids if scattered else None,
                    1,
                )

        # ---- Reduce via moe_unpermute ----
        if scattered or use_triton:
            # Scattered kernel writes to c[sorted_token_ids[block]], so
            # gemm2_out is in sequential slot order → identity mapping.
            if self._cached_arange is None or self._cached_arange.size(0) < P:
                self._cached_arange = torch.arange(
                    P, dtype=torch.int32, device=hidden_states.device
                )
            inv_permuted_idx = self._cached_arange[:P]
        else:
            # Contiguous mode: invert sorted_token_ids.
            if self._cached_arange is None or self._cached_arange.size(0) < num_slots:
                self._cached_arange = torch.arange(
                    num_slots, dtype=torch.int32, device=hidden_states.device
                )
            aligned_arange = self._cached_arange[:num_slots]

            if (
                self._cached_inv_perm_buf is None
                or self._cached_inv_perm_buf.size(0) < P + 1
            ):
                self._cached_inv_perm_buf = torch.empty(
                    P + 1, dtype=torch.int32, device=hidden_states.device
                )
            inv_perm_buf = self._cached_inv_perm_buf[: P + 1]
            inv_perm_buf.scatter_(
                0, sorted_token_ids.clamp(max=P).long(), aligned_arange
            )
            inv_permuted_idx = inv_perm_buf[:P]

        unpermute_weights = topk_weights
        if apply_router_weight_on_input:
            unpermute_weights = torch.ones_like(topk_weights)

        moe_unpermute(
            out=output,
            permuted_hidden_states=gemm2_out,
            topk_weights=unpermute_weights,
            inv_permuted_idx=inv_permuted_idx,
        )
