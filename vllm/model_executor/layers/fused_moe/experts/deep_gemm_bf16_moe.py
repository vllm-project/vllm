# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepGEMM BF16 grouped-GEMM MoE experts (unquantized).

Mirrors the FP8 ``DeepGemmExperts`` (``deep_gemm_moe.py``) but runs entirely in
bfloat16 using DeepGEMM's ``m_grouped_bf16_gemm_nt_contiguous`` grouped kernel,
with a plain silu-and-mul in between (no quantization). This is the ``Standard``
(contiguous) activation-format path, selected via the opt-in ``deep_gemm`` MoE
backend for unquantized bf16 MoE weights.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M_and_alignment,
    deepgemm_moe_permute,
    deepgemm_unpermute_and_reduce,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.utils.deep_gemm import (
    get_mk_alignment_for_contiguous_layout,
    is_deep_gemm_bf16_grouped_supported,
    is_deep_gemm_bf16_masked_supported,
    m_grouped_bf16_gemm_nt_contiguous,
    m_grouped_bf16_gemm_nt_masked,
    mk_alignment_scope,
)
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)


def _valid_deep_gemm_bf16_shape(M: int, N: int, K: int) -> bool:
    align = get_mk_alignment_for_contiguous_layout()[0]
    return align <= M and N % align == 0 and K % align == 0


def _valid_deep_gemm_bf16(
    hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> bool:
    """Whether the bf16 DeepGEMM grouped contiguous kernel supports this problem.

    All of M, N, K must be aligned to
    ``get_mk_alignment_for_contiguous_layout()`` (128) and every tensor must be
    contiguous bfloat16.
    """
    if not has_deep_gemm():
        logger.debug_once("DeepGemm(bf16) disabled: deep_gemm not available.")
        return False

    M = hidden_states.size(0)
    _, K, N = w2.size()
    align = get_mk_alignment_for_contiguous_layout()[0]

    if not _valid_deep_gemm_bf16_shape(M, N, K):
        logger.debug_once(
            "DeepGemm(bf16) disabled due to unaligned problem size. "
            "M=%s N=%s K=%s (need M>=%s and N,K multiples of %s). "
            "Falling back to the default backend.",
            M,
            N,
            K,
            align,
            align,
        )
        return False

    if (
        hidden_states.dtype != torch.bfloat16
        or w1.dtype != torch.bfloat16
        or w2.dtype != torch.bfloat16
    ):
        logger.debug_once(
            "DeepGemm(bf16) disabled: expected bfloat16 activations and weights. "
            "hidden_states=%s w1=%s w2=%s",
            hidden_states.dtype,
            w1.dtype,
            w2.dtype,
        )
        return False

    if not (
        hidden_states.is_contiguous() and w1.is_contiguous() and w2.is_contiguous()
    ):
        logger.debug_once(
            "DeepGemm(bf16) disabled: activations or weights not contiguous."
        )
        return False

    return True


class DeepGemmBf16Experts(mk.FusedMoEExpertsModular):
    """DeepGEMM-based fused MoE experts for unquantized bf16 weights.

    Uses ``m_grouped_bf16_gemm_nt_contiguous`` for both grouped GEMMs, with a
    plain silu-and-mul in between. This is the ``Standard`` (contiguous)
    activation-format analogue of the FP8 :class:`DeepGemmExperts`.
    """

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        # Unquantized only: no weight/activation scales, no block quant.
        assert quant_config.quant_dtype is None, (
            "DeepGemmBf16Experts only supports unquantized (bf16) weights."
        )
        assert not quant_config.per_act_token_quant
        assert not quant_config.per_out_ch_quant

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return is_deep_gemm_bf16_grouped_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # Unquantized bf16 only: the oracle passes (None, None) for this path.
        return weight_key is None and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # Phase 1: plain gated SiLU only (silu_and_mul). Other gated variants
        # (e.g. swigluoai) can be added later.
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # Match DeepGemmExperts: exclude the FlashInfer NVL fused kernels.
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

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
        block_m = get_mk_alignment_for_contiguous_layout()[0]
        M_sum, align_used = compute_aligned_M_and_alignment(
            M, topk, local_num_experts, block_m, expert_tokens_meta
        )
        assert M_sum % align_used == 0

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        # workspace13 holds the permuted bf16 activations (M_sum, K) and later
        # the post-activation input (M_sum, activation_out_dim); workspace2
        # holds either grouped-GEMM output (M_sum, N) or (M_sum, K).
        workspace1 = (M_sum, max(activation_out_dim, K))
        workspace2 = (M_sum, max(N, K))
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
        # Unquantized bf16: no activation scales are produced by prepare/finalize.
        assert a1q_scale is None
        assert a2_scale is None

        a1 = hidden_states
        _, N, K = w1.size()

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        assert w2.size(1) == K

        M_sum, _ = compute_aligned_M_and_alignment(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=get_mk_alignment_for_contiguous_layout()[0],
            expert_tokens_meta=expert_tokens_meta,
        )

        # Permute bf16 activations into the per-expert contiguous layout and
        # build m_indices (expert_ids). No scale scatter for bf16.
        a1_perm = _resize_cache(workspace13, (M_sum, K))
        a1, _, expert_ids, inv_perm, align_used = deepgemm_moe_permute(
            aq=a1,
            aq_scale=None,
            topk_ids=topk_ids,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
            aq_out=a1_perm,
        )
        assert a1.size(0) == M_sum

        # Cap DG's BLOCK_M heuristic at the workspace's per-expert alignment;
        # see DeepGemmExperts.apply for the IMA-under-cudagraph rationale.
        with mk_alignment_scope(align_used):
            mm1_out = _resize_cache(workspace2, (M_sum, N))
            m_grouped_bf16_gemm_nt_contiguous(a1, w1, mm1_out, expert_ids)

            activation_out_dim = self.adjust_N_for_activation(N, activation)
            act_out = _resize_cache(workspace13, (M_sum, activation_out_dim))
            # Plain (bf16) silu-and-mul; no requantization.
            self.activation(activation, act_out, mm1_out.view(-1, N))

            mm2_out = _resize_cache(workspace2, (M_sum, K))
            m_grouped_bf16_gemm_nt_contiguous(act_out, w2, mm2_out, expert_ids)

        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        deepgemm_unpermute_and_reduce(
            a=mm2_out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )


class DeepGemmBf16BatchedExperts(mk.FusedMoEExpertsModular):
    """DeepGEMM bf16 masked grouped-GEMM experts (``BatchedExperts`` / EP format).

    Unquantized bf16 analogue of ``BatchedDeepGemmExperts``: uses
    ``m_grouped_bf16_gemm_nt_masked`` for both grouped GEMMs with a plain
    silu-and-mul in between (no fp8 quantization). Selected via the opt-in
    ``deep_gemm`` MoE backend when the layer runs in the batched activation
    format (expert/data parallel with an all2all dispatcher).
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        assert quant_config.quant_dtype is None, (
            "DeepGemmBf16BatchedExperts only supports unquantized (bf16) weights."
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def _supports_current_device() -> bool:
        return is_deep_gemm_bf16_masked_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key is None and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the reduce impl.
        return TopKWeightAndReduceDelegate()

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
        assert self.num_dispatchers is not None
        assert self.max_num_tokens is not None
        num_dispatchers = self.num_dispatchers
        num_experts = local_num_experts
        max_num_tokens = M if self.max_num_tokens is None else self.max_num_tokens
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace13 = (num_experts, max_num_tokens * num_dispatchers, max(K, N))
        workspace2 = (num_experts, max_num_tokens * num_dispatchers, activation_out_dim)
        output = (num_experts, max_num_tokens * num_dispatchers, K)
        return (workspace13, workspace2, output)

    def estimate_expected_m(
        self, global_num_experts: int, max_tokens_per_expert: int, topk: int
    ) -> int:
        dp_meta = (
            get_forward_context().dp_metadata
            if is_forward_context_available()
            else None
        )
        if dp_meta is None:
            logger.warning_once(
                "DPMetadata unavailable. Defaulting expected_m to "
                f"{max_tokens_per_expert}.",
            )
            return max_tokens_per_expert

        total_num_tokens = dp_meta.num_tokens_across_dp_cpu.sum().item()
        total_num_tokens_replicated = total_num_tokens * topk

        # Assume even load balancing across experts.
        assert global_num_experts != 0
        estimate = round_up(int(total_num_tokens_replicated // global_num_experts), 16)
        estimate = max(estimate, 16)
        estimate = min(max_tokens_per_expert, estimate)
        return estimate

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
        # Unquantized bf16: no activation scales.
        assert a1q_scale is None
        assert a2_scale is None
        assert expert_tokens_meta is not None
        expert_num_tokens = expert_tokens_meta.expert_num_tokens

        assert hidden_states.ndim == 3
        a1 = hidden_states

        E, max_num_tokens, N, K, _ = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )
        assert w2.size(1) == K

        workspace1 = _resize_cache(workspace13, (E, max_num_tokens, N))

        expected_m = self.estimate_expected_m(
            global_num_experts=global_num_experts,
            max_tokens_per_expert=max_num_tokens,
            topk=topk_ids.size(-1),
        )

        # GroupGemm-0: (E, T, K) x (E, N, K) -> (E, T, N)
        m_grouped_bf16_gemm_nt_masked(a1, w1, workspace1, expert_num_tokens, expected_m)

        # Plain (bf16) silu-and-mul over the full padded batch. Rows past
        # expert_num_tokens are junk but are never read by the masked
        # down-GEMM below (it is gated by expert_num_tokens), so no masked
        # activation kernel is required.
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        act_out = _resize_cache(workspace2, (E, max_num_tokens, activation_out_dim))
        self.activation(
            activation,
            act_out.view(-1, activation_out_dim),
            workspace1.view(-1, N),
        )

        # GroupGemm-1: (E, T, N//2) x (E, K, N//2) -> (E, T, K)
        m_grouped_bf16_gemm_nt_masked(
            act_out, w2, output, expert_num_tokens, expected_m
        )
