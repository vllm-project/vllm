# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUTLASS based Fused MoE kernels."""

from collections.abc import Callable

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute,
    moe_unpermute,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)


def run_cutlass_moe_fp8(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    activation_callable: Callable,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    a1q_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    ab_strides1: torch.Tensor,
    ab_strides2: torch.Tensor,
    c_strides1: torch.Tensor,
    c_strides2: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    expert_num_tokens: torch.Tensor | None,
    out_dtype: torch.dtype,
    per_act_token: bool,
    per_out_ch: bool,
    use_batched_format: bool,
    topk_weights: torch.Tensor | None,
):
    a1q = hidden_states

    assert w1_scale is not None
    assert w2_scale is not None
    assert w1.dtype == torch.float8_e4m3fn
    assert w2.dtype == torch.float8_e4m3fn
    assert a1q.size(-1) == w1.size(2), "Hidden size mismatch w1"
    assert w1.size(1) == w2.size(2) * 2, "Hidden size mismatch w2"
    assert (
        w1_scale.dim() == 1 or w1_scale.size(1) == 1 or w1_scale.shape[1] == w1.size(1)
    ), "W1 scale shape mismatch"
    assert (
        w2_scale.dim() == 1 or w2_scale.size(1) == 1 or w2_scale.shape[1] == w2.size(1)
    ), "W2 scale shape mismatch"
    assert w1.size(0) == w2.size(0), "Expert number mismatch"
    assert (
        a1q_scale is None
        or a1q_scale.dim() == 0
        or a1q_scale.size(0) == 1
        or a1q_scale.size(0) == a1q.shape[0]
    ), "Input scale shape mismatch"
    assert w1.size(0) == w2.size(0), "Weights expert number mismatch"
    assert w1.size(0) == w1_scale.size(0), "w1 scales expert number mismatch"
    assert w1.size(0) == w2_scale.size(0), "w2 scales expert number mismatch"
    assert (
        a2_scale is None
        or a2_scale.dim() == 0
        or a2_scale.size(0) == 1
        or a2_scale.size(0) == a1q.shape[0]
    ), "Intermediate scale shape mismatch"
    assert out_dtype in [torch.half, torch.bfloat16], "Invalid output dtype"
    if expert_map is not None:
        assert expert_num_tokens is None

    # We have two modes: batched experts and non-batched experts.
    # In the non-batched mode, the input tokens are not padded: thus, the shape
    # of the input is [total_num_tokens, hidden_size]. The input and output
    # require shuffling by a_map and c_map such that the tokens assigned to
    # each expert are contiguous.
    # In the batched mode, the input tokens are padded per expert to ensure that
    # the batched dispatch and combine functions work correctly: thus, the shape
    # of the input is [num_experts, max_num_tokens_per_expert, hidden_size].
    # The batched input and output require no shuffling by a_map and c_map since
    # their tokens are already contiguous for each expert as a result of
    # the dispatch function.

    M = a1q.size(0)  # non batched expert M
    padded_M = a1q.size(1)  # batched expert M
    _, K, N = w2.shape
    device = a1q.device

    assert w1.size(2) == K
    assert global_num_experts != -1
    assert a1q_scale is not None

    if expert_map is not None:
        "Translate info from expert_map to topk_ids"
        local_topk_ids = torch.where(
            expert_map[topk_ids] != -1, expert_map[topk_ids], -1
        )
    else:
        local_topk_ids = topk_ids

    topk = local_topk_ids.size(1)
    local_E = w1.size(0)

    if use_batched_format:
        mm1_out = _resize_cache(workspace13, (local_E * padded_M, N * 2))
        act_out = _resize_cache(workspace2, (local_E * padded_M, N))
        quant_out = _resize_cache(
            workspace13.view(dtype=torch.float8_e4m3fn), (local_E * padded_M, N)
        )
        mm2_out = _resize_cache(workspace2, (local_E * padded_M, K))
    else:
        a1q_perm = _resize_cache(
            workspace2.view(dtype=torch.float8_e4m3fn), (M * topk, K)
        )
        mm1_out = _resize_cache(workspace13, (M * topk, N * 2))
        act_out = _resize_cache(workspace2, (M * topk, N))
        # original workspace are based on input hidden_states dtype (bf16)
        quant_out = _resize_cache(
            workspace13.view(dtype=torch.float8_e4m3fn), (M * topk, N)
        )
        mm2_out = _resize_cache(workspace2, (M * topk, K))

    if use_batched_format:
        assert expert_num_tokens is not None

        expert_offsets = torch.empty((local_E), dtype=torch.int32, device=device)
        problem_sizes1 = torch.empty((local_E, 3), dtype=torch.int32, device=device)
        problem_sizes2 = torch.empty((local_E, 3), dtype=torch.int32, device=device)

        ops.get_cutlass_pplx_moe_mm_data(
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
            expert_num_tokens,
            local_E,
            padded_M,
            N,
            K,
        )

        w1_scale = w1_scale.reshape(w1_scale.size(0), -1)
        w2_scale = w2_scale.reshape(w2_scale.size(0), -1)
        a1q = a1q.reshape(-1, a1q.size(2))
        a1q_scale = a1q_scale.reshape(-1, a1q_scale.size(2)).contiguous()
        # c3x get_group_gemm_starts expects int64 to avoid overflow
        # during offset calculations
        expert_offsets = expert_offsets.to(torch.int64)
    else:
        problem_sizes1 = torch.empty(
            (global_num_experts, 3), dtype=torch.int32, device=device
        )
        problem_sizes2 = torch.empty(
            (global_num_experts, 3), dtype=torch.int32, device=device
        )

        num_expert = global_num_experts if expert_map is None else expert_map.size(0)
        # permuted a1q reuses workspace2
        a1q, a1q_scale, expert_first_token_offset, inv_perm, _ = moe_permute(
            a1q,
            a1q_scale,
            topk_ids,
            num_expert,
            local_E,
            expert_map,
            permuted_hidden_states=a1q_perm,
        )
        expert_offsets = expert_first_token_offset[:-1]

        ops.get_cutlass_moe_mm_problem_sizes(
            local_topk_ids, problem_sizes1, problem_sizes2, global_num_experts, N, K
        )

    if not per_act_token and (expert_map is not None or use_batched_format):
        # this is necessary to avoid imprecise scale calculation caused by
        # random data in the unused workspace. The workspace is unused when
        # this rank handles only partial tokens, or when it is batched .
        mm1_out.fill_(0)

    ops.cutlass_moe_mm(
        mm1_out,
        a1q,
        w1,
        a1q_scale,
        w1_scale,
        expert_offsets,
        problem_sizes1,
        ab_strides1,
        ab_strides1,
        c_strides1,
        per_act_token,
        per_out_ch,
    )

    activation_callable(act_out, mm1_out)

    a2q, a2q_scale = ops.scaled_fp8_quant(
        act_out, a2_scale, use_per_token_if_dynamic=per_act_token, output=quant_out
    )

    ops.cutlass_moe_mm(
        mm2_out,
        a2q,
        w2,
        a2q_scale,
        w2_scale,
        expert_offsets,
        problem_sizes2,
        ab_strides2,
        ab_strides2,
        c_strides2,
        per_act_token,
        per_out_ch,
    )

    if use_batched_format:
        output.copy_(mm2_out.reshape(local_E, padded_M, K), non_blocking=True)
    else:
        # for non-chunking mode the output is resized from workspace13
        # so we need to make sure mm2_out uses workspace2.
        moe_unpermute(
            out=output,
            permuted_hidden_states=mm2_out,
            topk_weights=topk_weights,
            inv_permuted_idx=inv_perm,
            expert_first_token_offset=(
                expert_first_token_offset if expert_map is not None else None
            ),
        )


class CutlassExpertsFp8Base(mk.FusedMoEPermuteExpertsUnpermute):
    def __init__(
        self,
        e: int,
        n: int,
        k: int,
        out_dtype: torch.dtype | None,
        quant_config: FusedMoEQuantConfig,
        device: torch.dtype,
    ):
        assert quant_config.use_fp8_w8a8
        super().__init__(quant_config)

        # E: num_experts
        # N: intermediate size per partition
        # K: hidden dim
        ab_strides1_c_strides2 = torch.full((e,), k, device=device, dtype=torch.int64)
        ab_strides2 = torch.full((e,), n, device=device, dtype=torch.int64)
        c_strides1 = torch.full((e,), 2 * n, device=device, dtype=torch.int64)

        self.out_dtype = out_dtype
        self.ab_strides1 = ab_strides1_c_strides2
        self.ab_strides2 = ab_strides2
        self.c_strides1 = c_strides1
        self.c_strides2 = ab_strides1_c_strides2

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the impl.
        return TopKWeightAndReduceDelegate()

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert self.w1_zp is None, "w1_zp is not supported in CUTLASS MoE"
        assert self.w2_zp is None, "w2_zp is not supported in CUTLASS MoE"

        expert_num_tokens = None
        if expert_tokens_meta is not None:
            expert_num_tokens = expert_tokens_meta.expert_num_tokens

        activation_callable = lambda o, i: self.activation(activation, o, i)

        use_batched_format = (
            self.activation_formats[0] == mk.FusedMoEActivationFormat.BatchedExperts
        )

        in_dtype = hidden_states.dtype
        run_cutlass_moe_fp8(
            output,
            hidden_states,
            w1,
            w2,
            topk_ids,
            activation_callable,
            global_num_experts,
            expert_map,
            self.w1_scale,
            self.w2_scale,
            a1q_scale,
            a2_scale,
            self.ab_strides1,
            self.ab_strides2,
            self.c_strides1,
            self.c_strides2,
            workspace13,
            workspace2,
            expert_num_tokens,
            self.out_dtype if self.out_dtype is not None else in_dtype,
            self.per_act_token_quant,
            self.per_out_ch_quant,
            use_batched_format,
            topk_weights,
        )


class CutlassExpertsFp8(CutlassExpertsFp8Base):
    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (
            mk.FusedMoEActivationFormat.Standard,
            mk.FusedMoEActivationFormat.Standard,
        )

    def supports_chunking(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # topk weights and reduction are fused in moe_unpermute cuda kernel
        return TopKWeightAndReduceNoOP()

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return self.out_dtype if self.out_dtype is not None else act_dtype

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (M * topk, max(N, K))
        workspace2 = (M * topk, max(N // 2, K))
        output = (M, K)
        return (workspace1, workspace2, output)


class CutlassBatchedExpertsFp8(CutlassExpertsFp8Base):
    def __init__(
        self,
        max_experts_per_worker: int,
        num_dispatchers: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert max_experts_per_worker > 0
        self.max_experts_per_worker = max_experts_per_worker
        self.num_dispatchers = num_dispatchers

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (
            mk.FusedMoEActivationFormat.BatchedExperts,
            mk.FusedMoEActivationFormat.BatchedExperts,
        )

    def supports_chunking(self) -> bool:
        return False

    def supports_expert_map(self) -> bool:
        return False

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return self.out_dtype if self.out_dtype is not None else act_dtype

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        num_dp = self.num_dispatchers
        assert num_dp is not None
        workspace1 = (self.max_experts_per_worker, M * num_dp, max(N, K))
        workspace2 = (self.max_experts_per_worker, M * num_dp, max(N // 2, K))
        output = (self.max_experts_per_worker, M, K)
        return (workspace1, workspace2, output)


FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def run_cutlass_moe_fp4(
    output: torch.Tensor,
    a: torch.Tensor,
    a1_gscale: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    a2_gscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    m: int,
    n: int,
    k: int,
    e: int,
    device: torch.device,
    apply_router_weight_on_input: bool = False,
) -> None:
    """
    MoE implementation for FP4 Inputs

    # Gemm 1
    a: Input tensor: [m, k] (half/bfloat16)
    a1_gscale: Activation scale per expert: [e]  (float32)
    w1(gate up) (not an argument to cutlass_moe_fp4): [e, 2 * n, k]
    w1_fp4: [e, 2 * n, k // 2], dtype: torch.uint8 (stacked fp4: E2M1)
    (Note: `n` is the up projection output dim, `k` is the input dim in
     full precision)
    w1_blockscale: [e, 2 * n, k // block_size] (float8_e4m3)
                   (Block size = 16 for NVFP4)

    # Gemm 2
    a2_gscale: Activation scale per expert: [e]
    w2(down projection) (not an argument to cutlass_moe_fp4): [e, k, n]
    w2_fp4: [e, k, n // 2], dtype: torch.uint8 (stacked E2M1)
    w2_blockscale: [e, k, n // block_size], dtype: float8_e4m3

    topk_weights: [m, topk] dtype: float8
    topk_ids: [m, topk] dtype: float8

    m, n, k: Unquantized weight shapes, dtype: int
    e: number of experts, dtype: int

    assumes that topk < k < n to satisfy - up/down projection expectations.
    """
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_fp4.dtype == torch.uint8, "weight 1 must be uint8"
    assert w2_fp4.dtype == torch.uint8, "weight 2 must be uint8"
    assert (
        w1_fp4.ndim == 3
        and w2_fp4.ndim == 3
        and w1_blockscale.ndim == 3
        and w2_blockscale.ndim == 3
    ), "All Weights must be of rank 3 for cutlass_moe_fp4"
    m_a, k_a = a.shape
    e_w1, nx2_w1, half_k_w1 = w1_fp4.shape
    e_w2, k_w2, half_n_w2 = w2_fp4.shape

    assert e_w1 == e_w2 and e_w1 == e, (
        "Number of experts must match",
        f" between weights. {e_w1}, {e_w2}, {e}",
    )
    assert k_a == half_k_w1 * 2 and k == k_w2, (
        "Hidden size mismatch between a, w1 and w2"
    )
    assert nx2_w1 == n * 2 and half_n_w2 * 2 == n, "mismatch in expected `n`"
    assert m == m_a, "input shape mismatch"
    assert 2 * half_k_w1 == k_w2, "Hidden size mismatch w2 and w1"
    assert a.dtype in [torch.half, torch.bfloat16], "Invalid input dtype"
    assert topk_weights.size(0) == m and topk_ids.size(0) == m, (
        "topk must be provided for each row of a"
    )
    topk = topk_ids.size(1)
    out_dtype = a.dtype
    num_topk = topk_ids.size(1)

    expert_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
    blockscale_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
    # Problem size:  (num_experts, (m,2n,k))
    problem_sizes1 = torch.empty((e, 3), dtype=torch.int32, device=device)
    # Problem size:  (num_experts, (m,n,k))
    problem_sizes2 = torch.empty((e, 3), dtype=torch.int32, device=device)

    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    if apply_router_weight_on_input:
        # TODO: this only works for topK=1, will need to update for topK>1
        assert num_topk == 1, (
            "apply_router_weight_on_input is only implemented for topk=1"
        )
        a.mul_(topk_weights.to(out_dtype))

    # problem shapes should have [m, n, k]
    # Note that problem sizes are based on logical number of elements.
    ops.get_cutlass_moe_mm_data(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        e,
        n,
        k,
        blockscale_offsets,
    )

    a = ops.shuffle_rows(a, a_map)
    rep_a_fp4, rep_a_blockscale = ops.scaled_fp4_experts_quant(
        a,
        a1_gscale,
        expert_offsets,
        blockscale_offsets,
        num_topk,
    )
    c1 = _resize_cache(workspace13, (m * topk, n * 2))
    c2 = _resize_cache(workspace2, (m * topk, n))
    c3 = _resize_cache(workspace13, (m * topk, k))
    ops.cutlass_fp4_moe_mm(
        c1,
        rep_a_fp4,
        w1_fp4,
        rep_a_blockscale,
        w1_blockscale,
        w1_alphas,
        problem_sizes1,
        expert_offsets[:-1],
        blockscale_offsets[:-1],
    )
    del rep_a_fp4, rep_a_blockscale
    torch.ops._C.silu_and_mul(c2, c1)
    int_fp4, int_blockscale = ops.scaled_fp4_experts_quant(
        c2, a2_gscale, expert_offsets, blockscale_offsets, num_topk
    )

    ops.cutlass_fp4_moe_mm(
        c3,
        int_fp4,
        w2_fp4,
        int_blockscale,
        w2_blockscale,
        w2_alphas,
        problem_sizes2,
        expert_offsets[:-1],
        blockscale_offsets[:-1],
    )
    del int_fp4, int_blockscale

    c3 = ops.shuffle_rows(c3, c_map)

    assert output.dtype == out_dtype
    if not apply_router_weight_on_input:
        output.copy_(
            (
                c3.view(m, num_topk, k)
                * topk_weights.view(m, num_topk, 1).to(out_dtype)
            ).sum(dim=1),
            non_blocking=True,
        )
    else:
        output.copy_(c3.view(m, num_topk, k).sum(dim=1), non_blocking=True)
    return


# Split into batched and non-batched
class CutlassExpertsFp4(mk.FusedMoEPermuteExpertsUnpermute):
    def __init__(
        self,
        max_experts_per_worker: int,
        out_dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig,
        use_batched_format: bool = False,
    ):
        super().__init__(quant_config)
        self.max_experts_per_worker = max_experts_per_worker
        self.out_dtype = out_dtype
        self.use_batched_format = use_batched_format

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        if self.use_batched_format:
            return (
                mk.FusedMoEActivationFormat.BatchedExperts,
                mk.FusedMoEActivationFormat.BatchedExperts,
            )
        else:
            return (
                mk.FusedMoEActivationFormat.Standard,
                mk.FusedMoEActivationFormat.Standard,
            )

    def supports_expert_map(self) -> bool:
        return False

    def supports_chunking(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return self.out_dtype if self.out_dtype is not None else act_dtype

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1: tuple[int, ...] = ()
        workspace2: tuple[int, ...] = ()
        output: tuple[int, ...] = ()
        if self.use_batched_format:
            workspace1 = (self.max_experts_per_worker, M, max(N, K))
            workspace2 = (self.max_experts_per_worker, M, (N // 2))
            output = (self.max_experts_per_worker, M, K)
        else:
            workspace1 = (M * topk, max(2 * N, K))
            workspace2 = (M * topk, N)
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
        activation: str,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,  # unused
        a2_scale: torch.Tensor | None,  # unused
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        e, m, n, k, _ = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        n = w2.shape[2] * 2

        run_cutlass_moe_fp4(
            output=output,
            a=hidden_states,
            a1_gscale=self.a1_gscale,
            w1_fp4=w1,
            w1_blockscale=self.w1_scale,
            w1_alphas=self.g1_alphas,
            a2_gscale=self.a2_gscale,
            w2_fp4=w2,
            w2_blockscale=self.w2_scale,
            w2_alphas=self.g2_alphas,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            workspace13=workspace13,
            workspace2=workspace2,
            m=m,
            n=n,
            k=k,
            e=e,
            device=hidden_states.device,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )


# W4A8
def run_cutlass_moe_w4a8_fp8(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    activation_callable: Callable,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    a1q_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    w1_chan_scale: torch.Tensor,
    w2_chan_scale: torch.Tensor,
    a_strides1: torch.Tensor,
    a_strides2: torch.Tensor,
    b_strides1: torch.Tensor,
    b_strides2: torch.Tensor,
    c_strides1: torch.Tensor,
    c_strides2: torch.Tensor,
    s_strides1: torch.Tensor,
    s_strides2: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    expert_num_tokens: torch.Tensor | None,
    out_dtype: torch.dtype,
    per_act_token: bool,
    per_out_ch: bool,
    use_batched_format: bool,
    topk_weights: torch.Tensor | None,
    group_size: int,
):
    a1q = hidden_states
    M = a1q.size(0)
    local_E = w1.size(0)
    device = a1q.device
    _, K, N_packed = w2.shape
    N = N_packed * 8  # logical N, pack 8 int4 into 1 int32

    assert per_act_token, "W4A8 must use per-token scales"
    assert per_out_ch, "W4A8 must use per-channel scales"
    assert w1_scale is not None
    assert w2_scale is not None
    assert w1_scale.dtype == torch.float8_e4m3fn
    assert w2_scale.dtype == torch.float8_e4m3fn
    assert w1.dtype == torch.int32
    assert w2.dtype == torch.int32
    assert w1_chan_scale.dtype == torch.float32
    assert w2_chan_scale.dtype == torch.float32
    assert w1.size(0) == w2.size(0), "Weights expert number mismatch"
    assert a1q_scale is not None
    assert a2_scale is None
    assert out_dtype in [torch.bfloat16], f"Invalid output dtype: {out_dtype}"
    if expert_map is not None:
        assert expert_num_tokens is None
    assert not use_batched_format, "batched format not supported yet"
    assert group_size == 128, f"Only group size 128 supported but got {group_size=}"

    assert global_num_experts != -1
    assert w1.size(2) * 8 == K, (
        f"w1 hidden size mismatch: got {w1.size(2) * 8}, expected {K=}"
    )

    # Translate info from expert_map to topk_ids
    if expert_map is not None:
        local_topk_ids = torch.where(
            expert_map[topk_ids] != -1, expert_map[topk_ids], -1
        )
    else:
        local_topk_ids = topk_ids

    topk = local_topk_ids.size(1)
    a1q_perm = _resize_cache(workspace2.view(dtype=torch.float8_e4m3fn), (M * topk, K))
    mm1_out = _resize_cache(workspace13, (M * topk, N * 2))
    act_out = _resize_cache(workspace2, (M * topk, N))
    # original workspace are based on input hidden_states dtype (bf16)
    quant_out = _resize_cache(
        workspace13.view(dtype=torch.float8_e4m3fn), (M * topk, N)
    )
    mm2_out = _resize_cache(workspace2, (M * topk, K))

    problem_sizes1 = torch.empty(
        (global_num_experts, 3), dtype=torch.int32, device=device
    )
    problem_sizes2 = torch.empty(
        (global_num_experts, 3), dtype=torch.int32, device=device
    )

    num_expert = global_num_experts if expert_map is None else expert_map.size(0)
    # permuted a1q reuses workspace2
    a1q, a1q_scale, expert_first_token_offset, inv_perm, _ = moe_permute(
        a1q,
        a1q_scale,
        topk_ids,
        num_expert,
        local_E,
        expert_map,
        permuted_hidden_states=a1q_perm,
    )
    expert_offsets = expert_first_token_offset[:-1]

    # For RS gemm SwapAB is always enabled (swap logical M, N in the problem shape)
    ops.get_cutlass_moe_mm_problem_sizes(
        local_topk_ids,
        problem_sizes1,
        problem_sizes2,
        global_num_experts,
        N,
        K,
        force_swap_ab=True,
    )

    ops.cutlass_w4a8_moe_mm(
        mm1_out,
        a1q,
        w1,
        a1q_scale,
        w1_chan_scale,
        w1_scale,
        group_size,
        expert_offsets,
        problem_sizes1,
        a_strides1,
        b_strides1,
        c_strides1,
        s_strides1,
    )

    activation_callable(act_out, mm1_out)

    a2q, a2q_scale = ops.scaled_fp8_quant(
        act_out, a2_scale, use_per_token_if_dynamic=per_act_token, output=quant_out
    )

    ops.cutlass_w4a8_moe_mm(
        mm2_out,
        a2q,
        w2,
        a2q_scale,
        w2_chan_scale,
        w2_scale,
        group_size,
        expert_offsets,
        problem_sizes2,
        a_strides2,
        b_strides2,
        c_strides2,
        s_strides2,
    )

    # for non-chunking mode the output is resized from workspace13
    # so we need to make sure mm2_out uses workspace2.
    moe_unpermute(
        out=output,
        permuted_hidden_states=mm2_out,
        topk_weights=topk_weights,
        inv_permuted_idx=inv_perm,
        expert_first_token_offset=(
            expert_first_token_offset if expert_map is not None else None
        ),
    )


class CutlassExpertsW4A8Fp8(mk.FusedMoEPermuteExpertsUnpermute):
    def __init__(
        self,
        out_dtype: torch.dtype | None,
        a_strides1: torch.Tensor,
        a_strides2: torch.Tensor,
        b_strides1: torch.Tensor,
        b_strides2: torch.Tensor,
        c_strides1: torch.Tensor,
        c_strides2: torch.Tensor,
        s_strides1: torch.Tensor,
        s_strides2: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        group_size: int,
    ):
        super().__init__(quant_config)
        self.out_dtype = out_dtype
        self.a_strides1 = a_strides1
        self.a_strides2 = a_strides2
        self.b_strides1 = b_strides1
        self.b_strides2 = b_strides2
        self.c_strides1 = c_strides1
        self.c_strides2 = c_strides2
        self.s_strides1 = s_strides1
        self.s_strides2 = s_strides2
        self.group_size = group_size

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (
            mk.FusedMoEActivationFormat.Standard,
            mk.FusedMoEActivationFormat.Standard,
        )

    def supports_chunking(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # topk weights and reduction are fused in moe_unpermute cuda kernel
        return TopKWeightAndReduceNoOP()

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return self.out_dtype if self.out_dtype is not None else act_dtype

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (M * topk, max(N, K))
        workspace2 = (M * topk, max(N // 2, K))
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
        activation: str,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert self.w1_zp is None, "w1_zp is not supported in CUTLASS MoE"
        assert self.w2_zp is None, "w2_zp is not supported in CUTLASS MoE"

        expert_num_tokens = None
        activation_callable = lambda o, i: self.activation(activation, o, i)

        use_batched_format = (
            self.activation_formats[0] == mk.FusedMoEActivationFormat.BatchedExperts
        )
        assert not use_batched_format, "batched format not supported"

        in_dtype = hidden_states.dtype

        run_cutlass_moe_w4a8_fp8(
            output,
            hidden_states,
            w1,
            w2,
            topk_ids,
            activation_callable,
            global_num_experts,
            expert_map,
            self.w1_scale,
            self.w2_scale,
            a1q_scale,
            a2_scale,
            self.g1_alphas,  # per-channel scales
            self.g2_alphas,  # per-channel scales
            self.a_strides1,
            self.a_strides2,
            self.b_strides1,
            self.b_strides2,
            self.c_strides1,
            self.c_strides2,
            self.s_strides1,
            self.s_strides2,
            workspace13,
            workspace2,
            expert_num_tokens,
            self.out_dtype if self.out_dtype is not None else in_dtype,
            self.per_act_token_quant,
            self.per_out_ch_quant,
            use_batched_format,
            topk_weights,
            self.group_size,
        )


def cutlass_moe_w4a8_fp8(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    a_strides1: torch.Tensor,
    a_strides2: torch.Tensor,
    b_strides1: torch.Tensor,
    b_strides2: torch.Tensor,
    c_strides1: torch.Tensor,
    c_strides2: torch.Tensor,
    s_strides1: torch.Tensor,
    s_strides2: torch.Tensor,
    quant_config: FusedMoEQuantConfig,
    activation: str = "silu",
    expert_map: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    group_size: int = 128,
) -> torch.Tensor:
    """
    This function computes a w4a8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with CUTLASS
    mixed-dtype grouped gemm.

    Parameters:
    - a (torch.Tensor): The input tensor to the MoE layer.
        Shape: [M, K]
    - w1_q (torch.Tensor): The first set of fp8-quantized expert weights.
        Shape: [num_experts, 2*N, K // packed_factor]
    - w2_q (torch.Tensor): The second set of fp8-quantized expert weights.
        Shape: [num_experts, K, N // packed_factor]
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - topk_ids (torch.Tensor): The token->expert mappings.
    - a_strides1 (torch.Tensor): The input strides for the first gemm.
        Shape: [num_experts]
    - a_strides2 (torch.Tensor): The input strides for the second gemm.
        Shape: [num_experts]
    - b_strides1 (torch.Tensor): The packed layout for the first gemm weights.
        Shape: [num_experts, 3]
        dtype: torch.int32
    - b_strides2 (torch.Tensor): The packed layout for the second gemm weights.
        Shape: [num_experts, 3]
        dtype: torch.int32
    - c_strides1 (torch.Tensor): The output strides for the first gemm.
        Shape: [num_experts]
    - c_strides2 (torch.Tensor): The output strides for the second gemm.
        Shape: [num_experts]
    - s_strides1 (torch.Tensor): strides for the group-wise scales for the first gemm.
        Shape: [num_experts, 2]
        dtype: torch.int64
    - s_strides2 (torch.Tensor): strides for the group-wise scales for the second gemm.
        Shape: [num_experts, 2]
        dtype: torch.int64
    - per_act_token (Optional[bool]): Whether the scale is per-token or
                                      per-tensor.
    - activation (str): The activation function to use.
    - expert_map (Optional[torch.Tensor]): In the case of Expert parallel,
        every Rank is responsible for a subset of experts. expert_map is a
        mapping from global expert-id to local expert-id. When expert_map[i]
        is -1, it means that this Rank is not responsible for global
        expert-id i.
    - apply_router_weight_on_input (bool): When true, the topk weights are
        applied directly on the inputs. This is only applicable when topk is 1.
    - global_num_experts (int): The total number of experts.
    - group_size (int): The number of weights per scale factor

    Returns:
    - torch.Tensor: The bf16 output tensor after applying the MoE layer.
    """
    assert quant_config is not None

    num_experts = global_num_experts if global_num_experts != -1 else w1_q.size(0)

    fn = mk.FusedMoEModularKernel(
        MoEPrepareAndFinalizeNoEP(),
        CutlassExpertsW4A8Fp8(
            out_dtype=a.dtype,
            a_strides1=a_strides1,
            a_strides2=a_strides2,
            b_strides1=b_strides1,
            b_strides2=b_strides2,
            c_strides1=c_strides1,
            c_strides2=c_strides2,
            s_strides1=s_strides1,
            s_strides2=s_strides2,
            quant_config=quant_config,
            group_size=group_size,
        ),
    )

    return fn(
        a,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        activation=activation,
        global_num_experts=num_experts,
        expert_map=expert_map,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
