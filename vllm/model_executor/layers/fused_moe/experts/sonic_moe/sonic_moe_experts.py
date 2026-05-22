# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sonic-MoE fused-experts backend.

A grouped-GEMM-with-fused-SwiGLU MoE forward built on top of the quack kernels
(``quack.gemm_interface.gemm_gated`` for the gate+up projection, ``gemm`` for
the down projection). Permutation metadata is built by a small Triton kernel
adapted from the sonic-moe project; the weighted reduction across the top-K
axis is performed inline (`finalize_weight_and_reduce_impl` is a NoOP).

Opt-in only via ``--moe-backend sonic_moe``. Requires Hopper sm_90 or
Blackwell sm_100/sm_103.

This file is fully self-contained: the original sonic-moe Triton helpers
(``_keyed_add``, the two-stage bitmatrix kernels, ``TC_topk_router_metadata``)
are inlined verbatim with the upstream attribution preserved. No runtime
dependency on the ``sonicmoe`` Python package.
"""

from __future__ import annotations

import torch
from quack.gemm_interface import gemm, gemm_gated

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Triton routing-metadata kernels.
#
# Adapted from sonic-moe v0.1.2:
#   sonicmoe/functional/triton_kernels/{__init__,bitmatrix}.py
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
#
# Modifications:
#   - Drops the unused general_routing path (we only need TC top-K).
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _keyed_add(x, y):
    # we keep the key in the upper 16 bits of a uint32:
    key_mask: tl.constexpr = 0xFFFF0000

    kx = x & key_mask
    ky = y & key_mask
    z = tl.where(kx == ky, x + y - kx, y)
    return z


@triton.jit
def _bitmatrix_metadata_compute_stage1(
    expert_freq_ptr,
    expert_freq_offs_ptr,
    E: tl.constexpr,
    partial_sum_ptr,
    n_tiles,
    TK,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Assume grid size == E + 2
    pid = tl.program_id(0)
    if pid < E:
        # Convert partial_sum[e, *] from raw counts to exclusive prefix
        # sums over tiles. After this kernel, partial_sum[e, t] = number of
        # entries for expert e in tiles 0..t-1.
        expert_partial_sum_ptr = partial_sum_ptr + pid * n_tiles
        curr_sum = 0
        for start in range(0, n_tiles, BLOCK_M):
            offs = start + tl.arange(0, BLOCK_M)
            tile_counts = tl.load(
                expert_partial_sum_ptr + offs, mask=offs < n_tiles, other=0
            )
            excl_cumsum = tl.cumsum(tile_counts, 0) - tile_counts + curr_sum
            curr_sum += tl.sum(tile_counts, 0)
            tl.store(expert_partial_sum_ptr + offs, excl_cumsum, mask=offs < n_tiles)
    elif pid == E:
        # Exclusive prefix sum of per-expert total counts → expert_offs[e].
        curr_sum = 0
        for start in tl.static_range(0, E, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            expert_freq = tl.load(expert_freq_ptr + offs, mask=offs < E, other=0)
            excl_cumsum = tl.cumsum(expert_freq, 0) - expert_freq + curr_sum
            curr_sum += tl.sum(expert_freq, 0)
            tl.store(expert_freq_offs_ptr + offs, excl_cumsum, mask=offs < E)
    elif pid == E + 1:
        tl.store(expert_freq_offs_ptr + E, TK)


@triton.jit
def _bitmatrix_metadata_compute_stage2(
    s_scatter_idx_ptr,
    s_reverse_scatter_idx_ptr,
    x_gather_idx_ptr,
    topk_indices_ptr,
    T,
    partial_sum_ptr,
    n_tiles,
    expert_offs_ptr,
    K_POW2: tl.constexpr,
    K: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = TOKENS_PER_BLOCK * K_POW2
    IS_POW2_K: tl.constexpr = K == K_POW2
    tl.static_assert(BLOCK_SIZE <= 32768)

    pid_m = tl.program_id(0)
    offs_local = tl.arange(0, BLOCK_SIZE)
    offs_global = pid_m * BLOCK_SIZE + offs_local
    mask = offs_global < T * K_POW2

    if IS_POW2_K:
        expert = tl.load(topk_indices_ptr + offs_global, mask=mask, other=-1).to(
            tl.uint32
        )
    else:
        token_i_local = offs_local // K_POW2
        k_slot = offs_local % K_POW2
        token_i_global = pid_m * TOKENS_PER_BLOCK + token_i_local
        load_mask = mask & (k_slot < K)
        safe_k = tl.minimum(k_slot, K - 1)
        expert = tl.load(
            topk_indices_ptr + token_i_global * K + safe_k,
            mask=load_mask,
            other=-1,
        ).to(tl.uint32)

    kv_pairs = tl.sort(((expert << 16) | offs_local).to(tl.uint32), 0)
    expert = kv_pairs >> 16
    mask = expert != 0xFFFF

    scan_input = (kv_pairs & 0xFFFF0000) | 0x00000001
    inclusive_run_lengths = tl.associative_scan(scan_input, 0, _keyed_add)
    within_expert_rank = (inclusive_run_lengths - 1) & 0xFFFF

    s_reverse_scatter_idx = tl.load(
        partial_sum_ptr + pid_m + expert * n_tiles, mask=mask
    )
    s_reverse_scatter_idx += tl.load(expert_offs_ptr + expert, mask=mask)
    s_reverse_scatter_idx += within_expert_rank

    if IS_POW2_K:
        presort_offs = kv_pairs & 0xFFFF
        entry_idx = pid_m * BLOCK_SIZE + presort_offs
        tl.store(
            s_reverse_scatter_idx_ptr + entry_idx, s_reverse_scatter_idx, mask=mask
        )
        tl.store(s_scatter_idx_ptr + s_reverse_scatter_idx, entry_idx, mask=mask)
        tl.store(
            x_gather_idx_ptr + s_reverse_scatter_idx, entry_idx // K_POW2, mask=mask
        )
    else:
        presort_offs = kv_pairs & 0xFFFF
        token_i_global_s = pid_m * TOKENS_PER_BLOCK + presort_offs // K_POW2
        entry_idx = token_i_global_s * K + presort_offs % K_POW2
        tl.store(
            s_reverse_scatter_idx_ptr + entry_idx, s_reverse_scatter_idx, mask=mask
        )
        tl.store(s_scatter_idx_ptr + s_reverse_scatter_idx, entry_idx, mask=mask)
        tl.store(x_gather_idx_ptr + s_reverse_scatter_idx, token_i_global_s, mask=mask)


@triton.jit
def _compute_col_partial_sum_kernel(
    topk_indices_ptr,
    partial_sum_ptr,
    T,
    E: tl.constexpr,
    n_tiles,
    TOKENS_PER_TILE: tl.constexpr,
    K_POW2: tl.constexpr,
    K: tl.constexpr,
    E_POW2: tl.constexpr,
):
    tile_id = tl.program_id(0)

    for e_start in tl.static_range(0, E, E_POW2):
        e_offs = e_start + tl.arange(0, E_POW2)
        tl.store(
            partial_sum_ptr + e_offs * n_tiles + tile_id,
            tl.zeros([E_POW2], tl.int32),
            mask=e_offs < E,
        )

    tok_offs = tile_id * TOKENS_PER_TILE + tl.arange(0, TOKENS_PER_TILE)
    k_offs = tl.arange(0, K_POW2)
    tok_mask = tok_offs < T

    load_mask = tok_mask[:, None] & (k_offs[None, :] < K)
    safe_k = tl.minimum(k_offs, K - 1)
    expert_ids = tl.load(
        topk_indices_ptr + tok_offs[:, None] * K + safe_k[None, :],
        mask=load_mask,
        other=-1,
    )

    flat_experts = tl.reshape(expert_ids, [TOKENS_PER_TILE * K_POW2])
    flat_mask = tl.reshape(load_mask, [TOKENS_PER_TILE * K_POW2])
    safe_experts = tl.where(flat_mask, flat_experts, 0)

    tl.atomic_add(
        partial_sum_ptr + safe_experts * n_tiles + tile_id,
        tl.full([TOKENS_PER_TILE * K_POW2], 1, dtype=tl.int32),
        mask=flat_mask,
    )


def TC_topk_router_metadata_triton(
    topk_router_indices: torch.Tensor,
    E: int,
    expert_frequency: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    s_scatter_idx: torch.Tensor,
    s_reverse_scatter_idx: torch.Tensor,
) -> None:
    """Compute permutation metadata for grouped-by-expert MoE GEMMs.

    Given top-K-Choice routing indices of shape (T, K) with values in
    [0, E), produces:

    - expert_frequency[E]            : tokens routed to each expert
    - expert_frequency_offset[E+1]   : cumulative sum (cu_seqlens for grouped GEMM)
    - x_gather_idx[TK]               : grouped-slot -> source token index
    - s_scatter_idx[TK]              : grouped-slot -> original (token, k) entry
    - s_reverse_scatter_idx[TK]      : original (token, k) entry -> grouped-slot
    """
    T, K = topk_router_indices.size()
    TK = T * K
    device = topk_router_indices.device
    E_POW2 = triton.next_power_of_2(E)
    K_POW2 = triton.next_power_of_2(K)
    TOKENS_PER_BLOCK = 1024 // K_POW2
    n_tiles = triton.cdiv(T, TOKENS_PER_BLOCK)

    col_partial_sum_trans = torch.empty(E, n_tiles, dtype=torch.int32, device=device)
    _compute_col_partial_sum_kernel[(n_tiles,)](
        topk_router_indices,
        col_partial_sum_trans,
        T,
        E,
        n_tiles,
        TOKENS_PER_TILE=TOKENS_PER_BLOCK,
        K_POW2=K_POW2,
        K=K,
        E_POW2=E_POW2,
    )

    expert_frequency.copy_(col_partial_sum_trans.sum(dim=1, dtype=torch.int32))
    col_partial_sum = col_partial_sum_trans.T  # [n_tiles, E]

    _bitmatrix_metadata_compute_stage1[(E + 2,)](
        expert_frequency,
        expert_frequency_offset,
        E,
        col_partial_sum,
        n_tiles,
        TK,
        BLOCK_M=128,
        BLOCK_N=E_POW2,
    )

    _bitmatrix_metadata_compute_stage2[(n_tiles,)](
        s_scatter_idx,
        s_reverse_scatter_idx,
        x_gather_idx,
        topk_router_indices,
        T,
        col_partial_sum,
        n_tiles,
        expert_frequency_offset[:E],
        K_POW2=K_POW2,
        TOKENS_PER_BLOCK=TOKENS_PER_BLOCK,
        K=K,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Triton token-gather + weighted-sum kernel.
#
# Adapted from sonic-moe v0.1.2:
#   sonicmoe/functional/reduction_over_k_gather.py
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
#
# Computes, for each token t:
#   output[t] = Σ_{k=0..K-1}  y[M_perm[t*K + k]] * w[t*K + k]
# where y is the grouped-order down-GEMM output and M_perm = s_reverse_scatter_idx
# (natural slot -> grouped slot). Fused per-token gather + weighted reduce in
# one pass, no atomics, no `(M*topk, K)` intermediate.
# ─────────────────────────────────────────────────────────────────────────────


def _gather_sum_autotune_configs() -> list[triton.Config]:
    configs = []
    block_h = 256
    while block_h <= 4096:
        block_k = 1
        while block_k <= 128:
            if block_k * block_h <= 32768:
                for num_warps in (4, 8):
                    configs.append(
                        triton.Config(
                            {"BLOCK_H": block_h, "BLOCK_K": block_k},
                            num_warps=num_warps,
                            num_stages=4,
                        )
                    )
            block_k <<= 1
        block_h <<= 1
    return configs


def _prune_gather_sum_configs(configs, nargs, **kw):
    pruned = []
    for c in configs:
        block_h = c.kwargs["BLOCK_H"]
        block_k = c.kwargs["BLOCK_K"]
        if (
            block_h <= triton.next_power_of_2(kw["H"])
            and block_k <= triton.next_power_of_2(kw["MAX_K"])
            and min(kw["H"] * kw["MAX_K"], 1024) <= (block_h * block_k)
        ):
            pruned.append(c)
    return pruned if pruned else configs


@triton.autotune(
    configs=_gather_sum_autotune_configs(),
    key=["H", "MAX_K", "w_is_None", "is_varlen_K"],
    prune_configs_by={"early_config_prune": _prune_gather_sum_configs},
)
@triton.jit
def _token_gather_sum_kernel(
    x_ptr,  # (Mtotal, H)
    w_ptr,  # (Mtotal,) or null when w_is_None
    M_perm_ptr,  # (Mtotal,) int32
    M_offset_ptr,  # (T+1,) int32, only read when is_varlen_K
    out_ptr,  # (T, H)
    T,
    H: tl.constexpr,
    MAX_K: tl.constexpr,
    stride_xM: tl.constexpr,
    stride_xH: tl.constexpr,
    stride_outT: tl.constexpr,
    stride_outH: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    w_is_None: tl.constexpr,
    is_varlen_K: tl.constexpr,
):
    pid_t = tl.program_id(axis=0)
    t_idx = pid_t.to(tl.int64)

    if is_varlen_K:
        Ms = tl.load(M_offset_ptr + t_idx).to(tl.int64)
        Me = tl.load(M_offset_ptr + t_idx + 1).to(tl.int64)
        K_this_token = Me - Ms
    else:
        Ms = MAX_K * t_idx
        K_this_token = MAX_K  # type: ignore[no-redef]

    for h_tile in tl.static_range(triton.cdiv(H, BLOCK_H)):
        h_idx = (h_tile * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.int64)
        m_h = h_idx < H

        acc = tl.zeros([BLOCK_H], dtype=tl.float32)

        for k_tile in tl.range(tl.cdiv(K_this_token, BLOCK_K)):
            k_offset = k_tile * BLOCK_K
            k_idx = (k_offset + tl.arange(0, BLOCK_K)).to(tl.int64)
            m_k = k_idx < K_this_token
            m_abs = Ms + k_idx

            perm_idx = tl.load(M_perm_ptr + m_abs, mask=m_k, other=0).to(tl.int64)

            x_ptrs = x_ptr + perm_idx[:, None] * stride_xM + h_idx[None, :] * stride_xH
            x_mask = m_k[:, None] & m_h[None, :]
            x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

            if w_is_None:
                acc += tl.sum(x_vals, axis=0)
            else:
                w_vals = tl.load(w_ptr + m_abs, mask=m_k, other=0.0).to(tl.float32)
                acc += tl.sum(x_vals * w_vals[:, None], axis=0)

        out_ptrs = out_ptr + t_idx * stride_outT + h_idx * stride_outH
        tl.store(out_ptrs, acc, mask=m_h)


def token_gather_and_sum_varlen_K_triton(
    x: torch.Tensor,
    w: torch.Tensor | None,
    out: torch.Tensor,
    M_perm: torch.Tensor,
    M_offset: torch.Tensor | None,
    T: int,
    MAX_K: int,
    H: int,
    is_varlen_K: bool,
) -> None:
    """Per-token gather + weighted sum of grouped-order rows.

        out[t, :] = Σ_{k=0..K[t]-1}  x[M_perm[M_offset[t]+k], :] * w[M_offset[t]+k]

    When ``is_varlen_K=False``, ``K[t] = MAX_K`` for all t and ``M_offset`` is
    ignored (can be ``None``).
    """
    _token_gather_sum_kernel[(T,)](
        x,
        w,
        M_perm,
        M_offset,
        out,
        T=T,
        H=H,
        MAX_K=MAX_K,
        stride_xM=x.stride(0),
        stride_xH=x.stride(1),
        stride_outT=out.stride(0),
        stride_outH=out.stride(1),
        w_is_None=(w is None),
        is_varlen_K=is_varlen_K,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fused-experts class
# ─────────────────────────────────────────────────────────────────────────────


class SonicMoEExperts(mk.FusedMoEExpertsModular):
    """Grouped-GEMM + fused-SwiGLU MoE experts using quack kernels."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self.out_dtype = moe_config.in_dtype

    # ---- capability advertisement ----

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda() and (
            current_platform.is_device_capability(90)
            or current_platform.is_device_capability_family(100)
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        # quack's gemm_gated requires a gated activation; non-gated MLPs would
        # need gemm_act + a separate combine path that we do not implement.
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key is None and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # MoEActivation.SILU under is_act_and_mul=True == SwiGLU; quack maps it
        # to its "swiglu" gated activation in gemm_gated.
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    # ---- buffer + finalize plumbing ----

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        # Sonic MoE stores weights in quack's (E, K, N) layout after
        # convert_to_unquantized_kernel_format:
        #   w1: (E, H, 2*I)  -> N is the trailing dim
        #   w2: (E, I, H)
        # The default implementation extracts N from w1.shape[1], which would
        # return H (= K) here and cause incorrect workspace allocation.
        assert w1.dim() == 3 and w2.dim() == 3
        E, _, N = w1.shape
        K = a1.size(-1)
        assert a1.dim() == 2
        assert topk_ids.size(0) == a1.size(0)
        return E, a1.size(0), N, K, topk_ids.size(1)

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # apply() already multiplies by topk_weights and reduces over the topk
        # axis; the framework's finalize is a no-op.
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
        # N is the *unactivated* width of w1, which for a gated activation is
        # 2 * intermediate. activation_out_dim halves N for gated activations.
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        # workspace13: post-activation buffer `a`        (M*topk, I)
        # workspace2:  down-projection output buffer `y` (M*topk, K_hidden)
        # output:      final reduced result              (M, K_hidden)
        # workspace13 and output share storage in the framework's common
        # workspace; ordering in apply() makes that safe (see comments there).
        workspace1 = (M * topk, activation_out_dim)
        workspace2 = (M * topk, K)
        output = (M, K)
        return workspace1, workspace2, output

    # ---- forward ----

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
        # ---- shapes ----
        # After process_weights_after_loading the weights are in quack layout:
        #   w1: (E_local, H, 2*I)  with [gate | up] packed BLOCK along the last dim
        #   w2: (E_local, I, H)
        T, K_topk = topk_ids.shape
        E_local, H, two_I = w1.shape
        assert two_I % 2 == 0, (
            "SonicMoE expects gated w1 with packed gate|up, got shape "
            f"{tuple(w1.shape)}"
        )
        TK = T * K_topk
        device = hidden_states.device

        assert hidden_states.dim() == 2 and hidden_states.size(-1) == H
        assert w2.shape == (E_local, two_I // 2, H), (
            f"SonicMoE expects w2 shape {(E_local, two_I // 2, H)}, got "
            f"{tuple(w2.shape)}"
        )

        assert expert_map is None, (
            "SonicMoE does not support expert_map; supports_expert_map() returns False"
        )
        routed_ids = topk_ids.to(torch.int32).contiguous()

        # ---- routing metadata ----
        expert_frequency = torch.empty(E_local, dtype=torch.int32, device=device)
        expert_frequency_offset = torch.empty(
            E_local + 1, dtype=torch.int32, device=device
        )
        x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
        s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
        s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
        TC_topk_router_metadata_triton(
            routed_ids,
            E_local,
            expert_frequency,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
        )

        # ---- up projection: grouped GEMM + fused SwiGLU ----
        # w1 is (E, H, 2*I) with [gate | up] INTERLEAVED on the last dim
        # ([g0, u0, g1, u1, ...]); the interleave is materialized in
        # convert_to_unquantized_kernel_format. quack's varlen_m gemm_gated
        # ignores concat_layout=("B",) so we must hand it the interleaved
        # tensor directly.
        a = workspace13  # (TK, I)
        gemm_gated(
            hidden_states,
            w1,
            activation="swiglu",
            cu_seqlens_m=expert_frequency_offset,
            A_idx=x_gather_idx,
            postact_out=a,
            store_preact=False,
            tuned=False,
        )

        # ---- down projection: grouped GEMM (no activation) ----
        y = workspace2  # (TK, H)
        gemm(a, w2, out=y, cu_seqlens_m=expert_frequency_offset, tuned=False)

        # ---- weighted reduce: y (grouped) -> output (M, H) ----
        # Per natural slot (t, k), the corresponding grouped-row index in y is
        # s_reverse_scatter_idx[t*K + k]. The sonic-moe gather-sum kernel does
        # the natural-order gather + topk-weighted reduce in one pass — no
        # atomics, no (M*topk, K) intermediate.
        if apply_router_weight_on_input:
            flat_w: torch.Tensor | None = None
        else:
            flat_w = topk_weights.to(y.dtype).contiguous().view(-1)

        token_gather_and_sum_varlen_K_triton(
            x=y,
            w=flat_w,
            out=output,
            M_perm=s_reverse_scatter_idx,
            M_offset=None,
            T=T,
            MAX_K=K_topk,
            H=H,
            is_varlen_K=False,
        )
