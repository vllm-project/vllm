# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels of the Muse Spark ROCm decode MoE (`..decode_moe`)."""

import torch

from vllm.triton_utils import tl, triton

# AITER's Triton MXFP4 helpers (private); without them, or without the quant
# options used here, the decode MoE keeps AITER's sort and quant launches.
try:
    from aiter.ops.triton._triton_kernels.quant.quant import (
        _mxfp4_quant_op,
        mxfp4_swizzled_scale_offsets,
    )

    if not {"SCALING_MODE", "USE_ASM"} <= set(
        getattr(_mxfp4_quant_op, "arg_names", ())
    ):
        raise ImportError
except ImportError:
    _mxfp4_quant_op = mxfp4_swizzled_scale_offsets = None


@triton.jit(do_not_specialize_on_alignment=["M"])
def _decode_moe_prep_kernel(
    part_ptr,
    bias_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    x_ptr,
    nw_ptr,
    y_ptr,
    M,
    eps,
    gain_center,
    router_output_multiplier,
    route_scale,
    division_eps,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_PADDED: tl.constexpr,
    ROUTE_ROWS: tl.constexpr,
    HAS_MULTIPLIER: tl.constexpr,
    HAS_EPS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    INPUT_SCALE: tl.constexpr,
):
    """Programs [0, M): RMSNorm of one row. Programs >= M: routing of ROUTE_ROWS
    tokens each, one warp per token so the top-k reductions stay in a warp."""
    pid = tl.program_id(0)
    if pid < M:
        t = pid
        d = tl.arange(0, BLOCK_D)
        dm = d < D
        x = tl.load(x_ptr + t * D + d, mask=dm, other=0.0).to(tl.float32)
        if HAS_WEIGHT:
            w = tl.load(nw_ptr + d, mask=dm, other=0.0).to(tl.float32) + gain_center
            if INPUT_SCALE:
                x = x * w
        y = x * tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / D) + eps)
        if HAS_WEIGHT and not INPUT_SCALE:
            y = y * w
        tl.store(y_ptr + t * D + d, y.to(y_ptr.dtype.element_ty), mask=dm)
    else:
        rows = (pid - M) * ROUTE_ROWS + tl.arange(0, ROUTE_ROWS)
        row_mask = rows < M
        expert_offs = tl.arange(0, N)
        offs = rows[:, None] * N + expert_offs[None, :]
        # router logits: the split-K partials, summed in split order
        gating_scores = tl.load(part_ptr + offs, mask=row_mask[:, None], other=0.0)
        for s in tl.static_range(1, SPLITS):
            gating_scores += tl.load(
                part_ptr + s * M * N + offs, mask=row_mask[:, None], other=0.0
            )
        # sigmoid scores; the expert bias steers the selection only
        if HAS_MULTIPLIER:
            gating_scores = gating_scores * router_output_multiplier
        normalized_scores = tl.sigmoid(gating_scores)
        bias_values = tl.load(bias_ptr + expert_offs)
        scores_for_topk = normalized_scores + bias_values[None, :]
        slots = tl.arange(0, TOPK_PADDED)
        slot_mask = slots < TOPK
        selected_mask = tl.zeros([ROUTE_ROWS, N], dtype=tl.int1)
        ids = tl.zeros([ROUTE_ROWS, TOPK_PADDED], dtype=tl.int32)
        scores = tl.zeros([ROUTE_ROWS, TOPK_PADDED], dtype=tl.float32)
        score_sum = tl.zeros([ROUTE_ROWS], dtype=tl.float32)
        for k in tl.static_range(TOPK):
            masked_scores = tl.where(selected_mask, -float("inf"), scores_for_topk)
            max_idx = tl.argmax(masked_scores, axis=1)
            is_selected = expert_offs[None, :] == max_idx[:, None]
            selected_score = tl.sum(
                tl.where(is_selected, normalized_scores, 0.0), axis=1
            )
            selected_mask = selected_mask | is_selected
            slot = (slots == k)[None, :]
            ids = tl.where(slot, max_idx.to(tl.int32)[:, None], ids)
            scores = tl.where(slot, selected_score[:, None], scores)
            score_sum = score_sum + selected_score
        if HAS_EPS:
            score_sum = score_sum + division_eps
        weights = scores * (route_scale / score_sum)[:, None]
        out_offs = rows[:, None] * TOPK + slots[None, :]
        out_mask = row_mask[:, None] & slot_mask[None, :]
        tl.store(topk_ids_ptr + out_offs, ids, mask=out_mask)
        tl.store(topk_weights_ptr + out_offs, weights, mask=out_mask)


def supports_decode_moe_prep(num_experts: int, top_k: int) -> bool:
    return num_experts >= 8 and num_experts & (num_experts - 1) == 0 and 0 < top_k <= 16


def decode_moe_prep(
    router_logits: torch.Tensor,
    expert_bias: torch.Tensor,
    top_k: int,
    route_scale: float,
    router_output_multiplier: float | None,
    division_eps: float | None,
    pooled: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    gain_center: float,
    input_scale: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Router split-K reduce, sigmoid top-k routing (`MuseSparkRouter`) and the
    pre-expert RMSNorm of `pooled` in one launch.

    `router_logits` is `[T, E]` or its `[splits, T, E]` split-K partials.
    Returns `(topk_weights [T, k] fp32, topk_ids [T, k] int32, normed [T, D])`.
    """
    if router_logits.dim() == 2:
        router_logits = router_logits.unsqueeze(0)
    splits, T, E = router_logits.shape
    assert router_logits.dtype == torch.float32 and router_logits.is_contiguous()
    assert pooled.dim() == 2 and pooled.is_contiguous() and pooled.shape[0] == T
    D = pooled.shape[1]
    dev = pooled.device
    bias = expert_bias if expert_bias.dtype == torch.float32 else expert_bias.float()
    assert bias.is_contiguous() and bias.shape[0] == E
    topk_weights = torch.empty((T, top_k), dtype=torch.float32, device=dev)
    topk_ids = torch.empty((T, top_k), dtype=torch.int32, device=dev)
    normed = torch.empty_like(pooled)
    if T == 0:
        return topk_weights, topk_ids, normed
    block_d = triton.next_power_of_2(D)
    num_warps = 8 if block_d >= 4096 else 4
    _decode_moe_prep_kernel[(T + triton.cdiv(T, num_warps),)](
        router_logits,
        bias,
        topk_weights,
        topk_ids,
        pooled,
        norm_weight if norm_weight is not None else pooled,
        normed,
        T,
        norm_eps,
        gain_center,
        router_output_multiplier if router_output_multiplier is not None else 1.0,
        route_scale,
        division_eps if division_eps is not None else 0.0,
        N=E,
        SPLITS=splits,
        TOPK=top_k,
        TOPK_PADDED=triton.next_power_of_2(top_k),
        ROUTE_ROWS=num_warps,
        HAS_MULTIPLIER=router_output_multiplier is not None,
        HAS_EPS=division_eps is not None,
        D=D,
        BLOCK_D=block_d,
        HAS_WEIGHT=norm_weight is not None,
        INPUT_SCALE=input_scale and norm_weight is not None,
        num_warps=num_warps,
    )
    return topk_weights, topk_ids, normed


# Largest T for `sort_quant`: every program reads and bins all `T * top_k` routes.
SORT_QUANT_MAX_TOKENS = 512


@triton.jit(do_not_specialize_on_alignment=["T"])
def _sort_quant_kernel(
    x_ptr,
    ids_ptr,
    lmap_ptr,
    a1_ptr,
    scale_ptr,
    sorted_ids_ptr,
    sorted_eids_ptr,
    num_valid_ptr,
    T,
    D: tl.constexpr,
    CSPLIT: tl.constexpr,
    TOPK: tl.constexpr,
    LE: tl.constexpr,
    BR: tl.constexpr,
    BM: tl.constexpr,
    MAXB: tl.constexpr,
    SCALE_N_PAD: tl.constexpr,
):
    """Program `(t, c)`: MXFP4-quantizes column chunk c of row t and writes its
    scales to the sorted rows of t's local routes; every program recomputes the
    per-expert route counts and offsets from all routes (no inter-program sync).
    Programs `(t, 0)` write t's `sorted_ids`, program `(0, 0)` the padding
    entries, `sorted_expert_ids` and `num_valid_ids`."""
    t = tl.program_id(0)
    c = tl.program_id(1)
    DC: tl.constexpr = D // CSPLIT
    x = tl.load(x_ptr + t * D + c * DC + tl.arange(0, DC)).to(tl.float32)
    r = tl.arange(0, BR)
    rm = r < T * TOPK
    gid = tl.load(ids_ptr + r, mask=rm, other=-1)
    k = tl.arange(0, TOPK)
    my = tl.load(ids_ptr + t * TOPK + k)
    # local expert index, -1 for non-local or padding (< 0) ids
    lid = tl.load(lmap_ptr + gid, mask=rm & (gid >= 0), other=-1)
    myl = tl.load(lmap_ptr + my, mask=my >= 0, other=-1)
    # A token repeating an expert (non-finite router logits, e.g. CUDA-graph padding
    # rows) keeps one slot per expert, as moe_sorting does, so every sorted row is
    # written; here the last.
    BT: tl.constexpr = BR // TOPK
    g2 = tl.reshape(gid, [BT, TOPK])
    dup = tl.zeros([BT, TOPK], dtype=tl.int1)
    for j in tl.static_range(1, TOPK):
        col = tl.sum(tl.where(k[None, :] == j, g2, 0), axis=1)
        dup = dup | ((g2 == col[:, None]) & (k[None, :] < j))
    lid = tl.reshape(tl.where(dup, -1, tl.reshape(lid, [BT, TOPK])), [BR])
    my_later = (my[None, :] == my[:, None]) & (k[None, :] > k[:, None])
    myl = tl.where(tl.max(my_later.to(tl.int32), axis=1) > 0, -1, myl)
    # routes per local expert; non-local routes go to bins [LE, 2 LE) (a masked
    # tl.histogram miscounts on ROCm Triton 3.x)
    h2 = tl.reshape(tl.histogram(tl.where(lid >= 0, lid, LE), 2 * LE), [2, LE])
    counts = tl.sum(tl.where(tl.arange(0, 2)[:, None] == 0, h2, 0), axis=0)
    padded = (counts + BM - 1) // BM * BM
    start = tl.cumsum(padded, axis=0) - padded
    # sorted row of each local route of t: expert start + earlier routes of the
    # expert in (token, slot) order
    le = tl.arange(0, LE)
    mine = myl >= 0
    earlier = r[None, :] < (t * TOPK + k)[:, None]
    rank = tl.sum(((lid[None, :] == myl[:, None]) & earlier).to(tl.int32), axis=1)
    my_start = tl.sum(tl.where(le[None, :] == myl[:, None], start[None, :], 0), axis=1)
    pos = my_start + rank
    tl.store(sorted_ids_ptr + pos, t | (k << 24), mask=mine & (c == 0))
    # RoundUp e8m0 scale (2^ceil(log2(amax / 6))), round-half-even E2M1
    packed, sc = _mxfp4_quant_op(
        tl.reshape(x, [1, DC]), DC, 1, 32, SCALING_MODE=1, USE_ASM=True
    )
    tl.store(
        a1_ptr + t * (D // 2) + c * (DC // 2) + tl.arange(0, DC // 2),
        tl.reshape(packed, [DC // 2]),
    )
    g = c * (DC // 32) + tl.arange(0, DC // 32)
    offs = mxfp4_swizzled_scale_offsets(pos[:, None], g[None, :], SCALE_N_PAD, 32, 8)
    tl.store(
        scale_ptr + offs,
        tl.broadcast_to(tl.reshape(sc, [1, DC // 32]), [TOPK, DC // 32]),
        mask=mine[:, None],
    )
    if (t == 0) & (c == 0):
        pad_j = tl.arange(0, BM)
        tl.store(
            sorted_ids_ptr + (start + counts)[:, None] + pad_j[None, :],
            tl.zeros([LE, BM], dtype=tl.int32) + (T | (TOPK << 24)),
            mask=pad_j[None, :] < (padded - counts)[:, None],
        )
        blk = tl.arange(0, MAXB)
        tl.store(
            sorted_eids_ptr + (start // BM)[:, None] + blk[None, :],
            tl.zeros([LE, MAXB], dtype=tl.int32) + le[:, None],
            mask=blk[None, :] < (padded // BM)[:, None],
        )
        tl.store(num_valid_ptr, tl.sum(padded, axis=0))
        tl.store(num_valid_ptr + 1, T)


def _is_pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def supports_sort_quant(num_tokens: int, hidden: int, top_k: int, block_m: int) -> bool:
    return (
        _mxfp4_quant_op is not None
        and 0 < num_tokens <= SORT_QUANT_MAX_TOKENS
        and hidden >= 1024
        and _is_pow2(hidden)
        and _is_pow2(top_k)
        and top_k <= 16
        and _is_pow2(block_m)
    )


def sort_quant(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    local_index: torch.Tensor,
    num_local_experts: int,
    num_mask_experts: int,
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """AITER's `moe_sorting` of `topk_ids` over the local experts and the stage-1
    MXFP4 quant of the bf16 rows `x` with its scales in sorted layout, in one
    launch.

    Args:
        x: `[T, D]` bf16 rows, one per token.
        topk_ids: `[T, top_k]` int32 global expert ids (< 0: padding route).
        local_index: `[E]` int32 local expert index of each expert (-1: another
            rank's).
        num_local_experts: experts of this rank.
        num_mask_experts: entries of AITER's EP expert mask (`E + 1`).
        block_m: sort block (rows per stage-kernel tile).

    Returns:
        `(a1, a1_scale_sorted, sorted_ids, sorted_expert_ids, num_valid_ids)` in
        the shapes, dtypes and layouts of `moe_sorting` and
        `fused_dynamic_mx_quant_moe_sort_hip` (entries past `num_valid_ids[0]`
        are not written, as there).

    """
    T, D = x.shape
    top_k = topk_ids.shape[1]
    assert supports_sort_quant(T, D, top_k, block_m)
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    assert topk_ids.dtype == torch.int32 and topk_ids.is_contiguous()
    dev = x.device
    max_sorted = T * top_k + num_mask_experts * block_m - top_k
    num_blocks = triton.cdiv(max_sorted, block_m)
    sorted_ids = torch.empty(num_blocks * block_m, dtype=torch.int32, device=dev)
    sorted_eids = torch.empty(num_blocks, dtype=torch.int32, device=dev)
    num_valid = torch.empty(2, dtype=torch.int32, device=dev)
    a1 = torch.empty((T, D // 2), dtype=torch.float4_e2m1fn_x2, device=dev)
    scale_n_pad = (D // 32 + 7) // 8 * 8
    a1_scale = torch.empty(
        (triton.cdiv(sorted_ids.shape[0], 32) * 32, scale_n_pad),
        dtype=torch.float8_e8m0fnu,
        device=dev,
    )
    # Column chunks per row and warps, tuned on MI355X at D = 4096.
    if T <= 64:
        csplit, num_warps = 4, 4
    elif T <= 128:
        csplit, num_warps = 2, 4
    else:
        csplit, num_warps = 1, 8 if T <= 256 else 4
    _sort_quant_kernel[(T, csplit)](
        x,
        topk_ids,
        local_index,
        a1.view(torch.uint8),
        a1_scale.view(torch.uint8),
        sorted_ids,
        sorted_eids,
        num_valid,
        T,
        D=D,
        CSPLIT=csplit,
        TOPK=top_k,
        LE=triton.next_power_of_2(num_local_experts),
        BR=triton.next_power_of_2(T * top_k),
        BM=block_m,
        MAXB=triton.next_power_of_2(triton.cdiv(T, block_m)),
        SCALE_N_PAD=scale_n_pad,
        num_warps=num_warps,
    )
    return a1, a1_scale, sorted_ids, sorted_eids, num_valid
