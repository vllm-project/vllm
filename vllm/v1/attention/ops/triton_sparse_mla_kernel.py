# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton sparse MLA attention with split-KV for low-batch decode.

Stage 1 runs the sparse attention over a contiguous slice of the topk
axis (or the full axis when `num_kv_splits=1`) and writes a partial
`(out/e_sum, lse)` tile to a mid buffer. Stage 2 merges the splits via
online-softmax rescaling — pattern from `triton_decode_attention.py`.
"""

import functools

import torch

from vllm.triton_utils import LOG2E, LOGE2, tl, triton
from vllm.utils.platform_utils import num_compute_units

# DeepSeek-V3.2 / GLM-5 sparse MLA shape constants.
_BLOCK_DMODEL = 512
_BLOCK_DPE = 64
_BLOCK_DV = 512
_DIM_QK = _BLOCK_DMODEL + _BLOCK_DPE  # 576

_BLOCK_H = 16
# Smallest BLOCK_N the autotune sweep offers; only used for the topk-divisibility
# check at dispatch time.
_MIN_BLOCK_N = 16

# Merge kernel is launch-bound on a (1, 1) grid — one CTA per token starves the
# SMs. Spread across heads and DV tiles (pattern from FlashMLA's combine kernel
# at FlashMLA/csrc/smxx/decode/combine/combine.cu:22-27). BLOCK_H=1 so each
# of the 8 per-rank heads runs concurrently; BLOCK_DV_TILE=128 splits the 512
# output lanes into 4 tiles.
_MERGE_BLOCK_H = 1
_MERGE_BLOCK_DV_TILE = 128
assert _BLOCK_DV % _MERGE_BLOCK_DV_TILE == 0
_NUM_MERGE_DV_TILES = _BLOCK_DV // _MERGE_BLOCK_DV_TILE

# Separate config sweeps for the single-pass (prefill) and split-KV (decode)
# entry points. Per A100/SM80 sweeps:
#   - Single-pass ("final") kernel at prefill M>>1 prefers BLOCK_N=16 with few
#     warps; the wider configs in the combined sweep were landing 1.3–1.5×
#     slower because autotune's key omits M and the cached pick was a bad
#     compromise.
#   - Split kernel at decode M=1 prefers BLOCK_N=32/num_warps=4 across every
#     split count we tested.
# Each kernel only ever runs in its own regime (see `_choose_num_kv_splits`),
# so we can tune each independently.
_FINAL_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 16}, num_warps=nw, num_stages=ns)
    for nw in (2, 4)
    for ns in (2, 4)
]
_SPLIT_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=ns) for ns in (2, 4)
]

# Split count candidates that `_choose_num_kv_splits` can return; also the set
# `_warmup_autotune` pre-compiles so the first decode does not pay the sweep cost.
KV_SPLITS_CANDIDATES = (1, 2, 4, 8, 16)

# Split-KV heuristic tuning.
# At topk=2048 (DSv3.2/GLM-5.1) this unlocks 16-way split for decode, which
# benches ~1.3× faster than 8-way on A100 SM80 at BLOCK_N=32/num_warps=4.
_MIN_TOPK_PER_SPLIT = 128  # below this, per-split work is too small to amortize
_SPLIT_MAX_OCCUPANCY = 4  # skip split when baseline grid fills >=1/4 of SMs


@triton.jit
def _sparse_mla_compute_tile(
    q_buffer,
    k_buffer,  # V is the first BLOCK_DV lanes of each row of k_buffer.
    indices_ptr,
    cur_q,
    cur_head,
    cur_kv_head_id,
    mask_h,
    split_start,
    split_end,
    seq_kv,
    stride_q_token,
    stride_q_head,
    stride_kv_token,
    stride_kv_head,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
):
    """Shared stage-1 body: load Q, run the sparse online-softmax loop over
    `[split_start, split_end)` of the topk axis, return accumulators."""
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dpe = offs_dpe < BLOCK_DMODEL + BLOCK_DPE

    q = tl.load(
        q_buffer
        + cur_q * stride_q_token
        + cur_head[:, None] * stride_q_head
        + offs_d[None, :],
        mask=mask_h[:, None],
        other=0.0,
    )
    qpe = tl.load(
        q_buffer
        + cur_q * stride_q_token
        + cur_head[:, None] * stride_q_head
        + offs_dpe[None, :],
        mask=(mask_h[:, None]) & (mask_dpe[None, :]),
        other=0.0,
    )

    # Large negative but finite sentinel for masked-out positions. `-inf`
    # would give `-inf - -inf = NaN` when a whole BLOCK_N tile is masked
    # (common in short prefill where most topk slots are -1); a finite value
    # keeps `sentinel - sentinel = 0` and `exp2(0) = 1`, and the
    # corresponding v slots are already loaded as 0.
    NEG_LARGE = -1.0e30
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) + NEG_LARGE
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    for start_indice in range(split_start, split_end, BLOCK_N):
        offs_indice = start_indice + tl.arange(0, BLOCK_N)
        mask_indice = offs_indice < split_end
        indices = tl.load(
            indices_ptr
            + cur_q * stride_indices_token
            + cur_kv_head_id * stride_indices_head
            + offs_indice,
            mask=mask_indice,
            other=-1,
        )
        mask_kv = (indices >= 0) & (indices < seq_kv)

        offs_k = (
            indices[None, :] * stride_kv_token
            + cur_kv_head_id * stride_kv_head
            + offs_d[:, None]
        )
        k = tl.load(k_buffer + offs_k, mask=mask_kv[None, :], other=0.0)
        qk = tl.dot(q, k.to(q.dtype))

        offs_kpe = (
            indices[None, :] * stride_kv_token
            + cur_kv_head_id * stride_kv_head
            + offs_dpe[:, None]
        )
        kpe = tl.load(
            k_buffer + offs_kpe,
            mask=(mask_kv[None, :]) & (mask_dpe[:, None]),
            other=0.0,
        )
        qk += tl.dot(qpe, kpe.to(q.dtype))

        qk *= sm_scale
        qk = tl.where((mask_h[:, None]) & (mask_kv[None, :]), qk, NEG_LARGE)

        offs_v = (
            indices[:, None] * stride_kv_token
            + cur_kv_head_id * stride_kv_head
            + offs_dv[None, :]
        )
        v = tl.load(k_buffer + offs_v, mask=mask_kv[:, None], other=0.0)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    return acc, e_max, e_sum


@triton.autotune(configs=_FINAL_AUTOTUNE_CONFIGS, key=["index_topk", "kv_group_num"])
@triton.jit
def _sparse_mla_kernel_final(
    q_buffer,
    k_buffer,
    indices_ptr,
    out_ptr,
    seq_kv,
    h_q,
    stride_q_token,
    stride_q_head,
    stride_kv_token,
    stride_kv_head,
    stride_out_token,
    stride_out_head,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    index_topk: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
):
    """Single-pass fast path: full topk, write final bf16 output directly."""
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    acc, e_max, e_sum = _sparse_mla_compute_tile(
        q_buffer,
        k_buffer,
        indices_ptr,
        cur_q,
        cur_head,
        cur_kv_head_id,
        mask_h,
        0,
        index_topk,
        seq_kv,
        stride_q_token,
        stride_q_head,
        stride_kv_token,
        stride_kv_head,
        stride_indices_token,
        stride_indices_head,
        sm_scale,
        BLOCK_H,
        BLOCK_N,
        BLOCK_DV,
        BLOCK_DMODEL,
        BLOCK_DPE,
    )

    # Guard against queries with zero valid KV (e_sum == 0 → NaN from 0/0).
    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    offs_dv = tl.arange(0, BLOCK_DV)
    tl.store(
        out_ptr
        + cur_q * stride_out_token
        + cur_head[:, None] * stride_out_head
        + offs_dv[None, :],
        (acc / e_sum_safe[:, None]).to(tl.bfloat16),
        mask=mask_h[:, None],
    )


@triton.autotune(
    configs=_SPLIT_AUTOTUNE_CONFIGS,
    key=["index_topk", "NUM_KV_SPLITS", "kv_group_num"],
)
@triton.jit
def _sparse_mla_kernel_split(
    q_buffer,
    k_buffer,
    indices_ptr,
    mid_out_ptr,
    seq_kv,
    h_q,
    stride_q_token,
    stride_q_head,
    stride_kv_token,
    stride_kv_head,
    stride_mid_token,
    stride_mid_head,
    stride_mid_split,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    index_topk: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    LOGE2: tl.constexpr,
):
    """Stage 1 of split-KV: process one slice of the topk axis and write
    its `(out_partial, lse_partial)` into the mid buffer."""
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    split_topk: tl.constexpr = tl.cdiv(index_topk, NUM_KV_SPLITS)
    split_start = split_kv_id * split_topk
    split_end = tl.minimum(split_start + split_topk, index_topk)

    acc, e_max, e_sum = _sparse_mla_compute_tile(
        q_buffer,
        k_buffer,
        indices_ptr,
        cur_q,
        cur_head,
        cur_kv_head_id,
        mask_h,
        split_start,
        split_end,
        seq_kv,
        stride_q_token,
        stride_q_head,
        stride_kv_token,
        stride_kv_head,
        stride_indices_token,
        stride_indices_head,
        sm_scale,
        BLOCK_H,
        BLOCK_N,
        BLOCK_DV,
        BLOCK_DMODEL,
        BLOCK_DPE,
    )

    # Partial output and natural-log LSE for stage-2 merge.
    # When a split has no valid KV (`e_sum == 0`), guard the divide so the
    # mid buffer holds 0 instead of NaN; otherwise the `0 * NaN = NaN` term
    # in stage 2 would poison every other split.
    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    offs_dv = tl.arange(0, BLOCK_DV)
    mid_base_2d = (
        mid_out_ptr
        + cur_q * stride_mid_token
        + cur_head[:, None] * stride_mid_head
        + split_kv_id * stride_mid_split
    )
    tl.store(
        mid_base_2d + offs_dv[None, :],
        acc / e_sum_safe[:, None],
        mask=mask_h[:, None],
    )
    mid_lse_ptr = (
        mid_out_ptr
        + cur_q * stride_mid_token
        + cur_head * stride_mid_head
        + split_kv_id * stride_mid_split
        + BLOCK_DV
    )
    tl.store(mid_lse_ptr, (e_max + tl.log2(e_sum)) * LOGE2, mask=mask_h)


@triton.jit
def _sparse_mla_merge_kernel(
    mid_out_ptr,
    out_ptr,
    h_q,
    stride_mid_token,
    stride_mid_head,
    stride_mid_split,
    stride_out_token,
    stride_out_head,
    NUM_KV_SPLITS: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DV_TILE: tl.constexpr,
):
    """Stage 2: N-way online-softmax merge of per-split `(out, lse)` tiles.

    Grid is `(num_tokens, num_head_groups, num_dv_tiles)`. Each program handles
    `BLOCK_H` heads × `BLOCK_DV_TILE` output-dim lanes. The LSE reduction is
    identical across DV tiles for the same (token, head) — each program
    recomputes it locally, which is cheap (O(NUM_KV_SPLITS) scalars) and
    avoids inter-CTA synchronization.
    """
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_dv_tile = tl.program_id(2)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    offs_dv = cur_dv_tile * BLOCK_DV_TILE + tl.arange(0, BLOCK_DV_TILE)
    mask_dv = offs_dv < BLOCK_DV
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV_TILE], dtype=tl.float32)

    mid_base_2d = (
        mid_out_ptr + cur_q * stride_mid_token + cur_head[:, None] * stride_mid_head
    )
    mid_lse_1d = (
        mid_out_ptr + cur_q * stride_mid_token + cur_head * stride_mid_head + BLOCK_DV
    )

    for split_kv_id in range(NUM_KV_SPLITS):
        tv = tl.load(
            mid_base_2d + split_kv_id * stride_mid_split + offs_dv[None, :],
            mask=mask_h[:, None] & mask_dv[None, :],
            other=0.0,
        )
        tlogic = tl.load(
            mid_lse_1d + split_kv_id * stride_mid_split,
            mask=mask_h,
            other=-float("inf"),
        )
        n_e_max = tl.maximum(tlogic, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        exp_logic = tl.exp(tlogic - n_e_max)
        acc = acc * old_scale[:, None] + exp_logic[:, None] * tv
        e_sum = e_sum * old_scale + exp_logic
        e_max = n_e_max

    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    tl.store(
        out_ptr
        + cur_q * stride_out_token
        + cur_head[:, None] * stride_out_head
        + offs_dv[None, :],
        (acc / e_sum_safe[:, None]).to(tl.bfloat16),
        mask=mask_h[:, None] & mask_dv[None, :],
    )


@functools.lru_cache(maxsize=256)
def _choose_num_kv_splits(
    num_tokens: int, num_head_groups: int, index_topk: int, sm_count: int
) -> int:
    """Pick a power-of-2 split count that fills the device without dropping
    per-split work below _MIN_TOPK_PER_SPLIT. Returns 1 when the single-pass
    grid already reaches ~1/_SPLIT_MAX_OCCUPANCY utilization.
    """
    baseline = num_tokens * num_head_groups
    if baseline == 0 or baseline * _SPLIT_MAX_OCCUPANCY >= sm_count:
        return 1
    ideal = triton.next_power_of_2(max(1, index_topk // _MIN_TOPK_PER_SPLIT))
    max_splits = max(1, sm_count // baseline)
    max_splits = 1 << (max_splits.bit_length() - 1)  # floor to power of 2
    num_kv_splits = min(ideal, max_splits)
    while num_kv_splits > 1 and index_topk % num_kv_splits != 0:
        num_kv_splits //= 2
    return max(1, num_kv_splits)


def triton_sparse_mla_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    num_kv_splits: int | None = None,
    sm_count: int | None = None,
) -> torch.Tensor:
    """Sparse MLA attention over topk indices.

    Args:
        q:         [num_tokens, num_heads_q, dim_qk] bf16
        kv:        [seq_kv, num_heads_kv=1, dim_qk] bf16
        indices:   [num_tokens, num_heads_kv=1, topk] int32
        sm_scale:  softmax scale
        num_kv_splits: override auto-heuristic; None/0 = auto, 1 = force single-pass.
        sm_count:  device SM count, used by the split heuristic. If None,
            queried from the device — pass a cached value to avoid a dict
            lookup on every decode step.

    Returns:
        out:   [num_tokens, num_heads_q, _BLOCK_DV] bf16
    """
    num_tokens, num_heads_q, dim_qk = q.shape
    assert dim_qk == _DIM_QK, (
        f"sparse MLA kernel requires dim_qk={_DIM_QK} (DeepSeek-V3.2 / GLM-5), "
        f"got {dim_qk}"
    )
    assert kv.shape[1] == 1 and kv.shape[2] == _DIM_QK
    index_topk = indices.shape[2]
    assert index_topk % _MIN_BLOCK_N == 0, (
        f"topk ({index_topk}) must be a multiple of the smallest autotune "
        f"BLOCK_N ({_MIN_BLOCK_N})"
    )

    kv_group_num = num_heads_q
    num_head_groups = triton.cdiv(num_heads_q, min(_BLOCK_H, kv_group_num))

    if num_kv_splits is None or num_kv_splits == 0:
        if sm_count is None:
            sm_count = num_compute_units(q.device.index)
        num_kv_splits = _choose_num_kv_splits(
            num_tokens, num_head_groups, index_topk, sm_count
        )

    out = torch.empty(
        (num_tokens, num_heads_q, _BLOCK_DV),
        dtype=torch.bfloat16,
        device=q.device,
    )

    if num_kv_splits == 1:
        _sparse_mla_kernel_final[(num_tokens, num_head_groups)](
            q_buffer=q,
            k_buffer=kv,
            indices_ptr=indices,
            out_ptr=out,
            seq_kv=kv.shape[0],
            h_q=num_heads_q,
            stride_q_token=q.stride(0),
            stride_q_head=q.stride(1),
            stride_kv_token=kv.stride(0),
            stride_kv_head=kv.stride(1),
            stride_out_token=out.stride(0),
            stride_out_head=out.stride(1),
            stride_indices_token=indices.stride(0),
            stride_indices_head=indices.stride(1),
            sm_scale=sm_scale * LOG2E,
            index_topk=index_topk,
            kv_group_num=kv_group_num,
            BLOCK_H=_BLOCK_H,
            BLOCK_DV=_BLOCK_DV,
            BLOCK_DMODEL=_BLOCK_DMODEL,
            BLOCK_DPE=_BLOCK_DPE,
        )
        return out

    # Split-KV: partial fp32 output + LSE per (token, head, split).
    mid_out = torch.empty(
        (num_tokens, num_heads_q, num_kv_splits, _BLOCK_DV + 1),
        dtype=torch.float32,
        device=q.device,
    )
    _sparse_mla_kernel_split[(num_tokens, num_head_groups, num_kv_splits)](
        q_buffer=q,
        k_buffer=kv,
        indices_ptr=indices,
        mid_out_ptr=mid_out,
        seq_kv=kv.shape[0],
        h_q=num_heads_q,
        stride_q_token=q.stride(0),
        stride_q_head=q.stride(1),
        stride_kv_token=kv.stride(0),
        stride_kv_head=kv.stride(1),
        stride_mid_token=mid_out.stride(0),
        stride_mid_head=mid_out.stride(1),
        stride_mid_split=mid_out.stride(2),
        stride_indices_token=indices.stride(0),
        stride_indices_head=indices.stride(1),
        sm_scale=sm_scale * LOG2E,
        index_topk=index_topk,
        NUM_KV_SPLITS=num_kv_splits,
        kv_group_num=kv_group_num,
        BLOCK_H=_BLOCK_H,
        BLOCK_DV=_BLOCK_DV,
        BLOCK_DMODEL=_BLOCK_DMODEL,
        BLOCK_DPE=_BLOCK_DPE,
        LOGE2=LOGE2,
    )

    _sparse_mla_merge_kernel[(num_tokens, num_heads_q, _NUM_MERGE_DV_TILES)](
        mid_out_ptr=mid_out,
        out_ptr=out,
        h_q=num_heads_q,
        stride_mid_token=mid_out.stride(0),
        stride_mid_head=mid_out.stride(1),
        stride_mid_split=mid_out.stride(2),
        stride_out_token=out.stride(0),
        stride_out_head=out.stride(1),
        NUM_KV_SPLITS=num_kv_splits,
        kv_group_num=kv_group_num,
        BLOCK_H=_MERGE_BLOCK_H,
        BLOCK_DV=_BLOCK_DV,
        BLOCK_DV_TILE=_MERGE_BLOCK_DV_TILE,
        num_warps=2,
    )
    return out
