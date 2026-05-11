# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable sparse MLA Triton kernels."""

import math

import torch

from vllm.triton_utils import LOG2E, LOGE2, tl, triton
from vllm.v1.attention.backends.mla.sparse_mla_env import (
    triton_sparse_mla_head_block_size,
)

_SPLITKV_HEAD_BLOCK = 16
_SPLITKV_MERGE_HEAD_BLOCK = 1
_SPLITKV_BLOCK_N = 32
_SPLITKV_MERGE_BLOCK_D = 128
_SPLITKV_MIN_CANDIDATES_PER_SPLIT = 128
_SPLITKV_MEDIUM_BATCH_MIN_TOKENS = 16
_SPLITKV_MEDIUM_BATCH_CANDIDATES_PER_SPLIT = 512
_SPLITKV_MEDIUM_BATCH_MAX_SPLITS = 8
_SPLITKV_MAX_OCCUPANCY = 4


def sparse_mla_decode_head_block_size(num_decode_tokens: int) -> int:
    """Choose the SM12x sparse MLA head grouping for decode kernels.

    Single-token decode is latency sensitive and does best with one head per
    program. Once there are enough query tokens, grouping heads lets the kernel
    reuse each dequantized KV row across multiple heads.
    """

    configured_head_block_size = triton_sparse_mla_head_block_size()
    if configured_head_block_size is not None:
        return configured_head_block_size
    if num_decode_tokens <= 4:
        return 1
    if num_decode_tokens < 16:
        return 2
    return 4


def _next_power_of_2(value: int) -> int:
    return 1 << max(0, value - 1).bit_length()


def choose_sparse_mla_splitkv_splits(
    num_tokens: int,
    num_heads: int,
    num_candidates: int,
    sm_count: int,
    head_block_size: int = _SPLITKV_HEAD_BLOCK,
) -> int:
    if (
        num_tokens <= 0
        or num_heads <= 0
        or num_candidates <= 0
        or sm_count <= 0
        or head_block_size <= 0
    ):
        return 1

    num_head_groups = math.ceil(num_heads / min(head_block_size, num_heads))
    baseline = num_tokens * num_head_groups
    if baseline == 0:
        return 1

    ideal = _next_power_of_2(
        max(1, num_candidates // _SPLITKV_MIN_CANDIDATES_PER_SPLIT)
    )
    max_splits = max(1, (sm_count * _SPLITKV_MAX_OCCUPANCY) // baseline)
    max_splits = 1 << (max_splits.bit_length() - 1)
    num_splits = min(ideal, max_splits)
    if (
        num_tokens >= _SPLITKV_MEDIUM_BATCH_MIN_TOKENS
        and baseline <= sm_count * _SPLITKV_MAX_OCCUPANCY
    ):
        medium_batch_splits = _next_power_of_2(
            max(1, num_candidates // _SPLITKV_MEDIUM_BATCH_CANDIDATES_PER_SPLIT)
        )
        medium_batch_splits = min(
            ideal, medium_batch_splits, _SPLITKV_MEDIUM_BATCH_MAX_SPLITS
        )
        num_splits = max(num_splits, medium_batch_splits)
    while num_splits > 1 and num_candidates % num_splits != 0:
        num_splits //= 2
    return max(1, num_splits)


@triton.jit
def _splitkv_sparse_mla_stage1_kernel(
    q_ptr,
    kv_ptr,
    valid_ptr,
    mid_ptr,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kvt: tl.constexpr,
    stride_kvc: tl.constexpr,
    stride_kvd: tl.constexpr,
    stride_vt: tl.constexpr,
    stride_vc: tl.constexpr,
    stride_mt: tl.constexpr,
    stride_mh: tl.constexpr,
    stride_ms: tl.constexpr,
    num_heads: tl.constexpr,
    num_candidates: tl.constexpr,
    scale: tl.constexpr,
    num_splits: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOGE2_VALUE: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_group = tl.program_id(1)
    split_id = tl.program_id(2)

    offs_h = head_group * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    mask_h = offs_h < num_heads
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr
        + token_id * stride_qt
        + offs_h[:, None] * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=mask_h[:, None],
        other=0.0,
    )

    split_size: tl.constexpr = tl.cdiv(num_candidates, num_splits)
    split_start = split_id * split_size
    split_end = tl.minimum(split_start + split_size, num_candidates)

    neg_large = -1.0e30
    e_max = tl.full((HEAD_BLOCK,), neg_large, dtype=tl.float32)
    e_sum = tl.zeros((HEAD_BLOCK,), dtype=tl.float32)
    acc = tl.zeros((HEAD_BLOCK, BLOCK_D), dtype=tl.float32)

    for cand_start in range(split_start, split_end, BLOCK_N):
        offs_c = cand_start + tl.arange(0, BLOCK_N)
        mask_c = offs_c < split_end
        valid = tl.load(
            valid_ptr + token_id * stride_vt + offs_c * stride_vc,
            mask=mask_c,
            other=0,
        )
        mask_kv = mask_c & valid
        k = tl.load(
            kv_ptr
            + token_id * stride_kvt
            + offs_c[:, None] * stride_kvc
            + offs_d[None, :] * stride_kvd,
            mask=mask_kv[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k.to(q.dtype))) * scale
        qk = tl.where(mask_h[:, None] & mask_kv[None, :], qk, neg_large)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(k.dtype), k)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    mid_base = (
        mid_ptr
        + token_id * stride_mt
        + offs_h[:, None] * stride_mh
        + split_id * stride_ms
    )
    tl.store(
        mid_base + offs_d[None, :],
        acc / e_sum_safe[:, None],
        mask=mask_h[:, None],
    )
    tl.store(
        mid_ptr
        + token_id * stride_mt
        + offs_h * stride_mh
        + split_id * stride_ms
        + BLOCK_D,
        (e_max + tl.log2(e_sum)) * LOGE2_VALUE,
        mask=mask_h,
    )


@triton.jit
def _splitkv_sparse_mla_merge_kernel(
    mid_ptr,
    sink_ptr,
    output_ptr,
    stride_mt: tl.constexpr,
    stride_mh: tl.constexpr,
    stride_ms: tl.constexpr,
    stride_out_t: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    num_heads: tl.constexpr,
    num_splits: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_TILE: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_group = tl.program_id(1)
    d_tile = tl.program_id(2)

    offs_h = head_group * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    mask_h = offs_h < num_heads
    offs_d = d_tile * BLOCK_D_TILE + tl.arange(0, BLOCK_D_TILE)
    mask_d = offs_d < BLOCK_D

    e_max = tl.full((HEAD_BLOCK,), -float("inf"), dtype=tl.float32)
    e_sum = tl.zeros((HEAD_BLOCK,), dtype=tl.float32)
    acc = tl.zeros((HEAD_BLOCK, BLOCK_D_TILE), dtype=tl.float32)
    mid_base = mid_ptr + token_id * stride_mt + offs_h[:, None] * stride_mh
    mid_lse = mid_ptr + token_id * stride_mt + offs_h * stride_mh + BLOCK_D

    for split_id in range(num_splits):
        part = tl.load(
            mid_base + split_id * stride_ms + offs_d[None, :],
            mask=mask_h[:, None] & mask_d[None, :],
            other=0.0,
        )
        lse = tl.load(
            mid_lse + split_id * stride_ms,
            mask=mask_h,
            other=-float("inf"),
        )
        n_e_max = tl.maximum(lse, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        part_scale = tl.exp(lse - n_e_max)
        acc = acc * old_scale[:, None] + part * part_scale[:, None]
        e_sum = e_sum * old_scale + part_scale
        e_max = n_e_max

    sink = tl.load(sink_ptr + offs_h, mask=mask_h, other=-float("inf"))
    n_e_max = tl.maximum(sink, e_max)
    value_scale = tl.exp(e_max - n_e_max)
    sink_scale = tl.exp(sink - n_e_max)
    denom = e_sum * value_scale + sink_scale
    denom = tl.where(denom > 0, denom, 1.0)
    merged = acc * value_scale[:, None] / denom[:, None]

    tl.store(
        output_ptr
        + token_id * stride_out_t
        + offs_h[:, None] * stride_oh
        + offs_d[None, :] * stride_od,
        merged,
        mask=mask_h[:, None] & mask_d[None, :],
    )


def splitkv_sparse_mla_attention_with_sink(
    q: torch.Tensor,
    kv: torch.Tensor,
    valid_tokens: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    mid: torch.Tensor,
    num_splits: int,
    num_heads: int | None = None,
) -> None:
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]

    assert q.dim() == 3, f"Expected q shape [T, H, D], got {q.shape}"
    assert kv.dim() == 3, f"Expected kv shape [T, K, D], got {kv.shape}"
    assert valid_tokens.shape == kv.shape[:2]
    assert q.shape[0] == kv.shape[0]
    assert q.shape[-1] == kv.shape[-1]
    assert output.shape[0] == q.shape[0]
    assert output.shape[2] == q.shape[-1]
    assert q.is_cuda and kv.is_cuda and valid_tokens.is_cuda
    assert attn_sink.is_cuda and output.is_cuda and mid.is_cuda

    active_heads = num_heads if num_heads is not None else output.shape[1]
    assert active_heads <= q.shape[1]
    assert active_heads <= output.shape[1]
    assert active_heads <= attn_sink.shape[0]

    num_tokens, _, head_dim = q.shape
    num_candidates = kv.shape[1]
    assert mid.shape == (num_tokens, active_heads, num_splits, head_dim + 1)
    num_head_groups = triton.cdiv(active_heads, _SPLITKV_HEAD_BLOCK)
    _splitkv_sparse_mla_stage1_kernel[(num_tokens, num_head_groups, num_splits)](
        q,
        kv,
        valid_tokens,
        mid,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        valid_tokens.stride(0),
        valid_tokens.stride(1),
        mid.stride(0),
        mid.stride(1),
        mid.stride(2),
        active_heads,
        num_candidates,
        scale * LOG2E,
        num_splits,
        HEAD_BLOCK=_SPLITKV_HEAD_BLOCK,
        BLOCK_N=_SPLITKV_BLOCK_N,
        BLOCK_D=head_dim,
        LOGE2_VALUE=LOGE2,
        num_warps=4,
    )
    _splitkv_sparse_mla_merge_kernel[
        (num_tokens, active_heads, triton.cdiv(head_dim, _SPLITKV_MERGE_BLOCK_D))
    ](
        mid,
        attn_sink,
        output,
        mid.stride(0),
        mid.stride(1),
        mid.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        active_heads,
        num_splits,
        HEAD_BLOCK=_SPLITKV_MERGE_HEAD_BLOCK,
        BLOCK_D=head_dim,
        BLOCK_D_TILE=_SPLITKV_MERGE_BLOCK_D,
        num_warps=2,
    )


@triton.jit
def _merge_two_subsets_with_sink_kernel(
    out0_ptr,
    lse0_ptr,
    out1_ptr,
    lse1_ptr,
    sink_ptr,
    output_ptr,
    stride_out0_t: tl.constexpr,
    stride_out0_h: tl.constexpr,
    stride_out0_d: tl.constexpr,
    stride_lse0_t: tl.constexpr,
    stride_lse0_h: tl.constexpr,
    stride_out1_t: tl.constexpr,
    stride_out1_h: tl.constexpr,
    stride_out1_d: tl.constexpr,
    stride_lse1_t: tl.constexpr,
    stride_lse1_h: tl.constexpr,
    stride_output_t: tl.constexpr,
    stride_output_h: tl.constexpr,
    stride_output_d: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_head = tl.program_id(0)
    block_d = tl.program_id(1)
    token_idx = token_head // num_heads
    head_idx = token_head - token_idx * num_heads
    offsets = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offsets < head_dim

    lse0 = tl.load(lse0_ptr + token_idx * stride_lse0_t + head_idx * stride_lse0_h)
    lse1 = tl.load(lse1_ptr + token_idx * stride_lse1_t + head_idx * stride_lse1_h)
    sink = tl.load(sink_ptr + head_idx)
    merge_max = tl.maximum(tl.maximum(lse0, lse1), sink)

    weight0 = tl.exp(lse0 - merge_max)
    weight1 = tl.exp(lse1 - merge_max)
    weight_sink = tl.exp(sink - merge_max)
    denom = weight0 + weight1 + weight_sink

    out0 = tl.load(
        out0_ptr
        + token_idx * stride_out0_t
        + head_idx * stride_out0_h
        + offsets * stride_out0_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    out1 = tl.load(
        out1_ptr
        + token_idx * stride_out1_t
        + head_idx * stride_out1_h
        + offsets * stride_out1_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    merged = (out0 * weight0 + out1 * weight1) / denom
    tl.store(
        output_ptr
        + token_idx * stride_output_t
        + head_idx * stride_output_h
        + offsets * stride_output_d,
        merged,
        mask=mask,
    )


def merge_two_sparse_mla_subsets_with_sink(
    subset0_output: torch.Tensor,
    subset0_lse: torch.Tensor,
    subset1_output: torch.Tensor,
    subset1_lse: torch.Tensor,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
) -> None:
    assert subset0_output.shape == subset1_output.shape
    assert subset0_output.shape == output.shape
    assert subset0_lse.shape == subset1_lse.shape
    assert subset0_lse.shape == subset0_output.shape[:2]
    assert attn_sink.shape[0] == subset0_output.shape[1]
    assert subset0_output.is_cuda
    assert subset1_output.is_cuda
    assert output.is_cuda

    num_tokens, num_heads, head_dim = subset0_output.shape
    block_d = min(128, triton.next_power_of_2(head_dim))
    grid = (num_tokens * num_heads, triton.cdiv(head_dim, block_d))
    _merge_two_subsets_with_sink_kernel[grid](
        subset0_output,
        subset0_lse,
        subset1_output,
        subset1_lse,
        attn_sink,
        output,
        subset0_output.stride(0),
        subset0_output.stride(1),
        subset0_output.stride(2),
        subset0_lse.stride(0),
        subset0_lse.stride(1),
        subset1_output.stride(0),
        subset1_output.stride(1),
        subset1_output.stride(2),
        subset1_lse.stride(0),
        subset1_lse.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        num_heads,
        head_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )


@triton.jit
def _merge_single_subset_with_sink_kernel(
    subset_output_ptr,
    subset_lse_ptr,
    sink_ptr,
    output_ptr,
    stride_subset_t: tl.constexpr,
    stride_subset_h: tl.constexpr,
    stride_subset_d: tl.constexpr,
    stride_lse_t: tl.constexpr,
    stride_lse_h: tl.constexpr,
    stride_output_t: tl.constexpr,
    stride_output_h: tl.constexpr,
    stride_output_d: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_head = tl.program_id(0)
    block_d = tl.program_id(1)
    token_idx = token_head // num_heads
    head_idx = token_head - token_idx * num_heads
    offsets = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offsets < head_dim

    subset_lse = tl.load(
        subset_lse_ptr + token_idx * stride_lse_t + head_idx * stride_lse_h
    )
    sink = tl.load(sink_ptr + head_idx)
    merge_max = tl.maximum(subset_lse, sink)

    subset_weight = tl.exp(subset_lse - merge_max)
    sink_weight = tl.exp(sink - merge_max)
    denom = subset_weight + sink_weight
    subset_output = tl.load(
        subset_output_ptr
        + token_idx * stride_subset_t
        + head_idx * stride_subset_h
        + offsets * stride_subset_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    merged = subset_output * subset_weight / denom
    tl.store(
        output_ptr
        + token_idx * stride_output_t
        + head_idx * stride_output_h
        + offsets * stride_output_d,
        merged,
        mask=mask,
    )


def merge_sparse_mla_subset_with_sink(
    subset_output: torch.Tensor,
    subset_lse: torch.Tensor,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
) -> None:
    assert subset_output.shape == output.shape
    assert subset_lse.shape == subset_output.shape[:2]
    assert attn_sink.shape[0] == subset_output.shape[1]
    assert subset_output.is_cuda
    assert subset_lse.is_cuda
    assert attn_sink.is_cuda
    assert output.is_cuda

    num_tokens, num_heads, head_dim = subset_output.shape
    block_d = min(128, triton.next_power_of_2(head_dim))
    grid = (num_tokens * num_heads, triton.cdiv(head_dim, block_d))
    _merge_single_subset_with_sink_kernel[grid](
        subset_output,
        subset_lse,
        attn_sink,
        output,
        subset_output.stride(0),
        subset_output.stride(1),
        subset_output.stride(2),
        subset_lse.stride(0),
        subset_lse.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        num_heads,
        head_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )


@triton.jit
def _build_combined_decode_valid_mask_kernel(
    output_ptr,
    slot_ids_ptr,
    topk_lens_ptr,
    swa_lens_ptr,
    stride_output_t: tl.constexpr,
    stride_output_c: tl.constexpr,
    stride_slot_t: tl.constexpr,
    stride_slot_c: tl.constexpr,
    num_compressed_candidates: tl.constexpr,
    num_candidates: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_C)
    candidate_mask = offsets < num_candidates

    topk_lens = tl.load(topk_lens_ptr + token_idx)
    swa_lens = tl.load(swa_lens_ptr + token_idx)
    is_compressed = offsets < num_compressed_candidates
    swa_offsets = offsets - num_compressed_candidates
    slot_ids = tl.load(
        slot_ids_ptr + token_idx * stride_slot_t + offsets * stride_slot_c,
        mask=is_compressed,
        other=-1,
    )
    valid_compressed = is_compressed & (offsets < topk_lens) & (slot_ids >= 0)
    valid_swa = (~is_compressed) & (swa_offsets < swa_lens)
    valid = valid_compressed | valid_swa
    tl.store(
        output_ptr + token_idx * stride_output_t + offsets * stride_output_c,
        valid,
        mask=candidate_mask,
    )


def build_combined_sparse_mla_decode_valid_mask(
    output: torch.Tensor,
    compressed_slot_ids: torch.Tensor,
    topk_lens: torch.Tensor,
    swa_lens: torch.Tensor,
) -> None:
    """Build `[compressed, SWA]` validity mask for SM12x decode."""
    if compressed_slot_ids.dim() == 3:
        assert compressed_slot_ids.shape[1] == 1
        compressed_slot_ids = compressed_slot_ids[:, 0, :]

    assert output.dim() == 2
    assert output.dtype == torch.bool
    assert compressed_slot_ids.dim() == 2
    assert output.shape[0] == compressed_slot_ids.shape[0]
    assert output.shape[0] == topk_lens.shape[0]
    assert output.shape[0] == swa_lens.shape[0]
    assert output.shape[1] >= compressed_slot_ids.shape[1]
    assert output.is_cuda
    assert compressed_slot_ids.is_cuda
    assert topk_lens.is_cuda
    assert swa_lens.is_cuda

    num_candidates = output.shape[1]
    block_c = triton.next_power_of_2(num_candidates)
    _build_combined_decode_valid_mask_kernel[(output.shape[0],)](
        output,
        compressed_slot_ids,
        topk_lens,
        swa_lens,
        output.stride(0),
        output.stride(1),
        compressed_slot_ids.stride(0),
        compressed_slot_ids.stride(1),
        compressed_slot_ids.shape[1],
        num_candidates,
        BLOCK_C=block_c,
        num_warps=4,
    )


def matmul_sparse_mla_attention_with_sink(
    q: torch.Tensor,
    kv: torch.Tensor,
    valid_tokens: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    num_heads: int | None = None,
    score_buffer: torch.Tensor | None = None,
    head_block_size: int = 1,
    value_block_size: int | None = None,
    candidate_block_size: int | None = None,
) -> None:
    """Compute sink-aware sparse MLA over materialized BF16 KV.

    This path intentionally dequantizes/gathers KV once, computes scores with
    batched matrix multiplication, and finishes the sink-aware value reduction
    in Triton. It is useful for the SM12x decode path where the direct Triton
    kernel otherwise repeats fp8_ds_mla dequantization once per head group.
    """
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]

    assert q.dim() == 3, f"Expected q shape [T, H, D], got {q.shape}"
    assert kv.dim() == 3, f"Expected kv shape [T, K, D], got {kv.shape}"
    assert valid_tokens.shape == kv.shape[:2]
    assert q.shape[0] == kv.shape[0]
    assert q.shape[-1] == kv.shape[-1]
    assert output.shape[0] == q.shape[0]
    assert output.shape[2] == q.shape[-1]
    assert q.is_cuda and kv.is_cuda and valid_tokens.is_cuda
    assert attn_sink.is_cuda and output.is_cuda

    active_heads = num_heads if num_heads is not None else output.shape[1]
    assert active_heads <= q.shape[1]
    assert active_heads <= output.shape[1]
    assert active_heads <= attn_sink.shape[0]

    q_active = q[:, :active_heads]
    num_tokens = q.shape[0]
    num_candidates = kv.shape[1]
    if score_buffer is None:
        score_buffer = torch.empty(
            (num_tokens, active_heads, num_candidates),
            dtype=torch.float32,
            device=q.device,
        )
    assert score_buffer.shape == (num_tokens, active_heads, num_candidates)
    assert score_buffer.device == q.device
    assert score_buffer.dtype in (torch.float32, torch.bfloat16)
    if score_buffer.dtype == torch.float32:
        q_score = q_active.float()
        kv_score = kv.float()
    else:
        q_score = q_active.to(score_buffer.dtype)
        kv_score = kv.to(score_buffer.dtype)
    torch.bmm(q_score, kv_score.transpose(1, 2), out=score_buffer)
    score_buffer.mul_(scale)
    finish_materialized_sparse_mla_scores_with_sink(
        score_buffer,
        kv,
        valid_tokens,
        attn_sink,
        output,
        num_heads=active_heads,
        head_block_size=head_block_size,
        value_block_size=value_block_size,
        candidate_block_size=candidate_block_size,
    )


@triton.jit
def _finish_materialized_scores_with_sink_kernel(
    scores_ptr,
    kv_ptr,
    valid_tokens_ptr,
    attn_sink_ptr,
    output_ptr,
    stride_scores_t: tl.constexpr,
    stride_scores_h: tl.constexpr,
    stride_scores_c: tl.constexpr,
    stride_kv_t: tl.constexpr,
    stride_kv_c: tl.constexpr,
    stride_kv_d: tl.constexpr,
    stride_valid_t: tl.constexpr,
    stride_valid_c: tl.constexpr,
    stride_out_t: tl.constexpr,
    stride_out_h: tl.constexpr,
    stride_out_d: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim
    matrix_mask = head_mask[:, None] & dim_mask[None, :]

    running_max = tl.load(attn_sink_ptr + head_offsets, mask=head_mask, other=0.0).to(
        tl.float32
    )
    running_denom = tl.full((HEAD_BLOCK,), 1.0, tl.float32)
    running_acc = tl.zeros((HEAD_BLOCK, BLOCK_D), tl.float32)

    for candidate_idx in range(0, num_candidates):
        is_valid = tl.load(
            valid_tokens_ptr
            + token_idx * stride_valid_t
            + candidate_idx * stride_valid_c
        )
        if is_valid:
            score = tl.load(
                scores_ptr
                + token_idx * stride_scores_t
                + head_offsets * stride_scores_h
                + candidate_idx * stride_scores_c,
                mask=head_mask,
                other=-float("inf"),
            ).to(tl.float32)
            kv = tl.load(
                kv_ptr
                + token_idx * stride_kv_t
                + candidate_idx * stride_kv_c
                + dim_offsets * stride_kv_d,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = (
                running_acc * previous_weight[:, None]
                + kv[None, :] * candidate_weight[:, None]
            )
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    result = running_acc / running_denom[:, None]
    tl.store(
        output_ptr
        + token_idx * stride_out_t
        + head_offsets[:, None] * stride_out_h
        + dim_offsets[None, :] * stride_out_d,
        result,
        mask=matrix_mask,
    )


@triton.jit
def _finish_materialized_scores_with_sink_candidate_block_kernel(
    scores_ptr,
    kv_ptr,
    valid_tokens_ptr,
    attn_sink_ptr,
    output_ptr,
    stride_scores_t: tl.constexpr,
    stride_scores_h: tl.constexpr,
    stride_scores_c: tl.constexpr,
    stride_kv_t: tl.constexpr,
    stride_kv_c: tl.constexpr,
    stride_kv_d: tl.constexpr,
    stride_valid_t: tl.constexpr,
    stride_valid_c: tl.constexpr,
    stride_out_t: tl.constexpr,
    stride_out_h: tl.constexpr,
    stride_out_d: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dim_block_idx = tl.program_id(2)
    candidate_offsets = tl.arange(0, BLOCK_K)
    dim_offsets = dim_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < head_dim

    max_score = tl.load(attn_sink_ptr + head_idx).to(tl.float32)
    for candidate_start in range(0, num_candidates, BLOCK_K):
        candidates = candidate_start + candidate_offsets
        candidate_mask = candidates < num_candidates
        is_valid = tl.load(
            valid_tokens_ptr + token_idx * stride_valid_t + candidates * stride_valid_c,
            mask=candidate_mask,
            other=0,
        ).to(tl.int1)
        scores = tl.load(
            scores_ptr
            + token_idx * stride_scores_t
            + head_idx * stride_scores_h
            + candidates * stride_scores_c,
            mask=candidate_mask & is_valid,
            other=-float("inf"),
        ).to(tl.float32)
        max_score = tl.maximum(max_score, tl.max(scores, axis=0))

    denom = tl.exp(tl.load(attn_sink_ptr + head_idx).to(tl.float32) - max_score)
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for candidate_start in range(0, num_candidates, BLOCK_K):
        candidates = candidate_start + candidate_offsets
        candidate_mask = candidates < num_candidates
        is_valid = tl.load(
            valid_tokens_ptr + token_idx * stride_valid_t + candidates * stride_valid_c,
            mask=candidate_mask,
            other=0,
        ).to(tl.int1)
        scores = tl.load(
            scores_ptr
            + token_idx * stride_scores_t
            + head_idx * stride_scores_h
            + candidates * stride_scores_c,
            mask=candidate_mask & is_valid,
            other=-float("inf"),
        ).to(tl.float32)
        weights = tl.exp(scores - max_score)
        denom += tl.sum(weights, axis=0)
        kv = tl.load(
            kv_ptr
            + token_idx * stride_kv_t
            + candidates[:, None] * stride_kv_c
            + dim_offsets[None, :] * stride_kv_d,
            mask=(candidate_mask & is_valid)[:, None] & dim_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(kv.to(tl.float32) * weights[:, None], axis=0)

    tl.store(
        output_ptr
        + token_idx * stride_out_t
        + head_idx * stride_out_h
        + dim_offsets * stride_out_d,
        acc / denom,
        mask=dim_mask,
    )


@triton.jit
def _finish_materialized_scores_with_sink_value_block_kernel(
    scores_ptr,
    kv_ptr,
    valid_tokens_ptr,
    attn_sink_ptr,
    output_ptr,
    stride_scores_t: tl.constexpr,
    stride_scores_h: tl.constexpr,
    stride_scores_c: tl.constexpr,
    stride_kv_t: tl.constexpr,
    stride_kv_c: tl.constexpr,
    stride_kv_d: tl.constexpr,
    stride_valid_t: tl.constexpr,
    stride_valid_c: tl.constexpr,
    stride_out_t: tl.constexpr,
    stride_out_h: tl.constexpr,
    stride_out_d: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dim_block_idx = tl.program_id(2)
    dim_offsets = dim_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < head_dim

    running_max = tl.load(attn_sink_ptr + head_idx).to(tl.float32)
    running_denom = tl.full((), 1.0, tl.float32)
    running_acc = tl.zeros((BLOCK_D,), tl.float32)

    for candidate_idx in range(0, num_candidates):
        is_valid = tl.load(
            valid_tokens_ptr
            + token_idx * stride_valid_t
            + candidate_idx * stride_valid_c
        )
        if is_valid:
            score = tl.load(
                scores_ptr
                + token_idx * stride_scores_t
                + head_idx * stride_scores_h
                + candidate_idx * stride_scores_c
            ).to(tl.float32)
            kv = tl.load(
                kv_ptr
                + token_idx * stride_kv_t
                + candidate_idx * stride_kv_c
                + dim_offsets * stride_kv_d,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = running_acc * previous_weight + kv * candidate_weight
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    result = running_acc / running_denom
    tl.store(
        output_ptr
        + token_idx * stride_out_t
        + head_idx * stride_out_h
        + dim_offsets * stride_out_d,
        result,
        mask=dim_mask,
    )


def finish_materialized_sparse_mla_scores_with_sink(
    scores: torch.Tensor,
    kv: torch.Tensor,
    valid_tokens: torch.Tensor,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    num_heads: int | None = None,
    head_block_size: int = 1,
    value_block_size: int | None = None,
    candidate_block_size: int | None = None,
) -> None:
    assert scores.dim() == 3
    assert kv.dim() == 3
    assert valid_tokens.shape == kv.shape[:2]
    assert scores.shape[0] == kv.shape[0]
    assert scores.shape[2] == kv.shape[1]
    assert output.shape[0] == kv.shape[0]
    assert output.shape[2] == kv.shape[2]
    assert scores.dtype in (torch.float32, torch.bfloat16)
    assert head_block_size in (1, 2, 4)
    if value_block_size is not None:
        assert value_block_size in (64, 128, 256, 512)
    if candidate_block_size is not None:
        assert candidate_block_size in (16, 32, 64, 128)
    assert scores.is_cuda and kv.is_cuda and valid_tokens.is_cuda
    assert attn_sink.is_cuda and output.is_cuda

    active_heads = num_heads if num_heads is not None else output.shape[1]
    assert active_heads <= scores.shape[1]
    assert active_heads <= output.shape[1]
    assert active_heads <= attn_sink.shape[0]

    num_tokens, _, num_candidates = scores.shape
    head_dim = kv.shape[2]
    # Clamp BLOCK_D to the smallest power-of-2 >= head_dim (among the allowed
    # {64, 128, 256, 512}). Without this, a caller-supplied value_block_size
    # larger than head_dim wastes work on masked-off positions — e.g. DSv4
    # head_dim=192 with value_block_size=512 masks off 62.5% of D-axis work
    # in every program. Smaller caller values (intentional fine-grained
    # splits along D) are respected.
    def _smallest_block_d_covering(hd: int) -> int:
        for cand in (64, 128, 256, 512):
            if cand >= hd:
                return cand
        return 512  # head_dim > 512: BLOCK_D=512, grid splits along D

    if candidate_block_size is not None:
        target_block_d = _smallest_block_d_covering(head_dim)
        if value_block_size is None:
            block_d = target_block_d
        else:
            block_d = min(value_block_size, target_block_d)
        candidate_grid = (num_tokens, active_heads, triton.cdiv(head_dim, block_d))
        _finish_materialized_scores_with_sink_candidate_block_kernel[candidate_grid](
            scores,
            kv,
            valid_tokens,
            attn_sink,
            output,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            kv.stride(0),
            kv.stride(1),
            kv.stride(2),
            valid_tokens.stride(0),
            valid_tokens.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            head_dim,
            num_candidates,
            BLOCK_K=candidate_block_size,
            BLOCK_D=block_d,
            num_warps=8,
        )
        if output.shape[1] > active_heads:
            output[:, active_heads:].zero_()
        return

    if value_block_size is not None and value_block_size < head_dim:
        value_grid = (
            num_tokens,
            active_heads,
            triton.cdiv(head_dim, value_block_size),
        )
        _finish_materialized_scores_with_sink_value_block_kernel[value_grid](
            scores,
            kv,
            valid_tokens,
            attn_sink,
            output,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            kv.stride(0),
            kv.stride(1),
            kv.stride(2),
            valid_tokens.stride(0),
            valid_tokens.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            head_dim,
            num_candidates,
            BLOCK_D=value_block_size,
            num_warps=4,
        )
        if output.shape[1] > active_heads:
            output[:, active_heads:].zero_()
        return

    block_d = min(1024, triton.next_power_of_2(head_dim))
    head_grid = (num_tokens, triton.cdiv(active_heads, head_block_size))
    _finish_materialized_scores_with_sink_kernel[head_grid](
        scores,
        kv,
        valid_tokens,
        attn_sink,
        output,
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        valid_tokens.stride(0),
        valid_tokens.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        active_heads,
        head_dim,
        num_candidates,
        HEAD_BLOCK=head_block_size,
        BLOCK_D=block_d,
        num_warps=8,
    )
    if output.shape[1] > active_heads:
        output[:, active_heads:].zero_()


@triton.jit
def _accumulate_gathered_attention_chunk_kernel(
    q_ptr,
    kv_ptr,
    slot_ids_ptr,
    lens_ptr,
    max_score_ptr,
    denom_ptr,
    acc_ptr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_kv_t: tl.constexpr,
    stride_kv_c: tl.constexpr,
    stride_kv_d: tl.constexpr,
    stride_slot_t: tl.constexpr,
    stride_slot_c: tl.constexpr,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_acc_t: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates,
    candidate_offset,
    scale: tl.constexpr,
    HAS_SLOT_IDS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_D)
    dim_mask = offsets < head_dim

    q = tl.load(
        q_ptr + token_idx * stride_q_t + head_idx * stride_q_h + offsets * stride_q_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    state_offset = token_idx * stride_state_t + head_idx * stride_state_h
    acc_offset = (
        token_idx * stride_acc_t + head_idx * stride_acc_h + offsets * stride_acc_d
    )
    running_max = tl.load(max_score_ptr + state_offset)
    running_denom = tl.load(denom_ptr + state_offset)
    running_acc = tl.load(acc_ptr + acc_offset, mask=dim_mask, other=0.0).to(tl.float32)
    valid_len = tl.load(lens_ptr + token_idx)

    for candidate_idx in range(0, num_candidates):
        is_valid = (candidate_offset + candidate_idx) < valid_len
        if HAS_SLOT_IDS:
            slot_id = tl.load(
                slot_ids_ptr + token_idx * stride_slot_t + candidate_idx * stride_slot_c
            )
            is_valid = is_valid & (slot_id >= 0)

        if is_valid:
            kv = tl.load(
                kv_ptr
                + token_idx * stride_kv_t
                + candidate_idx * stride_kv_c
                + offsets * stride_kv_d,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            score = tl.sum(q * kv, axis=0) * scale
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = running_acc * previous_weight + kv * candidate_weight
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    tl.store(max_score_ptr + state_offset, running_max)
    tl.store(denom_ptr + state_offset, running_denom)
    tl.store(acc_ptr + acc_offset, running_acc, mask=dim_mask)


def accumulate_gathered_sparse_mla_attention_chunk(
    q: torch.Tensor,
    kv: torch.Tensor,
    lens: torch.Tensor,
    scale: float,
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
    candidate_offset: int = 0,
    slot_ids: torch.Tensor | None = None,
) -> None:
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]
    assert q.dim() == 3, f"Expected q shape [T, H, D], got {q.shape}"
    assert kv.dim() == 3, f"Expected kv shape [T, K, D], got {kv.shape}"
    assert q.shape[0] == kv.shape[0]
    assert q.shape[-1] == kv.shape[-1]
    assert lens.shape[0] == q.shape[0]
    assert max_score.shape[0] == q.shape[0]
    assert max_score.shape[1] <= q.shape[1]
    assert denom.shape == max_score.shape
    assert acc.shape == (*max_score.shape, q.shape[-1])
    assert max_score.dtype == torch.float32
    assert denom.dtype == torch.float32
    assert acc.dtype == torch.float32
    assert q.is_cuda and kv.is_cuda and lens.is_cuda
    assert max_score.is_cuda and denom.is_cuda and acc.is_cuda

    if slot_ids is not None:
        if slot_ids.dim() == 3:
            assert slot_ids.shape[1] == 1
            slot_ids = slot_ids[:, 0]
        assert slot_ids.dim() == 2
        assert slot_ids.shape == kv.shape[:2]
        assert slot_ids.is_cuda

    num_tokens, _, head_dim = q.shape
    num_heads = max_score.shape[1]
    num_candidates = kv.shape[1]
    block_d = min(1024, triton.next_power_of_2(head_dim))
    grid = (num_tokens, num_heads)
    _accumulate_gathered_attention_chunk_kernel[grid](
        q,
        kv,
        slot_ids,
        lens,
        max_score,
        denom,
        acc,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        slot_ids.stride(0) if slot_ids is not None else 0,
        slot_ids.stride(1) if slot_ids is not None else 0,
        max_score.stride(0),
        max_score.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        num_heads,
        head_dim,
        num_candidates,
        candidate_offset,
        scale,
        HAS_SLOT_IDS=slot_ids is not None,
        BLOCK_D=block_d,
        num_warps=8,
    )


@triton.jit
def _accumulate_indexed_attention_chunk_kernel(
    q_ptr,
    kv_flat_ptr,
    indices_ptr,
    lens_ptr,
    max_score_ptr,
    denom_ptr,
    acc_ptr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_kv_t,
    stride_kv_d: tl.constexpr,
    stride_indices_t: tl.constexpr,
    stride_indices_c: tl.constexpr,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_acc_t: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates,
    candidate_offset,
    scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_D)
    dim_mask = offsets < head_dim

    q = tl.load(
        q_ptr + token_idx * stride_q_t + head_idx * stride_q_h + offsets * stride_q_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    state_offset = token_idx * stride_state_t + head_idx * stride_state_h
    acc_offset = (
        token_idx * stride_acc_t + head_idx * stride_acc_h + offsets * stride_acc_d
    )
    running_max = tl.load(max_score_ptr + state_offset)
    running_denom = tl.load(denom_ptr + state_offset)
    running_acc = tl.load(acc_ptr + acc_offset, mask=dim_mask, other=0.0).to(tl.float32)
    valid_len = tl.load(lens_ptr + token_idx)

    for candidate_idx in range(0, num_candidates):
        kv_index = tl.load(
            indices_ptr
            + token_idx * stride_indices_t
            + candidate_idx * stride_indices_c
        )
        is_valid = ((candidate_offset + candidate_idx) < valid_len) & (kv_index >= 0)

        if is_valid:
            kv = tl.load(
                kv_flat_ptr
                + kv_index.to(tl.int64) * stride_kv_t
                + offsets * stride_kv_d,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            score = tl.sum(q * kv, axis=0) * scale
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = running_acc * previous_weight + kv * candidate_weight
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    tl.store(max_score_ptr + state_offset, running_max)
    tl.store(denom_ptr + state_offset, running_denom)
    tl.store(acc_ptr + acc_offset, running_acc, mask=dim_mask)


def accumulate_indexed_sparse_mla_attention_chunk(
    q: torch.Tensor,
    kv_flat: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    scale: float,
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
    candidate_offset: int = 0,
) -> None:
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]

    assert q.dim() == 3, f"Expected q shape [T, H, D], got {q.shape}"
    assert kv_flat.dim() == 2
    assert indices.dim() == 2
    assert indices.shape[0] == q.shape[0]
    assert kv_flat.shape[-1] == q.shape[-1]
    assert lens.shape[0] == q.shape[0]
    assert max_score.shape[0] == q.shape[0]
    assert max_score.shape[1] <= q.shape[1]
    assert denom.shape == max_score.shape
    assert acc.shape == (*max_score.shape, q.shape[-1])
    assert max_score.dtype == torch.float32
    assert denom.dtype == torch.float32
    assert acc.dtype == torch.float32
    assert q.is_cuda and kv_flat.is_cuda and indices.is_cuda and lens.is_cuda
    assert max_score.is_cuda and denom.is_cuda and acc.is_cuda

    num_tokens, _, head_dim = q.shape
    num_heads = max_score.shape[1]
    num_candidates = indices.shape[1]
    block_d = min(1024, triton.next_power_of_2(head_dim))
    grid = (num_tokens, num_heads)
    _accumulate_indexed_attention_chunk_kernel[grid](
        q,
        kv_flat,
        indices,
        lens,
        max_score,
        denom,
        acc,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_flat.stride(0),
        kv_flat.stride(1),
        indices.stride(0),
        indices.stride(1),
        max_score.stride(0),
        max_score.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        num_heads,
        head_dim,
        num_candidates,
        candidate_offset,
        scale,
        BLOCK_D=block_d,
        num_warps=8,
    )


@triton.jit
def _accumulate_fp8ds_global_slots_attention_chunk_kernel(
    q_ptr,
    k_cache_ptr,
    slot_ids_ptr,
    lens_ptr,
    max_score_ptr,
    denom_ptr,
    acc_ptr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_slot_t: tl.constexpr,
    stride_slot_c: tl.constexpr,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_acc_t: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    block_stride: tl.constexpr,
    fp8_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates,
    candidate_offset,
    scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_D)
    dim_mask = offsets < head_dim

    q = tl.load(
        q_ptr + token_idx * stride_q_t + head_idx * stride_q_h + offsets * stride_q_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    state_offset = token_idx * stride_state_t + head_idx * stride_state_h
    acc_offset = (
        token_idx * stride_acc_t + head_idx * stride_acc_h + offsets * stride_acc_d
    )
    running_max = tl.load(max_score_ptr + state_offset)
    running_denom = tl.load(denom_ptr + state_offset)
    running_acc = tl.load(acc_ptr + acc_offset, mask=dim_mask, other=0.0).to(tl.float32)
    valid_len = tl.load(lens_ptr + token_idx)

    fp8_mask = offsets < fp8_dim
    rope_mask = (offsets >= fp8_dim) & dim_mask
    rope_offsets = tl.maximum(offsets - fp8_dim, 0)

    for candidate_idx in range(0, num_candidates):
        slot_id = tl.load(
            slot_ids_ptr + token_idx * stride_slot_t + candidate_idx * stride_slot_c
        )
        is_valid = ((candidate_offset + candidate_idx) < valid_len) & (slot_id >= 0)

        if is_valid:
            block_idx = slot_id // cache_block_size
            pos_in_block = slot_id % cache_block_size
            cache_block_ptr = k_cache_ptr + block_idx.to(tl.int64) * block_stride
            token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
            token_scale_ptr = (
                cache_block_ptr
                + cache_block_size * token_data_size
                + pos_in_block * scale_dim
            )

            x_uint8 = tl.load(token_data_ptr + offsets, mask=fp8_mask, other=0)
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            x_float = x_fp8.to(tl.float32)
            scale_offsets = offsets // quant_block
            encoded_scale = tl.load(
                token_scale_ptr + scale_offsets,
                mask=fp8_mask,
                other=127,
            )
            dequant_scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
            x_dequant = x_float * dequant_scale

            rope_ptr = (token_data_ptr + fp8_dim).to(tl.pointer_type(tl.bfloat16))
            rope = tl.load(rope_ptr + rope_offsets, mask=rope_mask, other=0.0).to(
                tl.float32
            )
            kv = tl.where(fp8_mask, x_dequant, rope)
            kv = tl.where(dim_mask, kv, 0.0)

            score = tl.sum(q * kv, axis=0) * scale
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = running_acc * previous_weight + kv * candidate_weight
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    tl.store(max_score_ptr + state_offset, running_max)
    tl.store(denom_ptr + state_offset, running_denom)
    tl.store(acc_ptr + acc_offset, running_acc, mask=dim_mask)


def accumulate_fp8ds_global_slots_sparse_mla_attention_chunk(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    slot_ids: torch.Tensor,
    lens: torch.Tensor,
    block_size: int,
    scale: float,
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
    candidate_offset: int = 0,
) -> None:
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]
    if slot_ids.dim() == 3:
        assert slot_ids.shape[1] == 1
        slot_ids = slot_ids[:, 0]

    assert q.dim() == 3, f"Expected q shape [T, H, D], got {q.shape}"
    assert q.shape[-1] == 512
    assert slot_ids.dim() == 2
    assert slot_ids.shape[0] == q.shape[0]
    assert lens.shape[0] == q.shape[0]
    assert max_score.shape[0] == q.shape[0]
    assert max_score.shape[1] <= q.shape[1]
    assert denom.shape == max_score.shape
    assert acc.shape == (*max_score.shape, q.shape[-1])
    assert max_score.dtype == torch.float32
    assert denom.dtype == torch.float32
    assert acc.dtype == torch.float32
    assert k_cache.dtype == torch.uint8
    assert q.is_cuda and k_cache.is_cuda and slot_ids.is_cuda and lens.is_cuda
    assert max_score.is_cuda and denom.is_cuda and acc.is_cuda

    token_fp8_dim = 448
    token_bf16_dim = 64
    token_scale_dim = 8
    quant_block_size = 64
    token_data_size = token_fp8_dim + token_bf16_dim * 2

    num_tokens, _, head_dim = q.shape
    num_heads = max_score.shape[1]
    num_candidates = slot_ids.shape[1]
    block_d = min(1024, triton.next_power_of_2(head_dim))
    grid = (num_tokens, num_heads)
    _accumulate_fp8ds_global_slots_attention_chunk_kernel[grid](
        q,
        k_cache,
        slot_ids,
        lens,
        max_score,
        denom,
        acc,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        slot_ids.stride(0),
        slot_ids.stride(1),
        max_score.stride(0),
        max_score.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        block_size,
        token_data_size,
        k_cache.stride(0),
        token_fp8_dim,
        token_scale_dim,
        quant_block_size,
        num_heads,
        head_dim,
        num_candidates,
        candidate_offset,
        scale,
        BLOCK_D=block_d,
        num_warps=8,
    )


@triton.jit
def _accumulate_fp8ds_global_slots_attention_chunk_multihead_kernel(
    q_ptr,
    k_cache_ptr,
    slot_ids_ptr,
    lens_ptr,
    max_score_ptr,
    denom_ptr,
    acc_ptr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_slot_t: tl.constexpr,
    stride_slot_c: tl.constexpr,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_acc_t: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    block_stride: tl.constexpr,
    fp8_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates,
    candidate_offset,
    scale: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim
    matrix_mask = head_mask[:, None] & dim_mask[None, :]

    q = tl.load(
        q_ptr
        + token_idx * stride_q_t
        + head_offsets[:, None] * stride_q_h
        + dim_offsets[None, :] * stride_q_d,
        mask=matrix_mask,
        other=0.0,
    ).to(tl.float32)

    state_offsets = token_idx * stride_state_t + head_offsets * stride_state_h
    acc_offsets = (
        token_idx * stride_acc_t
        + head_offsets[:, None] * stride_acc_h
        + dim_offsets[None, :] * stride_acc_d
    )
    running_max = tl.load(
        max_score_ptr + state_offsets,
        mask=head_mask,
        other=-float("inf"),
    )
    running_denom = tl.load(denom_ptr + state_offsets, mask=head_mask, other=0.0)
    running_acc = tl.load(acc_ptr + acc_offsets, mask=matrix_mask, other=0.0).to(
        tl.float32
    )
    valid_len = tl.load(lens_ptr + token_idx)

    fp8_mask = dim_offsets < fp8_dim
    rope_mask = (dim_offsets >= fp8_dim) & dim_mask
    rope_offsets = tl.maximum(dim_offsets - fp8_dim, 0)

    for candidate_idx in range(0, num_candidates):
        slot_id = tl.load(
            slot_ids_ptr + token_idx * stride_slot_t + candidate_idx * stride_slot_c
        )
        is_valid = ((candidate_offset + candidate_idx) < valid_len) & (slot_id >= 0)

        if is_valid:
            block_idx = slot_id // cache_block_size
            pos_in_block = slot_id % cache_block_size
            cache_block_ptr = k_cache_ptr + block_idx.to(tl.int64) * block_stride
            token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
            token_scale_ptr = (
                cache_block_ptr
                + cache_block_size * token_data_size
                + pos_in_block * scale_dim
            )

            x_uint8 = tl.load(token_data_ptr + dim_offsets, mask=fp8_mask, other=0)
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            x_float = x_fp8.to(tl.float32)
            scale_offsets = dim_offsets // quant_block
            encoded_scale = tl.load(
                token_scale_ptr + scale_offsets,
                mask=fp8_mask,
                other=127,
            )
            dequant_scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
            x_dequant = x_float * dequant_scale

            rope_ptr = (token_data_ptr + fp8_dim).to(tl.pointer_type(tl.bfloat16))
            rope = tl.load(rope_ptr + rope_offsets, mask=rope_mask, other=0.0).to(
                tl.float32
            )
            kv = tl.where(fp8_mask, x_dequant, rope)
            kv = tl.where(dim_mask, kv, 0.0)

            score = tl.sum(q * kv[None, :], axis=1) * scale
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = (
                running_acc * previous_weight[:, None]
                + kv[None, :] * candidate_weight[:, None]
            )
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    tl.store(max_score_ptr + state_offsets, running_max, mask=head_mask)
    tl.store(denom_ptr + state_offsets, running_denom, mask=head_mask)
    tl.store(acc_ptr + acc_offsets, running_acc, mask=matrix_mask)


def accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    slot_ids: torch.Tensor,
    lens: torch.Tensor,
    block_size: int,
    scale: float,
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
    candidate_offset: int = 0,
    head_block_size: int = 2,
) -> None:
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]
    if slot_ids.dim() == 3:
        assert slot_ids.shape[1] == 1
        slot_ids = slot_ids[:, 0]

    assert q.dim() == 3, f"Expected q shape [T, H, D], got {q.shape}"
    assert q.shape[-1] == 512
    assert slot_ids.dim() == 2
    assert slot_ids.shape[0] == q.shape[0]
    assert lens.shape[0] == q.shape[0]
    assert max_score.shape[0] == q.shape[0]
    assert max_score.shape[1] <= q.shape[1]
    assert denom.shape == max_score.shape
    assert acc.shape == (*max_score.shape, q.shape[-1])
    assert head_block_size in (1, 2, 4)
    assert max_score.dtype == torch.float32
    assert denom.dtype == torch.float32
    assert acc.dtype == torch.float32
    assert k_cache.dtype == torch.uint8
    assert q.is_cuda and k_cache.is_cuda and slot_ids.is_cuda and lens.is_cuda
    assert max_score.is_cuda and denom.is_cuda and acc.is_cuda

    token_fp8_dim = 448
    token_bf16_dim = 64
    token_scale_dim = 8
    quant_block_size = 64
    token_data_size = token_fp8_dim + token_bf16_dim * 2

    num_tokens, _, head_dim = q.shape
    num_heads = max_score.shape[1]
    num_candidates = slot_ids.shape[1]
    block_d = min(1024, triton.next_power_of_2(head_dim))
    grid = (num_tokens, triton.cdiv(num_heads, head_block_size))
    _accumulate_fp8ds_global_slots_attention_chunk_multihead_kernel[grid](
        q,
        k_cache,
        slot_ids,
        lens,
        max_score,
        denom,
        acc,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        slot_ids.stride(0),
        slot_ids.stride(1),
        max_score.stride(0),
        max_score.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        block_size,
        token_data_size,
        k_cache.stride(0),
        token_fp8_dim,
        token_scale_dim,
        quant_block_size,
        num_heads,
        head_dim,
        num_candidates,
        candidate_offset,
        scale,
        HEAD_BLOCK=head_block_size,
        BLOCK_D=block_d,
        num_warps=8,
    )


@triton.jit
def _accumulate_fp8ds_paged_attention_chunk_kernel(
    q_ptr,
    k_cache_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    block_table_ptr,
    max_score_ptr,
    denom_ptr,
    acc_ptr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_block_table_t,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_acc_t: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    block_stride: tl.constexpr,
    fp8_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates,
    candidate_offset,
    scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_D)
    dim_mask = offsets < head_dim

    q = tl.load(
        q_ptr + token_idx * stride_q_t + head_idx * stride_q_h + offsets * stride_q_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    state_offset = token_idx * stride_state_t + head_idx * stride_state_h
    acc_offset = (
        token_idx * stride_acc_t + head_idx * stride_acc_h + offsets * stride_acc_d
    )
    running_max = tl.load(max_score_ptr + state_offset)
    running_denom = tl.load(denom_ptr + state_offset)
    running_acc = tl.load(acc_ptr + acc_offset, mask=dim_mask, other=0.0).to(tl.float32)

    seq_len = tl.load(seq_lens_ptr + token_idx)
    gather_len = tl.load(gather_lens_ptr + token_idx)
    start_pos = seq_len - gather_len
    fp8_mask = offsets < fp8_dim
    rope_mask = (offsets >= fp8_dim) & dim_mask
    rope_offsets = tl.maximum(offsets - fp8_dim, 0)

    for candidate_idx in range(0, num_candidates):
        gather_idx = candidate_offset + candidate_idx
        is_valid = gather_idx < gather_len

        if is_valid:
            pos = start_pos + gather_idx
            block_in_seq = pos // cache_block_size
            pos_in_block = pos % cache_block_size
            physical_block = tl.load(
                block_table_ptr + token_idx * stride_block_table_t + block_in_seq
            )
            cache_block_ptr = k_cache_ptr + physical_block.to(tl.int64) * block_stride
            token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
            token_scale_ptr = (
                cache_block_ptr
                + cache_block_size * token_data_size
                + pos_in_block * scale_dim
            )

            x_uint8 = tl.load(token_data_ptr + offsets, mask=fp8_mask, other=0)
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            x_float = x_fp8.to(tl.float32)
            scale_offsets = offsets // quant_block
            encoded_scale = tl.load(
                token_scale_ptr + scale_offsets,
                mask=fp8_mask,
                other=127,
            )
            dequant_scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
            x_dequant = x_float * dequant_scale

            rope_ptr = (token_data_ptr + fp8_dim).to(tl.pointer_type(tl.bfloat16))
            rope = tl.load(rope_ptr + rope_offsets, mask=rope_mask, other=0.0).to(
                tl.float32
            )
            kv = tl.where(fp8_mask, x_dequant, rope)
            kv = tl.where(dim_mask, kv, 0.0)

            score = tl.sum(q * kv, axis=0) * scale
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = running_acc * previous_weight + kv * candidate_weight
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    tl.store(max_score_ptr + state_offset, running_max)
    tl.store(denom_ptr + state_offset, running_denom)
    tl.store(acc_ptr + acc_offset, running_acc, mask=dim_mask)


def accumulate_fp8ds_paged_sparse_mla_attention_chunk(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    scale: float,
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
    candidate_offset: int,
    num_candidates: int,
) -> None:
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]

    assert q.dim() == 3, f"Expected q shape [T, H, D], got {q.shape}"
    assert q.shape[-1] == 512
    assert seq_lens.shape[0] == q.shape[0]
    assert gather_lens.shape[0] == q.shape[0]
    assert block_table.shape[0] == q.shape[0]
    assert max_score.shape[0] == q.shape[0]
    assert max_score.shape[1] <= q.shape[1]
    assert denom.shape == max_score.shape
    assert acc.shape == (*max_score.shape, q.shape[-1])
    assert max_score.dtype == torch.float32
    assert denom.dtype == torch.float32
    assert acc.dtype == torch.float32
    assert k_cache.dtype == torch.uint8
    assert q.is_cuda and k_cache.is_cuda
    assert seq_lens.is_cuda and gather_lens.is_cuda and block_table.is_cuda
    assert max_score.is_cuda and denom.is_cuda and acc.is_cuda

    token_fp8_dim = 448
    token_bf16_dim = 64
    token_scale_dim = 8
    quant_block_size = 64
    token_data_size = token_fp8_dim + token_bf16_dim * 2

    num_tokens, _, head_dim = q.shape
    num_heads = max_score.shape[1]
    block_d = min(1024, triton.next_power_of_2(head_dim))
    grid = (num_tokens, num_heads)
    _accumulate_fp8ds_paged_attention_chunk_kernel[grid](
        q,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        max_score,
        denom,
        acc,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        block_table.stride(0),
        max_score.stride(0),
        max_score.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        block_size,
        token_data_size,
        k_cache.stride(0),
        token_fp8_dim,
        token_scale_dim,
        quant_block_size,
        num_heads,
        head_dim,
        num_candidates,
        candidate_offset,
        scale,
        BLOCK_D=block_d,
        num_warps=8,
    )


@triton.jit
def _accumulate_fp8ds_paged_attention_chunk_multihead_kernel(
    q_ptr,
    k_cache_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    block_table_ptr,
    max_score_ptr,
    denom_ptr,
    acc_ptr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_block_table_t,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_acc_t: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    block_stride: tl.constexpr,
    fp8_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates,
    candidate_offset,
    scale: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim
    matrix_mask = head_mask[:, None] & dim_mask[None, :]

    q = tl.load(
        q_ptr
        + token_idx * stride_q_t
        + head_offsets[:, None] * stride_q_h
        + dim_offsets[None, :] * stride_q_d,
        mask=matrix_mask,
        other=0.0,
    ).to(tl.float32)

    state_offsets = token_idx * stride_state_t + head_offsets * stride_state_h
    acc_offsets = (
        token_idx * stride_acc_t
        + head_offsets[:, None] * stride_acc_h
        + dim_offsets[None, :] * stride_acc_d
    )
    running_max = tl.load(
        max_score_ptr + state_offsets,
        mask=head_mask,
        other=-float("inf"),
    )
    running_denom = tl.load(denom_ptr + state_offsets, mask=head_mask, other=0.0)
    running_acc = tl.load(acc_ptr + acc_offsets, mask=matrix_mask, other=0.0).to(
        tl.float32
    )

    seq_len = tl.load(seq_lens_ptr + token_idx)
    gather_len = tl.load(gather_lens_ptr + token_idx)
    start_pos = seq_len - gather_len
    fp8_mask = dim_offsets < fp8_dim
    rope_mask = (dim_offsets >= fp8_dim) & dim_mask
    rope_offsets = tl.maximum(dim_offsets - fp8_dim, 0)

    for candidate_idx in range(0, num_candidates):
        gather_idx = candidate_offset + candidate_idx
        is_valid = gather_idx < gather_len

        if is_valid:
            pos = start_pos + gather_idx
            block_in_seq = pos // cache_block_size
            pos_in_block = pos % cache_block_size
            physical_block = tl.load(
                block_table_ptr + token_idx * stride_block_table_t + block_in_seq
            )
            cache_block_ptr = k_cache_ptr + physical_block.to(tl.int64) * block_stride
            token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
            token_scale_ptr = (
                cache_block_ptr
                + cache_block_size * token_data_size
                + pos_in_block * scale_dim
            )

            x_uint8 = tl.load(token_data_ptr + dim_offsets, mask=fp8_mask, other=0)
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            x_float = x_fp8.to(tl.float32)
            scale_offsets = dim_offsets // quant_block
            encoded_scale = tl.load(
                token_scale_ptr + scale_offsets,
                mask=fp8_mask,
                other=127,
            )
            dequant_scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
            x_dequant = x_float * dequant_scale

            rope_ptr = (token_data_ptr + fp8_dim).to(tl.pointer_type(tl.bfloat16))
            rope = tl.load(rope_ptr + rope_offsets, mask=rope_mask, other=0.0).to(
                tl.float32
            )
            kv = tl.where(fp8_mask, x_dequant, rope)
            kv = tl.where(dim_mask, kv, 0.0)

            score = tl.sum(q * kv[None, :], axis=1) * scale
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = (
                running_acc * previous_weight[:, None]
                + kv[None, :] * candidate_weight[:, None]
            )
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    tl.store(max_score_ptr + state_offsets, running_max, mask=head_mask)
    tl.store(denom_ptr + state_offsets, running_denom, mask=head_mask)
    tl.store(acc_ptr + acc_offsets, running_acc, mask=matrix_mask)


def accumulate_fp8ds_paged_sparse_mla_attention_chunk_multihead(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    scale: float,
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
    candidate_offset: int,
    num_candidates: int,
    head_block_size: int = 2,
) -> None:
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]

    assert q.dim() == 3, f"Expected q shape [T, H, D], got {q.shape}"
    assert q.shape[-1] == 512
    assert seq_lens.shape[0] == q.shape[0]
    assert gather_lens.shape[0] == q.shape[0]
    assert block_table.shape[0] == q.shape[0]
    assert max_score.shape[0] == q.shape[0]
    assert max_score.shape[1] <= q.shape[1]
    assert denom.shape == max_score.shape
    assert acc.shape == (*max_score.shape, q.shape[-1])
    assert head_block_size in (1, 2, 4)
    assert max_score.dtype == torch.float32
    assert denom.dtype == torch.float32
    assert acc.dtype == torch.float32
    assert k_cache.dtype == torch.uint8
    assert q.is_cuda and k_cache.is_cuda
    assert seq_lens.is_cuda and gather_lens.is_cuda and block_table.is_cuda
    assert max_score.is_cuda and denom.is_cuda and acc.is_cuda

    token_fp8_dim = 448
    token_bf16_dim = 64
    token_scale_dim = 8
    quant_block_size = 64
    token_data_size = token_fp8_dim + token_bf16_dim * 2

    num_tokens, _, head_dim = q.shape
    num_heads = max_score.shape[1]
    block_d = min(1024, triton.next_power_of_2(head_dim))
    grid = (num_tokens, triton.cdiv(num_heads, head_block_size))
    _accumulate_fp8ds_paged_attention_chunk_multihead_kernel[grid](
        q,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        max_score,
        denom,
        acc,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        block_table.stride(0),
        max_score.stride(0),
        max_score.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        block_size,
        token_data_size,
        k_cache.stride(0),
        token_fp8_dim,
        token_scale_dim,
        quant_block_size,
        num_heads,
        head_dim,
        num_candidates,
        candidate_offset,
        scale,
        HEAD_BLOCK=head_block_size,
        BLOCK_D=block_d,
        num_warps=8,
    )


@triton.jit
def _fp8ds_paged_attention_with_sink_multihead_kernel(
    q_ptr,
    k_cache_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    block_table_ptr,
    sink_ptr,
    output_ptr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_block_table_t,
    stride_output_t: tl.constexpr,
    stride_output_h: tl.constexpr,
    stride_output_d: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    block_stride: tl.constexpr,
    fp8_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    candidate_offset: tl.constexpr,
    num_candidates: tl.constexpr,
    scale: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim
    matrix_mask = head_mask[:, None] & dim_mask[None, :]

    q = tl.load(
        q_ptr
        + token_idx * stride_q_t
        + head_offsets[:, None] * stride_q_h
        + dim_offsets[None, :] * stride_q_d,
        mask=matrix_mask,
        other=0.0,
    ).to(tl.float32)
    running_max = tl.full((HEAD_BLOCK,), -float("inf"), tl.float32)
    running_denom = tl.zeros((HEAD_BLOCK,), tl.float32)
    running_acc = tl.zeros((HEAD_BLOCK, BLOCK_D), tl.float32)

    seq_len = tl.load(seq_lens_ptr + token_idx)
    gather_len = tl.load(gather_lens_ptr + token_idx)
    start_pos = seq_len - gather_len
    fp8_mask = dim_offsets < fp8_dim
    rope_mask = (dim_offsets >= fp8_dim) & dim_mask
    rope_offsets = tl.maximum(dim_offsets - fp8_dim, 0)

    for candidate_idx in range(0, num_candidates):
        gather_idx = candidate_offset + candidate_idx
        is_valid = gather_idx < gather_len
        if is_valid:
            pos = start_pos + gather_idx
            block_in_seq = pos // cache_block_size
            pos_in_block = pos % cache_block_size
            physical_block = tl.load(
                block_table_ptr + token_idx * stride_block_table_t + block_in_seq
            )
            cache_block_ptr = k_cache_ptr + physical_block.to(tl.int64) * block_stride
            token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
            token_scale_ptr = (
                cache_block_ptr
                + cache_block_size * token_data_size
                + pos_in_block * scale_dim
            )

            x_uint8 = tl.load(token_data_ptr + dim_offsets, mask=fp8_mask, other=0)
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            x_float = x_fp8.to(tl.float32)
            scale_offsets = dim_offsets // quant_block
            encoded_scale = tl.load(
                token_scale_ptr + scale_offsets,
                mask=fp8_mask,
                other=127,
            )
            dequant_scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
            x_dequant = x_float * dequant_scale

            rope_ptr = (token_data_ptr + fp8_dim).to(tl.pointer_type(tl.bfloat16))
            rope = tl.load(rope_ptr + rope_offsets, mask=rope_mask, other=0.0).to(
                tl.float32
            )
            kv = tl.where(fp8_mask, x_dequant, rope)
            kv = tl.where(dim_mask, kv, 0.0)

            score = tl.sum(q * kv[None, :], axis=1) * scale
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = (
                running_acc * previous_weight[:, None]
                + kv[None, :] * candidate_weight[:, None]
            )
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    sink = tl.load(sink_ptr + head_offsets, mask=head_mask, other=-float("inf"))
    has_tokens = running_denom > 0.0
    has_sink = sink > -float("inf")
    valid_max = tl.where(has_tokens, running_max, -float("inf"))
    valid_sink = tl.where(has_sink, sink, -float("inf"))
    merge_max = tl.maximum(valid_max, valid_sink)
    has_any = has_tokens | has_sink
    safe_merge_max = tl.where(has_any, merge_max, 0.0)
    safe_running_max = tl.where(has_tokens, running_max, safe_merge_max)
    safe_sink = tl.where(has_sink, sink, safe_merge_max)
    subset_scale = tl.where(has_tokens, tl.exp(safe_running_max - safe_merge_max), 0.0)
    sink_weight = tl.where(has_sink, tl.exp(safe_sink - safe_merge_max), 0.0)
    total_weight = running_denom * subset_scale + sink_weight
    inv_total = tl.where(total_weight > 0.0, 1.0 / total_weight, 0.0)
    final = running_acc * subset_scale[:, None] * inv_total[:, None]

    tl.store(
        output_ptr
        + token_idx * stride_output_t
        + head_offsets[:, None] * stride_output_h
        + dim_offsets[None, :] * stride_output_d,
        final,
        mask=matrix_mask,
    )


def fp8ds_paged_sparse_mla_attention_with_sink_multihead(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    candidate_offset: int,
    num_candidates: int,
    scale: float,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    head_block_size: int = 1,
    num_heads: int | None = None,
) -> None:
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]

    assert q.dim() == 3, f"Expected q shape [T, H, D], got {q.shape}"
    assert q.shape[-1] == 512
    assert seq_lens.shape[0] == q.shape[0]
    assert gather_lens.shape[0] == q.shape[0]
    assert block_table.shape[0] == q.shape[0]
    assert output.shape[0] == q.shape[0]
    assert output.shape[2] == q.shape[-1]
    assert head_block_size in (1, 2, 4)
    assert k_cache.dtype == torch.uint8
    assert q.is_cuda and k_cache.is_cuda
    assert seq_lens.is_cuda and gather_lens.is_cuda and block_table.is_cuda
    assert attn_sink.is_cuda and output.is_cuda

    token_fp8_dim = 448
    token_bf16_dim = 64
    token_scale_dim = 8
    quant_block_size = 64
    token_data_size = token_fp8_dim + token_bf16_dim * 2

    num_tokens, _, head_dim = q.shape
    active_heads = num_heads if num_heads is not None else output.shape[1]
    assert active_heads <= q.shape[1]
    assert active_heads <= output.shape[1]
    assert active_heads <= attn_sink.shape[0]
    block_d = min(1024, triton.next_power_of_2(head_dim))
    grid = (num_tokens, triton.cdiv(active_heads, head_block_size))
    _fp8ds_paged_attention_with_sink_multihead_kernel[grid](
        q,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        attn_sink,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        block_table.stride(0),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        block_size,
        token_data_size,
        k_cache.stride(0),
        token_fp8_dim,
        token_scale_dim,
        quant_block_size,
        active_heads,
        head_dim,
        candidate_offset,
        num_candidates,
        scale,
        HEAD_BLOCK=head_block_size,
        BLOCK_D=block_d,
        num_warps=8,
    )


@triton.jit
def _fp8ds_global_paged_attention_with_sink_multihead_kernel(
    q_ptr,
    compressed_k_cache_ptr,
    slot_ids_ptr,
    topk_lens_ptr,
    swa_k_cache_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    block_table_ptr,
    sink_ptr,
    output_ptr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_slot_t: tl.constexpr,
    stride_slot_c: tl.constexpr,
    stride_block_table_t,
    stride_output_t: tl.constexpr,
    stride_output_h: tl.constexpr,
    stride_output_d: tl.constexpr,
    compressed_cache_block_size: tl.constexpr,
    compressed_block_stride: tl.constexpr,
    swa_cache_block_size: tl.constexpr,
    swa_block_stride: tl.constexpr,
    token_data_size: tl.constexpr,
    fp8_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_compressed_candidates: tl.constexpr,
    num_swa_candidates: tl.constexpr,
    scale: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim
    matrix_mask = head_mask[:, None] & dim_mask[None, :]

    q = tl.load(
        q_ptr
        + token_idx * stride_q_t
        + head_offsets[:, None] * stride_q_h
        + dim_offsets[None, :] * stride_q_d,
        mask=matrix_mask,
        other=0.0,
    ).to(tl.float32)
    running_max = tl.full((HEAD_BLOCK,), -float("inf"), tl.float32)
    running_denom = tl.zeros((HEAD_BLOCK,), tl.float32)
    running_acc = tl.zeros((HEAD_BLOCK, BLOCK_D), tl.float32)

    fp8_mask = dim_offsets < fp8_dim
    rope_mask = (dim_offsets >= fp8_dim) & dim_mask
    rope_offsets = tl.maximum(dim_offsets - fp8_dim, 0)
    topk_len = tl.load(topk_lens_ptr + token_idx)

    for candidate_idx in range(0, num_compressed_candidates):
        slot_id = tl.load(
            slot_ids_ptr + token_idx * stride_slot_t + candidate_idx * stride_slot_c
        )
        is_valid = (candidate_idx < topk_len) & (slot_id >= 0)
        if is_valid:
            block_idx = slot_id // compressed_cache_block_size
            pos_in_block = slot_id % compressed_cache_block_size
            cache_block_ptr = (
                compressed_k_cache_ptr
                + block_idx.to(tl.int64) * compressed_block_stride
            )
            token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
            token_scale_ptr = (
                cache_block_ptr
                + compressed_cache_block_size * token_data_size
                + pos_in_block * scale_dim
            )

            x_uint8 = tl.load(token_data_ptr + dim_offsets, mask=fp8_mask, other=0)
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            x_float = x_fp8.to(tl.float32)
            scale_offsets = dim_offsets // quant_block
            encoded_scale = tl.load(
                token_scale_ptr + scale_offsets,
                mask=fp8_mask,
                other=127,
            )
            dequant_scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
            x_dequant = x_float * dequant_scale
            rope_ptr = (token_data_ptr + fp8_dim).to(tl.pointer_type(tl.bfloat16))
            rope = tl.load(rope_ptr + rope_offsets, mask=rope_mask, other=0.0).to(
                tl.float32
            )
            kv = tl.where(fp8_mask, x_dequant, rope)
            kv = tl.where(dim_mask, kv, 0.0)

            score = tl.sum(q * kv[None, :], axis=1) * scale
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = (
                running_acc * previous_weight[:, None]
                + kv[None, :] * candidate_weight[:, None]
            )
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    seq_len = tl.load(seq_lens_ptr + token_idx)
    gather_len = tl.load(gather_lens_ptr + token_idx)
    start_pos = seq_len - gather_len
    for candidate_idx in range(0, num_swa_candidates):
        is_valid = candidate_idx < gather_len
        if is_valid:
            pos = start_pos + candidate_idx
            block_in_seq = pos // swa_cache_block_size
            pos_in_block = pos % swa_cache_block_size
            physical_block = tl.load(
                block_table_ptr + token_idx * stride_block_table_t + block_in_seq
            )
            cache_block_ptr = (
                swa_k_cache_ptr + physical_block.to(tl.int64) * swa_block_stride
            )
            token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
            token_scale_ptr = (
                cache_block_ptr
                + swa_cache_block_size * token_data_size
                + pos_in_block * scale_dim
            )

            x_uint8 = tl.load(token_data_ptr + dim_offsets, mask=fp8_mask, other=0)
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            x_float = x_fp8.to(tl.float32)
            scale_offsets = dim_offsets // quant_block
            encoded_scale = tl.load(
                token_scale_ptr + scale_offsets,
                mask=fp8_mask,
                other=127,
            )
            dequant_scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
            x_dequant = x_float * dequant_scale
            rope_ptr = (token_data_ptr + fp8_dim).to(tl.pointer_type(tl.bfloat16))
            rope = tl.load(rope_ptr + rope_offsets, mask=rope_mask, other=0.0).to(
                tl.float32
            )
            kv = tl.where(fp8_mask, x_dequant, rope)
            kv = tl.where(dim_mask, kv, 0.0)

            score = tl.sum(q * kv[None, :], axis=1) * scale
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = (
                running_acc * previous_weight[:, None]
                + kv[None, :] * candidate_weight[:, None]
            )
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    sink = tl.load(sink_ptr + head_offsets, mask=head_mask, other=-float("inf"))
    has_tokens = running_denom > 0.0
    has_sink = sink > -float("inf")
    valid_max = tl.where(has_tokens, running_max, -float("inf"))
    valid_sink = tl.where(has_sink, sink, -float("inf"))
    merge_max = tl.maximum(valid_max, valid_sink)
    has_any = has_tokens | has_sink
    safe_merge_max = tl.where(has_any, merge_max, 0.0)
    safe_running_max = tl.where(has_tokens, running_max, safe_merge_max)
    safe_sink = tl.where(has_sink, sink, safe_merge_max)
    subset_scale = tl.where(has_tokens, tl.exp(safe_running_max - safe_merge_max), 0.0)
    sink_weight = tl.where(has_sink, tl.exp(safe_sink - safe_merge_max), 0.0)
    total_weight = running_denom * subset_scale + sink_weight
    inv_total = tl.where(total_weight > 0.0, 1.0 / total_weight, 0.0)
    final = running_acc * subset_scale[:, None] * inv_total[:, None]

    tl.store(
        output_ptr
        + token_idx * stride_output_t
        + head_offsets[:, None] * stride_output_h
        + dim_offsets[None, :] * stride_output_d,
        final,
        mask=matrix_mask,
    )


def fp8ds_global_paged_sparse_mla_attention_with_sink_multihead(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    slot_ids: torch.Tensor,
    topk_lens: torch.Tensor,
    compressed_block_size: int,
    swa_k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    block_table: torch.Tensor,
    swa_block_size: int,
    num_compressed_candidates: int,
    num_swa_candidates: int,
    scale: float,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    head_block_size: int = 1,
    num_heads: int | None = None,
) -> None:
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]
    if slot_ids.dim() == 3:
        assert slot_ids.shape[1] == 1
        slot_ids = slot_ids[:, 0]

    assert q.dim() == 3, f"Expected q shape [T, H, D], got {q.shape}"
    assert q.shape[-1] == 512
    assert slot_ids.dim() == 2
    assert slot_ids.shape[0] == q.shape[0]
    assert topk_lens.shape[0] == q.shape[0]
    assert seq_lens.shape[0] == q.shape[0]
    assert gather_lens.shape[0] == q.shape[0]
    assert block_table.shape[0] == q.shape[0]
    assert output.shape[0] == q.shape[0]
    assert output.shape[2] == q.shape[-1]
    assert head_block_size in (1, 2, 4)
    assert compressed_k_cache.dtype == torch.uint8
    assert swa_k_cache.dtype == torch.uint8
    assert q.is_cuda and compressed_k_cache.is_cuda and swa_k_cache.is_cuda
    assert slot_ids.is_cuda and topk_lens.is_cuda
    assert seq_lens.is_cuda and gather_lens.is_cuda and block_table.is_cuda
    assert attn_sink.is_cuda and output.is_cuda

    token_fp8_dim = 448
    token_bf16_dim = 64
    token_scale_dim = 8
    quant_block_size = 64
    token_data_size = token_fp8_dim + token_bf16_dim * 2

    num_tokens, _, head_dim = q.shape
    active_heads = num_heads if num_heads is not None else output.shape[1]
    assert active_heads <= q.shape[1]
    assert active_heads <= output.shape[1]
    assert active_heads <= attn_sink.shape[0]
    block_d = min(1024, triton.next_power_of_2(head_dim))
    grid = (num_tokens, triton.cdiv(active_heads, head_block_size))
    _fp8ds_global_paged_attention_with_sink_multihead_kernel[grid](
        q,
        compressed_k_cache,
        slot_ids,
        topk_lens,
        swa_k_cache,
        seq_lens,
        gather_lens,
        block_table,
        attn_sink,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        slot_ids.stride(0),
        slot_ids.stride(1),
        block_table.stride(0),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        compressed_block_size,
        compressed_k_cache.stride(0),
        swa_block_size,
        swa_k_cache.stride(0),
        token_data_size,
        token_fp8_dim,
        token_scale_dim,
        quant_block_size,
        active_heads,
        head_dim,
        num_compressed_candidates,
        num_swa_candidates,
        scale,
        HEAD_BLOCK=head_block_size,
        BLOCK_D=block_d,
        num_warps=8,
    )


@triton.jit
def _finish_attention_state_kernel(
    max_score_ptr,
    denom_ptr,
    acc_ptr,
    output_ptr,
    lse_ptr,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_acc_t: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    stride_output_t: tl.constexpr,
    stride_output_h: tl.constexpr,
    stride_output_d: tl.constexpr,
    stride_lse_t: tl.constexpr,
    stride_lse_h: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_head = tl.program_id(0)
    block_d = tl.program_id(1)
    token_idx = token_head // num_heads
    head_idx = token_head - token_idx * num_heads
    offsets = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = offsets < head_dim

    state_offset = token_idx * stride_state_t + head_idx * stride_state_h
    running_max = tl.load(max_score_ptr + state_offset)
    running_denom = tl.load(denom_ptr + state_offset)
    is_valid = running_denom > 0.0
    inv_denom = tl.where(is_valid, 1.0 / running_denom, 0.0)
    subset_lse = tl.where(
        is_valid,
        running_max + tl.log(running_denom),
        -float("inf"),
    )

    acc = tl.load(
        acc_ptr
        + token_idx * stride_acc_t
        + head_idx * stride_acc_h
        + offsets * stride_acc_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    subset_output = acc * inv_denom
    tl.store(
        output_ptr
        + token_idx * stride_output_t
        + head_idx * stride_output_h
        + offsets * stride_output_d,
        subset_output,
        mask=dim_mask,
    )
    if block_d == 0:
        tl.store(
            lse_ptr + token_idx * stride_lse_t + head_idx * stride_lse_h,
            subset_lse,
        )


def finish_gathered_sparse_mla_attention(
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    assert max_score.shape == denom.shape
    assert acc.shape[:2] == max_score.shape
    assert output.shape == acc.shape
    assert lse.shape == max_score.shape
    assert max_score.dtype == torch.float32
    assert denom.dtype == torch.float32
    assert acc.dtype == torch.float32
    assert output.dtype == torch.float32
    assert lse.dtype == torch.float32
    assert max_score.is_cuda and denom.is_cuda and acc.is_cuda
    assert output.is_cuda and lse.is_cuda

    num_tokens, num_heads, head_dim = acc.shape
    block_d = min(128, triton.next_power_of_2(head_dim))
    grid = (num_tokens * num_heads, triton.cdiv(head_dim, block_d))
    _finish_attention_state_kernel[grid](
        max_score,
        denom,
        acc,
        output,
        lse,
        max_score.stride(0),
        max_score.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        num_heads,
        head_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )


@triton.jit
def _finish_attention_state_with_sink_kernel(
    max_score_ptr,
    denom_ptr,
    acc_ptr,
    sink_ptr,
    output_ptr,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_acc_t: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    stride_output_t: tl.constexpr,
    stride_output_h: tl.constexpr,
    stride_output_d: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_head = tl.program_id(0)
    block_d = tl.program_id(1)
    token_idx = token_head // num_heads
    head_idx = token_head - token_idx * num_heads
    offsets = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = offsets < head_dim

    state_offset = token_idx * stride_state_t + head_idx * stride_state_h
    running_max = tl.load(max_score_ptr + state_offset)
    running_denom = tl.load(denom_ptr + state_offset)
    sink = tl.load(sink_ptr + head_idx)
    has_tokens = running_denom > 0.0
    has_sink = sink > -float("inf")
    valid_max = tl.where(has_tokens, running_max, -float("inf"))
    valid_sink = tl.where(has_sink, sink, -float("inf"))
    merge_max = tl.maximum(valid_max, valid_sink)
    has_any = has_tokens | has_sink
    safe_merge_max = tl.where(has_any, merge_max, 0.0)
    safe_running_max = tl.where(has_tokens, running_max, safe_merge_max)
    safe_sink = tl.where(has_sink, sink, safe_merge_max)
    subset_scale = tl.where(has_tokens, tl.exp(safe_running_max - safe_merge_max), 0.0)
    subset_weight = running_denom * subset_scale
    sink_weight = tl.where(has_sink, tl.exp(safe_sink - safe_merge_max), 0.0)
    total_weight = subset_weight + sink_weight
    inv_total = tl.where(total_weight > 0.0, 1.0 / total_weight, 0.0)

    acc_values = tl.load(
        acc_ptr
        + token_idx * stride_acc_t
        + head_idx * stride_acc_h
        + offsets * stride_acc_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    acc_values = tl.where(has_tokens, acc_values, 0.0)
    output = acc_values * subset_scale * inv_total
    tl.store(
        output_ptr
        + token_idx * stride_output_t
        + head_idx * stride_output_h
        + offsets * stride_output_d,
        output,
        mask=dim_mask,
    )


@triton.jit
def _finish_two_attention_states_with_sink_kernel(
    max0_ptr,
    denom0_ptr,
    acc0_ptr,
    max1_ptr,
    denom1_ptr,
    acc1_ptr,
    sink_ptr,
    output_ptr,
    stride_state0_t: tl.constexpr,
    stride_state0_h: tl.constexpr,
    stride_acc0_t: tl.constexpr,
    stride_acc0_h: tl.constexpr,
    stride_acc0_d: tl.constexpr,
    stride_state1_t: tl.constexpr,
    stride_state1_h: tl.constexpr,
    stride_acc1_t: tl.constexpr,
    stride_acc1_h: tl.constexpr,
    stride_acc1_d: tl.constexpr,
    stride_output_t: tl.constexpr,
    stride_output_h: tl.constexpr,
    stride_output_d: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_head = tl.program_id(0)
    block_d = tl.program_id(1)
    token_idx = token_head // num_heads
    head_idx = token_head - token_idx * num_heads
    offsets = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = offsets < head_dim

    state0_offset = token_idx * stride_state0_t + head_idx * stride_state0_h
    state1_offset = token_idx * stride_state1_t + head_idx * stride_state1_h
    max0 = tl.load(max0_ptr + state0_offset)
    denom0 = tl.load(denom0_ptr + state0_offset)
    max1 = tl.load(max1_ptr + state1_offset)
    denom1 = tl.load(denom1_ptr + state1_offset)
    sink = tl.load(sink_ptr + head_idx)

    has0 = denom0 > 0.0
    has1 = denom1 > 0.0
    has_sink = sink > -float("inf")
    valid_max0 = tl.where(has0, max0, -float("inf"))
    valid_max1 = tl.where(has1, max1, -float("inf"))
    valid_sink = tl.where(has_sink, sink, -float("inf"))
    merge_max = tl.maximum(tl.maximum(valid_max0, valid_max1), valid_sink)
    has_any = has0 | has1 | has_sink
    safe_merge_max = tl.where(has_any, merge_max, 0.0)
    safe_max0 = tl.where(has0, max0, safe_merge_max)
    safe_max1 = tl.where(has1, max1, safe_merge_max)
    safe_sink = tl.where(has_sink, sink, safe_merge_max)
    scale0 = tl.where(has0, tl.exp(safe_max0 - safe_merge_max), 0.0)
    scale1 = tl.where(has1, tl.exp(safe_max1 - safe_merge_max), 0.0)
    sink_weight = tl.where(has_sink, tl.exp(safe_sink - safe_merge_max), 0.0)
    total_weight = denom0 * scale0 + denom1 * scale1 + sink_weight
    inv_total = tl.where(total_weight > 0.0, 1.0 / total_weight, 0.0)

    acc0 = tl.load(
        acc0_ptr
        + token_idx * stride_acc0_t
        + head_idx * stride_acc0_h
        + offsets * stride_acc0_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    acc1 = tl.load(
        acc1_ptr
        + token_idx * stride_acc1_t
        + head_idx * stride_acc1_h
        + offsets * stride_acc1_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    acc0 = tl.where(has0, acc0, 0.0)
    acc1 = tl.where(has1, acc1, 0.0)
    output = (acc0 * scale0 + acc1 * scale1) * inv_total
    tl.store(
        output_ptr
        + token_idx * stride_output_t
        + head_idx * stride_output_h
        + offsets * stride_output_d,
        output,
        mask=dim_mask,
    )


def finish_two_sparse_mla_attention_states_with_sink(
    max_score0: torch.Tensor,
    denom0: torch.Tensor,
    acc0: torch.Tensor,
    max_score1: torch.Tensor,
    denom1: torch.Tensor,
    acc1: torch.Tensor,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
) -> None:
    assert max_score0.shape == denom0.shape
    assert max_score1.shape == denom1.shape
    assert max_score0.shape == max_score1.shape
    assert acc0.shape == acc1.shape
    assert acc0.shape[:2] == max_score0.shape
    assert output.shape[0] == acc0.shape[0]
    assert output.shape[1] >= acc0.shape[1]
    assert output.shape[2] == acc0.shape[2]
    assert attn_sink.shape[0] >= acc0.shape[1]
    assert max_score0.dtype == torch.float32
    assert denom0.dtype == torch.float32
    assert acc0.dtype == torch.float32
    assert max_score1.dtype == torch.float32
    assert denom1.dtype == torch.float32
    assert acc1.dtype == torch.float32
    assert max_score0.is_cuda and denom0.is_cuda and acc0.is_cuda
    assert max_score1.is_cuda and denom1.is_cuda and acc1.is_cuda
    assert attn_sink.is_cuda and output.is_cuda

    num_tokens, num_heads, head_dim = acc0.shape
    block_d = min(128, triton.next_power_of_2(head_dim))
    grid = (num_tokens * num_heads, triton.cdiv(head_dim, block_d))
    _finish_two_attention_states_with_sink_kernel[grid](
        max_score0,
        denom0,
        acc0,
        max_score1,
        denom1,
        acc1,
        attn_sink,
        output,
        max_score0.stride(0),
        max_score0.stride(1),
        acc0.stride(0),
        acc0.stride(1),
        acc0.stride(2),
        max_score1.stride(0),
        max_score1.stride(1),
        acc1.stride(0),
        acc1.stride(1),
        acc1.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        num_heads,
        head_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )


def finish_sparse_mla_attention_with_sink(
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
) -> None:
    assert max_score.shape == denom.shape
    assert acc.shape[:2] == max_score.shape
    assert output.shape[0] == acc.shape[0]
    assert output.shape[1] >= acc.shape[1]
    assert output.shape[2] == acc.shape[2]
    assert attn_sink.shape[0] >= acc.shape[1]
    assert max_score.dtype == torch.float32
    assert denom.dtype == torch.float32
    assert acc.dtype == torch.float32
    assert max_score.is_cuda and denom.is_cuda and acc.is_cuda
    assert attn_sink.is_cuda and output.is_cuda

    num_tokens, num_heads, head_dim = acc.shape
    block_d = min(128, triton.next_power_of_2(head_dim))
    grid = (num_tokens * num_heads, triton.cdiv(head_dim, block_d))
    _finish_attention_state_with_sink_kernel[grid](
        max_score,
        denom,
        acc,
        attn_sink,
        output,
        max_score.stride(0),
        max_score.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        num_heads,
        head_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )
