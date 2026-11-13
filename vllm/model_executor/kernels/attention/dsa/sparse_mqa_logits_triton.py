# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton sparse MQA-logits path for the DeepSeek V4.1 two-level indexer.

SM120 (and other non-SM100 NVIDIA GPUs) cannot run DeepGEMM's sparse MQA
kernels or the DeepSelect top-k, so candidate-consuming indexer layers fall
back to scoring the whole compressed context with the dense kernel and
masking the logits down to the candidate blocks afterwards, discarding ~98%
of the scoring work at long context (see
https://github.com/vllm-project/vllm/issues/61830).

This module is the SM120 counterpart of `sparse_mqa_logits`: Triton kernels
score only the candidate positions (fp8 dot with fp32 accumulation, relu,
per-head weights, per-position fp32 scale) into a compact
`[rows, num_sparse_blocks * sparse_block_kv]` fp32 buffer, then the
persistent top-k (`torch.ops._C.persistent_topk`) and the shared sparse
remap kernel produce request-local top-k indices. The paged-decode kernel
reads K in place from the indexer cache, an approach first implemented for
SM90 in PR #62463 (`candidate_paged_logits.py`); here the same technique is
generalized to the sparse-block expansion shared with the SM100 path and
extended to the packed prefill workspace.

Only the FP8 indexer cache is supported (the MXFP4 cache keeps its UE8M0
scales, which this path does not decode). Both the flattened per-token
decode layout and the native `[B, next_n]` layout are handled through a
`block_table` row-repeat factor, and every per-row bound stays on the
device, so the path is safe under full CUDA graph capture.
"""

import torch

from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
    _sparse_topk_remap_kernel,
    candidate_blocks_to_sparse_indices,
)
from vllm.model_executor.layers.indexer_topk import RADIX_TOPK_WORKSPACE_SIZE
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.worker.workspace import current_workspace_manager

# One fp32 scale per cached position, stored after the block's value bytes.
INDEXER_SCALE_BYTES = 4
_SCALE_BYTES = tl.constexpr(4)

# Compact columns per program. A multiple of the sparse block size, and at
# least 16 so the MMA has a full M tile.
_BLOCK_N = 128


def has_triton_sparse_mqa() -> bool:
    """Whether this platform can run the Triton sparse MQA-logits path."""
    if not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
    ):
        return False
    try:
        torch.ops._C.persistent_topk
    except AttributeError:
        return False
    return True


@triton.jit(do_not_specialize_on_alignment=["ks_ptr", "ke_ptr", "end_ptr"])
def _prefill_sparse_logits_kernel(
    q_ptr,  # fp8 [rows, H, D]
    k_ptr,  # fp8 [total_kv, D] packed prefill workspace
    k_scale_ptr,  # fp32 [total_kv] per-position scales
    w_ptr,  # fp32 [rows, H]
    si_ptr,  # int32 [rows, S] sparse block ids (packed-workspace coords)
    end_ptr,  # int32 [rows] valid compact column count
    ks_ptr,  # int32 [rows] packed K range start
    ke_ptr,  # int32 [rows] packed K range end
    out_ptr,  # fp32 [rows, S * SBK]
    stride_q_row,
    stride_q_head,
    stride_w_row,
    stride_si_row,
    stride_out_row,
    width,
    H: tl.constexpr,
    H_PAD: tl.constexpr,
    D: tl.constexpr,
    SBK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Score one tile of candidate columns for one prefill row.

    Compact column ``j * SBK + o`` scores the packed-workspace position
    ``si[row, j] * SBK + (ks % SBK) + o`` (see
    ``candidate_blocks_to_sparse_indices``); columns past the row's valid
    prefix ``end`` become ``-inf`` so the top-k never selects them.
    """
    row = tl.program_id(0).to(tl.int64)
    tile_start = tl.program_id(1) * BLOCK_N
    cols = tile_start + tl.arange(0, BLOCK_N)
    in_bounds = cols < width

    end = tl.load(end_ptr + row)
    if tile_start >= end:
        # Dead tile: the top-k bounds never read past ``end``, but keep the
        # -inf contract for the whole buffer. The grid stays width-sized, so
        # this is CUDA-graph safe.
        tl.store(
            out_ptr + row * stride_out_row + cols,
            tl.full((BLOCK_N,), -float("inf"), tl.float32),
            mask=in_bounds,
        )
        return

    ks = tl.load(ks_ptr + row)
    ke = tl.load(ke_ptr + row)
    live = in_bounds & (cols < end)
    blk = tl.load(
        si_ptr + row * stride_si_row + cols // SBK, mask=live, other=0
    ).to(tl.int64)
    pos = blk * SBK + ks % SBK + cols % SBK
    live = live & (pos < ke)
    safe_pos = tl.where(live, pos, 0)

    d = tl.arange(0, D)
    h = tl.arange(0, H_PAD)
    h_live = h < H

    # [BLOCK_N, D] fp8: one coalesced 128-byte segment per column.
    k_block = tl.load(
        k_ptr + safe_pos[:, None] * D + d[None, :], mask=live[:, None], other=0.0
    )
    # [D, H_PAD] fp8; zero-padded heads contribute relu(0) * 0 = 0.
    q_block = tl.load(
        q_ptr + row * stride_q_row + h[None, :] * stride_q_head + d[:, None],
        mask=h_live[None, :],
        other=0.0,
    )
    weights = tl.load(w_ptr + row * stride_w_row + h, mask=h_live, other=0.0).to(
        tl.float32
    )

    # [BLOCK_N, H_PAD] on fp8 tensor cores with fp32 accumulation.
    scores = tl.dot(k_block, q_block)
    scores = tl.maximum(scores, 0.0)
    scores = scores * weights[None, :]
    acc = tl.sum(scores, axis=1)
    acc = acc * tl.load(k_scale_ptr + safe_pos, mask=live, other=0.0)

    tl.store(
        out_ptr + row * stride_out_row + cols,
        tl.where(live, acc, -float("inf")),
        mask=in_bounds,
    )


@triton.jit(
    # ``cache_block_stride`` and ``cache_block_size`` are deliberately left
    # specializable: Triton needs their divisibility to prove that a
    # position's value bytes are 16-byte aligned and emit vector loads for the
    # candidate gather. Hiding it costs roughly 20x on the K stream.
    do_not_specialize=[
        "num_blocks",
        "bt_width",
        "width",
        "bt_row_repeat",
    ],
    do_not_specialize_on_alignment=["end_ptr", "causal_ptr"],
)
def _paged_sparse_logits_kernel(
    q_ptr,  # fp8 [rows, H, D]
    kv_bytes,  # uint8 flat storage view of the paged indexer cache
    kv_scales,  # fp32 reinterpretation of the same flat storage
    w_ptr,  # fp32 [rows, H]
    si_ptr,  # int32 [rows, S] sparse block ids (context-relative)
    end_ptr,  # int32 [rows] valid compact column count
    causal_ptr,  # int32 [rows] causal context length, compressed positions
    bt_ptr,  # int32 [bt_rows, P] page table
    out_ptr,  # fp32 [rows, width]
    stride_q_row,
    stride_q_head,
    stride_w_row,
    stride_si_row,
    stride_bt_row,
    stride_bt_col,
    stride_out_row,
    bt_row_repeat,
    cache_block_stride,  # bytes per physical cache block
    cache_block_size,  # positions per cache block
    num_blocks,
    bt_width,
    width,
    H: tl.constexpr,
    H_PAD: tl.constexpr,
    D: tl.constexpr,
    SBK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Score one tile of candidate columns for one decode row.

    Compact column ``j * SBK + o`` scores request-local compressed position
    ``si[row, j] * SBK + o``; columns past the row's valid prefix, and
    columns past the row's causal length, become ``-inf``.
    """
    row = tl.program_id(0).to(tl.int64)
    tile_start = tl.program_id(1) * BLOCK_N
    cols = tile_start + tl.arange(0, BLOCK_N)
    in_bounds = cols < width

    end = tl.load(end_ptr + row)
    if tile_start >= end:
        tl.store(
            out_ptr + row * stride_out_row + cols,
            tl.full((BLOCK_N,), -float("inf"), tl.float32),
            mask=in_bounds,
        )
        return

    causal = tl.load(causal_ptr + row)
    live = in_bounds & (cols < end)
    blk = tl.load(
        si_ptr + row * stride_si_row + cols // SBK, mask=live, other=-1
    ).to(tl.int64)
    pos = blk * SBK + cols % SBK
    # The expansion pads the tail with a repeat of the largest block, so the
    # causal bound is what rejects those columns.
    live = live & (blk >= 0) & (pos < causal)

    page = tl.where(live, pos // cache_block_size, 0)
    live = live & (page < bt_width)
    phys = tl.load(
        bt_ptr + (row // bt_row_repeat) * stride_bt_row + page * stride_bt_col,
        mask=live,
        other=-1,
    ).to(tl.int64)
    live = live & (phys >= 0) & (phys < num_blocks)
    safe_phys = tl.where(live, phys, 0)
    page_off = tl.where(live, pos % cache_block_size, 0)

    d = tl.arange(0, D)
    h = tl.arange(0, H_PAD)
    h_live = h < H

    k_offset = safe_phys * cache_block_stride + page_off * D
    k_block = tl.load(
        kv_bytes + k_offset[:, None] + d[None, :],
        mask=live[:, None],
        other=0,
    ).to(tl.float8e4nv, bitcast=True)
    q_block = tl.load(
        q_ptr + row * stride_q_row + h[None, :] * stride_q_head + d[:, None],
        mask=h_live[None, :],
        other=0.0,
    )
    weights = tl.load(w_ptr + row * stride_w_row + h, mask=h_live, other=0.0).to(
        tl.float32
    )

    scores = tl.dot(k_block, q_block)
    scores = tl.maximum(scores, 0.0)
    scores = scores * weights[None, :]
    acc = tl.sum(scores, axis=1)

    # Scales follow the block's value bytes, one fp32 per position.
    sc_idx = (
        (safe_phys * cache_block_stride) // _SCALE_BYTES
        + (cache_block_size * D) // _SCALE_BYTES
        + page_off
    )
    acc = acc * tl.load(kv_scales + sc_idx, mask=live, other=0.0)

    tl.store(
        out_ptr + row * stride_out_row + cols,
        tl.where(live, acc, -float("inf")),
        mask=in_bounds,
    )


def _flat_cache_views(
    kv_cache: torch.Tensor, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-copy ``(uint8, fp32)`` one-dimensional views of the cache storage.

    Packed KV-cache views leave gaps between logical pages, because other
    layers share the same physical manager block, so ``view(-1)`` is invalid.
    Span the gaps instead and let the kernel use the physical block stride.
    """
    cache_block_size = kv_cache.shape[1]
    page_bytes = cache_block_size * (head_dim + INDEXER_SCALE_BYTES)
    assert kv_cache.dtype == torch.uint8 and kv_cache.ndim == 3
    assert kv_cache.shape[2] == head_dim + INDEXER_SCALE_BYTES, (
        "FP8 indexer cache entry must be head_dim+4 bytes, got shape "
        f"{tuple(kv_cache.shape)} with head_dim={head_dim}"
    )
    assert kv_cache.stride(2) == 1 and kv_cache.stride(0) >= page_bytes
    assert kv_cache.storage_offset() % INDEXER_SCALE_BYTES == 0
    assert kv_cache.stride(0) % INDEXER_SCALE_BYTES == 0
    storage_span = (kv_cache.shape[0] - 1) * kv_cache.stride(0) + page_bytes
    assert storage_span % INDEXER_SCALE_BYTES == 0
    kv_bytes = torch.as_strided(kv_cache, (storage_span,), (1,))
    return kv_bytes, kv_bytes.view(torch.float32)


def _sparse_logits(
    q: torch.Tensor,
    end: torch.Tensor,
    width: int,
    launch,
) -> torch.Tensor:
    """Allocate the compact fp32 logits and run one scorer launch."""
    rows = q.shape[0]
    num_heads = q.shape[1]
    logits = torch.empty((rows, width), dtype=torch.float32, device=q.device)
    block_n = max(_BLOCK_N, 16)
    launch(logits, block_n)
    return logits


def sparse_topk_remap_triton(
    logits: torch.Tensor,
    sparse_indices: torch.Tensor,
    end: torch.Tensor,
    row_ks: torch.Tensor,
    sparse_block_kv: int,
    topk_tokens: int,
    topk_indices: torch.Tensor,
    *,
    col_indices: torch.Tensor,
) -> None:
    """Row top-k over sparse logits, remapped to request-local positions.

    Same contract as `sparse_topk_remap`, but with the persistent top-k
    kernel instead of DeepSelect, so it runs on SM120. `logits` is fp32.
    """
    rows, width = logits.shape
    assert logits.dtype == torch.float32 and width >= topk_tokens
    assert end.dtype == torch.int32 and end.shape == (rows,)
    assert topk_tokens in (512, 1024, 2048), topk_tokens
    cols = col_indices[:rows]
    (workspace,) = current_workspace_manager().get_simultaneous(
        ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
    )
    # ``width`` is fixed by the model config, so passing it as the host-side
    # bound keeps this call free of any context-dependent value and therefore
    # safe to capture in a CUDA graph.
    torch.ops._C.persistent_topk(logits, end, cols, workspace, topk_tokens, width)
    _sparse_topk_remap_kernel[(rows,)](
        cols,
        cols.stride(0),
        sparse_indices,
        sparse_indices.stride(0),
        row_ks,
        row_ks.stride(0),
        topk_indices,
        topk_indices.stride(0),
        topk_tokens,
        width,
        SBK=sparse_block_kv,
        K_POW2=triton.next_power_of_2(topk_tokens),
    )


def sparse_mqa_logits_prefill_chunk_triton(
    q: torch.Tensor,
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
    sparse_block_kv: int,
    topk_tokens: int,
    topk_indices: torch.Tensor,
    *,
    sparse_indices: torch.Tensor,
    end: torch.Tensor,
    col_indices: torch.Tensor,
    plan_ready: bool = False,
) -> None:
    """Triton sparse-MQA logits + top-k for one prefill chunk (packed KV).

    Args mirror `sparse_mqa_logits_prefill_chunk` except:

        k_scale: [total_kv] fp32 per-position scales (the FP8 cache layout,
            not the SM100 UE8M0 packing).
        plan_ready: True when another consumer layer of this step already
            expanded the candidates into `sparse_indices`/`end`.
    """
    assert weights.dtype == torch.float32, weights.dtype
    assert k_scale.dtype == torch.float32
    rows = q.shape[0]
    if not plan_ready:
        candidate_blocks_to_sparse_indices(
            candidate_blocks,
            cu_seqlen_ks,
            cu_seqlen_ke,
            candidate_block_size,
            sparse_block_kv,
            out=(sparse_indices, end),
        )
    width = sparse_indices.shape[1] * sparse_block_kv

    def launch(logits: torch.Tensor, block_n: int) -> None:
        _prefill_sparse_logits_kernel[(rows, triton.cdiv(width, block_n))](
            q,
            k_quant,
            k_scale,
            weights,
            sparse_indices,
            end,
            cu_seqlen_ks,
            cu_seqlen_ke,
            logits,
            q.stride(0),
            q.stride(1),
            weights.stride(0),
            sparse_indices.stride(0),
            logits.stride(0),
            width,
            H=q.shape[1],
            H_PAD=triton.next_power_of_2(q.shape[1]),
            D=q.shape[2],
            SBK=sparse_block_kv,
            BLOCK_N=block_n,
        )

    logits = _sparse_logits(q, end, width, launch)
    sparse_topk_remap_triton(
        logits,
        sparse_indices,
        end,
        cu_seqlen_ks,
        sparse_block_kv,
        topk_tokens,
        topk_indices,
        col_indices=col_indices,
    )


def sparse_mqa_logits_paged_decode_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
    sparse_block_kv: int,
    topk_tokens: int,
    topk_indices: torch.Tensor,
    *,
    row_ks: torch.Tensor,
    sparse_indices: torch.Tensor,
    end: torch.Tensor,
    col_indices: torch.Tensor,
    head_dim: int,
    plan_ready: bool = False,
) -> None:
    """Triton sparse-MQA logits + top-k for paged decode rows.

    Args mirror `sparse_mqa_logits_paged_decode` except:

        q: [rows, H, D] fp8 queries (the Q scale is folded into `weights`).
        kv_cache: [num_blocks, block_size, head_dim + 4] uint8 paged indexer
            cache; per block, all value bytes then all fp32 scales.
        context_lens: [rows] int32 per-row (compressed) context lengths.
        block_table: [bt_rows, P] int32 page table; ``rows`` must be a whole
            multiple of ``bt_rows`` (native multi-token decode repeats each
            request's table row).
        plan_ready: See `sparse_mqa_logits_prefill_chunk_triton`.
    """
    assert weights.dtype == torch.float32, weights.dtype
    rows = q.shape[0]
    assert rows % block_table.shape[0] == 0
    assert context_lens.shape == (rows,) and context_lens.dtype == torch.int32
    if not plan_ready:
        candidate_blocks_to_sparse_indices(
            candidate_blocks,
            row_ks,
            context_lens,
            candidate_block_size,
            sparse_block_kv,
            out=(sparse_indices, end),
        )
    width = sparse_indices.shape[1] * sparse_block_kv
    kv_bytes, kv_scales = _flat_cache_views(kv_cache, head_dim)

    def launch(logits: torch.Tensor, block_n: int) -> None:
        _paged_sparse_logits_kernel[(rows, triton.cdiv(width, block_n))](
            q,
            kv_bytes,
            kv_scales,
            weights,
            sparse_indices,
            end,
            context_lens,
            block_table,
            logits,
            q.stride(0),
            q.stride(1),
            weights.stride(0),
            sparse_indices.stride(0),
            block_table.stride(0),
            block_table.stride(1),
            logits.stride(0),
            rows // block_table.shape[0],
            kv_cache.stride(0),
            kv_cache.shape[1],
            kv_cache.shape[0],
            block_table.shape[1],
            width,
            H=q.shape[1],
            H_PAD=triton.next_power_of_2(q.shape[1]),
            D=head_dim,
            SBK=sparse_block_kv,
            BLOCK_N=block_n,
        )

    logits = _sparse_logits(q, end, width, launch)
    sparse_topk_remap_triton(
        logits,
        sparse_indices,
        end,
        row_ks,
        sparse_block_kv,
        topk_tokens,
        topk_indices,
        col_indices=col_indices,
    )
