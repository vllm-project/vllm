# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM90 candidate-only decode scoring for the DeepSeek V4.1 two-level indexer.

The V4.1 indexer publishes candidate blocks at ``candidate_source_layer_id``;
the later consumer indexers are meant to select their top-k *inside those
candidate blocks only*. On SM100 that is DeepGEMM's sparse MQA-logits kernels.
SM90 has no equivalent, so the consumers score the whole compressed context
with the dense paged kernel and then mask the logits down to the candidate
blocks, discarding the work outside them. With a 16K-position candidate set
against a 1M-position context that is ~98% of the scoring.

This module scores only the candidate positions, reading K **in place** from
the paged FP8 indexer cache. Logits land in a compact ``[rows, num_cands *
candidate_block_size]`` buffer whose columns are candidate-relative, the usual
decode top-k runs over that buffer, and the selected columns are remapped back
to request-local compressed positions.

Two properties keep replays correct under full CUDA-graph capture:

* **No per-step host decision.** Context lengths stay on the device; the host
  only sees shapes that are fixed by the model and scheduler config. A full
  CUDA graph therefore cannot bake a context-dependent choice, and replays
  stay correct at any live context length.
* **K is read in place.** Unlike a scheme that gathers candidate K into a
  scratch buffer first, no K data is copied that the dense path would not
  also read. The only extra work is per-row candidate bookkeeping (expansion
  and sorting of block ids), computed once per step and shared by the
  candidate-consuming layers, whose inputs are the same step-level tensors.

Numerics follow the decode reference (``fp8_paged_mqa_logits``): dot, relu,
head weights, sum over heads, then one dequantization multiply per column.
Scaling after the relu is exact because the per-position scale is positive,
and it costs ``num_heads`` times fewer multiplies.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
    _sparse_topk_remap_kernel,
    candidate_blocks_to_sparse_indices,
)
from vllm.model_executor.layers.indexer_topk import (
    RADIX_TOPK_WORKSPACE_SIZE,
    get_indexer_topk,
)
from vllm.triton_utils import tl, triton
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

# One fp32 scale per cached position, stored after the block's value bytes.
INDEXER_SCALE_BYTES = 4
# The same value for use inside the kernel; Triton only reads globals that are
# instantiated as constexpr.
_SCALE_BYTES = tl.constexpr(4)

# Compact columns per program. A multiple of the candidate block size, and at
# least 16 so the MMA has a full M tile.
_BLOCK_N = 128


@triton.jit
def _prepare_candidate_rows_kernel(
    seq_lens,
    row_ks,
    row_ends,
    rows,
    next_n: tl.constexpr,
    per_row_lens: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per-row causal context length, and the zero K-range start the paged
    candidate expansion expects (its block ids are context-relative)."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < rows
    if per_row_lens:
        lens = tl.load(seq_lens + offsets, mask=mask, other=0)
    else:
        # One length per request: speculative row ``s`` of a request sees
        # ``next_n - 1 - s`` fewer positions than the request's newest row.
        request = offsets // next_n
        step = offsets % next_n
        base = tl.load(seq_lens + request, mask=mask, other=0)
        lens = tl.maximum(base - next_n + 1 + step, 0)
    tl.store(row_ks + offsets, 0, mask=mask)
    tl.store(row_ends + offsets, lens, mask=mask)


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
    ]
)
def _paged_candidate_logits_kernel(
    q_ptr,  # fp8 [rows, H, D]
    kv_bytes,  # uint8 flat storage view of the paged indexer cache
    kv_scales,  # fp32 reinterpretation of the same flat storage
    w_ptr,  # fp32 [rows, H]
    blocks_ptr,  # int32 [rows, S] candidate block ids, sorted, padded tail
    end_ptr,  # int32 [rows] valid compact column count
    causal_ptr,  # int32 [rows] causal context length, compressed positions
    bt_ptr,  # int32 [bt_rows, P] page table
    out_ptr,  # fp32 [rows, width]
    stride_q_row,
    stride_q_head,
    stride_w_row,
    stride_blocks_row,
    stride_bt_row,
    stride_bt_col,
    stride_out_row,
    bt_row_repeat,
    cache_block_stride,  # bytes per physical cache block
    cache_block_size,  # positions per cache block
    num_blocks,
    bt_width,
    width,
    H: tl.constexpr,  # indexer heads actually present
    H_PAD: tl.constexpr,  # padded to a power of two for the MMA
    D: tl.constexpr,  # indexer head dim
    CBS: tl.constexpr,  # positions per candidate block
    BLOCK_N: tl.constexpr,  # compact columns per program
):
    """Score one tile of candidate columns for one decode row.

    Compact column ``c`` scores request-local compressed position
    ``blocks[row, c // CBS] * CBS + c % CBS``. Columns past the row's valid
    candidate prefix, columns whose block is padding, and columns past the
    row's causal length all become ``-inf`` so the top-k never selects them.
    """
    row = tl.program_id(0).to(tl.int64)
    tile_start = tl.program_id(1) * BLOCK_N
    cols = tile_start + tl.arange(0, BLOCK_N)
    in_bounds = cols < width

    end = tl.load(end_ptr + row)
    if tile_start >= end:
        # Dead tile: the top-k bounds never read past ``end``, but keep the
        # -inf contract for the whole buffer without paying for the gather
        # and MMA. The grid stays width-sized, so this is CUDA-graph safe.
        tl.store(
            out_ptr + row * stride_out_row + cols,
            tl.full((BLOCK_N,), -float("inf"), tl.float32),
            mask=in_bounds,
        )
        return

    causal = tl.load(causal_ptr + row)
    live = in_bounds & (cols < end)

    blk = tl.load(
        blocks_ptr + row * stride_blocks_row + cols // CBS, mask=live, other=-1
    ).to(tl.int64)
    pos = blk * CBS + cols % CBS
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

    # Values are contiguous per position, so a column's D bytes are one
    # coalesced segment and consecutive columns of a candidate block adjoin.
    k_offset = safe_phys * cache_block_stride + page_off * D
    k_block = tl.load(
        kv_bytes + k_offset[:, None] + d[None, :],
        mask=live[:, None],
        other=0,
    ).to(tl.float8e4nv, bitcast=True)
    # Zero-padded heads contribute relu(0) * 0 = 0, so the padding is exact.
    # Loaded as the MMA's K-major operand directly; the tile is small and this
    # avoids an in-kernel transpose of an fp8 operand.
    q_block = tl.load(
        q_ptr + row * stride_q_row + h[None, :] * stride_q_head + d[:, None],
        mask=h_live[None, :],
        other=0.0,
    )
    weights = tl.load(w_ptr + row * stride_w_row + h, mask=h_live, other=0.0).to(
        tl.float32
    )

    # [BLOCK_N, H_PAD] on fp8 tensor cores with fp32 accumulation. Scoring the
    # columns as the MMA's M dimension keeps the large operand untransposed.
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


def paged_candidate_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    candidate_blocks: torch.Tensor,
    end: torch.Tensor,
    causal_lens: torch.Tensor,
    block_table: torch.Tensor,
    candidate_block_size: int,
    head_dim: int,
    out: torch.Tensor,
) -> None:
    """Score each row's candidate positions into a compact logits buffer.

    Args:
        q: [rows, H, D] fp8 queries. On the SM90 FP8 path the per-token scale
            is already folded into ``weights``.
        kv_cache: [num_blocks, block_size, head_dim + 4] uint8 paged indexer
            cache; per block, all value bytes then all fp32 scales.
        weights: [rows, H] per-head weights.
        candidate_blocks: [rows, S] int32 candidate block ids, sorted with a
            padded tail, as produced by ``candidate_blocks_to_sparse_indices``.
        end: [rows] int32 valid compact column count per row.
        causal_lens: [rows] int32 causal context length in compressed
            positions.
        block_table: [bt_rows, P] int32 page table; ``rows`` must be a whole
            multiple of ``bt_rows`` for speculative row expansion.
        candidate_block_size: Compressed positions per candidate block.
        head_dim: Indexer head dim.
        out: [rows, S * candidate_block_size] fp32 output.

    """
    rows, width = out.shape
    num_heads = q.shape[1]
    assert q.shape[0] == rows and weights.shape == (rows, num_heads)
    assert q.shape[2] == head_dim
    assert out.dtype == torch.float32 and out.stride(1) == 1
    assert candidate_blocks.shape[0] == rows
    assert candidate_blocks.shape[1] * candidate_block_size == width
    assert candidate_blocks.dtype == torch.int32
    assert end.dtype == torch.int32 and causal_lens.dtype == torch.int32
    assert block_table.dtype == torch.int32 and block_table.ndim == 2
    assert rows % block_table.shape[0] == 0
    assert q.stride(2) == 1

    kv_bytes, kv_scales = _flat_cache_views(kv_cache, head_dim)
    block_n = max(_BLOCK_N, candidate_block_size)
    assert block_n % candidate_block_size == 0
    _paged_candidate_logits_kernel[(rows, triton.cdiv(width, block_n))](
        q,
        kv_bytes,
        kv_scales,
        weights,
        candidate_blocks,
        end,
        causal_lens,
        block_table,
        out,
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        candidate_blocks.stride(0),
        block_table.stride(0),
        block_table.stride(1),
        out.stride(0),
        rows // block_table.shape[0],
        kv_cache.stride(0),
        kv_cache.shape[1],
        kv_cache.shape[0],
        block_table.shape[1],
        width,
        H=num_heads,
        H_PAD=triton.next_power_of_2(num_heads),
        D=head_dim,
        CBS=candidate_block_size,
        BLOCK_N=block_n,
    )


def candidate_paged_workspace_specs(
    rows: int,
    num_candidates: int,
    candidate_block_size: int,
    topk_tokens: int,
) -> list[tuple[tuple[int, ...], torch.dtype]]:
    """Simultaneously-live scratch for candidate-only decode scoring.

    The radix reservation must stay first: nested cooperative/persistent top-k
    requests the same workspace prefix while the remaining buffers are live.
    """
    width = num_candidates * candidate_block_size
    return [
        ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        ((rows,), torch.int32),  # row_ks
        ((rows,), torch.int32),  # causal lens, then candidate-column ends
        ((rows, num_candidates), torch.int32),  # sorted candidate block ids
        ((rows,), torch.int32),  # valid compact columns per row
        ((rows, width), torch.float32),  # compact logits
        ((rows, topk_tokens), torch.int32),  # compact top-k columns
    ]


def candidate_paged_prepare(
    seq_lens: torch.Tensor,
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
    next_n: int,
    rows: int,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Expand the step's shared candidate blocks into a per-row scoring plan.

    ``out`` is ``(row_ks, causal_lens, sorted_blocks, end)``, each sized for at
    least ``rows``. The inputs are step-level, so every candidate-consuming
    layer sees an identical plan; the caller computes it once per step and
    the later layers reuse it.
    """
    row_ks, causal_lens, sorted_blocks, end = (t[:rows] for t in out)
    assert candidate_blocks.shape[0] >= rows, (candidate_blocks.shape[0], rows)
    _prepare_candidate_rows_kernel[(triton.cdiv(rows, 128),)](
        seq_lens,
        row_ks,
        causal_lens,
        rows,
        next_n=next_n,
        per_row_lens=seq_lens.numel() == rows,
        BLOCK=128,
    )
    # Sorting the candidate ids makes the valid columns a prefix, which is
    # what the top-k backends' per-row bounds require. ``end`` accounts for a
    # partial final block.
    candidate_blocks_to_sparse_indices(
        candidate_blocks[:rows],
        row_ks,
        causal_lens,
        candidate_block_size,
        candidate_block_size,
        out=(sorted_blocks, end),
    )


def candidate_paged_score_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
    next_n: int,
    topk_tokens: int,
    topk_backend: str,
    topk_indices: torch.Tensor,
    head_dim: int,
    plan: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Select each decode row's top-k inside its candidate blocks.

    Writes request-local compressed positions into ``topk_indices``, ``-1``
    padded, matching what the dense score-then-mask path produces.

    Args:
        q: [rows, H, D] fp8 queries in the padded decode layout
            ``[B, next_n]`` flattened.
        kv_cache: [num_blocks, block_size, head_dim + 4] uint8 indexer cache.
        weights: [rows, H] per-head weights with the Q scale folded in.
        seq_lens: [B, next_n] or [B, 1] int32 causal context lengths.
        block_table: [B, P] int32 page table.
        candidate_blocks: [B or rows, S] int32 request-local candidate block
            ids, ``-1`` padded, unsorted.
        candidate_block_size: Compressed positions per candidate block.
        next_n: Query rows per request in the padded layout.
        topk_tokens: Positions to keep per row.
        topk_backend: Decode top-k backend, see ``get_indexer_topk``.
        topk_indices: [rows, topk_tokens] int32 output.
        head_dim: Indexer head dim.
        plan: Optional ``(row_ks, causal_lens, sorted_blocks, end)`` from
            ``candidate_paged_prepare``. All candidate-consuming layers in a
            step share one plan, so callers pass it to skip the expansion.

    """
    rows = q.shape[0]
    num_candidates = candidate_blocks.shape[1]
    width = num_candidates * candidate_block_size
    # The candidate buffer is indexed per decode row, like the logits the
    # dense path masks, and may be a longer buffer view than this step needs.
    # Candidates are indexed per decode row, matching the dense path's mask
    # (``candidates + row * stride_row``); only the causal lengths are shared
    # across a request's speculative rows. The buffer is sized for
    # max_num_batched_tokens, so it is usually longer than this step needs.
    assert candidate_blocks.shape[0] >= rows, (candidate_blocks.shape[0], rows)

    ws_manager = current_workspace_manager()
    if plan is None:
        (
            _radix_reserve,
            row_ks,
            causal_lens,
            sorted_blocks,
            end,
            logits,
            local_topk,
        ) = ws_manager.get_simultaneous(
            *candidate_paged_workspace_specs(
                rows, num_candidates, candidate_block_size, topk_tokens
            )
        )
        candidate_paged_prepare(
            seq_lens,
            candidate_blocks,
            candidate_block_size,
            next_n,
            rows,
            out=(row_ks, causal_lens, sorted_blocks, end),
        )
    else:
        row_ks, causal_lens, sorted_blocks, end = (t[:rows] for t in plan)
        assert sorted_blocks.shape == (rows, num_candidates)
        assert end.shape == (rows,) and end.dtype == torch.int32
        _radix_reserve, logits, local_topk = ws_manager.get_simultaneous(
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            ((rows, width), torch.float32),
            ((rows, topk_tokens), torch.int32),
        )

    paged_candidate_logits(
        q,
        kv_cache,
        weights,
        sorted_blocks,
        end,
        causal_lens,
        block_table,
        candidate_block_size,
        head_dim,
        logits,
    )
    # ``width`` is fixed by the model config, so passing it as the host-side
    # bound keeps this call free of any context-dependent value and therefore
    # safe to capture in a CUDA graph.
    get_indexer_topk(topk_backend)(
        logits,
        end.view(-1, next_n),
        next_n,
        local_topk,
        topk_tokens,
        width,
    )
    _sparse_topk_remap_kernel[(rows,)](
        local_topk,
        local_topk.stride(0),
        sorted_blocks,
        sorted_blocks.stride(0),
        row_ks,
        row_ks.stride(0),
        topk_indices,
        topk_indices.stride(0),
        topk_tokens,
        width,
        SBK=candidate_block_size,
        K_POW2=triton.next_power_of_2(topk_tokens),
    )


def candidate_paged_decode_available(
    use_fp4_cache: bool,
    dcp_world_size: int,
    use_pcp: bool,
    candidate_block_size: int,
    num_candidates: int,
    topk_tokens: int,
    head_dim: int,
) -> bool:
    """Whether candidate-only decode scoring applies to this configuration.

    Every input is fixed by the model, platform and scheduler config, so the
    answer is a CUDA-graph invariant: a captured graph cannot diverge from the
    path chosen while it was captured.

    Restricted to SM90 with the FP8 indexer cache and no context parallelism.
    SM100 has the native sparse kernels, and the other platforms keep the
    dense score-then-mask fallback.
    """
    import vllm.envs as envs
    from vllm.platforms import current_platform

    if not envs.VLLM_DSA_CANDIDATE_PAGED_DECODE:
        return False
    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(90)
    ):
        return False
    if use_fp4_cache or dcp_world_size != 1 or use_pcp:
        return False
    if num_candidates <= 0 or num_candidates & (num_candidates - 1):
        # The candidate expansion's in-register sort needs a power of two.
        return False
    if candidate_block_size <= 0 or candidate_block_size & (candidate_block_size - 1):
        # Compact columns are indexed by shifting, and the MMA tile must be a
        # whole number of candidate blocks.
        return False
    if topk_tokens > num_candidates * candidate_block_size:
        return False
    # The MMA's K dimension is the head dim.
    return head_dim > 0 and not (head_dim & (head_dim - 1))
