# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM90 candidate-only scoring for the DeepSeek V4.1 two-level indexer.

On SM100 the candidate-consuming indexer layers score only the candidate
blocks published by the candidate-source layer (DeepGEMM's sparse MQA-logits
kernels, SM100-only). On SM90 those kernels do not exist, so consumers
currently score the whole context with the dense paged MQA-logits kernel and
then mask the logits down to the candidate blocks, discarding ~98% of the
scoring work at long context.

This module provides the SM90 substitute for decode: gather each row's
candidate K from the paged FP8 indexer cache into a contiguous workspace,
score it with the existing dense (non-paged) MQA-logits kernel, run the
usual decode top-k, and remap the selected columns back to request-local
compressed positions. For unique valid candidates this preserves the scoring
problem without requantizing K. Different scoring kernels and top-k tie
breaking can still change floating-point results or the selected indices.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
    _sparse_topk_remap_kernel,
    candidate_blocks_to_sparse_indices,
    pick_sparse_block_kv,
)
from vllm.model_executor.layers.indexer_topk import (
    RADIX_TOPK_WORKSPACE_SIZE,
    get_indexer_topk,
)
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import fp8_fp4_mqa_logits
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)


@triton.jit(
    do_not_specialize=[
        "cache_block_stride",
        "cache_block_size",
        "num_cols",
        "bt_row_repeat",
    ]
)
def _gather_candidate_k_kernel(
    kv_bytes,  # uint8 flat storage view of the paged indexer cache
    kv_scales,  # fp32 reinterpretation of the same flat storage
    block_table,  # [B, P] int32 per-request page table
    sparse_indices,  # [rows, S] int32 candidate sparse-block ids
    end,  # [rows] int32 valid candidate column count
    out_k,  # [rows, S*SBK, D] uint8 (fp8 payload bytes)
    out_s,  # [rows, S*SBK] fp32
    bt_row_stride,
    bt_col_stride,
    bt_row_repeat,
    si_stride,
    cache_block_stride,  # bytes per physical cache block
    cache_block_size,  # tokens per cache block
    num_cols,  # S*SBK
    SBK: tl.constexpr,  # positions per sparse block
    D: tl.constexpr,  # head_dim
    TILE: tl.constexpr,
):
    """Gather candidate K (fp8 values + fp32 scales) for one row tile.

    The paged FP8 indexer cache stores per block: ``block_size * head_dim``
    value bytes followed by ``block_size * 4`` scale bytes (one fp32 scale
    per entry). Column j of row r scores compressed position
    ``sparse_indices[r, j // SBK] * SBK + j % SBK``; invalid tail columns
    (j >= end) are zero-filled so the top-k bound can simply exclude them.
    """
    row = tl.program_id(0).to(tl.int64)
    cols = (tl.program_id(1) * TILE + tl.arange(0, TILE)).to(tl.int64)
    in_bounds = cols < num_cols
    e = tl.load(end + row).to(tl.int64)
    valid = in_bounds & (cols < e)

    sb = tl.load(
        sparse_indices + row * si_stride + cols // SBK, mask=valid, other=0
    ).to(tl.int64)
    pos = sb * SBK + cols % SBK
    page = pos // cache_block_size
    off = pos % cache_block_size
    bt_row = row // bt_row_repeat
    phys = tl.load(
        block_table + bt_row * bt_row_stride + page * bt_col_stride,
        mask=valid,
        other=0,
    ).to(tl.int64)

    val_base = phys * cache_block_stride + off * D
    d = tl.arange(0, D)
    vals = tl.load(
        kv_bytes + val_base[:, None] + d[None, :], mask=valid[:, None], other=0
    )
    tl.store(
        out_k + (row * num_cols + cols)[:, None] * D + d[None, :],
        vals,
        mask=in_bounds[:, None],
    )

    # Scales live right after the values section of the block (fp32 each).
    sc_idx = (phys * cache_block_stride) // 4 + (cache_block_size * D) // 4 + off
    sc = tl.load(kv_scales + sc_idx, mask=valid, other=0.0)
    tl.store(out_s + row * num_cols + cols, sc, mask=in_bounds)


def gather_candidate_k_decode(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    sparse_indices: torch.Tensor,
    end: torch.Tensor,
    head_dim: int,
    sparse_block_kv: int,
    out_k: torch.Tensor,
    out_s: torch.Tensor,
) -> None:
    """Gather candidate K for decode rows from the paged FP8 indexer cache."""
    rows, num_cols = out_s.shape
    assert kv_cache.dtype == torch.uint8 and kv_cache.ndim == 3
    assert block_table.dtype == torch.int32 and block_table.ndim == 2
    assert sparse_indices.dtype == torch.int32 and end.dtype == torch.int32
    cache_block_size = kv_cache.shape[1]
    page_bytes = cache_block_size * (head_dim + 4)
    assert kv_cache.shape[2] == head_dim + 4, (
        "FP8 indexer cache entry must be head_dim+4 bytes, "
        f"got shape {tuple(kv_cache.shape)} with head_dim={head_dim}"
    )
    assert kv_cache.stride(2) == 1 and kv_cache.stride(0) >= page_bytes
    assert kv_cache.storage_offset() % 4 == 0 and kv_cache.stride(0) % 4 == 0
    assert block_table.stride(1) > 0
    assert rows % block_table.shape[0] == 0
    bt_row_repeat = rows // block_table.shape[0]

    # Packed KV-cache views have gaps between logical pages (other layers share
    # the same physical manager block), so ``view(-1)`` is invalid. Build a
    # zero-copy one-dimensional storage view that includes those gaps and use
    # the physical block stride in the kernel.
    storage_span = (kv_cache.shape[0] - 1) * kv_cache.stride(0) + page_bytes
    kv_bytes = torch.as_strided(kv_cache, (storage_span,), (1,))
    assert storage_span % 4 == 0
    _gather_candidate_k_kernel[(rows, triton.cdiv(num_cols, 64))](
        kv_bytes,
        kv_bytes.view(torch.float32),
        block_table,
        sparse_indices,
        end,
        out_k,
        out_s,
        block_table.stride(0),
        block_table.stride(1),
        bt_row_repeat,
        sparse_indices.stride(0),
        kv_cache.stride(0),
        cache_block_size,
        num_cols,
        SBK=sparse_block_kv,
        D=head_dim,
        TILE=64,
        num_warps=4,
    )


@triton.jit
def _prepare_candidate_rows_kernel(
    seq_lens,
    row_ks,
    row_ends,
    col_starts,
    rows,
    next_n: tl.constexpr,
    num_cols: tl.constexpr,
    per_row_lens: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < rows
    if per_row_lens:
        lens = tl.load(seq_lens + offsets, mask=mask, other=0)
    else:
        request = offsets // next_n
        step = offsets % next_n
        base = tl.load(seq_lens + request, mask=mask, other=0)
        lens = tl.maximum(base - next_n + 1 + step, 0)
    tl.store(row_ks + offsets, 0, mask=mask)
    tl.store(row_ends + offsets, lens, mask=mask)
    tl.store(col_starts + offsets, offsets * num_cols, mask=mask)


def candidate_gather_workspace_specs(
    rows: int,
    num_candidates: int,
    candidate_block_size: int,
    head_dim: int,
    topk_tokens: int,
) -> list[tuple[tuple[int, ...], torch.dtype]]:
    """Simultaneously-live scratch for candidate-only decode scoring.

    The radix reservation must stay first: nested cooperative/persistent top-k
    requests the same workspace prefix while the remaining buffers are live.
    """
    sparse_block_kv = pick_sparse_block_kv(candidate_block_size)
    ratio = candidate_block_size // sparse_block_kv
    assert ratio in (1, 2) and ratio * sparse_block_kv == candidate_block_size
    num_cols = num_candidates * candidate_block_size
    num_sparse = num_candidates * ratio
    return [
        ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        ((rows,), torch.int32),  # row_ks
        ((rows, num_sparse), torch.int32),
        ((rows,), torch.int32),  # row_ends, then candidate-column ends
        ((rows, num_cols, head_dim), torch.uint8),
        ((rows, num_cols), torch.float32),
        ((rows,), torch.int32),  # flattened candidate-row starts
        ((rows,), torch.int32),  # flattened candidate-row ends
        ((rows, topk_tokens), torch.int32),
    ]


def candidate_gather_score_decode(
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
    max_seq_len: int,
    fp8_dtype: torch.dtype,
) -> None:
    """Score only candidate blocks for decode rows (SM90 substitute for the
    SM100 sparse MQA-logits path) and write request-local top-k indices.

    Args:
        q: [rows, H, D] fp8 query values (per-token scale folded into
            ``weights`` on the SM90 FP8 path). Rows follow the padded decode
            layout ``[B, next_n]`` flattened.
        kv_cache: [num_blocks, block_size, head_dim + 4] uint8 indexer cache.
        weights: [rows, H] per-head weights.
        seq_lens: [B, next_n] or [B, 1] int32 per-row effective context lens.
        block_table: [rows, P] int32 per-row page table.
        candidate_blocks: [rows, K] int32 candidate block ids (-1 padded), in
            units of ``candidate_block_size`` compressed positions. K must be
            a power of two.
        candidate_block_size: Compressed positions per candidate block.
        next_n: Query rows per request in the padded layout.
        topk_tokens: Number of tokens to keep per row.
        topk_backend: Decode top-k backend (see ``get_indexer_topk``).
        topk_indices: [rows, topk_tokens] int32 output buffer.
        head_dim: Indexer head dim.
        max_seq_len: Host-side maximum context length for this batch.
        fp8_dtype: The platform FP8 dtype for the K workspace.

    """
    rows = q.shape[0]
    num_candidates = candidate_blocks.shape[1]
    assert candidate_blocks.dtype == torch.int32
    assert block_table.shape[0] in (rows, rows // next_n)
    assert candidate_blocks.shape[0] in (rows, rows // next_n)
    sparse_block_kv = pick_sparse_block_kv(candidate_block_size)
    num_cols = num_candidates * candidate_block_size

    (
        _radix_reserve,
        row_ks,
        sparse_indices,
        end,
        k_gathered,
        s_gathered,
        col_starts,
        col_ends,
        local_topk,
    ) = current_workspace_manager().get_simultaneous(
        *candidate_gather_workspace_specs(
            rows,
            num_candidates,
            candidate_block_size,
            head_dim,
            topk_tokens,
        )
    )
    _prepare_candidate_rows_kernel[(triton.cdiv(rows, 128),)](
        seq_lens,
        row_ks,
        end,
        col_starts,
        rows,
        next_n=next_n,
        num_cols=num_cols,
        per_row_lens=seq_lens.numel() == rows,
        BLOCK=128,
    )
    candidate_blocks_to_sparse_indices(
        candidate_blocks,
        row_ks,
        end,
        candidate_block_size,
        sparse_block_kv,
        out=(sparse_indices, end),
    )

    gather_candidate_k_decode(
        kv_cache,
        block_table,
        sparse_indices,
        end,
        head_dim,
        sparse_block_kv,
        k_gathered,
        s_gathered,
    )

    torch.add(col_starts, end, out=col_ends)
    logits = fp8_fp4_mqa_logits(
        (q, None),
        (k_gathered.view(-1, head_dim).view(fp8_dtype), s_gathered.view(-1)),
        weights,
        col_starts,
        col_ends,
        num_cols,
    )
    # SM90 returns fp32 [rows, num_cols]; only the first ``end`` columns of
    # each row are valid.

    get_indexer_topk(topk_backend)(
        logits,
        end.view(-1, next_n),
        next_n,
        local_topk,
        topk_tokens,
        min(max_seq_len, num_cols),
    )
    _sparse_topk_remap_kernel[(rows,)](
        local_topk,
        local_topk.stride(0),
        sparse_indices,
        sparse_indices.stride(0),
        row_ks,
        row_ks.stride(0),
        topk_indices,
        topk_indices.stride(0),
        topk_tokens,
        num_cols,
        SBK=sparse_block_kv,
        K_POW2=triton.next_power_of_2(topk_tokens),
    )


def candidate_gather_decode_profitable(
    max_seq_len: int,
    num_rows: int,
    num_candidates: int,
    candidate_block_size: int,
) -> bool:
    """Apply user-tunable host-side shape gates before candidate scoring.

    Dense scoring reads the whole context for every decode row, while
    candidate scoring reads a fixed candidate width per row and is dominated
    by a per-step fixed cost. The crossover therefore tracks the scanned work
    ``num_rows * max_seq_len`` relative to the candidate width, not the
    context length alone. The crossover constant is hardware-dependent, so no
    device-specific value is baked in here.
    """
    import vllm.envs as envs

    if envs.VLLM_DSA_CANDIDATE_GATHER_FORCE:
        return True

    candidate_width = num_candidates * candidate_block_size
    min_ratio = max(1, envs.VLLM_DSA_CANDIDATE_GATHER_MIN_WORK_RATIO)
    min_rows = max(1, envs.VLLM_DSA_CANDIDATE_GATHER_MIN_ROWS)
    if num_rows < min_rows:
        return False
    return num_rows * max_seq_len >= min_ratio * candidate_width


def candidate_gather_decode_available(
    use_fp4_cache: bool,
    dcp_world_size: int,
    use_pcp: bool,
    candidate_block_size: int,
    num_candidates: int,
    topk_tokens: int,
    topk_backend: str,
) -> bool:
    """Whether the SM90 gather-based candidate scoring path applies.

    Restricted to SM90 FP8-cache decode without context parallelism; SM100
    has the native sparse kernels and other platforms keep the dense
    score-then-mask fallback.
    """
    import vllm.envs as envs
    from vllm.platforms import current_platform
    from vllm.utils.deep_gemm import has_deep_gemm

    if not envs.VLLM_DSA_CANDIDATE_GATHER_DECODE:
        return False
    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(90)
    ):
        return False
    if use_fp4_cache or dcp_world_size != 1 or use_pcp:
        return False
    if topk_backend == "deep_select":
        return False
    if not has_deep_gemm():
        return False
    if num_candidates <= 0 or num_candidates & (num_candidates - 1):
        # The candidate-expansion kernel's in-register sort needs pow2.
        return False
    if candidate_block_size <= 0 or topk_tokens > num_candidates * candidate_block_size:
        return False
    try:
        sparse_block_kv = pick_sparse_block_kv(candidate_block_size)
    except ValueError:
        return False
    ratio = candidate_block_size // sparse_block_kv
    return ratio in (1, 2) and ratio * sparse_block_kv == candidate_block_size
