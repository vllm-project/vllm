# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepGEMM sparse MQA-logits path for the DeepSeek V4.1 two-level indexer.

The candidate-source indexer publishes the top candidate blocks (in units of
``candidate_block_size`` compressed positions). Instead of scoring every KV
position and masking the dense logits down to those blocks, this path scores
only the candidate tokens with DeepGEMM's sparse MQA logits kernels
(``fp8_fp4_(paged_)sparse_mqa_logits``, DeepGEMM >= 2.8, SM100 only) and runs
the row top-k directly in the compressed sparse space, remapping the selected
columns back to request-local positions.

Only the MXFP4 indexer cache is supported: the kernels require UE8M0-packed
(granularity-32) Q/KV scales, which is exactly the MXFP4 cache layout. The
FP8 indexer cache stores one fp32 scale per quant block instead.
"""

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    fp8_fp4_paged_sparse_mqa_logits,
    fp8_fp4_sparse_mqa_logits,
    get_paged_sparse_mqa_logits_metadata,
    get_sparse_mqa_logits_metadata,
    has_deep_gemm_sparse_mqa,
)

# Sparse-block sizes (in tokens) supported by the DeepGEMM kernels.
_SPARSE_BLOCK_KV_CHOICES = (16, 8)

# vllm's radix top-k kernels (used for the decode side) support only these k.
_TOPK_KERNEL_SUPPORTED = (512, 1024, 2048)

# Scratch for the persistent radix top-k (same size the dense path uses).
_RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

_INT32_BIG = tl.constexpr(2**31 - 1)


@triton.jit
def _expand_candidates_kernel(
    cand_ptr,
    ks_ptr,
    ke_ptr,
    si_ptr,
    end_ptr,
    K: tl.constexpr,
    S: tl.constexpr,
    RATIO: tl.constexpr,
    SBK: tl.constexpr,
):
    """One program per row: expand + filter + sort candidate blocks.

    Each row's valid sparse blocks come out unique and strictly increasing
    (candidates are unique in production), padded by repeating the last valid
    block — DeepGEMM's padding convention. ``end`` counts valid sparse-token
    columns; valid columns form a prefix because block ids are sorted.
    """
    row = tl.program_id(0)
    ks = tl.load(ks_ptr + row)
    ke = tl.load(ke_ptr + row)
    cols = tl.arange(0, K)
    cand = tl.load(cand_ptr + row * K + cols)  # K is pow2, no mask needed
    if RATIO == 2:
        cand = tl.interleave(cand * RATIO, cand * RATIO + 1)

    # Keep only blocks inside the row's KV range [0, ke - ks); the lower
    # bound is free since candidates are non-negative.
    kv_len = tl.maximum(ke - ks, 0)
    max_block = (kv_len + SBK - 1) // SBK
    valid = (cand >= 0) & (cand < max_block)
    vals = tl.where(valid, cand, _INT32_BIG)
    vals = tl.sort(vals)

    # Anchor at the row's ks rounded down to a sparse-block boundary
    # (identity for the paged path, where ks == 0).
    base = ks // SBK
    si = tl.where(vals != _INT32_BIG, vals, 0) + base
    n_valid = tl.sum(valid.to(tl.int32))
    last_rel = tl.max(tl.where(valid, cand, -1))
    pad_val = tl.where(n_valid > 0, last_rel + base, base)
    tl.store(
        si_ptr + row * S + tl.arange(0, S),
        tl.where(tl.arange(0, S) < n_valid, si, pad_val),
    )

    # Valid sparse-token column count; only the last block can be partial.
    # Absolute start of the last block: (base + last_rel) * SBK + ks % SBK,
    # which is exactly last_rel * SBK + ks.
    last_start = last_rel * SBK + ks
    fill = tl.minimum(tl.maximum(ke - last_start, 1), SBK)
    end = tl.where(n_valid > 0, (n_valid - 1) * SBK + fill, 0)
    tl.store(end_ptr + row, end)


@triton.jit
def _sparse_topk_remap_kernel(
    col_ptr,
    si_ptr,
    ks_ptr,
    out_ptr,
    k,
    S,
    col_stride,
    out_stride,
    SBK: tl.constexpr,
    K_POW2: tl.constexpr,
):
    """Remap sparse top-k columns to request-local positions.

    Column ``j * SBK + o`` scores the token at
    ``(si[row, j] - ks // SBK) * SBK + o``; -1 columns stay -1.
    """
    row = tl.program_id(0)
    ks_base = tl.load(ks_ptr + row) // SBK
    cols = tl.arange(0, K_POW2)
    c = tl.load(col_ptr + row * col_stride + cols, mask=cols < k, other=-1)
    valid = c >= 0
    cc = tl.where(valid, c, 0)
    blocks = tl.load(si_ptr + row * S + cc // SBK)
    pos = (blocks - ks_base) * SBK + cc % SBK
    tl.store(out_ptr + row * out_stride + cols, tl.where(valid, pos, -1), mask=cols < k)


def sparse_mqa_logits_supported(candidate_block_size: int, use_fp4_cache: bool) -> bool:
    """Whether the sparse-MQA candidate path can run on this system.

    The path is O(candidate_blocks) regardless of context length while the
    dense path is O(context) — it pays off for long-context deployments,
    which is what the opt-in env flag implies.
    """
    return (
        envs.VLLM_DSA_SPARSE_MQA_LOGITS
        and use_fp4_cache
        and candidate_block_size > 0
        and candidate_block_size % _SPARSE_BLOCK_KV_CHOICES[-1] == 0
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and has_deep_gemm_sparse_mqa()
    )


def pick_sparse_block_kv(candidate_block_size: int) -> int:
    """Largest sparse-block size (tokens) dividing the candidate block size."""
    for sbk in _SPARSE_BLOCK_KV_CHOICES:
        if candidate_block_size % sbk == 0:
            return sbk
    raise ValueError(
        f"candidate_block_size={candidate_block_size} is not a multiple of "
        f"any supported sparse_block_kv {_SPARSE_BLOCK_KV_CHOICES}."
    )


def candidate_blocks_to_sparse_indices(
    candidate_blocks: torch.Tensor,
    row_ks: torch.Tensor,
    row_ke: torch.Tensor,
    candidate_block_size: int,
    sparse_block_kv: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand request-local candidate blocks into DeepGEMM sparse block ids.

    Args:
        candidate_blocks: [rows, K] int32 request-local candidate block ids
            (-1 padded), in units of ``candidate_block_size`` positions.
        row_ks/row_ke: [rows] int32 per-row K range; bounds are in the same
            (packed-workspace) coordinates the sparse kernel iterates over.
            Pass zeros for the paged path, whose blocks are context-relative.
        candidate_block_size: Positions per candidate block.
        sparse_block_kv: Positions per sparse block (8 or 16).

    Returns:
        sparse_indices: [rows, K * ratio] int32 block ids for DeepGEMM.
            Each row's valid prefix is unique and strictly increasing;
            remaining slots repeat the last valid block (the kernel's padding
            convention). With the unaligned-ks variant, block ``i`` starts at
            ``i * sparse_block_kv + ks % sparse_block_kv``.
        end: [rows] int32 count of valid sparse-token columns; valid columns
            always form a prefix because block ids are strictly increasing.
    """
    rows, num_candidates = candidate_blocks.shape
    ratio = candidate_block_size // sparse_block_kv
    num_sparse = num_candidates * ratio
    device = candidate_blocks.device

    # Fused Triton path: the whole expansion + filter + sort in one launch.
    if (
        candidate_blocks.is_cuda
        and candidate_blocks.is_contiguous()
        and row_ks.dtype == row_ke.dtype == candidate_blocks.dtype == torch.int32
        and row_ks.is_contiguous()
        and row_ke.is_contiguous()
        and num_candidates & (num_candidates - 1) == 0  # tl.sort needs pow2
        and ratio in (1, 2)
    ):
        sparse_indices = torch.empty(rows, num_sparse, dtype=torch.int32, device=device)
        end = torch.empty(rows, dtype=torch.int32, device=device)
        _expand_candidates_kernel[(rows,)](
            candidate_blocks,
            row_ks,
            row_ke,
            sparse_indices,
            end,
            K=num_candidates,
            S=num_sparse,
            RATIO=ratio,
            SBK=sparse_block_kv,
            num_warps=8,
        )
        return sparse_indices, end

    # Expand each candidate block into its `ratio` sparse blocks. All math
    # stays in int32: garbage candidate values (cudagraph warmup runs on
    # uninitialized buffers) overflow to negatives and get filtered below.
    offsets = torch.arange(ratio, device=device, dtype=torch.int32)
    blocks = (candidate_blocks.unsqueeze(-1) * ratio + offsets).flatten(1)

    # Keep only blocks inside the row's KV range [0, ke - ks); the lower
    # bound is free since candidates are non-negative.
    kv_lens = (row_ke - row_ks).clamp(min=0)
    max_block = (kv_lens + sparse_block_kv - 1) // sparse_block_kv
    valid = (blocks >= 0) & (blocks < max_block.unsqueeze(1))

    # Sort ascending; invalid entries take the sentinel value and so end up
    # last, forming the padding region. Candidates are unique per row by
    # construction, and the kernel tolerates repeated padding blocks (that is
    # DeepGEMM's own padding convention), so no dedup pass is needed.
    big = torch.iinfo(torch.int32).max
    sorted_blocks, _ = torch.where(valid, blocks, big).sort(dim=1)
    n_valid = (sorted_blocks != big).sum(dim=1, dtype=torch.int32)

    # Relative -> kernel block id: anchor at the row's ks rounded down to a
    # sparse-block boundary (identity for the paged path, where ks == 0).
    base = row_ks // sparse_block_kv
    sparse_indices = torch.where(
        sorted_blocks != big, sorted_blocks, 0
    ) + base.unsqueeze(1)

    # Pad with the last valid block; empty rows repeat the first in-range
    # block (matches the DeepGEMM test convention of repeating/0-padding).
    last_slot = (n_valid - 1).clamp(min=0).unsqueeze(1).long()
    last_block = sparse_indices.gather(1, last_slot).squeeze(1)
    pad = torch.arange(num_sparse, device=device) >= n_valid.unsqueeze(1)
    sparse_indices = torch.where(
        pad, last_block.unsqueeze(1), sparse_indices
    ).contiguous()

    # Valid sparse-token column count. Blocks are sorted, so only the last
    # valid block can extend past ke.
    ks_mod = row_ks % sparse_block_kv
    last_start = last_block * sparse_block_kv + ks_mod
    last_fill = (row_ke - last_start).clamp(min=1, max=sparse_block_kv)
    end = torch.where(
        n_valid > 0,
        (n_valid - 1) * sparse_block_kv + last_fill,
        torch.zeros_like(n_valid),
    )

    return sparse_indices, end


def sparse_topk_remap(
    logits: torch.Tensor,
    sparse_indices: torch.Tensor,
    end: torch.Tensor,
    row_ks: torch.Tensor,
    sparse_block_kv: int,
    topk_tokens: int,
    topk_indices: torch.Tensor,
    *,
    decode: bool,
) -> None:
    """Row top-k over sparse logits, remapped to request-local positions.

    Column ``j * sparse_block_kv + o`` of ``logits`` scores the token at
    request-local position
    ``(sparse_indices[row, j] - ks // sparse_block_kv) * sparse_block_kv + o``.

    The valid columns form a prefix (block ids are strictly increasing), so
    the per-row ``end`` bounds can drive vllm's radix top-k kernels directly —
    the same kernels and tie behavior as the dense path. Those kernels are
    fp32-only, hence the cast of the bf16 sparse logits. Slots beyond a row's
    valid count come back as -1 from the kernels.
    """
    rows, width = logits.shape
    device = logits.device
    k = min(topk_tokens, width)

    logits_f32 = logits.float()
    end_i32 = end.to(torch.int32)
    col_indices = torch.empty(rows, k, dtype=torch.int32, device=device)
    if decode:
        workspace = torch.empty(
            _RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=device
        )
        torch.ops._C.persistent_topk(
            logits_f32, end_i32, col_indices, workspace, k, width
        )
    else:
        zeros = torch.zeros(rows, dtype=torch.int32, device=device)
        ops.top_k_per_row_prefill(
            logits_f32, zeros, end_i32, col_indices, rows, width, 1, k
        )

    # Remap sparse column -> request-local position, keeping the kernels' -1
    # padding. A selected column is ``sparse_idx[c // sbk] * sbk + c % sbk``
    # in kernel coordinates; subtract the row's aligned ks to make it
    # request-local (identity for decode, where ks == 0).
    if col_indices.is_cuda and (k & (k - 1)) == 0:
        _sparse_topk_remap_kernel[(rows,)](
            col_indices,
            sparse_indices,
            row_ks,
            topk_indices,
            k,
            sparse_indices.shape[1],
            col_indices.stride(0),
            topk_indices.stride(0),
            SBK=sparse_block_kv,
            K_POW2=triton.next_power_of_2(k),
        )
        return

    valid_col = col_indices >= 0
    cols64 = col_indices.long().clamp(min=0)
    blocks = sparse_indices.gather(1, cols64 // sparse_block_kv)
    pos = (blocks.long() - (row_ks // sparse_block_kv).unsqueeze(1)) * sparse_block_kv
    pos = pos + cols64 % sparse_block_kv
    pos = torch.where(valid_col, pos, -1)
    topk_indices[:, :k] = pos.to(torch.int32)


def sparse_mqa_logits_prefill_chunk(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
    topk_tokens: int,
    topk_indices: torch.Tensor,
) -> None:
    """Sparse-MQA logits + top-k for one prefill chunk (packed KV workspace).

    Args:
        q: [rows, H, D] packed Q values (MXFP4 viewed as int8).
        q_scale: [rows, H] int32 packed UE8M0 Q scales.
        k_quant: [total_kv, D] packed K workspace (MXFP4 viewed as int8).
        k_scale: [total_kv] int32 packed UE8M0 K scales.
        weights: [rows, H] raw per-head weights (cast to bf16 internally);
            the sparse kernels require bf16 and do not fold the Q scale in.
        cu_seqlen_ks/cu_seqlen_ke: [rows] int32 per-token K bounds in the
            packed workspace.
        candidate_blocks: [rows, K] int32 request-local candidate block ids.
        topk_indices: [rows, topk_tokens] output buffer.
    """
    sparse_block_kv = pick_sparse_block_kv(candidate_block_size)
    sparse_indices, end = candidate_blocks_to_sparse_indices(
        candidate_blocks,
        cu_seqlen_ks,
        cu_seqlen_ke,
        candidate_block_size,
        sparse_block_kv,
    )
    metadata = get_sparse_mqa_logits_metadata(
        cu_seqlen_ks,
        cu_seqlen_ke,
        k_quant.shape[0],
        sparse_indices,
        q.dtype,
        sparse_block_kv,
    )
    logits = fp8_fp4_sparse_mqa_logits(
        (q, q_scale),
        (k_quant, k_scale),
        weights.to(torch.bfloat16),
        metadata,
        sparse_indices.shape[1],
        sparse_block_kv,
    )
    sparse_topk_remap(
        logits,
        sparse_indices,
        end,
        cu_seqlen_ks,
        sparse_block_kv,
        topk_tokens,
        topk_indices,
        decode=False,
    )


def sparse_mqa_logits_paged_decode(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    row_indices: torch.Tensor | None,
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
    topk_tokens: int,
    topk_indices: torch.Tensor,
) -> bool:
    """Sparse-MQA logits + top-k for paged decode; False if unsupported here.

    Args:
        q: [batch, next_n, H, D] packed Q values (MXFP4 viewed as int8);
            flattened to [batch * next_n, 1, H, D] for the kernel, which only
            supports one Q row per query.
        q_scale: [batch, next_n, H] int32 packed UE8M0 Q scales.
        kv_cache: 4D [num_pages, page_kv, 1, head_bytes] uint8 fused cache
            view; the page stride must be 512B-aligned.
        weights: [batch * next_n, H] raw per-head weights.
        seq_lens: [batch] or [batch, next_n] int32 per-row context lengths.
        block_table: [batch, P] int32 page table (expanded per row here when
            next_n > 1) or an already per-row [batch * next_n, P] table.
        row_indices: Optional varlen row -> request map; pairing only affects
            scheduling, so an arange is used when absent.
        candidate_blocks: [batch * next_n, K] int32 candidate block ids.
        topk_indices: [batch * next_n, topk_tokens] output buffer.
    """
    if topk_tokens not in _TOPK_KERNEL_SUPPORTED:
        return False
    sparse_block_kv = pick_sparse_block_kv(candidate_block_size)
    page_kv = kv_cache.shape[1]
    if kv_cache.stride(0) % 512 != 0 or page_kv % sparse_block_kv != 0:
        return False

    batch_size, next_n = q.shape[0], q.shape[1]
    num_rows = batch_size * next_n
    context_lens = seq_lens.reshape(-1)
    if context_lens.numel() != num_rows:
        return False
    context_lens = context_lens.contiguous()
    device = q.device

    row_ks = torch.zeros(num_rows, dtype=torch.int32, device=device)
    sparse_indices, end = candidate_blocks_to_sparse_indices(
        candidate_blocks,
        row_ks,
        context_lens,
        candidate_block_size,
        sparse_block_kv,
    )

    if block_table.shape[0] != num_rows:
        if row_indices is not None:
            # The varlen-flatten path always expands the block table per row
            # already; a request-level table with a row->request map means
            # the native layout, whose pairing we can't express here.
            return False
        repeat = num_rows // block_table.shape[0]
        if repeat * block_table.shape[0] != num_rows:
            return False
        block_table = block_table.repeat_interleave(repeat, dim=0)
    # See fp8_fp4_paged_mqa_logits: force stride(-1) == 1.
    if block_table.stride(-1) != 1:
        block_table = block_table.clone(memory_format=torch.contiguous_format)
    if row_indices is None:
        row_indices = torch.arange(num_rows, dtype=torch.int32, device=device)
    row_indices = row_indices.contiguous()

    metadata = get_paged_sparse_mqa_logits_metadata(
        context_lens,
        block_table,
        row_indices,
        page_kv,
        sparse_indices,
        q.dtype,
        sparse_block_kv,
    )
    logits = fp8_fp4_paged_sparse_mqa_logits(
        (q.view(num_rows, 1, *q.shape[2:]), q_scale.view(num_rows, 1, -1)),
        kv_cache,
        weights.to(torch.bfloat16),
        metadata,
        sparse_indices.shape[1],
        sparse_block_kv,
    )
    sparse_topk_remap(
        logits,
        sparse_indices,
        end,
        row_ks,
        sparse_block_kv,
        topk_tokens,
        topk_indices,
        decode=True,
    )
    return True
