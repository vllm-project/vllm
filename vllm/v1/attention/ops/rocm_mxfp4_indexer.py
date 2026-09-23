# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 sparse indexer on aiter's paged MXFP4 MQA-logits kernel.

gfx950 only. The kernel reads the preshuffled paged indexer K cache in place,
so no layer gathers K into a contiguous buffer, prefill included. The
candidate source takes its block maxima from the same walk that writes its
logits, and the candidate consumers can walk the candidate pool instead of
the whole context: the pool is resolved once per step and shared by all four.
"""

import functools
import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    apply_candidate_mask,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _apply_candidate_mask_strided,
    _get_aiter_top_k_kernel,
    _launch_aiter_top_k_per_row_decode,
    _max_decode_logits_rows,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerPrefillChunkMetadata,
    )
    from vllm.v1.attention.backends.mla.rocm_mxfp4_indexer import (
        DeepseekV41RocmMxfp4IndexerMetadata,
        RocmMxfp4PrefillPlan,
    )

MXFP4_BLOCK_SIZE = 32
_AITER_MODULE = "aiter.ops.triton.attention.pa_mqa_logits_mxfp4"


@functools.cache
def _aiter():
    return importlib.import_module(_AITER_MODULE)


@functools.cache
def rocm_mxfp4_indexer_unsupported_reason() -> str | None:
    """Why this platform cannot run the ROCm MXFP4 indexer, or None."""
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return "the ROCm MXFP4 indexer kernels are gfx950 only"
    try:
        _aiter()
    except ImportError as e:
        return f"aiter's paged MXFP4 MQA-logits kernel is unavailable ({e})"
    return None


def rocm_mxfp4_n_per_tile(num_heads: int, head_dim: int) -> int:
    """The MFMA N the kernel tiles keys by, which the cache order is built on."""
    return _aiter().mfma_nonk_dim(num_heads, head_dim)


def check_rocm_mxfp4_cache_geometry(
    num_heads: int, head_dim: int, page_entries: int
) -> None:
    _aiter().cache_format(num_heads, head_dim, page_entries)


def _kv_view(kv_cache: torch.Tensor, head_dim: int) -> torch.Tensor:
    """The indexer cache as [pages, entries, 1, bytes]. The page stride is the
    block-major pool's, not the page's own size."""
    num_pages, entries, width = kv_cache.shape
    assert kv_cache.dtype == torch.uint8
    assert width == head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE, width
    return torch.as_strided(
        kv_cache, (num_pages, entries, 1, width), (kv_cache.stride(0), width, width, 1)
    )


@functools.cache
def _check_page_stride(page_stride: int, entries: int, width: int, head_dim: int):
    from aiter.ops.triton.attention.pa_mqa_logits_mxfp4_gather import cache_strides

    probe = torch.empty_strided(
        (2, entries, 1, width),
        (page_stride, width, width, 1),
        dtype=torch.uint8,
        device="meta",
    )
    used = cache_strides(probe, head_dim)[1]
    if used != page_stride:
        raise RuntimeError(
            f"aiter's paged MXFP4 MQA-logits kernel would step {used} B between "
            f"pages, but the indexer cache pages are {page_stride} B apart: they "
            "sit inside vLLM's block-major KV pool. aiter has to take the page "
            "stride from kv_cache.stride(0)."
        )


@functools.cache
def _gather_reaches(span: int, block: int, head_dim: int) -> bool:
    # build_candidate_gather resolves each candidate block to an int32 offset
    # from the cache base, in 16 B units for values and gather_s_unit B for
    # scales. The base is the pool's first block, so the span is the pool's.
    from aiter.ops.triton.attention.pa_mqa_logits_mxfp4_gather import gather_s_unit

    unit = min(16, gather_s_unit(block, head_dim // MXFP4_BLOCK_SIZE))
    return span <= unit * (2**31 - 1)


def reserve_rocm_mxfp4_indexer_workspace(
    hidden_states: torch.Tensor, logits_width: int, candidate_block_size: int = 0
) -> None:
    """Profiling run: claim the decode logits workspace and the peak prefill
    logits, block scores included when the layer writes them."""
    rows = _max_decode_logits_rows(hidden_states.shape[0])
    specs = [((rows, logits_width), torch.float32)]
    budget = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
    if candidate_block_size:
        nblocks = triton.cdiv(logits_width, candidate_block_size)
        specs.append(((rows, nblocks), torch.float32))
        budget += budget // candidate_block_size
    current_workspace_manager().get_simultaneous(*specs)
    torch.empty(budget, dtype=torch.uint8, device=hidden_states.device)


def _topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
    k: int,
    compress_ratio: int = 0,
    max_len: int = 0,
) -> None:
    """Top-k over each row's [0, lengths[row]); -1 pads rows shorter than k."""
    rows = logits.shape[0]
    if rows == 0:
        return
    kernel = None
    if logits.shape[1] >= k:
        kernel = _get_aiter_top_k_kernel(
            is_prefill=False,
            compress_ratio=compress_ratio,
            num_rows=rows,
            max_valid_seq_len=max_len,
            num_columns=logits.shape[1],
            topk_tokens=k,
        )
    if kernel is not None:
        _launch_aiter_top_k_per_row_decode(kernel, logits, lengths, out, k)
    else:
        torch.ops._C.top_k_per_row_decode(
            logits, 1, lengths, out, rows, logits.stride(0), logits.stride(1), k
        )


@triton.jit
def _remap_compact_topk_kernel(
    idx_ptr,
    idx_stride,
    pos_ptr,
    pos_stride,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    PADDED_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, PADDED_K)
    live = cols < K
    slot = tl.load(idx_ptr + row * idx_stride + cols, mask=live, other=-1)
    hit = slot >= 0
    slot = tl.where(hit, slot, 0)
    start = tl.load(
        pos_ptr + row * pos_stride + slot // BLOCK, mask=live & hit, other=0
    )
    tl.store(
        idx_ptr + row * idx_stride + cols,
        tl.where(hit, start + slot % BLOCK, -1).to(tl.int32),
        mask=live,
    )


def _remap_compact_topk(
    indices: torch.Tensor, positions: torch.Tensor, block: int
) -> None:
    """Candidate slots to context positions, in place: slot j sits in pool
    block j // block, which starts at positions[j // block]."""
    rows, k = indices.shape
    if rows == 0:
        return
    _remap_compact_topk_kernel[(rows,)](
        indices,
        indices.stride(0),
        positions,
        positions.stride(0),
        K=k,
        BLOCK=block,
        PADDED_K=triton.next_power_of_2(k),
        num_warps=4,
    )


@dataclass
class _Layer:
    """One indexer layer's inputs and outputs for this step."""

    metadata: "DeepseekV41RocmMxfp4IndexerMetadata"
    kv: torch.Tensor
    q: torch.Tensor
    q_scale: torch.Tensor
    weights: torch.Tensor
    topk_buffer: torch.Tensor
    topk_tokens: int
    compress_ratio: int
    candidates: torch.Tensor | None
    block: int
    full_graph: bool

    @property
    def num_heads(self) -> int:
        return self.q.shape[1]

    @property
    def head_dim(self) -> int:
        return self.q.shape[2] * 2

    def rows(self, lo: int, hi: int, seqs: int = 1) -> tuple[torch.Tensor, ...]:
        """Q, its scales and the weights of token rows [lo, hi), as ``seqs``
        sequences of next_n rows each."""
        return (
            self.q[lo:hi].view(seqs, -1, *self.q.shape[1:]),
            self.q_scale[lo:hi].view(seqs, -1, *self.q_scale.shape[1:]),
            self.weights[lo:hi],
        )

    def decode_rows(self, n: int) -> tuple[torch.Tensor, ...]:
        """The first n token rows, one sequence each."""
        return (
            self.q[:n].unsqueeze(1),
            self.q_scale[:n].unsqueeze(1),
            self.weights[:n],
        )


def _layer(
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_values: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    topk_indices_buffer: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    compress_ratio: int,
    candidate_blocks: torch.Tensor | None,
    candidate_block_size: int,
) -> _Layer:
    from vllm.v1.attention.backends.mla.rocm_mxfp4_indexer import (
        DeepseekV41RocmMxfp4IndexerMetadata,
    )

    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    assert isinstance(attn_metadata, dict)
    metadata = attn_metadata[k_cache_prefix]
    assert isinstance(metadata, DeepseekV41RocmMxfp4IndexerMetadata), (
        "the ROCm MXFP4 indexer needs DeepseekV41RocmMxfp4IndexerBackend metadata"
    )
    kv = _kv_view(kv_cache, head_dim)
    _check_page_stride(kv.stride(0), kv.shape[1], kv.shape[3], head_dim)
    num_heads = q_values.shape[1]
    return _Layer(
        metadata=metadata,
        kv=kv,
        q=q_values,
        # The fused Q kernel returns one int32 per head; the kernel reads the
        # four ue8m0 bytes behind it.
        q_scale=q_scale.view(torch.uint8).view(q_scale.shape[0], num_heads, -1),
        weights=weights,
        topk_buffer=topk_indices_buffer,
        topk_tokens=topk_tokens,
        compress_ratio=compress_ratio,
        candidates=candidate_blocks,
        block=candidate_block_size,
        full_graph=forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL,
    )


def _block_scores(layer: _Layer, scores: torch.Tensor | None) -> dict:
    if scores is None:
        return {}
    return dict(
        calc_block_scores=True, block_scores=scores, candidate_block_size=layer.block
    )


def _dense_prefill(
    layer: _Layer,
    chunk: "DeepseekV32IndexerPrefillChunkMetadata",
    plan: "RocmMxfp4PrefillPlan",
    candidate_write: bool,
) -> None:
    pa = _aiter()
    t0, t1 = chunk.token_start, chunk.token_end
    logits = layer.q.new_empty((t1 - t0, plan.width), dtype=torch.float32)
    scores = None
    if candidate_write:
        nblocks = triton.cdiv(plan.width, layer.block)
        scores = layer.q.new_empty((t1 - t0, nblocks), dtype=torch.float32)
    # A request's rows go in as one sequence's next_n, which is what lets a
    # workgroup share a KV tile; requests with equal rows share a launch.
    for lo, hi, req, seqs in plan.launches:
        q, q_scale, weights = layer.rows(t0 + lo, t0 + hi, seqs)
        pa.paged_mxfp4_mqa_logits(
            q,
            q_scale,
            layer.kv,
            weights,
            plan.context_lens[req : req + seqs],
            chunk.block_table[req : req + seqs],
            plan.width,
            out_logits=logits[lo:hi],
            clean_logits=False,
            cu_ends=plan.row_ends[lo:hi],
            **_block_scores(layer, None if scores is None else scores[lo:hi]),
        )
    candidates = None if layer.candidates is None else layer.candidates[t0:t1]
    if scores is not None:
        assert candidates is not None and plan.block_ends is not None
        _topk(scores, plan.block_ends, candidates, candidates.shape[1])
    elif candidates is not None:
        apply_candidate_mask(logits, None, plan.row_ends, candidates, layer.block)
    _topk(
        logits,
        plan.row_ends,
        layer.topk_buffer[t0:t1, : layer.topk_tokens],
        layer.topk_tokens,
        layer.compress_ratio,
        plan.width,
    )


def _dense_decode(layer: _Layer, logits_width: int, candidate_write: bool) -> None:
    metadata = layer.metadata
    lengths = metadata.decode_row_lens
    assert metadata.decode is not None and lengths is not None
    block_table = metadata.decode.block_table
    rows = lengths.shape[0]
    assert block_table.shape[0] == rows, "one block-table row per query row"
    specs = [((rows, logits_width), torch.float32)]
    if candidate_write:
        specs.append(((rows, triton.cdiv(logits_width, layer.block)), torch.float32))
    logits, *scores = current_workspace_manager().get_simultaneous(*specs)
    q, q_scale, weights = layer.decode_rows(rows)
    _aiter().paged_mxfp4_mqa_logits(
        q,
        q_scale,
        layer.kv,
        weights,
        lengths,
        block_table,
        logits_width,
        out_logits=logits,
        clean_logits=False,
        **_block_scores(layer, scores[0] if scores else None),
    )
    candidates = None if layer.candidates is None else layer.candidates[:rows]
    if scores:
        assert candidates is not None and metadata.decode_block_ends is not None
        _topk(scores[0], metadata.decode_block_ends, candidates, candidates.shape[1])
    elif candidates is not None:
        _apply_candidate_mask_strided(logits, None, lengths, candidates, layer.block)
    # FULL graphs replay every context length, so the top-k is picked for the
    # longest one.
    max_len = (
        logits_width
        if layer.full_graph
        else metadata.max_seq_len // layer.compress_ratio
    )
    _topk(
        logits,
        lengths,
        layer.topk_buffer[:rows, : layer.topk_tokens],
        layer.topk_tokens,
        layer.compress_ratio,
        max_len,
    )


def _gather_prefill(
    layer: _Layer,
    chunk: "DeepseekV32IndexerPrefillChunkMetadata",
    plan: "RocmMxfp4PrefillPlan",
    num_cols: int,
) -> None:
    pa = _aiter()
    assert layer.candidates is not None
    t0, t1 = chunk.token_start, chunk.token_end
    compact = layer.q.new_empty((t1 - t0, num_cols), dtype=torch.float32)
    for i, (lo, hi, req) in enumerate(plan.requests):
        if plan.gathers[i] is None:
            # The first consumer resolves the pool; the other three reuse it.
            plan.gathers[i] = pa.build_candidate_gather(
                layer.candidates[t0 + lo : t0 + hi],
                plan.row_ends[lo:hi],
                chunk.block_table[req : req + 1].expand(hi - lo, -1),
                layer.kv,
                layer.num_heads,
                layer.head_dim,
                layer.block,
            )
        gather, slot_ends = plan.gathers[i]
        q, q_scale, weights = layer.rows(t0 + lo, t0 + hi)
        pa.paged_mxfp4_mqa_logits(
            q,
            q_scale,
            layer.kv,
            weights,
            plan.context_lens[req : req + 1],
            chunk.block_table[req : req + 1],
            num_cols,
            out_logits=compact[lo:hi],
            clean_logits=False,
            cu_ends=slot_ends,
            use_gather=True,
            candidates=gather,
        )
        out = layer.topk_buffer[t0 + lo : t0 + hi, : layer.topk_tokens]
        _topk(compact[lo:hi], slot_ends, out, layer.topk_tokens)
        _remap_compact_topk(out, gather["positions"], layer.block)


def _gather_decode(layer: _Layer, num_cols: int) -> None:
    pa = _aiter()
    assert layer.candidates is not None
    metadata = layer.metadata
    lengths = metadata.decode_row_lens
    assert metadata.decode is not None and lengths is not None
    block_table = metadata.decode.block_table
    rows = lengths.shape[0]
    if metadata.decode_gather is None:
        metadata.decode_gather = pa.build_candidate_gather(
            layer.candidates[:rows],
            lengths,
            block_table,
            layer.kv,
            layer.num_heads,
            layer.head_dim,
            layer.block,
        )
    gather, slot_ends = metadata.decode_gather
    (compact,) = current_workspace_manager().get_simultaneous(
        ((rows, num_cols), torch.float32)
    )
    q, q_scale, weights = layer.decode_rows(rows)
    pa.paged_mxfp4_mqa_logits(
        q,
        q_scale,
        layer.kv,
        weights,
        lengths,
        block_table,
        num_cols,
        out_logits=compact,
        clean_logits=False,
        cu_ends=slot_ends,
        use_gather=True,
        candidates=gather,
    )
    out = layer.topk_buffer[:rows, : layer.topk_tokens]
    _topk(compact, slot_ends, out, layer.topk_tokens)
    _remap_compact_topk(out, gather["positions"], layer.block)


@eager_break_during_capture
def rocm_mxfp4_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_values: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    topk_indices_buffer: torch.Tensor,
    compress_ratio: int = 1,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    candidate_write: bool = False,
) -> torch.Tensor:
    """Dense indexer: every layer that scores the whole context, the
    candidate source included. With ``candidate_blocks`` and not
    ``candidate_write`` the scores are masked to the pool first, as the
    shared path does."""
    if not isinstance(get_forward_context().attn_metadata, dict):
        reserve_rocm_mxfp4_indexer_workspace(
            hidden_states,
            max_model_len,
            candidate_block_size if candidate_write else 0,
        )
        return topk_indices_buffer
    layer = _layer(
        k_cache_prefix,
        kv_cache,
        q_values,
        q_scale,
        weights,
        topk_indices_buffer,
        topk_tokens,
        head_dim,
        compress_ratio,
        candidate_blocks,
        candidate_block_size,
    )
    metadata = layer.metadata
    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if metadata.prefill is not None:
        for chunk, plan in zip(metadata.prefill.chunks, metadata.prefill_plans):
            _dense_prefill(layer, chunk, plan, candidate_write)
    if metadata.decode is not None:
        _dense_decode(layer, max_model_len, candidate_write)
    return topk_indices_buffer


@eager_break_during_capture
def rocm_mxfp4_sparse_mqa_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_values: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    topk_indices_buffer: torch.Tensor,
    compress_ratio: int,
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
    num_candidate_cols: int,
) -> torch.Tensor:
    """Candidate consumer: score only the source's pool where the builder's
    length gate says it pays, else the dense walk masked to the pool."""
    if not isinstance(get_forward_context().attn_metadata, dict):
        reserve_rocm_mxfp4_indexer_workspace(
            hidden_states, max(max_model_len, num_candidate_cols)
        )
        return topk_indices_buffer
    layer = _layer(
        k_cache_prefix,
        kv_cache,
        q_values,
        q_scale,
        weights,
        topk_indices_buffer,
        topk_tokens,
        head_dim,
        compress_ratio,
        candidate_blocks,
        candidate_block_size,
    )
    metadata = layer.metadata
    kv = layer.kv
    span = (kv.shape[0] - 1) * kv.stride(0) + kv.shape[1] * kv.shape[3]
    reachable = _gather_reaches(span, candidate_block_size, head_dim)
    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if metadata.prefill is not None:
        for chunk, plan in zip(metadata.prefill.chunks, metadata.prefill_plans):
            if reachable and plan.use_gather:
                _gather_prefill(layer, chunk, plan, num_candidate_cols)
            else:
                _dense_prefill(layer, chunk, plan, candidate_write=False)
    if metadata.decode is not None:
        # A FULL graph cannot follow the per-step gate; the gather is the side
        # that stays flat in the context length.
        if reachable and (metadata.decode_use_gather or layer.full_graph):
            _gather_decode(layer, num_candidate_cols)
        else:
            _dense_decode(layer, max_model_len, candidate_write=False)
    return topk_indices_buffer
