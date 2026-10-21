# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sparse-attention indexers in a specialized decode step.

``decode_rows`` runs a dense indexer (``SparseAttnIndexer``) on a copy of the
varlen decode branch of ``sparse_attn_indexer``, with two additions:

- Row sharding. The indexer is replicated across the TP ranks. In a uniform
  decode step of at least ``ROW_SHARD_MIN_ROWS`` rows, each rank scores,
  selects candidates for and top-k's only its contiguous slice of the rows
  (an even number of them, so the last ranks' slices may be short or empty);
  TP all-gathers of the int32 rows (see ``_all_gather_rows``) rebuild
  ``topk_indices_buffer`` and, in the candidate source, the candidate blocks.
  A slice may split a request: varlen rows are independent.
- Selection from the BF16 logits. For up to 64 rows per rank, ``decode_topk``
  (``decode_topk_cute``) selects the token top-k, and in the candidate source
  the candidate blocks, from SM100's BF16 logits without upcasting them. The
  candidate source's logits are twice as wide (compress ratio 1), so at a 1M
  context they need 16 CTAs a row and take that path only up to two waves of
  such clusters (14 rows on a GB300); otherwise the candidate source selects
  with ``cooperative_topk`` (see ``candidate_topk``) on the upcast logits.

Steps it does not take run the stock indexer.

``decode_rows`` runs in two parts. ``plan_decode_rows`` takes the gates,
resets the top-k rows and builds the slice's DeepGEMM schedule: none of it
reads the layer's index query or keys, so the attention step issues it while
the current stream would otherwise wait for them. ``finish_decode_rows``
scores, selects and gathers once they are ready.

``CandidateOverlap`` moves the candidate consumers' candidate expansion and
sparse MQA-logits schedule onto a side stream.

The row-sharded path issues TP collectives, so its gates read only host state
that every TP rank shares: the step's row and request counts.
"""

from dataclasses import dataclass

import torch
from torch import nn

from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    apply_candidate_mask,
    select_candidate_blocks,
)
from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
    candidate_blocks_to_sparse_indices,
)
from vllm.model_executor.layers.indexer_topk import get_indexer_topk
from vllm.model_executor.layers.sparse_attn_indexer import (
    SparseAttnIndexer,
    as_fp32_logits,
    kv_cache_as_quant_view,
)
from vllm.model_executor.layers.sparse_mqa_indexer import SparseMQAIndexer
from vllm.models.deepseek_v41.attention import DeepseekV4Indexer
from vllm.models.deepseek_v41.common.ops import fused_indexer_q_rope_quant
from vllm.triton_utils import triton
from vllm.utils.deep_gemm import (
    fp8_fp4_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
    get_paged_sparse_mqa_logits_metadata,
    mqa_logits_mx_inputs,
    mqa_logits_use_mx_scales,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.backends.mla.sparse_indexer import (
    DeepseekV41SparseIndexerMetadata,
    SparseMQARowsMetadata,
)

from .candidate_topk import (
    select_candidate_blocks_cooperative,
    supports_cooperative_candidates,
)
from .decode_topk_cute import decode_topk, supports_decode_topk

# Fewest rows of a row-sharded decode step (two requests of six rows): a
# single request's slices save less than the all-gathers cost.
ROW_SHARD_MIN_ROWS = 12


def row_shard_rows(num_rows: int, num_requests: int, world_size: int) -> int | None:
    """Rows each rank's slice spans in a pure-decode step, or None when the
    step stays replicated. Rank ``r`` scores rows ``[r n, (r + 1) n)``
    clipped to the step's rows.

    Slices start on an even row: DeepGEMM's paged MQA logits needs its int32
    ``indices`` 8-byte aligned. Varlen rows are independent, so a slice may
    split a request. Steps whose rows are not a multiple of their requests
    (some capture and profiling runs) stay replicated.
    """
    if world_size < 2 or not num_requests or num_rows % num_requests:
        return None
    if num_rows < ROW_SHARD_MIN_ROWS:
        return None
    rows = -(-num_rows // world_size)
    return rows + rows % 2


def row_shard_bounds(
    num_rows: int, num_requests: int, world_size: int, rank: int
) -> tuple[int, int] | None:
    """Rows ``[start, end)`` that ``rank`` scores in a pure-decode step (an
    empty range for a rank past the last row), or None when the step stays
    replicated."""
    rows = row_shard_rows(num_rows, num_requests, world_size)
    if rows is None:
        return None
    start = min(rank * rows, num_rows)
    return start, min(start + rows, num_rows)


def rope_quant_q(
    indexer: DeepseekV4Indexer,
    q: torch.Tensor,
    indexer_weights: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """The second half of ``indexer.forward_q``: RoPE and quantize the
    projected queries ``q`` and scale the per-head ``indexer_weights``."""
    q_quant, weights = fused_indexer_q_rope_quant(
        positions,
        q.view(-1, indexer.n_head, indexer.head_dim),
        rotary_emb.cos_sin_cache,
        indexer_weights,
        indexer.softmax_scale,
        indexer.n_head**-0.5,
        use_fp4=indexer.use_fp4_kv,
        weights_out_dtype=indexer.indexer_weights_dtype,
    )
    if mqa_logits_use_mx_scales():
        # The SM100 logits kernels take BF16 weights. Cast them here, on the
        # query's side stream, rather than in the scoring path
        # (``mqa_logits_mx_inputs``); a no-op for BF16 weights.
        weights = weights.to(torch.bfloat16)
    if isinstance(q_quant, tuple):
        return q_quant[0], q_quant[1], weights
    return q_quant, None, weights


def _all_gather_rows(buffer: torch.Tensor, rows: int, num_rows: int) -> None:
    """Gather every rank's ``rows``-row slice of ``buffer`` into its first
    ``num_rows`` rows; rows past ``num_rows`` (a short or empty last slice)
    are dropped."""
    group = get_tp_group()
    local = buffer[group.rank_in_group * rows : (group.rank_in_group + 1) * rows]
    # NCCL, not the custom all-gather: its MNNVL Lamport kernel never waits for
    # a rank's own multicast copy, so a late copy can be read as a peer's slot
    # by a later gather of another size, and a step's gathers differ in size
    # (the top-k rows, then the candidate blocks).
    gathered = group.all_gather(local, 0)
    buffer[:num_rows].copy_(gathered[:num_rows])


@dataclass
class DecodeRowsPlan:
    """The rows a rank scores in ``finish_decode_rows`` and their schedule."""

    decode: DeepSeekV32IndexerDecodeMetadata
    max_seq_len: int
    num_rows: int
    # Rows of each rank's slice, or None when the step stays replicated.
    shard: int | None
    start: int
    end: int
    fast_candidates: bool
    # DeepGEMM's paged MQA-logits schedule of rows ``[start, end)``.
    schedule: torch.Tensor


def decode_rows(
    op: SparseAttnIndexer,
    hidden_states: torch.Tensor,
    q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
) -> torch.Tensor:
    """``op(hidden_states, q_quant, None, weights)`` for a pure-decode step,
    row-sharded and with cooperative candidates where they apply."""
    plan = plan_decode_rows(op, hidden_states.shape[0])
    return finish_decode_rows(op, hidden_states, q_quant, weights, plan)


def plan_decode_rows(op: SparseAttnIndexer, num_tokens: int) -> DecodeRowsPlan | None:
    """The first part of ``decode_rows`` for a step of ``num_tokens`` tokens:
    reset the top-k rows and build the slice's schedule. None when the stock
    indexer runs instead (and nothing was issued)."""
    attn_metadata = get_forward_context().attn_metadata
    metadata = (
        attn_metadata.get(op.k_cache.prefix)
        if isinstance(attn_metadata, dict)
        else None
    )
    if (
        not isinstance(metadata, DeepseekV32IndexerMetadata)
        or metadata.num_prefills > 0
        or metadata.decode is None
        # Varlen paged MQA logits without DCP.
        or metadata.decode.indices is None
        or metadata.decode.global_seq_lens is not None
        or not op.skip_k_cache_insert
    ):
        return None
    num_rows = metadata.num_decode_tokens
    tp_group = get_tp_group()
    shard = row_shard_rows(num_rows, metadata.num_decodes, tp_group.world_size)
    if shard is not None and any(
        buffer is not None and buffer.shape[0] < tp_group.world_size * shard
        for buffer in (op.topk_indices_buffer, op.candidate_blocks)
    ):
        # The rows past the step's that the slices span do not fit.
        shard = None
    bounds = (
        None
        if shard is None
        else row_shard_bounds(
            num_rows, metadata.num_decodes, tp_group.world_size, tp_group.rank_in_group
        )
    )
    start, end = bounds or (0, num_rows)
    fast_candidates = (
        op.candidate_write
        and op.candidate_blocks is not None
        and supports_cooperative_candidates(
            end - start,
            triton.cdiv(op.max_model_len, op.candidate_block_size),
            op.candidate_blocks.shape[1],
        )
    )
    if bounds is None and not fast_candidates:
        return None

    decode = metadata.decode
    assert decode.indices is not None
    op.topk_indices_buffer[:num_tokens] = -1
    schedule = decode.schedule_metadata
    if end > start and end - start < decode.seq_lens.shape[0]:
        # The slice's own schedule; a FULL graph records it.
        kv_cache = kv_cache_as_quant_view(
            op.k_cache.kv_cache, op.head_dim, op.use_fp4_cache
        )
        schedule = get_paged_mqa_logits_metadata(
            decode.seq_lens[start:end],
            kv_cache.shape[1],
            schedule.shape[0] - 1,
            indices=decode.indices[start:end],
        )
    return DecodeRowsPlan(
        decode,
        metadata.max_seq_len,
        num_rows,
        shard,
        start,
        end,
        fast_candidates,
        schedule,
    )


def finish_decode_rows(
    op: SparseAttnIndexer,
    hidden_states: torch.Tensor,
    q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    plan: DecodeRowsPlan | None,
) -> torch.Tensor:
    """The rest of ``decode_rows`` once the index query and keys are ready:
    score and select the planned rows and gather the slices."""
    if plan is None:
        return op(hidden_states, q_quant, None, weights)
    q_values, q_scale = q_quant if isinstance(q_quant, tuple) else (q_quant, None)
    if plan.end > plan.start:
        _decode_rows(op, q_values, q_scale, weights, plan)
    if plan.shard is not None:
        _all_gather_rows(
            op.topk_indices_buffer[:, : op.topk_tokens], plan.shard, plan.num_rows
        )
        if op.candidate_write:
            assert op.candidate_blocks is not None
            _all_gather_rows(op.candidate_blocks, plan.shard, plan.num_rows)
    return op.topk_indices_buffer


def _decode_rows(
    op: SparseAttnIndexer,
    q: torch.Tensor,
    q_scale: torch.Tensor | None,
    weights: torch.Tensor,
    plan: DecodeRowsPlan,
) -> None:
    """The varlen decode branch of `sparse_attn_indexer` on the planned rows
    ``[start, end)``: paged MQA logits, candidate selection or mask, and
    top-k into the same rows of the shared buffers."""
    decode, start, end = plan.decode, plan.start, plan.end
    assert decode.indices is not None
    kv_cache = kv_cache_as_quant_view(
        op.k_cache.kv_cache, op.head_dim, op.use_fp4_cache
    )
    seq_lens = decode.seq_lens[start:end]
    indices = decode.indices[start:end]
    q = q[start:end].unsqueeze(1)
    if op.use_fp4_cache:
        q = q.view(torch.int8)
    if q_scale is not None:
        q_scale = q_scale[start:end].unsqueeze(1)
    q_scale, decode_weights = mqa_logits_mx_inputs(q, q_scale, weights[start:end])
    row_ends = seq_lens.view(-1)
    raw = fp8_fp4_paged_mqa_logits(
        (q, q_scale),
        kv_cache,
        decode_weights,
        seq_lens,
        decode.block_table[start:end],
        plan.schedule,
        max_model_len=op.max_model_len,
        indices=indices,
    )
    topk_rows = op.topk_indices_buffer[start:end, : op.topk_tokens]
    if _selects_from_bf16(op, raw, plan.fast_candidates):
        # Straight from the BF16 logits: no upcast.
        if op.candidate_blocks is not None:
            decode_topk(raw, row_ends, op.candidate_blocks[start:end], block_mode=True)
        decode_topk(raw, row_ends, topk_rows, block_mode=False)
        return
    logits = as_fp32_logits(raw, row_ends)
    if op.candidate_blocks is not None:
        candidates = op.candidate_blocks[start:end]
        if plan.fast_candidates:
            select_candidate_blocks_cooperative(
                logits, row_ends, op.candidate_block_size, candidates
            )
        elif op.candidate_write:
            select_candidate_blocks(
                logits,
                None,
                row_ends,
                candidates.shape[1],
                op.candidate_block_size,
                candidates,
            )
        else:
            apply_candidate_mask(
                logits, None, row_ends, candidates, op.candidate_block_size
            )
    get_indexer_topk(op.topk_backend)(
        logits,
        seq_lens,
        1,
        topk_rows,
        op.topk_tokens,
        plan.max_seq_len,
    )


def _selects_from_bf16(
    op: SparseAttnIndexer, logits: torch.Tensor, fast_candidates: bool
) -> bool:
    """Whether ``decode_topk`` selects the token top-k, and the candidate
    blocks if the layer writes them, from these logits."""
    rows = logits.shape[0]
    if not (
        op.topk_backend == "auto"
        and supports_decode_topk(logits, rows, op.topk_tokens, block_mode=False)
    ):
        return False
    if op.candidate_blocks is None:
        return True
    return (
        fast_candidates
        and op.candidate_block_size == 8
        and supports_decode_topk(
            logits, rows, op.candidate_blocks.shape[1], block_mode=True
        )
    )


def decode_schedule(
    op: SparseMQAIndexer,
    rows: SparseMQARowsMetadata,
    sparse_block_kv: int,
    scratch: bool = False,
) -> torch.Tensor:
    """Expand the decode rows' candidates into ``rows.sparse_indices`` /
    ``rows.end`` and return their DeepGEMM schedule; the same kernels and
    inputs as the inline step of `sparse_mqa_logits_paged_decode`. With
    ``scratch``, expand zero candidates into private outputs instead."""
    assert rows.block_table is not None and rows.row_indices is not None
    candidates = op.candidate_blocks[: rows.row_ke.shape[0]]
    sparse_indices, end = rows.sparse_indices, rows.end
    if scratch:
        candidates = torch.zeros_like(candidates)
        sparse_indices, end = torch.empty_like(sparse_indices), torch.empty_like(end)
    candidate_blocks_to_sparse_indices(
        candidates,
        rows.row_ks,
        rows.row_ke,
        op.candidate_block_size,
        sparse_block_kv,
        out=(sparse_indices, end),
    )
    return get_paged_sparse_mqa_logits_metadata(
        rows.row_ke,
        rows.block_table,
        rows.row_indices,
        op.k_cache.kv_cache.shape[1],
        sparse_indices,
        torch.int8,
        sparse_block_kv,
    )


class CandidateOverlap:
    """The candidate consumers' expansion and schedule on a side stream.

    The first candidate consumer of a decode step expands the candidate blocks
    into DeepGEMM sparse block ids and builds the paged sparse MQA-logits
    schedule, which later consumers reuse. Neither reads Q, the KV cache or the
    hidden states, so the candidate source forks them onto ``stream`` once it
    has written the candidates (``fork``) and every consumer waits for them
    (``join``) before scoring. The kernels and inputs are those of the inline
    path, so the outputs match.

    The side stream issues no collectives. The fork waits on the main stream,
    so a step's writes to the builder's ``sparse_indices`` / ``end`` cannot
    overtake the previous step's reads, and the schedule allocated on
    ``stream`` is recorded as used by the main stream.

    DeepGEMM's schedule kernel keeps a per-stream workspace that it creates on
    first use and refuses to create during a capture, so ``warm`` runs the
    schedule on ``stream`` once from an eager decode step, on zero candidates
    and private outputs; the executor forks only after that.
    """

    def __init__(self, consumer: SparseMQAIndexer) -> None:
        self.consumer = consumer
        self.stream = torch.cuda.Stream()
        self.done = torch.cuda.Event()
        self.warmed = False
        self._forked = False

    def _decode_rows(self) -> tuple[SparseMQARowsMetadata, int] | None:
        attn_metadata = get_forward_context().attn_metadata
        metadata = (
            attn_metadata.get(self.consumer.k_cache.prefix)
            if isinstance(attn_metadata, dict)
            else None
        )
        if not isinstance(metadata, DeepseekV41SparseIndexerMetadata):
            return None
        rows = metadata.sparse_decode
        if rows is None or not rows.row_ke.numel():
            return None
        return rows, metadata.sparse_block_kv

    def warm(self) -> None:
        """Create the side stream's schedule workspace from an eager step."""
        found = None if self.warmed else self._decode_rows()
        if found is None:
            return
        main = torch.cuda.current_stream()
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            decode_schedule(self.consumer, *found, scratch=True)
        main.wait_stream(self.stream)
        self.warmed = True

    def fork(self) -> None:
        """Start the expansion; the candidate source calls this right after
        its indexer has written the candidates."""
        found = self._decode_rows()
        if found is None or found[0].kernel_metadata is not None:
            return
        rows, sparse_block_kv = found
        main = torch.cuda.current_stream()
        self.stream.wait_stream(main)
        try:
            with torch.cuda.stream(self.stream):
                schedule = decode_schedule(self.consumer, rows, sparse_block_kv)
                self.done.record()
        except BaseException:
            # Rejoin so that a capture reports this error, not unjoined work.
            main.wait_stream(self.stream)
            raise
        schedule.record_stream(main)
        rows.kernel_metadata = schedule
        self._forked = True

    def join(self) -> None:
        """Make the current stream wait for a pending fork, if any."""
        if self._forked:
            torch.cuda.current_stream().wait_event(self.done)
            self._forked = False
