# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sparse-attention indexers in a specialized decode step.

``decode_rows`` runs a dense indexer (``SparseAttnIndexer``) on a copy of the
varlen decode branch of ``sparse_attn_indexer``, with two additions:

- Row sharding. The indexer is replicated across the TP ranks. In a step of at
  least ``ROW_SHARD_MIN_REQUESTS`` requests whose count the ranks divide, each
  rank scores, selects candidates for and top-k's only its contiguous slice of
  the rows; TP all-gathers of the int32 rows rebuild ``topk_indices_buffer``
  and, in the candidate source, the candidate blocks.
- The candidate source selects its candidate blocks with ``cooperative_topk``
  for up to 64 rows per rank (see ``candidate_topk``).

Steps it does not take run the stock indexer.

``CandidateOverlap`` moves the candidate consumers' candidate expansion and
sparse MQA-logits schedule onto a side stream.

The row-sharded path issues TP collectives, so its gates read only host state
that every TP rank shares: the step's row and request counts.
"""

import torch
from torch import nn

from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    apply_candidate_mask,
    select_candidate_blocks,
    select_candidate_blocks_cooperative,
    supports_cooperative_candidates,
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
from vllm.models.common.ops.sequence_parallel import sp_all_gather
from vllm.models.deepseek_v41.attention import DeepseekV4Indexer
from vllm.models.deepseek_v41.common.ops import fused_indexer_q_rope_quant
from vllm.triton_utils import triton
from vllm.utils.deep_gemm import (
    fp8_fp4_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
    get_paged_sparse_mqa_logits_metadata,
    mqa_logits_mx_inputs,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.backends.mla.sparse_indexer import (
    DeepseekV41SparseIndexerMetadata,
    SparseMQARowsMetadata,
)

# Fewest requests of a row-sharded decode step.
ROW_SHARD_MIN_REQUESTS = 8
# Most rows of a step whose candidate expansion overlaps.
OVERLAP_MAX_ROWS = 16


def row_shard_bounds(
    num_rows: int, num_requests: int, world_size: int, rank: int
) -> tuple[int, int] | None:
    """Rows ``[start, end)`` that ``rank`` scores in a pure-decode step, or
    None when the step stays replicated.

    Each slice holds whole requests of a uniform step; varlen rows are
    independent, so the split is exact either way. Slices start on an even
    row: DeepGEMM's paged MQA logits needs its int32 ``indices`` 8-byte
    aligned. DSpark decode steps always split so; capture and profiling runs
    of one row a request may not, and stay replicated.
    """
    rows = num_rows // world_size
    if (
        world_size < 2
        or num_requests < ROW_SHARD_MIN_REQUESTS
        or num_requests % world_size
        or num_rows % num_requests
        or rows % 2
    ):
        return None
    return rank * rows, (rank + 1) * rows


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
    if isinstance(q_quant, tuple):
        return q_quant[0], q_quant[1], weights
    return q_quant, None, weights


def _all_gather_rows(local: torch.Tensor, out: torch.Tensor) -> None:
    # The custom all-gather takes floating-point rows and moves the int32
    # bits unchanged: its Lamport variant rewrites only the INT32_MIN pattern.
    out.copy_(sp_all_gather(local.view(torch.float32)).view(torch.int32))


def decode_rows(
    op: SparseAttnIndexer,
    hidden_states: torch.Tensor,
    q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
) -> torch.Tensor:
    """``op(hidden_states, q_quant, None, weights)`` for a pure-decode step,
    row-sharded and with cooperative candidates where they apply."""
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
        return op(hidden_states, q_quant, None, weights)
    num_rows = metadata.num_decode_tokens
    tp_group = get_tp_group()
    bounds = row_shard_bounds(
        num_rows, metadata.num_decodes, tp_group.world_size, tp_group.rank_in_group
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
        return op(hidden_states, q_quant, None, weights)

    q_values, q_scale = q_quant if isinstance(q_quant, tuple) else (q_quant, None)
    topk_indices_buffer = op.topk_indices_buffer
    topk_indices_buffer[: hidden_states.shape[0]] = -1
    _decode_rows(
        op,
        q_values,
        q_scale,
        weights,
        metadata.decode,
        metadata.max_seq_len,
        start,
        end,
        fast_candidates,
    )
    if bounds is not None:
        _all_gather_rows(
            topk_indices_buffer[start:end, : op.topk_tokens],
            topk_indices_buffer[:num_rows, : op.topk_tokens],
        )
        if op.candidate_write:
            assert op.candidate_blocks is not None
            _all_gather_rows(
                op.candidate_blocks[start:end], op.candidate_blocks[:num_rows]
            )
    return topk_indices_buffer


def _decode_rows(
    op: SparseAttnIndexer,
    q: torch.Tensor,
    q_scale: torch.Tensor | None,
    weights: torch.Tensor,
    decode: DeepSeekV32IndexerDecodeMetadata,
    max_seq_len: int,
    start: int,
    end: int,
    fast_candidates: bool,
) -> None:
    """The varlen decode branch of `sparse_attn_indexer` on rows
    ``[start, end)``: paged MQA logits, candidate selection or mask, and
    top-k into the same rows of the shared buffers."""
    assert decode.indices is not None
    kv_cache = kv_cache_as_quant_view(
        op.k_cache.kv_cache, op.head_dim, op.use_fp4_cache
    )
    seq_lens = decode.seq_lens[start:end]
    indices = decode.indices[start:end]
    schedule = decode.schedule_metadata
    if end - start < decode.seq_lens.shape[0]:
        # The slice's own schedule; a FULL graph records it.
        schedule = get_paged_mqa_logits_metadata(
            seq_lens, kv_cache.shape[1], schedule.shape[0] - 1, indices=indices
        )
    q = q[start:end].unsqueeze(1)
    if op.use_fp4_cache:
        q = q.view(torch.int8)
    if q_scale is not None:
        q_scale = q_scale[start:end].unsqueeze(1)
    q_scale, decode_weights = mqa_logits_mx_inputs(q, q_scale, weights[start:end])
    row_ends = seq_lens.view(-1)
    logits = as_fp32_logits(
        fp8_fp4_paged_mqa_logits(
            (q, q_scale),
            kv_cache,
            decode_weights,
            seq_lens,
            decode.block_table[start:end],
            schedule,
            max_model_len=op.max_model_len,
            indices=indices,
        ),
        row_ends,
    )
    if op.candidate_blocks is not None:
        candidates = op.candidate_blocks[start:end]
        if fast_candidates:
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
        op.topk_indices_buffer[start:end, : op.topk_tokens],
        op.topk_tokens,
        max_seq_len,
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
