# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sparse indexer layer of DeepSeek-V4.1 with --monokernel on gfx942.

The layer runs vLLM's ROCm indexer op, rocm_aiter_sparse_attn_indexer, on
every step except a step of only decode rows. Such a step runs the decode path
of that op here, with the gfx942 launches of topk.py in front of vLLM's
logits, candidate selection, candidate mask and top-k. Each launch returns
False for a call that it cannot take, and then vLLM's op at that place runs.
The model takes this layer only with --monokernel on gfx942
(DeepseekV41ROCMAiterMLAAttention._indexer_cls in rocm.py).
"""

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.sparse_attn_indexer import (
    SparseAttnIndexer,
    _select_candidate_blocks,
)
from vllm.models.deepseek_v41.amd.ops import enabled, topk
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _apply_candidate_mask_strided,
    _indexer_k_is_c4a_block_flat,
    rocm_aiter_sparse_attn_indexer,
    rocm_fp8_paged_mqa_logits,
)


class Gfx942SparseAttnIndexer(SparseAttnIndexer):
    """vLLM's sparse indexer layer, whose HIP path runs
    gfx942_sparse_attn_indexer in place of vLLM's ROCm op."""

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        from vllm._aiter_ops import rocm_aiter_ops

        if (
            not rocm_aiter_ops.is_enabled()
            or self.use_fp4_cache
            or not isinstance(q_quant, torch.Tensor)
        ):
            # vLLM's layer checks these cases itself. On gfx942 it needs
            # AITER and an FP8 Q, and it refuses the call otherwise.
            return super().forward_hip(hidden_states, q_quant, k, weights)
        return gfx942_sparse_attn_indexer(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache,
            q_quant,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            skip_k_cache_insert=self.skip_k_cache_insert,
            compress_ratio=self.compress_ratio,
            candidate_blocks=self.candidate_blocks,
            candidate_block_size=self.candidate_block_size,
            candidate_write=self.candidate_write,
        )


@eager_break_during_capture
def gfx942_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor | None,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool = False,
    compress_ratio: int = 1,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    candidate_write: bool = False,
) -> torch.Tensor:
    """rocm_aiter_sparse_attn_indexer with the same arguments, whose steps of
    only decode rows run _decode_step. The profiling run, prefill and mixed
    steps, a layer that writes its own K cache and a process without
    --monokernel on gfx942 run vLLM's op unchanged.

    Like vLLM's op, this is a break of a breakable CUDA graph capture. Its
    Python runs again at every replay, so the choice between the two paths
    is made for each step and not once at capture."""
    attn_metadata = get_forward_context().attn_metadata
    meta = (
        attn_metadata.get(k_cache_prefix) if isinstance(attn_metadata, dict) else None
    )
    if (
        not isinstance(meta, DeepseekV32IndexerMetadata)
        or meta.num_prefills > 0
        or meta.num_decodes == 0
        or not skip_k_cache_insert
        or not enabled()
    ):
        return rocm_aiter_sparse_attn_indexer(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_fp8,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert=skip_k_cache_insert,
            compress_ratio=compress_ratio,
            candidate_blocks=candidate_blocks,
            candidate_block_size=candidate_block_size,
            candidate_write=candidate_write,
        )
    return _decode_step(
        meta,
        hidden_states.shape[0],
        kv_cache,
        q_fp8,
        weights,
        topk_tokens,
        max_model_len,
        topk_indices_buffer,
        compress_ratio,
        candidate_blocks,
        candidate_block_size,
        candidate_write,
    )


def _decode_step(
    meta: DeepseekV32IndexerMetadata,
    num_tokens: int,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    max_model_len: int,
    topk_indices_buffer: torch.Tensor,
    compress_ratio: int,
    candidate_blocks: torch.Tensor | None,
    candidate_block_size: int,
    candidate_write: bool,
) -> torch.Tensor:
    """The decode path of rocm_aiter_sparse_attn_indexer, for a step of only
    decode rows whose K cache the model already wrote, with the gfx942
    launches of topk.py in front of vLLM's ops."""
    from vllm._aiter_ops import rocm_aiter_ops

    decode = meta.decode
    assert decode is not None
    num_decode_tokens = meta.num_decode_tokens
    # A step whose decode rows are not padded skips vLLM's -1 fill, because
    # the decode top-k below then writes every entry of every row
    # (topk.skip_decode_fill). This saves one launch on each index layer.
    if not topk.skip_decode_fill(False, num_decode_tokens, num_tokens, decode):
        topk_indices_buffer[:num_tokens] = -1

    # The logits kernels read the cache as [blocks, block size, heads, dim]
    # with one head.
    kv_cache = kv_cache.unsqueeze(-2)
    decode_lens = decode.decode_lens
    if decode.requires_padding:
        # vLLM counts a prefill chunk shorter than 1 + drafts as a decode, and
        # its rows are then padded to the step's row count per request.
        q = pack_seq_triton(q_fp8[:num_decode_tokens], decode_lens)
    else:
        q = q_fp8[:num_decode_tokens].reshape(
            decode_lens.shape[0], -1, *q_fp8.shape[1:]
        )
    batch_size, next_n = q.shape[:2]
    assert batch_size == decode.seq_lens.shape[0]
    num_padded_tokens = batch_size * next_n
    topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

    # Layers 24 to 36 compute only the logits of their candidate blocks,
    # straight into compact rows, and take their top 512 from those rows. The
    # logits are bit-identical to the dense call's below, and the top-512
    # sets are the same (tests/kernels/models/amd/test_dsv41_gfx942_cand_logits.py).
    # At 128k context the dense call scores 8 times more positions than the
    # top-k reads. The launch only takes calls whose dense logits would come
    # from AITER's gfx942 Gluon kernel in rocm_fp8_paged_mqa_logits, because
    # it copies that kernel's arithmetic.
    if (
        candidate_blocks is not None
        and not candidate_write
        and topk.candidate_logits_top_k(
            q,
            kv_cache,
            weights,
            decode.seq_lens,
            decode.block_table,
            candidate_blocks,
            candidate_block_size,
            topk_indices,
            topk_tokens,
            kv_cache.shape[1] > 1
            and not _indexer_k_is_c4a_block_flat(compress_ratio)
            and rocm_aiter_ops.is_enabled(),
            decode.requires_padding,
        )
    ):
        return topk_indices_buffer

    logits = rocm_fp8_paged_mqa_logits(
        q,
        kv_cache,
        weights[:num_padded_tokens],
        decode.seq_lens,
        decode.block_table,
        decode.schedule_metadata,
        max_model_len=max_model_len,
        compress_ratio=compress_ratio,
    )
    num_rows = logits.shape[0]

    candidates_done = False
    if candidate_blocks is not None:
        decode_candidates = candidate_blocks[:num_rows]
        if candidate_write:
            # Layer 20 writes the candidate blocks of layers 24 to 36 in one
            # launch. They are the blocks that vLLM's selection writes, in
            # another order (tests/kernels/models/amd/test_dsv41_gfx942_topk.py).
            if not topk.select_candidates(
                logits,
                next_n,
                decode.seq_lens,
                candidate_block_size,
                decode_candidates,
            ):
                row_starts, row_ends = _row_bounds(decode.seq_lens, num_rows, next_n)
                _select_candidate_blocks(
                    logits,
                    row_starts,
                    row_ends,
                    decode_candidates.shape[1],
                    candidate_block_size,
                    decode_candidates,
                )
        elif topk.candidate_top_k(
            logits,
            next_n,
            decode.seq_lens,
            decode_candidates,
            candidate_block_size,
            topk_indices,
            topk_tokens,
        ):
            # The launch wrote each row's top 512 among its candidate blocks.
            # Those are the columns that vLLM's mask and top-k choose
            # (tests/kernels/models/amd/test_dsv41_gfx942_topk.py), so both
            # are skipped. The mask writes -inf to about 7 of every 8 logits
            # of a 128k row, and the top-k then reads the whole row.
            candidates_done = True
        else:
            row_starts, row_ends = _row_bounds(decode.seq_lens, num_rows, next_n)
            _apply_candidate_mask_strided(
                logits,
                row_starts,
                row_ends,
                decode_candidates,
                candidate_block_size,
            )

    if not candidates_done:
        _decode_top_k(
            logits,
            meta,
            next_n,
            topk_indices,
            topk_tokens,
            compress_ratio,
            max_model_len,
        )

    if decode.requires_padding:
        # Drop the padded rows again, as vLLM's op does.
        unpacked = unpack_seq_triton(
            topk_indices.reshape(batch_size, next_n, topk_indices.shape[-1]),
            decode_lens,
        )
        topk_indices_buffer[:num_decode_tokens, : unpacked.shape[-1]] = unpacked
    return topk_indices_buffer


def _row_bounds(
    seq_lens: torch.Tensor, num_rows: int, next_n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The zero row starts and the int64 row ends that vLLM's candidate
    kernels take. Only the branches that run those kernels make them, so a
    layer whose call a gfx942 launch takes saves these two launches."""
    visible = seq_lens.reshape(-1)
    if visible.numel() != num_rows:
        visible = visible.repeat_interleave(next_n)
    visible = visible[:num_rows].to(torch.int64)
    return torch.zeros_like(visible), visible


def _decode_top_k(
    logits: torch.Tensor,
    meta: DeepseekV32IndexerMetadata,
    next_n: int,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    compress_ratio: int,
    max_model_len: int,
) -> None:
    """The decode top-k of vLLM's op: AITER's indexer top-k where AITER takes
    the call, and otherwise the gfx942 top-512 of topk.decode_top_k or, when
    that returns False, vLLM's top_k_per_row_decode."""
    from vllm._aiter_ops import rocm_aiter_ops

    decode = meta.decode
    assert decode is not None
    num_rows = logits.shape[0]
    # FULL graphs are not keyed by context length, so the choice of the
    # captured kernel takes a bound that holds at every replay.
    if get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL:
        max_compressed_seq_len = max_model_len
    else:
        max_compressed_seq_len = meta.max_seq_len // compress_ratio
    if rocm_aiter_ops.is_indexer_top_k_supported(
        is_prefill=False,
        compress_ratio=compress_ratio,
        num_rows=num_rows,
        max_valid_seq_len=max_compressed_seq_len,
    ) and not rocm_aiter_ops.dsv4_indexer_prefers_native_top_k(
        num_rows=num_rows,
        num_columns=logits.shape[1],
        topk_tokens=topk_tokens,
    ):
        rocm_aiter_ops.indexer_top_k_decode(
            logits,
            1,
            decode.seq_lens.reshape(-1),
            topk_indices,
            topk_tokens,
        )
    elif not topk.decode_top_k(
        logits, next_n, decode.seq_lens, topk_indices, topk_tokens
    ):
        # On gfx942 vLLM's kernel takes its generic path, which adds every
        # -inf logit to one shared-memory histogram bin with an atomic. After
        # the DSpark candidate mask most of a row is -inf, and the call takes
        # 100 us at 128k context. So topk.decode_top_k runs the gfx942
        # top-512 instead where it can. It gives the same index sets, except
        # for ties at the cut (tests/kernels/models/amd/test_dsv41_gfx942_topk.py).
        torch.ops._C.top_k_per_row_decode(
            logits,
            next_n,
            decode.seq_lens,
            topk_indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
        )
