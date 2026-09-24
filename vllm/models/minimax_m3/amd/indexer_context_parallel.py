# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 context-parallel Triton indexer for ROCm.

Each TP rank scores its own round-robin shard of KV blocks and writes the
results into a global-shape ``[heads, tokens, max_block]`` tensor pre-filled
with ``-inf``. A MAX allreduce across the TP group fills every global position
from its owning rank. The complete score tensor is forwarded to
``minimax_m3_index_decode`` via ``precomputed_score`` to skip re-scoring.

Enabled by ``VLLM_ROCM_MINIMAX_INDEXER_CP=1`` (ROCm, TP>1 only).
"""

import torch
import torch.distributed as dist
import triton

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.amd.ops.indexer_context_parallel import (
    indexer_context_scores,
)
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerMetadata,
    MiniMaxM3IndexerTritonImpl,
)
from vllm.models.minimax_m3.common.ops.index_topk import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_index_decode,
)

logger = init_logger(__name__)


def _round_up_16(n: int) -> int:
    return (n + 15) & ~15


class MiniMaxM3IndexerTritonCPImpl(MiniMaxM3IndexerTritonImpl):
    """Triton indexer with context-parallel decode scoring for ROCm.

    Decode: each rank scores its 1/world_size shard of global KV blocks,
    scatters into global-shape score tensor, MAX allreduce reconstructs all
    positions, ``minimax_m3_index_decode(precomputed_score=...)`` skips
    re-scoring and runs top-k directly.

    Prefill: unchanged, delegates to base Triton impl.
    """

    def forward(
        self,
        index_query: torch.Tensor,
        *,
        attention_block_table: torch.Tensor | None = None,
        sparse_block_table_out: torch.Tensor | None = None,
        sparse_context_lens_out: torch.Tensor | None = None,
        block_page_stride: int | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None

        index_md = attn_metadata[self.index_cache.prefix]
        assert isinstance(index_md, MiniMaxM3IndexerMetadata)
        num_tokens = index_md.num_actual_tokens
        nd = index_md.num_decode_tokens
        iq = index_query[:num_tokens].view(
            -1, self.num_index_heads, self.index_head_dim
        )
        kv = self.index_cache.kv_cache

        buf = self.topk_indices_buffer
        buf_htk = (
            buf
            if buf is None or not hasattr(buf, "transpose")
            else buf.transpose(0, 1)
        )

        decode_topk: torch.Tensor | None = None
        prefill_topk: torch.Tensor | None = None

        if index_md.num_decodes > 0:
            d = index_md.decode
            assert d is not None
            world_size = get_tensor_model_parallel_world_size()
            rank = get_tp_group().rank_in_group

            max_block = triton.cdiv(d.max_seq_len, SPARSE_BLOCK_SIZE)

            # During cudagraph capture, fall through to base impl
            # (dist.all_reduce not compatible with static graph recording).
            if torch.compiler.is_compiling() or max_block == 0 or rank >= max_block:
                decode_topk, prefill_topk = super().forward(
                    index_query,
                    attention_block_table=attention_block_table,
                    sparse_block_table_out=sparse_block_table_out,
                    sparse_context_lens_out=sparse_context_lens_out,
                    block_page_stride=block_page_stride,
                )
                return decode_topk, prefill_topk
            stride = _round_up_16(max_block)

            # Global score tensor pre-filled with -inf.
            # Shape matches minimax_m3_index_decode_score output.
            global_score = torch.full(
                (self.num_index_heads, nd, stride),
                float("-inf"),
                dtype=torch.float32,
                device=iq.device,
            )

            # Each rank scores its round-robin shard of global blocks.
            # indexer_context_scores returns [heads, tokens, local_blocks]
            # where local_blocks = ceil(max_block / world_size).
            local_scores = indexer_context_scores(
                iq[:nd],
                kv,
                d.block_table,
                d.seq_lens,
                d.max_seq_len,
                rank,
                world_size,
                d.max_decode_query_len,
                self.scale,
            )

            # Scatter local scores into global tensor at owned column indices:
            # global_block = local_block * world_size + rank
            owned_cols = torch.arange(
                rank, max_block, world_size, device=iq.device
            )
            local_blocks = local_scores.shape[2]
            n_owned = min(len(owned_cols), local_blocks)
            global_score[:, :, owned_cols[:n_owned]] = local_scores[:, :, :n_owned]


            # MAX allreduce: each rank's -inf placeholders are replaced by the
            # owning rank's real scores. Result is the full global score matrix.
            dist.all_reduce(global_score, op=dist.ReduceOp.MAX, group=get_tp_group().device_group)

            decode_topk = minimax_m3_index_decode(
                iq[:nd],
                kv,
                d.block_table,
                d.seq_lens,
                d.max_seq_len,
                self.topk_blocks,
                self.init_blocks,
                self.local_blocks,
                self.num_kv_heads,
                d.decode_query_len,
                d.max_decode_query_len,
                out=buf_htk,
                precomputed_score=global_score,
            )

        if index_md.num_prefills > 0:
            _, prefill_topk = super().forward(
                index_query,
                attention_block_table=attention_block_table,
                sparse_block_table_out=sparse_block_table_out,
                sparse_context_lens_out=sparse_context_lens_out,
                block_page_stride=block_page_stride,
            )

        return decode_topk, prefill_topk

