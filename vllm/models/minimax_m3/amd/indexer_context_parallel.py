# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 context-parallel Triton indexer for ROCm.

Each TP rank scores its own round-robin shard of KV blocks, then scores are
allreduced (max) across ranks to reconstruct the full score matrix. The
standard top-k selector runs on the allreduced result.

Activated when VLLM_ROCM_MINIMAX_INDEXER_CP=1 and TP>1 on ROCm, via
select_indexer_impl_cls in common/indexer.py.

PR 2 replaces the O(blocks) allreduce with O(topk) candidate exchange.
"""

import torch

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
from vllm.models.minimax_m3.common.ops.index_topk import minimax_m3_index_decode

logger = init_logger(__name__)


class MiniMaxM3IndexerTritonCPImpl(MiniMaxM3IndexerTritonImpl):
    """Triton indexer with context-parallel decode scoring for ROCm.

    Decode: each rank scores 1/world_size block shard, allreduce (max)
    reconstructs full scores, standard top-k runs on result.
    Prefill: unchanged (delegates to base Triton impl).
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
        buf_htk = buf if buf is None else buf.transpose(0, 1)

        decode_topk: torch.Tensor | None = None
        prefill_topk: torch.Tensor | None = None

        if index_md.num_decodes > 0:
            d = index_md.decode
            assert d is not None
            world_size = get_tensor_model_parallel_world_size()
            rank = get_tp_group().rank_in_group

            local_scores = indexer_context_scores(
                iq[:nd], kv, d.block_table, d.seq_lens,
                d.max_seq_len, rank, world_size,
                d.max_decode_query_len, self.scale,
            )

            import torch.distributed as dist
            dist.all_reduce(
                local_scores,
                op=dist.ReduceOp.MAX,
                group=get_tp_group().device_group,
            )

            fused_sparse_kwargs: dict = {}
            if attention_block_table is not None:
                fused_sparse_kwargs = {
                    "attention_block_table": attention_block_table,
                    "sparse_block_table_out": sparse_block_table_out,
                    "sparse_context_lens_out": sparse_context_lens_out,
                    "block_page_stride": block_page_stride,
                }
            decode_topk = minimax_m3_index_decode(
                iq[:nd], kv, d.block_table, d.seq_lens, d.max_seq_len,
                self.topk_blocks, self.init_blocks, self.local_blocks,
                self.num_kv_heads, d.decode_query_len, d.max_decode_query_len,
                out=buf_htk, precomputed_score=local_scores,
                **fused_sparse_kwargs,
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
