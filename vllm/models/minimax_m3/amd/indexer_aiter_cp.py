# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 context-parallel AITER fp8 indexer for ROCm.

Extends MiniMaxM3IndexerAiterImpl with context parallelism for decode:
each TP rank scores its round-robin shard of KV blocks via the AITER
pa_sparse_block_score_decode kernel, then scores are allreduced (MAX)
across the TP group before pa_sparse_block_topk selects the global top-k.

This is the production path on MI350X with fp8 index cache.
PR A (#57702, Triton CP) covers bf16 index cache.

Activated when VLLM_ROCM_MINIMAX_INDEXER_CP=1 and TP>1 on ROCm,
dispatched via select_aiter_indexer_impl_cls in amd/indexer_aiter.py.
"""

import torch

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.amd.indexer_aiter import (
    MiniMaxM3IndexerAiterImpl,
    MiniMaxM3IndexerAiterMetadata,
)

logger = init_logger(__name__)


class MiniMaxM3IndexerAiterCPImpl(MiniMaxM3IndexerAiterImpl):
    """AITER fp8 MFMA indexer with context-parallel decode scoring for ROCm.

    Decode: each rank scores only its round-robin block shard via
    pa_sparse_block_score_decode. A MAX allreduce reconstructs the full
    score matrix, then pa_sparse_block_topk selects the global top-k.

    Round-robin assignment: rank R owns global blocks at positions
    [R, R+world, R+2*world, ...]. The shard block table is a column-slice
    of the full block table covering only those positions. The kernel writes
    only the owned block columns of the shared score buffer; non-owned
    columns stay at -inf and receive the correct value after allreduce.

    Prefill: unchanged — delegates to MiniMaxM3IndexerAiterImpl.forward.
    """

    def forward(
        self,
        index_query: torch.Tensor,
        *,
        decode_page16_block_table: torch.Tensor | None = None,
        prefill_page16_block_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        from aiter.ops.msa_attention import (
            pa_sparse_block_score_decode,
            pa_sparse_block_topk,
        )

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None

        md = attn_metadata[self.index_cache.prefix]
        assert isinstance(md, MiniMaxM3IndexerAiterMetadata)
        num_tokens = md.num_actual_tokens
        nd = md.num_decode_tokens
        iq = index_query[:num_tokens].view(
            -1, self.num_index_heads, self.index_head_dim
        )
        kv = self.index_cache.kv_cache

        buf = self.topk_indices_buffer
        if buf is None:
            buf = torch.empty(
                (self.num_index_heads, num_tokens, self.topk_blocks),
                dtype=torch.int32,
                device=iq.device,
            )

        decode_topk: torch.Tensor | None = None
        prefill_topk: torch.Tensor | None = None

        if md.num_decodes > 0:
            d = md.decode
            assert d is not None
            assert decode_page16_block_table is not None

            world_size = get_tensor_model_parallel_world_size()
            rank = get_tp_group().rank_in_group
            max_blocks = d.block_table.shape[1]

            # Allocate the full score buffer at -inf so non-owned block
            # columns are neutral for the MAX allreduce.
            score = self._new_score(nd, d.max_seq_len)
            score.fill_(float("-inf"))

            # Build the shard block table: rank R owns global block indices
            # [R, R+world, R+2*world, ...]. Slice those columns from the full
            # block table; the kernel writes scores only for those blocks.
            owned_cols = torch.arange(
                rank, max_blocks, world_size,
                device=d.block_table.device, dtype=torch.long,
            )
            if owned_cols.numel() > 0:
                shard_block_table = d.block_table[:, owned_cols].contiguous()
                # local_shard_seq_len: max sequence length covered by this
                # rank's shard. The last owned block ends at:
                #   owned_cols[-1] * block_size + block_size
                last_owned_block = owned_cols[-1].item()
                shard_max_seq_len = min(
                    (int(last_owned_block) + 1) * self.block_size,
                    d.max_seq_len,
                )
                pa_sparse_block_score_decode(
                    iq[:nd],
                    kv,
                    score,
                    shard_block_table,
                    d.seq_lens,
                    init_blocks=self.init_blocks,
                    local_blocks=self.local_blocks,
                    query_len=d.decode_query_len,
                    max_seq_len=shard_max_seq_len,
                )

            # MAX allreduce reconstructs the full score matrix.
            # Each owned column now has the correct score; non-owned stay -inf
            # until they receive the owning rank's value via MAX.
            get_tp_group().all_reduce(score)

            decode_topk = buf[:, :nd, :]
            sparse_bt, sparse_ctx = self._table_rows(0, nd)
            pa_sparse_block_topk(
                score,
                decode_topk,
                decode_page16_block_table,
                d.seq_lens,
                sparse_bt,
                sparse_ctx,
                max_seq_len=d.max_seq_len,
                block_size=self.block_size,
                query_len=d.decode_query_len,
                num_kv_heads=self.num_kv_heads,
                pages_per_block=self.pages_per_block,
            )

        if md.num_prefills > 0:
            _, prefill_topk = super().forward(
                index_query,
                decode_page16_block_table=None,
                prefill_page16_block_table=prefill_page16_block_table,
            )

        return decode_topk, prefill_topk
