# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 context-parallel AITER indexer for ROCm.

Each TP rank scores its own round-robin shard of KV blocks via
``minimax_m3_index_decode_score`` with a shard block table, scatters the
results into a global-shape score tensor pre-filled with ``-inf``, and a MAX
allreduce across the TP group fills the remaining positions. The complete score
tensor is forwarded to ``pa_sparse_block_topk`` directly; the separate
``pa_sparse_block_score_decode`` call is skipped.

Enabled by ``VLLM_ROCM_MINIMAX_INDEXER_CP=1`` when the AITER indexer is
selected (fp8 index cache, gfx950, TP>1).
"""

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.amd.indexer_aiter import (
    MiniMaxM3IndexerAiterImpl,
    select_aiter_indexer_impl_cls,
)
from vllm.models.minimax_m3.common.indexer import MiniMaxM3IndexerMetadata

logger = init_logger(__name__)


class MiniMaxM3IndexerAiterCPImpl(MiniMaxM3IndexerAiterImpl):
    """AITER indexer with context-parallel decode scoring for ROCm.

    Decode: each rank scores its 1/world_size shard via the fp8 MFMA kernel,
    scatters into a global score tensor, MAX allreduce reconstructs all
    positions, ``pa_sparse_block_topk`` runs on the complete scores.

    Prefill: unchanged, delegates to base AITER impl.
    """

    def forward(
        self,
        index_query: torch.Tensor,
        *,
        decode_page16_block_table: torch.Tensor | None = None,
        prefill_page16_block_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        from vllm._aiter_ops import rocm_aiter_ops

        pa_sparse_block_score_decode = rocm_aiter_ops.pa_sparse_block_score_decode
        pa_sparse_block_topk = rocm_aiter_ops.pa_sparse_block_topk

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None

        md = attn_metadata[self.index_cache.prefix]
        assert isinstance(md, MiniMaxM3IndexerMetadata)
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

            # During cudagraph capture or when this rank owns no blocks
            # (short sequences where rank >= max_blocks), fall through to
            # the base AITER impl which handles these cases correctly.
            if (torch.compiler.is_compiling()
                    or max_blocks == 0
                    or rank >= max_blocks):
                return super().forward(
                    index_query,
                    decode_page16_block_table=decode_page16_block_table,
                    prefill_page16_block_table=prefill_page16_block_table,
                )

            # Round-robin shard: rank r owns global blocks r, r+W, r+2W, ...
            owned_cols = torch.arange(
                rank, max_blocks, world_size,
                dtype=d.block_table.dtype,
                device=d.block_table.device,
            )
            shard_bt = d.block_table[:, owned_cols]

            # Allocate global score tensor pre-filled with -inf.
            score = self._new_score(nd, d.max_seq_len)
            score.fill_(float("-inf"))

            # Score owned blocks. shard_score uses d.max_seq_len so its
            # block-axis width (score_block_width) matches the global score,
            # making owned_cols column indices directly valid for both tensors.
            shard_score = self._new_score(nd, d.max_seq_len)
            pa_sparse_block_score_decode(
                iq[:nd],
                kv,
                shard_score,
                shard_bt,
                d.seq_lens,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                query_len=d.decode_query_len,
                max_seq_len=d.max_seq_len,
            )
            # Scatter per-block scores into global tensor at owned column positions.
            n_owned = min(len(owned_cols), shard_score.shape[-1])
            score[..., owned_cols[:n_owned]] = shard_score[..., :n_owned]

            # MAX allreduce reconstructs full global scores.
            dist.all_reduce(
                score,
                op=dist.ReduceOp.MAX,
                group=get_tp_group().device_group,
            )

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


def select_aiter_cp_indexer_impl_cls(
    **kwargs,
) -> type[MiniMaxM3IndexerAiterCPImpl] | None:
    """Return the CP AITER impl if the env var and base AITER conditions hold."""
    if not (
        get_tensor_model_parallel_world_size() > 1
        and envs.VLLM_ROCM_MINIMAX_INDEXER_CP
    ):
        return None
    if select_aiter_indexer_impl_cls(**kwargs) is None:
        return None
    logger.info_once(
        "MiniMax M3 indexer: selected AITER CP (context-parallel, ROCm) "
        "[topk_blocks=%d, tp=%d]",
        kwargs.get("topk_blocks", "?"),
        get_tensor_model_parallel_world_size(),
    )
    return MiniMaxM3IndexerAiterCPImpl