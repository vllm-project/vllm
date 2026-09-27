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

from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.amd.ops.index_topk import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
from vllm.models.minimax_m3.amd.ops.indexer_context_parallel import (
    indexer_context_scores,
)
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerMetadata,
    MiniMaxM3IndexerTritonImpl,
)
from vllm.platforms import current_platform


def _round_up_16(n: int) -> int:
    return (n + 15) & ~15


class MiniMaxM3IndexerTritonCPImpl(MiniMaxM3IndexerTritonImpl):
    """Triton indexer with context-parallel decode scoring for ROCm.

    Decode: each rank scores its 1/world_size shard of global KV blocks,
    scatters into global-shape score tensor, MAX allreduce reconstructs all
    positions, ``minimax_m3_index_decode(precomputed_score=...)`` skips
    re-scoring and runs top-k directly.

    Prefill: unchanged, uses the same kernels as the base Triton impl.

    CUDAGraph: v2 FULL capture records the model with
    ``cudagraph_runtime_mode=NONE`` (see ``ModelCudaGraphManager.capture``), so
    a ``CUDAGraphMode.FULL`` guard never fires and a stream-capturing skip
    would bake the non-CP indexer into decode graphs. Persistent buffers keep
    this path allocation-free so CP is what gets captured.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        vllm_config = get_current_vllm_config()
        max_model_len = vllm_config.model_config.max_model_len
        # Cover cudagraph capture sizes as well as max_num_seqs.
        comp = vllm_config.compilation_config
        max_cg = getattr(comp, "max_cudagraph_capture_size", 0) or 0
        max_nd = max(vllm_config.scheduler_config.max_num_seqs, max_cg)
        max_blocks = triton.cdiv(max_model_len, SPARSE_BLOCK_SIZE)
        max_stride = _round_up_16(max_blocks)
        world_size = get_tensor_model_parallel_world_size()
        rank = get_tp_group().rank_in_group
        max_local = triton.cdiv(max_blocks, world_size)
        self._cp_world_size = world_size
        self._cp_rank = rank
        self.register_buffer(
            "_global_score_buf",
            torch.full(
                (self.num_index_heads, max_nd, max_stride),
                float("-inf"),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_local_score_buf",
            torch.empty(
                (self.num_index_heads, max_nd, max_local),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_owned_cols",
            torch.arange(rank, max_blocks, world_size, dtype=torch.int64),
            persistent=False,
        )

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
            buf if buf is None or current_platform.is_rocm() else buf.transpose(0, 1)
        )

        decode_topk: torch.Tensor | None = None
        prefill_topk: torch.Tensor | None = None

        if index_md.num_decodes > 0:
            d = index_md.decode
            assert d is not None
            world_size = self._cp_world_size
            rank = self._cp_rank
            max_block = triton.cdiv(d.max_seq_len, SPARSE_BLOCK_SIZE)

            # Dynamo tracing cannot capture the TP MAX allreduce; decode
            # graphs are recorded from eager warmup+capture, not from compile.
            # Do not skip on CUDAGraphMode.FULL or stream-capturing: v2 records
            # FULL graphs with runtime_mode=NONE, and skipping capture would
            # put the non-CP indexer into the replayed decode graph.
            if (
                torch.compiler.is_compiling()
                or max_block == 0
                or kv.ndim != 3
                or kv.numel() == 0
            ):
                decode_topk, prefill_topk = super().forward(
                    index_query,
                    attention_block_table=attention_block_table,
                    sparse_block_table_out=sparse_block_table_out,
                    sparse_context_lens_out=sparse_context_lens_out,
                    block_page_stride=block_page_stride,
                )
                return decode_topk, prefill_topk

            n_owned = (
                0 if rank >= max_block else (max_block - 1 - rank) // world_size + 1
            )
            # Reset the persistent global buffer (stable address) then scatter.
            self._global_score_buf.fill_(float("-inf"))
            local_scores = indexer_context_scores(
                iq[:nd],
                kv,
                d.block_table,
                d.seq_lens,
                d.max_seq_len,
                rank,
                world_size,
                d.decode_query_len,
                self.scale,
                out=self._local_score_buf,
            )
            if n_owned:
                n_tok = min(nd, local_scores.shape[1], self._global_score_buf.shape[1])
                n_col = min(n_owned, local_scores.shape[2], self._owned_cols.numel())
                owned = self._owned_cols[:n_col]
                self._global_score_buf[:, :n_tok, owned] = local_scores[:, :n_tok, :n_col]

            # MAX allreduce the full contiguous buffer, not a [:, :nd, :stride]
            # view: that view is non-contiguous when nd < max_nd, and NCCL
            # contig copies inside a CUDA graph HSA-fault on ROCm.
            dist.all_reduce(
                self._global_score_buf,
                op=dist.ReduceOp.MAX,
                group=get_tp_group().device_group,
            )
            stride = _round_up_16(max_block)
            fused_sparse_kwargs = {}
            if attention_block_table is not None:
                fused_sparse_kwargs = {
                    "attention_block_table": attention_block_table,
                    "sparse_block_table_out": sparse_block_table_out,
                    "sparse_context_lens_out": sparse_context_lens_out,
                    "block_page_stride": block_page_stride,
                }
            decode_backend_kwargs = {}
            if current_platform.is_rocm():
                decode_backend_kwargs["completion_counter"] = (
                    self.topk_completion_counter
                )
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
                precomputed_score=self._global_score_buf[:, :nd, :stride],
                **fused_sparse_kwargs,
                **decode_backend_kwargs,
            )

        if index_md.num_prefills > 0:
            p = index_md.prefill
            assert p is not None
            score = minimax_m3_index_score(
                iq[nd:],
                kv,
                p.block_table,
                p.cu_seqlens_q,
                p.seq_lens,
                p.context_lens,
                p.max_query_len,
                p.max_seq_len,
                self.num_kv_heads,
            )
            prefill_topk = minimax_m3_index_topk(
                score,
                p.cu_seqlens_q,
                p.context_lens,
                p.max_query_len,
                self.topk_blocks,
                self.init_blocks,
                self.local_blocks,
                out=buf_htk[:, nd:, :] if buf_htk is not None else None,
            )

        return decode_topk, prefill_topk
