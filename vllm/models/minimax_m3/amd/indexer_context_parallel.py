# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 context-parallel Triton indexer for ROCm.

Each TP rank scores its own round-robin shard of KV blocks (1/P of the index
cache), packs that shard's full top-k as ``(score, global block id)`` keys,
all-gathers ``P * k`` keys per token, and merges. Prefill is unchanged.

This is the same candidate-exchange as PR #57909, not a MAX allreduce of the
full score tensor: the payload is ``P * k * 8`` bytes per token and does not
grow with context length.

Enabled by ``VLLM_ROCM_MINIMAX_INDEXER_CP=1`` (ROCm, TP>1 only).
"""

import torch
import torch.distributed as dist

from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.amd.ops.index_topk import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
from vllm.models.minimax_m3.amd.ops.indexer_context_parallel import (
    indexer_context_scores,
)
from vllm.models.minimax_m3.amd.ops.indexer_cp_exchange import (
    aiter_all_gather_keys,
    local_topk_keys,
    merge_topk_keys,
)
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerMetadata,
    MiniMaxM3IndexerTritonImpl,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton


class MiniMaxM3IndexerTritonCPImpl(MiniMaxM3IndexerTritonImpl):
    """Triton indexer with context-parallel decode scoring for ROCm.

    Decode: score 1/world_size of the blocks, local top-k of the full k,
    all-gather packed keys, merge. Prefill uses the base Triton kernels.

    CUDAGraph: v2 FULL capture records the model with
    ``cudagraph_runtime_mode=NONE`` (see ``ModelCudaGraphManager.capture``), so
    a ``CUDAGraphMode.FULL`` guard never fires and a stream-capturing skip
    would bake the non-CP indexer into decode graphs. Persistent buffers keep
    this path allocation-free so CP is what gets captured.

    Every kernel shape is therefore taken from ``max_model_len`` rather than
    the batch's ``max_seq_len``: capture runs ``_dummy_run`` with
    ``seq_lens = max_query_len``, so shapes derived from the batch would bake
    a one-block context into the graph and replay it at full length.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        vllm_config = get_current_vllm_config()
        max_model_len = vllm_config.model_config.max_model_len
        device = (
            torch.device("cuda", torch.accelerator.current_device_index())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        comp = vllm_config.compilation_config
        max_cg = getattr(comp, "max_cudagraph_capture_size", 0) or 0
        spec = getattr(vllm_config, "speculative_config", None)
        n_spec = int(getattr(spec, "num_speculative_tokens", 0) or 0) if spec else 0
        max_nd = max(vllm_config.scheduler_config.max_num_seqs, max_cg) * (n_spec + 1)
        max_blocks = triton.cdiv(max_model_len, SPARSE_BLOCK_SIZE)
        world_size = get_tensor_model_parallel_world_size()
        rank = get_tp_group().rank_in_group
        max_local = triton.cdiv(max_blocks, world_size)
        self._cp_world_size = world_size
        self._cp_rank = rank
        self._cp_max_seq_len = max_model_len
        self._cp_max_blocks = max_blocks
        self.register_buffer(
            "_local_score_buf",
            torch.empty(
                (self.num_index_heads, max_nd, max_local),
                dtype=torch.float32,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_local_keys_buf",
            torch.empty(
                (self.num_index_heads, max_nd, self.topk_blocks),
                dtype=torch.int64,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_gathered_keys_buf",
            torch.empty(
                (world_size, self.num_index_heads, max_nd, self.topk_blocks),
                dtype=torch.int64,
                device=device,
            ),
            persistent=False,
        )

    def _packed_local_keys(self, nd: int) -> torch.Tensor:
        """Contiguous ``[heads, nd, topk]`` prefix of the persistent key buffer."""
        heads, _, topk = self._local_keys_buf.shape
        return self._local_keys_buf.view(-1)[: heads * nd * topk].view(heads, nd, topk)

    def _packed_gathered_keys(self, nd: int) -> torch.Tensor:
        """Contiguous ``[world, heads, nd, topk]`` prefix of the gather buffer."""
        world, heads, _, topk = self._gathered_keys_buf.shape
        n = heads * nd * topk
        return self._gathered_keys_buf.view(-1)[: world * n].view(
            world, heads, nd, topk
        )

    def _exchange_keys(self, nd: int) -> torch.Tensor:
        """All-gather this rank's packed keys. Payload is independent of ctx."""
        keys = self._packed_local_keys(nd)
        gathered = aiter_all_gather_keys(keys)
        if gathered is not None:
            return gathered
        dest = self._packed_gathered_keys(nd)
        dist.all_gather_into_tensor(
            dest.reshape(-1, nd, keys.shape[-1]),
            keys,
            group=get_tp_group().device_group,
        )
        return dest

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
            max_block = triton.cdiv(d.max_seq_len, SPARSE_BLOCK_SIZE)

            # Dynamo tracing cannot capture the TP collective; decode graphs
            # are recorded from eager warmup+capture, not from compile.
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

            local_scores = indexer_context_scores(
                iq[:nd],
                kv,
                d.block_table,
                d.seq_lens,
                self._cp_max_seq_len,
                self._cp_rank,
                self._cp_world_size,
                d.decode_query_len,
                self.scale,
                out=self._local_score_buf,
            )
            local_topk_keys(
                local_scores,
                d.seq_lens,
                topk=self.topk_blocks,
                rank=self._cp_rank,
                world=self._cp_world_size,
                query_len=d.decode_query_len,
                global_blocks=self._cp_max_blocks,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                out=self._packed_local_keys(nd),
            )
            if buf_htk is None:
                decode_topk = torch.empty(
                    (self.num_index_heads, nd, self.topk_blocks),
                    dtype=torch.int32,
                    device=iq.device,
                )
            else:
                decode_topk = buf_htk[:, :nd]
            merge_topk_keys(
                self._exchange_keys(nd),
                decode_topk,
                d.seq_lens,
                query_len=d.decode_query_len,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                attention_block_table=attention_block_table,
                sparse_block_table_out=sparse_block_table_out,
                sparse_context_lens_out=sparse_context_lens_out,
                block_page_stride=block_page_stride,
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
