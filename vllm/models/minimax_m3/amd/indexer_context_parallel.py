# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 context-parallel Triton indexer for ROCm.

M3's index heads shard like its KV heads, so at TP=4 each rank owns one of
the four index heads. Partitioning the *blocks* across ranks therefore needs
every rank to score every head:

1. all-gather the index queries (``heads * 128`` bf16 per token),
2. score all global heads on this rank's round-robin block shard (1/P of the
   index cache),
3. pack each head's shard top-k as ``(score, global block id)`` keys and
   all-gather them,
4. merge per global head, keeping only the heads this rank owns.

Each head's merge sees every shard's top-k *for that head*, so it is exact: a
global winner is in its own shard's top-k. Prefill is unchanged.

Enabled by ``--attention-config '{"minimax_m3_indexer_cp": true}'`` (ROCm,
TP>1, bf16 index cache).
"""

import math

import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import GroupCoordinator, get_tp_group
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
    all_gather,
    local_topk_keys,
    merge_topk_keys,
)
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerMetadata,
    MiniMaxM3IndexerTritonImpl,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton


def _packed(buf: torch.Tensor, dim: int, n: int) -> torch.Tensor:
    """Contiguous view of ``buf`` with ``shape[dim]`` cut to ``n``.

    Uses the buffer's leading storage rather than slicing, so collectives and
    kernels that need contiguous memory get it at a stable address.
    """
    shape = list(buf.shape)
    shape[dim] = n
    return buf.view(-1)[: math.prod(shape)].view(shape)


class IndexerCPDecode(nn.Module):
    """Decode top-k for this rank's index heads, scored context-parallel.

    Holds the persistent buffers so the path is allocation-free under CUDA
    graph capture. Every shape is sized from ``max_seq_len`` (pass
    ``max_model_len``): capture runs ``_dummy_run`` with a one-block context,
    and shapes derived from the batch would be baked into the graph.
    """

    def __init__(
        self,
        *,
        total_heads: int,
        rank: int,
        world_size: int,
        max_tokens: int,
        max_seq_len: int,
        topk: int,
        init_blocks: int,
        local_blocks: int,
        scale: float,
        group: GroupCoordinator,
        device: torch.device,
        head_dim: int = 128,
    ) -> None:
        super().__init__()
        if total_heads >= world_size:
            if total_heads % world_size:
                raise ValueError("index heads must divide evenly across TP ranks")
            local_heads, replicas = total_heads // world_size, 1
        else:
            if world_size % total_heads:
                raise ValueError("TP ranks must replicate index heads evenly")
            local_heads, replicas = 1, world_size // total_heads
        self.total_heads = total_heads
        self.local_heads = local_heads
        self.replicas = replicas
        # Same layout as the KV-head shard/replication in the fused QKV linear.
        self.head_offset = (rank // replicas) * local_heads
        self.rank = rank
        self.world_size = world_size
        self.max_seq_len = max_seq_len
        self.max_blocks = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
        self.topk = topk
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.scale = scale
        self.group = group

        max_local = triton.cdiv(self.max_blocks, world_size)

        def buf(name: str, *shape: int, dtype: torch.dtype) -> None:
            self.register_buffer(
                name,
                torch.empty(shape, dtype=dtype, device=device),
                persistent=False,
            )

        bf16, f32, i64 = torch.bfloat16, torch.float32, torch.int64
        buf("_q_local", max_tokens, local_heads, head_dim, dtype=bf16)
        buf("_q_gathered", world_size, max_tokens, local_heads, head_dim, dtype=bf16)
        buf("_q_full", max_tokens, total_heads, head_dim, dtype=bf16)
        buf("_scores", total_heads, max_tokens, max_local, dtype=f32)
        buf("_keys", total_heads, max_tokens, topk, dtype=i64)
        buf("_keys_gathered", world_size, total_heads, max_tokens, topk, dtype=i64)

    def _full_queries(self, local_q: torch.Tensor) -> torch.Tensor:
        """All-gather ``[n, local_heads, D]`` queries to ``[n, total_heads, D]``."""
        n = local_q.shape[0]
        q = _packed(self._q_local, 0, n)
        q.copy_(local_q)
        gathered = all_gather(q, _packed(self._q_gathered, 1, n), self.group)
        # Replicated ranks carry the same head; keep the first of each group.
        owners = gathered[:: self.replicas]
        full = _packed(self._q_full, 0, n)
        full.view(n, -1, self.local_heads, full.shape[-1]).copy_(
            owners.permute(1, 0, 2, 3)
        )
        return full

    def forward(
        self,
        local_q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        query_len: int,
        out: torch.Tensor,
        *,
        attention_block_table: torch.Tensor | None = None,
        sparse_block_table_out: torch.Tensor | None = None,
        sparse_context_lens_out: torch.Tensor | None = None,
        block_page_stride: int | None = None,
    ) -> torch.Tensor:
        """Write ``[local_heads, n, topk]`` block ids for this rank's heads."""
        n = local_q.shape[0]
        scores = indexer_context_scores(
            self._full_queries(local_q),
            kv_cache,
            block_table,
            seq_lens,
            self.max_seq_len,
            self.rank,
            self.world_size,
            query_len,
            self.scale,
            out=self._scores,
        )
        keys = _packed(self._keys, 1, n)
        local_topk_keys(
            scores,
            seq_lens,
            topk=self.topk,
            rank=self.rank,
            world=self.world_size,
            query_len=query_len,
            global_blocks=self.max_blocks,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            out=keys,
        )
        gathered = all_gather(keys, _packed(self._keys_gathered, 2, n), self.group)
        own = slice(self.head_offset, self.head_offset + self.local_heads)
        return merge_topk_keys(
            gathered[:, own],
            out,
            seq_lens,
            query_len=query_len,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            attention_block_table=attention_block_table,
            sparse_block_table_out=sparse_block_table_out,
            sparse_context_lens_out=sparse_context_lens_out,
            block_page_stride=block_page_stride,
        )


class MiniMaxM3IndexerTritonCPImpl(MiniMaxM3IndexerTritonImpl):
    """Triton indexer with context-parallel decode scoring for ROCm.

    Decode runs through :class:`IndexerCPDecode`. Prefill uses the base Triton
    kernels.

    CUDAGraph: v2 FULL capture records the model with
    ``cudagraph_runtime_mode=NONE`` (see ``ModelCudaGraphManager.capture``), so
    a ``CUDAGraphMode.FULL`` guard never fires and a stream-capturing skip
    would bake the non-CP indexer into decode graphs. Persistent buffers keep
    this path allocation-free so CP is what gets captured.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        vllm_config = get_current_vllm_config()
        hf_config = vllm_config.model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        total_heads = text_config.sparse_attention_config["sparse_num_index_heads"]
        comp = vllm_config.compilation_config
        max_cg = getattr(comp, "max_cudagraph_capture_size", 0) or 0
        spec = getattr(vllm_config, "speculative_config", None)
        n_spec = int(getattr(spec, "num_speculative_tokens", 0) or 0) if spec else 0
        max_tokens = max(vllm_config.scheduler_config.max_num_seqs, max_cg) * (
            n_spec + 1
        )
        device = (
            torch.device("cuda", torch.accelerator.current_device_index())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.cp_decode = IndexerCPDecode(
            total_heads=total_heads,
            rank=get_tp_group().rank_in_group,
            world_size=get_tensor_model_parallel_world_size(),
            max_tokens=max_tokens,
            max_seq_len=vllm_config.model_config.max_model_len,
            topk=self.topk_blocks,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            scale=self.scale,
            group=get_tp_group(),
            device=device,
            head_dim=self.index_head_dim,
        )
        assert self.cp_decode.local_heads == self.num_index_heads

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
                return super().forward(
                    index_query,
                    attention_block_table=attention_block_table,
                    sparse_block_table_out=sparse_block_table_out,
                    sparse_context_lens_out=sparse_context_lens_out,
                    block_page_stride=block_page_stride,
                )

            if buf_htk is None:
                decode_topk = torch.empty(
                    (self.num_index_heads, nd, self.topk_blocks),
                    dtype=torch.int32,
                    device=iq.device,
                )
            else:
                decode_topk = buf_htk[:, :nd]
            self.cp_decode(
                iq[:nd],
                kv,
                d.block_table,
                d.seq_lens,
                d.decode_query_len,
                decode_topk,
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
