# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention backend for MiniMax M3 sparse ("lightning indexer") attention.

MiniMax M3 sparse layers run GQA attention restricted to a small set of KV
blocks chosen by a lightning indexer: index heads score KV blocks, the top-k
blocks (plus fixed init/local blocks) are selected, and the main attention
attends only to those blocks. Index keys live in a separate side cache
(``MiniMaxM3IndexerCache``).
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImplBase,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.minimax_m3_sparse_ops import (
    minimax_m3_index_topk,
    minimax_m3_index_topk_decode,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
)

logger = init_logger(__name__)


class MiniMaxM3SparseBackend(AttentionBackend):
    """Block-sparse GQA backend for MiniMax M3 sparse attention layers."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float16]

    @staticmethod
    def get_name() -> str:
        return "MINIMAX_M3_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["MiniMaxM3SparseImpl"]:
        return MiniMaxM3SparseImpl

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3SparseMetadataBuilder"]:
        return MiniMaxM3SparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Page size must equal the sparse block size so one sparse block maps
        # to one KV page (see minimax_m3_sparse_ops).
        return [128]

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Main GQA cache: paged K and V (FlashAttention layout). NOTE: cannot
        # key off num_kv_heads==1 to detect the index cache, since GQA itself
        # has num_kv_heads==1 at TP>=num_kv_heads. The index cache uses a
        # dedicated backend (MiniMaxM3IndexerBackend) instead.
        return (num_blocks, 2, block_size, num_kv_heads, head_size)


class MiniMaxM3IndexerBackend(MiniMaxM3SparseBackend):
    """Backend for the lightning-indexer side cache (key-only, single vector).

    Shares the impl/builder/metadata with the main sparse backend but stores a
    single index-key vector per token, matching the MLAAttentionSpec page.
    """

    @staticmethod
    def get_name() -> str:
        return "MINIMAX_M3_SPARSE_INDEXER"

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)


class MiniMaxM3IndexerCache(nn.Module, AttentionLayerBase):
    """Side KV cache for the lightning indexer's per-token index keys.

    Stores a single index-key vector per token (no value: M3 disables the index
    value projection). Modeled on ``DeepseekV32IndexerCache``; registers itself
    in the static forward context so the KV-cache manager allocates its cache.
    """

    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig | None = None,
    ) -> None:
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.dtype = dtype
        self.prefix = prefix
        self.cache_config = cache_config
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Key-only cache: one vector per token. MLAAttentionSpec budgets a
        # single vector; FullAttentionSpec would reserve 2x for K+V.
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
        )

    def forward(self) -> None: ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return MiniMaxM3IndexerBackend


@dataclass
class MiniMaxM3SparsePrefillMetadata:
    """Per-prefill state for index scoring + block-sparse attention.

    ``cu_seqlens_q``/``prefix_lens`` are precomputed in the builder so the
    forward path stays free of host syncs and per-step tensor allocations.
    """

    cu_seqlens_q: torch.Tensor  # [num_prefills + 1] int32, rebased to 0
    seq_lens: torch.Tensor  # [num_prefills] int32
    prefix_lens: torch.Tensor  # [num_prefills] int32 (cached/context tokens)
    block_table: torch.Tensor
    max_query_len: int
    max_seq_len: int


@dataclass
class MiniMaxM3SparseDecodeMetadata:
    """Per-decode state; all kernel args precomputed (cudagraph-safe)."""

    cu_seqlens_q: torch.Tensor  # [num_decodes + 1] int32
    seq_lens: torch.Tensor  # [num_decodes] int32
    prefix_lens: torch.Tensor  # [num_decodes] int32 (cached/context tokens)
    block_table: torch.Tensor
    max_query_len: int


@dataclass
class MiniMaxM3SparseMetadata(AttentionMetadata):
    """Sparse-attention metadata, split into prefill and decode sub-metadata."""

    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor

    # Total query tokens in the (decode-first) batch.
    num_actual_tokens: int

    # Split counts (batch is reordered decode-first).
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    # Sparse selection parameters.
    topk_blocks: int
    block_size: int
    init_blocks: int
    local_blocks: int
    score_type: str

    prefill: MiniMaxM3SparsePrefillMetadata | None = None
    decode: MiniMaxM3SparseDecodeMetadata | None = None


class MiniMaxM3SparseMetadataBuilder(AttentionMetadataBuilder[MiniMaxM3SparseMetadata]):
    # Full cudagraphs for pure single-token decode batches; mixed/prefill
    # batches fall back to piecewise/eager.
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    # Decode == a single query token; reorder pulls decodes to the front.
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        hf_config = vllm_config.model_config.hf_config.get_text_config()
        sparse_cfg = hf_config.sparse_attention_config
        self.topk_blocks: int = sparse_cfg["sparse_topk_blocks"]
        self.block_size: int = sparse_cfg["sparse_block_size"]
        self.init_blocks: int = sparse_cfg.get("sparse_init_block", 0)
        self.local_blocks: int = sparse_cfg.get("sparse_local_block", 0)
        self.score_type: str = sparse_cfg.get("sparse_score_type", "max")

        # Persistent buffer for the derived decode prefix lengths, so full
        # decode cudagraphs read from a stable address across replays.
        compilation_config = vllm_config.compilation_config
        self.use_full_cuda_graph = (
            compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self.decode_cudagraph_max_bs = vllm_config.scheduler_config.max_num_seqs
        if compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                compilation_config.max_cudagraph_capture_size,
            )
        self._decode_prefix_lens = torch.empty(
            self.decode_cudagraph_max_bs, dtype=torch.int32, device=device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3SparseMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
            )
        )
        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        # Cached/context tokens per request (seq_len - query_len), on device.
        num_computed = common_attn_metadata.compute_num_computed_tokens()

        # Batch is reordered decode-first, then prefill.
        prefill_metadata: MiniMaxM3SparsePrefillMetadata | None = None
        if num_prefills > 0:
            prefill_metadata = MiniMaxM3SparsePrefillMetadata(
                cu_seqlens_q=(query_start_loc[num_decodes:] - num_decode_tokens).to(
                    torch.int32
                ),
                seq_lens=seq_lens[num_decodes:],
                prefix_lens=num_computed[num_decodes:].to(torch.int32),
                block_table=block_table[num_decodes:],
                max_query_len=common_attn_metadata.max_query_len,
                max_seq_len=common_attn_metadata.max_seq_len,
            )

        decode_metadata: MiniMaxM3SparseDecodeMetadata | None = None
        if num_decodes > 0:
            # Decode query lengths from the CPU mirror (no device->host sync).
            q_cpu = query_start_loc_cpu
            decode_max_query_len = int(
                (q_cpu[1 : num_decodes + 1] - q_cpu[:num_decodes]).max()
            )
            # cu_seqlens_q aliases the runner's persistent query_start_loc.
            decode_prefix = num_computed[:num_decodes].to(torch.int32)
            if (
                self.use_full_cuda_graph
                and num_prefills == 0
                and num_decodes <= self.decode_cudagraph_max_bs
            ):
                # Persist derived prefix lens; the captured graph reads a
                # stable buffer address across replays.
                self._decode_prefix_lens[:num_decodes].copy_(
                    decode_prefix, non_blocking=True
                )
                decode_prefix = self._decode_prefix_lens[:num_decodes]
            decode_metadata = MiniMaxM3SparseDecodeMetadata(
                cu_seqlens_q=query_start_loc[: num_decodes + 1],
                seq_lens=seq_lens[:num_decodes],
                prefix_lens=decode_prefix,
                block_table=block_table[:num_decodes],
                max_query_len=decode_max_query_len,
            )

        return MiniMaxM3SparseMetadata(
            seq_lens=seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_actual_tokens=num_decode_tokens + num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            topk_blocks=self.topk_blocks,
            block_size=self.block_size,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            score_type=self.score_type,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )


class MiniMaxM3SparseImpl(AttentionImplBase[MiniMaxM3SparseMetadata]):
    """Block-sparse GQA attention for MiniMax M3 (forward not yet implemented).

    Inherits ``AttentionImplBase`` (not ``AttentionImpl``) because the sparse
    path needs a custom forward signature: the owning layer pre-inserts K/V and
    index-K into their caches, so forward takes only the queries.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        kv_cache_dtype: str = "auto",
        attn_type: str = AttentionType.DECODER,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type

        # Sparse selection parameters and index-branch dims.
        self.topk_blocks: int | None = kwargs.get("topk_blocks")
        self.block_size: int | None = kwargs.get("sparse_block_size")
        self.init_blocks: int = kwargs.get("init_blocks", 0)
        self.local_blocks: int = kwargs.get("local_blocks", 0)
        self.score_type: str = kwargs.get("score_type", "max")
        self.num_index_heads: int | None = kwargs.get("num_index_heads")
        self.index_head_dim: int | None = kwargs.get("index_head_dim")

    def _run_prefill(
        self,
        q: torch.Tensor,  # [tot, num_heads, head_dim]
        iq: torch.Tensor,  # [tot, num_idx_heads, head_dim]
        out: torch.Tensor,  # [tot, num_heads, head_dim]
        cu_seqlens_q: torch.Tensor,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        main_block_table: torch.Tensor,
        index_block_table: torch.Tensor,
        max_query_len: int,
        max_seq_len: int,
    ) -> None:
        # 1. Index block-score + top-k (reads the index-K cache).
        topk_idx = minimax_m3_index_topk(
            iq,
            self._index_kv_cache,
            index_block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            max_query_len,
            max_seq_len,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            self.num_kv_heads,
            self.scale,
        )
        # 2. GQA block-sparse attention over the selected blocks (main cache).
        minimax_m3_sparse_attn(
            q,
            self._kv_cache,
            topk_idx,
            main_block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            max_query_len,
            self.num_kv_heads,
            self.scale,
            out,
        )

    def _run_decode(
        self,
        q: torch.Tensor,  # [batch, num_heads, head_dim]
        iq: torch.Tensor,  # [batch, num_idx_heads, head_dim]
        out: torch.Tensor,  # [batch, num_heads, head_dim]
        cu_seqlens_q: torch.Tensor,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        main_block_table: torch.Tensor,
        index_block_table: torch.Tensor,
        max_seq_len: int,
    ) -> None:
        # Dedicated split-K decode kernels: parallelize over the KV dimension
        # (one query token per request leaves the prefill kernels idle).
        # 1. Index block-score (split-K over seq blocks) + top-k.
        topk_idx = minimax_m3_index_topk_decode(
            iq,
            self._index_kv_cache,
            index_block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            max_seq_len,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            self.num_kv_heads,
            self.scale,
        )
        # 2. GQA block-sparse attention (split-K over the selected blocks).
        minimax_m3_sparse_attn_decode(
            q,
            self._kv_cache,
            topk_idx,
            main_block_table,
            seq_lens,
            self.num_kv_heads,
            self.scale,
            out,
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        index_query: torch.Tensor,
        kv_cache: torch.Tensor,
        index_kv_cache: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        # K/V and index-K are pre-inserted into their caches; per-request
        # metadata comes from the forward context. Only queries are passed in.
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            # Profiling run: caches unbound, output left as-is.
            return output
        main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        index_md = attn_metadata[layer.index_cache.prefix]  # type: ignore[attr-defined]
        assert isinstance(main_md, MiniMaxM3SparseMetadata)
        assert isinstance(index_md, MiniMaxM3SparseMetadata)

        nd = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        hd = self.head_size
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        iq = index_query[:num_tokens].view(
            -1, self.num_index_heads, self.index_head_dim
        )
        out = output[:num_tokens].view(-1, self.num_heads, hd)
        # Stash caches for _run_phase (avoid threading through every call).
        self._kv_cache = kv_cache
        self._index_kv_cache = index_kv_cache

        # Decode slice [:nd]: each token is a 1-token "prefill" at seq_len-1.
        # All kernel args are precomputed in the builder (cudagraph-safe).
        if main_md.num_decodes > 0:
            d, idx_d = main_md.decode, index_md.decode
            assert d is not None and idx_d is not None
            self._run_decode(
                q[:nd],
                iq[:nd],
                out[:nd],
                d.cu_seqlens_q,
                d.seq_lens,
                d.prefix_lens,
                d.block_table,
                idx_d.block_table,
                main_md.max_seq_len,
            )

        # Prefill slice [nd:]: cu_seqlens_q already rebased to 0.
        if main_md.num_prefills > 0:
            p, idx_p = main_md.prefill, index_md.prefill
            assert p is not None and idx_p is not None
            self._run_prefill(
                q[nd:],
                iq[nd:],
                out[nd:],
                p.cu_seqlens_q,
                p.seq_lens,
                p.prefix_lens,
                p.block_table,
                idx_p.block_table,
                p.max_query_len,
                p.max_seq_len,
            )
        return output
