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
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImplBase,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
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
        # The lightning-indexer side cache (MiniMaxM3IndexerCache) stores a
        # single key vector per token (num_kv_heads == 1); the main GQA cache
        # stores paged K and V in the standard FlashAttention layout.
        if num_kv_heads == 1:
            return (num_blocks, block_size, head_size)
        return (num_blocks, 2, block_size, num_kv_heads, head_size)


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
        return MiniMaxM3SparseBackend


@dataclass
class MiniMaxM3SparsePrefillMetadata:
    """Per-prefill state for index scoring + block-sparse attention."""

    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int
    max_seq_len: int


@dataclass
class MiniMaxM3SparseDecodeMetadata:
    """Per-decode state; top-k block indices are reused across decode steps."""

    seq_lens: torch.Tensor
    block_table: torch.Tensor
    decode_lens: torch.Tensor
    # Top-k block indices, reused across decode steps.
    topk_block_indices: torch.Tensor | None = None


@dataclass
class MiniMaxM3SparseMetadata(AttentionMetadata):
    """Sparse-attention metadata, split into prefill and decode sub-metadata."""

    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor

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

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3SparseMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
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

        # Batch is reordered decode-first, then prefill.
        prefill_metadata: MiniMaxM3SparsePrefillMetadata | None = None
        if num_prefills > 0:
            prefill_metadata = MiniMaxM3SparsePrefillMetadata(
                query_start_loc=query_start_loc[num_decodes:] - num_decode_tokens,
                seq_lens=seq_lens[num_decodes:],
                block_table=block_table[num_decodes:],
                max_query_len=common_attn_metadata.max_query_len,
                max_seq_len=common_attn_metadata.max_seq_len,
            )
            # TODO: index top-k workspace / chunking prep for prefill scoring.

        decode_metadata: MiniMaxM3SparseDecodeMetadata | None = None
        if num_decodes > 0:
            decode_lens = torch.diff(query_start_loc[: num_decodes + 1])
            decode_metadata = MiniMaxM3SparseDecodeMetadata(
                seq_lens=seq_lens[:num_decodes],
                block_table=block_table[:num_decodes],
                decode_lens=decode_lens,
            )
            # TODO: top-k schedule metadata / cached block-index buffer prep.

        return MiniMaxM3SparseMetadata(
            seq_lens=seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=common_attn_metadata.slot_mapping,
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
        raise NotImplementedError(
            "MiniMax M3 sparse attention forward is not yet implemented"
        )
