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

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.models.minimax_m3.common.ops.index_topk import (
    minimax_m3_index_topk,
    minimax_m3_index_topk_decode,
)
from vllm.models.minimax_m3.common.ops.sparse_attn import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImplBase,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    get_kv_cache_layout,
    split_decodes_and_prefills,
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
    # Sparse kernels operate on a bf16 KV cache only.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["bfloat16"]

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
        # to one KV page (see common.ops.sparse_attn).
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
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets us from
        # `get_kv_cache_shape` to the actual memory layout we want.
        if include_num_layers_dimension:
            # M3 does not use cross-layer (per-layer-stacked) KV blocks for now.
            raise NotImplementedError
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order


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

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            # M3 does not use cross-layer (per-layer-stacked) KV blocks.
            raise NotImplementedError
        return (0, 1, 2)


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

    ``cu_seqlens_q``/``context_lens`` are precomputed in the builder so the
    forward path stays free of host syncs and per-step tensor allocations.
    """

    cu_seqlens_q: torch.Tensor  # [num_prefills + 1] int32, rebased to 0
    cu_seqlens_k: torch.Tensor  # [num_prefills + 1] int32, cumulative KV lengths
    seq_lens: torch.Tensor  # [num_prefills] int32, total KV lengths
    context_lens: torch.Tensor  # [num_prefills] int32 (cached/context tokens)
    block_table: torch.Tensor
    max_query_len: int
    max_seq_len: int
    total_kv_blocks: int


@dataclass
class MiniMaxM3SparseDecodeMetadata:
    """Per-decode state (cudagraph-safe); split-K kernels key only on seq_lens."""

    seq_lens: torch.Tensor  # [num_decodes] int32
    block_table: torch.Tensor


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

    prefill: MiniMaxM3SparsePrefillMetadata | None = None
    decode: MiniMaxM3SparseDecodeMetadata | None = None


class MiniMaxM3SparseMetadataBuilder(AttentionMetadataBuilder[MiniMaxM3SparseMetadata]):
    # Full cudagraphs for uniform single-query decode batches.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    # The split-K decode kernel doesn't support spec decode yet (it handles one
    # query token per request only). Keep the threshold at 1 so multi-query
    # verify batches route to the prefill kernels instead of the decode kernel.
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # Stable per-request context-length buffer for decode cudagraph replays.
        # Sized to max_num_batched_tokens (>= num_reqs).
        self.context_len_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
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

        # Per-request context tokens (seq_len - query_len) into the stable
        # buffer; batch is decode-first ([:num_decodes] decode, rest prefill).
        context_lens = self.context_len_buffer[:num_reqs]
        context_lens.copy_(
            common_attn_metadata.compute_num_computed_tokens(), non_blocking=True
        )

        prefill_metadata: MiniMaxM3SparsePrefillMetadata | None = None
        if num_prefills > 0:
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            assert seq_lens_cpu is not None
            prefill_seq_lens_cpu = seq_lens_cpu[num_decodes:]
            prefill_total_kv_blocks = (
                ((prefill_seq_lens_cpu + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE)
                .sum()
                .item()
            )
            prefill_kv_lens = seq_lens[num_decodes:]
            prefill_cu_seqlens_k = torch.empty(
                num_prefills + 1, dtype=torch.int32, device=seq_lens.device
            )
            prefill_cu_seqlens_k[0] = 0
            torch.cumsum(prefill_kv_lens, dim=0, out=prefill_cu_seqlens_k[1:])
            prefill_metadata = MiniMaxM3SparsePrefillMetadata(
                cu_seqlens_q=(query_start_loc[num_decodes:] - num_decode_tokens).to(
                    torch.int32
                ),
                cu_seqlens_k=prefill_cu_seqlens_k,
                seq_lens=prefill_kv_lens,
                context_lens=context_lens[num_decodes:],
                block_table=block_table[num_decodes:],
                max_query_len=common_attn_metadata.max_query_len,
                max_seq_len=common_attn_metadata.max_seq_len,
                total_kv_blocks=prefill_total_kv_blocks,
            )

        decode_metadata: MiniMaxM3SparseDecodeMetadata | None = None
        if num_decodes > 0:
            decode_metadata = MiniMaxM3SparseDecodeMetadata(
                seq_lens=seq_lens[:num_decodes],
                block_table=block_table[:num_decodes],
            )

        return MiniMaxM3SparseMetadata(
            seq_lens=seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_actual_tokens=num_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
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
        *,
        topk_blocks: int,
        sparse_block_size: int,
        num_index_heads: int,
        index_head_dim: int,
        init_blocks: int = 0,
        local_blocks: int = 0,
        score_type: str = "max",
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype

        # Sparse selection parameters and index-branch dims.
        self.topk_blocks = topk_blocks
        self.block_size = sparse_block_size
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.score_type = score_type
        self.num_index_heads = num_index_heads
        self.index_head_dim = index_head_dim
        can_run_prefill_cutedsl = (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
            and has_cutedsl()
            and self.head_size == 128
            and self.block_size == 128
            and self.topk_blocks in (4, 8, 16, 32)
        )
        self._prefill_gqa_sparse = (
            self._prefill_gqa_sparse_cutedsl
            if can_run_prefill_cutedsl
            else self._prefill_gqa_sparse_triton
        )

    def _run_prefill(
        self,
        q: torch.Tensor,  # [tot, num_heads, head_dim]
        iq: torch.Tensor,  # [tot, num_idx_heads, head_dim]
        out: torch.Tensor,  # [tot, num_heads, head_dim]
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        seq_lens: torch.Tensor,
        context_lens: torch.Tensor,
        main_block_table: torch.Tensor,
        index_block_table: torch.Tensor,
        max_query_len: int,
        max_seq_len: int,
        total_kv_blocks: int,
    ) -> None:
        # 1. Index block-score + top-k (reads the index-K cache).
        topk_idx = minimax_m3_index_topk(
            iq,
            self._index_kv_cache,
            index_block_table,
            cu_seqlens_q,
            seq_lens,
            context_lens,
            max_query_len,
            max_seq_len,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            self.num_kv_heads,
            self.scale,
        )
        # 2. GQA block-sparse attention over the selected blocks (main cache).
        self._prefill_gqa_sparse(
            q,
            out,
            topk_idx,
            cu_seqlens_q,
            cu_seqlens_k,
            seq_lens,
            context_lens,
            main_block_table,
            max_query_len,
            max_seq_len,
            total_kv_blocks,
        )

    def _prefill_gqa_sparse_triton(
        self,
        q: torch.Tensor,
        out: torch.Tensor,
        topk_idx: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        seq_lens: torch.Tensor,
        context_lens: torch.Tensor,
        main_block_table: torch.Tensor,
        max_query_len: int,
        max_seq_len: int,
        total_kv_blocks: int,
    ) -> None:
        minimax_m3_sparse_attn(
            q,
            self._kv_cache,
            topk_idx,
            main_block_table,
            cu_seqlens_q,
            seq_lens,
            context_lens,
            max_query_len,
            self.num_kv_heads,
            self.scale,
            out,
        )

    def _prefill_gqa_sparse_cutedsl(
        self,
        q: torch.Tensor,
        out: torch.Tensor,
        topk_idx: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        seq_lens: torch.Tensor,
        context_lens: torch.Tensor,
        main_block_table: torch.Tensor,
        max_query_len: int,
        max_seq_len: int,
        total_kv_blocks: int,
    ) -> None:
        from vllm.models.minimax_m3.nvidia.ops.prefill_gqa_sparse import (
            minimax_m3_sparse_attn_cutedsl,
        )

        minimax_m3_sparse_attn_cutedsl(
            q,
            self._kv_cache,
            topk_idx,
            main_block_table,
            cu_seqlens_q,
            cu_seqlens_k,
            seq_lens,
            max_query_len,
            max_seq_len,
            self.num_kv_heads,
            self.scale,
            out,
            total_kv_blocks=total_kv_blocks,
        )

    def _run_decode(
        self,
        q: torch.Tensor,  # [batch, num_heads, head_dim]
        iq: torch.Tensor,  # [batch, num_idx_heads, head_dim]
        out: torch.Tensor,  # [batch, num_heads, head_dim]
        seq_lens: torch.Tensor,
        main_block_table: torch.Tensor,
        index_block_table: torch.Tensor,
        max_seq_len: int,
    ) -> None:
        # Split-K decode kernels (parallelize over KV; one query token/request).
        # 1. Index block-score + top-k.
        topk_idx = minimax_m3_index_topk_decode(
            iq,
            self._index_kv_cache,
            index_block_table,
            seq_lens,
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

    @eager_break_during_capture
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
                d.seq_lens,
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
                p.cu_seqlens_k,
                p.seq_lens,
                p.context_lens,
                p.block_table,
                idx_p.block_table,
                p.max_query_len,
                p.max_seq_len,
                p.total_kv_blocks,
            )
        return output
