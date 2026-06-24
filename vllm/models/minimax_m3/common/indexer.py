# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 lightning indexer: side cache, metadata, and impl.

The indexer scores KV blocks with the index heads and selects the top-k blocks
(plus fixed init/local blocks) that the main block-sparse attention
(``sparse_attention.py``) then attends to. It owns its own side cache
(``MiniMaxM3IndexerCache``, one index-key vector per token), metadata, and
metadata builder, mirroring how DeepSeek V4 keeps the indexer separate from the
main attention.

``MiniMaxM3Indexer`` is the ``nn.Module`` the attention layer holds (like
``DeepseekV4Indexer``); it picks a kernel impl in ``__init__`` (via
``select_indexer_impl_cls``) and delegates ``forward`` to it.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.attention import IndexerKVDType
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.models.minimax_m3.common.ops.index_topk import (
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
)

logger = init_logger(__name__)


class MiniMaxM3IndexerBackend(AttentionBackend):
    """Indexer side-cache backend (key-only)."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float16]
    # bf16 today; mirrors the main backend to keep spec validation permissive.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_name() -> str:
        return "MINIMAX_M3_SPARSE_INDEXER"

    @staticmethod
    def get_impl_cls() -> type["MiniMaxM3IndexerImpl"]:
        # Concrete impl chosen by select_indexer_impl_cls; base for introspection.
        return MiniMaxM3IndexerImpl

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3IndexerMetadataBuilder"]:
        return MiniMaxM3IndexerTritonMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
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
    """Side KV cache for the indexer's per-token index keys (key-only).

    Registers itself in the static forward context so the KV-cache manager
    allocates it (like ``DeepseekV32IndexerCache``).
    """

    def __init__(
        self,
        head_dim: int,
        prefix: str,
        cache_config: CacheConfig | None = None,
        indexer_kv_dtype: IndexerKVDType = "bf16",
        backend_cls: type[AttentionBackend] = MiniMaxM3IndexerBackend,
    ) -> None:
        super().__init__()
        if indexer_kv_dtype in ("fp8", "fp8_e4m3"):
            cache_dtype = torch.float8_e4m3fn
        elif indexer_kv_dtype == "bf16":
            cache_dtype = torch.bfloat16
        else:
            raise NotImplementedError(
                f"indexer_kv_dtype={indexer_kv_dtype!r} is not supported by the "
                "MiniMax M3 indexer cache (only 'bf16' or 'fp8'/'fp8_e4m3')."
            )
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.indexer_kv_dtype = indexer_kv_dtype
        # Side-cache storage dtype: bf16, or e4m3 for the fp8 score path.
        self.dtype = cache_dtype
        self.prefix = prefix
        self.cache_config = cache_config
        # Impl-chosen backend -> each impl gets its own builder (get_attn_backend).
        self.backend_cls = backend_cls
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Key-only: MLAAttentionSpec budgets one vector/token (not 2x for K+V).
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
        )

    def forward(self) -> None: ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls


@dataclass
class MiniMaxM3IndexerPrefillMetadata:
    """Per-prefill index-scoring state."""

    cu_seqlens_q: torch.Tensor  # [num_prefills + 1] int32, rebased to 0
    seq_lens: torch.Tensor  # [num_prefills] int32, total KV lengths
    context_lens: torch.Tensor  # [num_prefills] int32 (cached/context tokens)
    block_table: torch.Tensor
    max_query_len: int
    max_seq_len: int


@dataclass
class MiniMaxM3IndexerDecodeMetadata:
    """Per-decode state (cudagraph-safe). ``decode_query_len`` is the uniform
    per-request query length (1, or 1 + num_speculative_tokens)."""

    seq_lens: torch.Tensor  # [num_decodes] int32
    block_table: torch.Tensor
    max_seq_len: int
    decode_query_len: int
    max_decode_query_len: int


@dataclass
class MiniMaxM3IndexerMetadata(AttentionMetadata):
    """Indexer metadata, split into prefill and decode sub-metadata."""

    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor

    num_actual_tokens: int  # total query tokens (decode-first batch)

    # Split counts; identical to the main metadata's (same reorder threshold).
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    prefill: MiniMaxM3IndexerPrefillMetadata | None = None
    decode: MiniMaxM3IndexerDecodeMetadata | None = None


class MiniMaxM3IndexerMetadataBuilder(
    AttentionMetadataBuilder[MiniMaxM3IndexerMetadata]
):
    """Abstract base: shared setup only. The Triton and MSA builders are
    parallel subclasses that each own their full ``build`` (no shared code)."""

    # Full cudagraphs for uniform decode batches (incl. spec-decode verify).
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    # Raised to 1 + num_speculative_tokens by _init_reorder_batch_threshold when
    # spec decode is on; matches the main builder so the splits agree.
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        hf_config = vllm_config.model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        sparse_cfg = text_config.sparse_attention_config
        # Index-query head count from model config (cache spec has 1 vec/token).
        total_index_heads = sparse_cfg["sparse_num_index_heads"]
        tp_size = get_tensor_model_parallel_world_size()
        if total_index_heads >= tp_size:
            assert total_index_heads % tp_size == 0
        else:
            assert tp_size % total_index_heads == 0
        self.num_index_heads = max(1, total_index_heads // tp_size)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        assert self.reorder_batch_threshold is not None
        self.max_decode_query_len = self.reorder_batch_threshold

        # Stable context-length buffer for decode cudagraph replays.
        self.context_len_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
        )


class MiniMaxM3IndexerTritonMetadataBuilder(MiniMaxM3IndexerMetadataBuilder):
    """Triton indexer metadata: no SM100 fmha_sm100 plan."""

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3IndexerMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )
        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        # Decode-first batch: context lengths into the stable cudagraph buffer.
        context_lens = self.context_len_buffer[:num_reqs]
        context_lens.copy_(
            common_attn_metadata.compute_num_computed_tokens(), non_blocking=True
        )

        prefill_metadata: MiniMaxM3IndexerPrefillMetadata | None = None
        if num_prefills > 0:
            prefill_metadata = MiniMaxM3IndexerPrefillMetadata(
                cu_seqlens_q=(query_start_loc[num_decodes:] - num_decode_tokens).to(
                    torch.int32
                ),
                seq_lens=seq_lens[num_decodes:],
                context_lens=context_lens[num_decodes:],
                block_table=block_table[num_decodes:],
                max_query_len=common_attn_metadata.max_query_len,
                max_seq_len=common_attn_metadata.max_seq_len,
            )

        decode_metadata: MiniMaxM3IndexerDecodeMetadata | None = None
        if num_decodes > 0:
            qsl_cpu = common_attn_metadata.query_start_loc_cpu
            query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
            decode_query_len = int(query_lens_cpu[0].item())
            assert decode_query_len > 0
            assert torch.all(
                (query_lens_cpu == decode_query_len) | (query_lens_cpu == 0)
            )
            assert num_decode_tokens == num_decodes * decode_query_len
            decode_metadata = MiniMaxM3IndexerDecodeMetadata(
                seq_lens=seq_lens[:num_decodes],
                block_table=block_table[:num_decodes],
                max_seq_len=common_attn_metadata.max_seq_len,
                decode_query_len=decode_query_len,
                max_decode_query_len=self.max_decode_query_len,
            )

        return MiniMaxM3IndexerMetadata(
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


class MiniMaxM3IndexerImpl(nn.Module):
    """Abstract base for the indexer kernel impls.

    Each impl owns its side cache and reports its backend via
    ``indexer_backend_cls`` (so each gets its own builder). The Triton and MSA
    subclasses each own a full ``forward`` returning ``(decode_topk,
    prefill_topk)`` -- no shared forward code.
    """

    # Set by each impl so the side cache reports the matching backend + builder.
    indexer_backend_cls: ClassVar[type[AttentionBackend]] = MiniMaxM3IndexerBackend

    def __init__(
        self,
        *,
        num_kv_heads: int,
        scale: float,
        topk_blocks: int,
        sparse_block_size: int,
        num_index_heads: int,
        index_head_dim: int,
        prefix: str,
        init_blocks: int = 0,
        local_blocks: int = 0,
        score_type: str = "max",
        cache_config: CacheConfig | None = None,
        indexer_kv_dtype: IndexerKVDType = "bf16",
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.scale = scale
        self.topk_blocks = topk_blocks
        self.block_size = sparse_block_size
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.score_type = score_type
        self.num_index_heads = num_index_heads
        self.index_head_dim = index_head_dim
        self.indexer_kv_dtype = indexer_kv_dtype
        # Shared, stable-address top-k output buffer (set by the model for the
        # cudagraph-safe MSA impl); None -> impl allocates fresh (eager).
        self.topk_indices_buffer = topk_indices_buffer
        # Owns the side cache (registers itself in the static forward context).
        self.index_cache = MiniMaxM3IndexerCache(
            head_dim=index_head_dim,
            prefix=f"{prefix}.index_cache",
            cache_config=cache_config,
            indexer_kv_dtype=indexer_kv_dtype,
            backend_cls=type(self).indexer_backend_cls,
        )

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return ``(decode_topk, prefill_topk)``; implemented per kernel impl."""
        raise NotImplementedError


class MiniMaxM3IndexerTritonImpl(MiniMaxM3IndexerImpl):
    """Triton indexer score + top-k for both prefill and decode."""

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None  # profiling run; caches unbound
        index_md = attn_metadata[self.index_cache.prefix]
        assert isinstance(index_md, MiniMaxM3IndexerMetadata)
        num_tokens = index_md.num_actual_tokens
        nd = index_md.num_decode_tokens
        iq = index_query[:num_tokens].view(
            -1, self.num_index_heads, self.index_head_dim
        )
        kv = self.index_cache.kv_cache

        # Both sides write into the single shared persistent topk_indices_buffer
        # (decode at [:, :nd], prefill at [:, nd:]) and return views into it; the
        # kernels' out= writes out[:, :total_q]. None -> allocate fresh.
        buf = self.topk_indices_buffer
        decode_topk: torch.Tensor | None = None
        prefill_topk: torch.Tensor | None = None
        if index_md.num_decodes > 0:
            d = index_md.decode
            assert d is not None
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
                out=buf,
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
                out=buf[:, nd:, :] if buf is not None else None,
            )
        return decode_topk, prefill_topk


def select_indexer_impl_cls(
    *,
    topk_blocks: int,
    indexer_kv_dtype: IndexerKVDType = "bf16",
) -> type[MiniMaxM3IndexerImpl]:
    """Pick the indexer impl off the platform, top-k count, and cache dtype.

    On Blackwell (SM100) with ``topk_blocks`` in ``(4, 8, 16, 32)`` (matching the
    main MSA attend), the fmha_sm100 score path + Triton top-k is used for both
    bf16 and fp8 index caches. Everything else falls back to the Triton indexer
    (bf16 only).
    """
    if indexer_kv_dtype in ("mxfp4", "nvfp4"):
        raise NotImplementedError(
            f"indexer_kv_dtype={indexer_kv_dtype!r} needs the (not-yet-added) "
            "CuteDSL indexer impl."
        )
    is_sm100 = (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    )
    use_msa = (
        is_sm100
        and topk_blocks in (4, 8, 16, 32)
        and indexer_kv_dtype in ("bf16", "fp8", "fp8_e4m3")
    )
    if use_msa:
        # Lazy import so AMD / non-SM100 never import fmha_sm100.
        from vllm.models.minimax_m3.nvidia.indexer_msa import (
            MiniMaxM3IndexerMSAImpl,
        )

        logger.info_once(
            "MiniMax M3 indexer: selected MSA (fmha_sm100 score + Triton top-k) "
            "[topk_blocks=%d, indexer_kv_dtype=%s]",
            topk_blocks,
            indexer_kv_dtype,
        )
        return MiniMaxM3IndexerMSAImpl
    if indexer_kv_dtype != "bf16":
        raise NotImplementedError(
            f"indexer_kv_dtype={indexer_kv_dtype!r} is not supported by the "
            "Triton indexer impl."
        )
    logger.info_once(
        "MiniMax M3 indexer: selected Triton (no fmha_sm100) "
        "[topk_blocks=%d, indexer_kv_dtype=%s, sm100=%s]",
        topk_blocks,
        indexer_kv_dtype,
        is_sm100,
    )
    return MiniMaxM3IndexerTritonImpl


class MiniMaxM3Indexer(nn.Module):
    """Indexer module held by the attention layer (like ``DeepseekV4Indexer``).

    Picks the kernel impl in ``__init__`` (``select_indexer_impl_cls``) and
    delegates ``forward``; exposes the impl's side cache via ``index_cache``.
    """

    def __init__(
        self,
        *,
        num_kv_heads: int,
        scale: float,
        topk_blocks: int,
        sparse_block_size: int,
        num_index_heads: int,
        index_head_dim: int,
        prefix: str,
        init_blocks: int = 0,
        local_blocks: int = 0,
        score_type: str = "max",
        cache_config: CacheConfig | None = None,
        indexer_kv_dtype: IndexerKVDType = "bf16",
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        impl_cls = select_indexer_impl_cls(
            topk_blocks=topk_blocks,
            indexer_kv_dtype=indexer_kv_dtype,
        )
        self.impl = impl_cls(
            num_kv_heads=num_kv_heads,
            scale=scale,
            topk_blocks=topk_blocks,
            sparse_block_size=sparse_block_size,
            num_index_heads=num_index_heads,
            index_head_dim=index_head_dim,
            prefix=prefix,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            score_type=score_type,
            cache_config=cache_config,
            indexer_kv_dtype=indexer_kv_dtype,
            topk_indices_buffer=topk_indices_buffer,
        )

    @property
    def index_cache(self) -> MiniMaxM3IndexerCache:
        return self.impl.index_cache

    @property
    def num_index_heads(self) -> int:
        return self.impl.num_index_heads

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.impl(index_query)
