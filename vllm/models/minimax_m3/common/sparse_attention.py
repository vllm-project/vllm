# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Main block-sparse GQA attention for MiniMax M3 sparse layers.

The lightning indexer (``indexer.py``) selects the top-k KV blocks; this module
holds the main attention that attends only to those blocks: the paged K/V cache
backend, its metadata + builder, and the impl that consumes the indexer's
``topk_idx``. The Triton attend kernel lives here; the SM100 (MSA)
``build_k2q_csr`` + ``sparse_atten_func`` attend lives in
``nvidia/sparse_attention_msa.py``.

``MiniMaxM3SparseBackend`` and ``MiniMaxM3SparseMetadata`` are referenced by the
attention-backend registry (by dotted path) and by spec-decode, so they must
keep these names and stay in this module.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.common.ops.sparse_attn import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)
from vllm.platforms import current_platform
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
from vllm.v1.kv_cache_interface import AttentionSpec, is_quantized_kv_cache


class MiniMaxM3SparseBackend(AttentionBackend):
    """Block-sparse GQA backend for MiniMax M3 sparse attention layers."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float16]
    # bf16 or fp8 (e4m3/e5m2): the Triton kernels dequant fp8 before the dots.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_name() -> str:
        return "MINIMAX_M3_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["MiniMaxM3SparseImpl"]:
        # Concrete impl chosen by select_main_impl_cls; base for introspection.
        return MiniMaxM3SparseImpl

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3SparseMetadataBuilder"]:
        return MiniMaxM3SparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Page size == sparse block size (one sparse block per KV page).
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
        # Permutation from get_kv_cache_shape to the actual memory layout.
        if include_num_layers_dimension:
            raise NotImplementedError  # no cross-layer KV blocks in M3
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order


@dataclass
class MiniMaxM3SparsePrefillMetadata:
    """Per-prefill state; ``cu_seqlens_k``/``total_kv_blocks`` feed the MSA CSR."""

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
    """Per-decode state (cudagraph-safe). ``decode_query_len`` is the uniform
    per-request query length (1, or 1 + num_speculative_tokens)."""

    seq_lens: torch.Tensor  # [num_decodes] int32
    block_table: torch.Tensor
    decode_query_len: int


@dataclass
class MiniMaxM3SparseMetadata(AttentionMetadata):
    """Sparse-attention metadata, split into prefill and decode sub-metadata."""

    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor

    num_actual_tokens: int  # total query tokens (decode-first batch)

    # Split counts (batch reordered decode-first).
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    prefill: MiniMaxM3SparsePrefillMetadata | None = None
    decode: MiniMaxM3SparseDecodeMetadata | None = None


class MiniMaxM3SparseMetadataBuilder(AttentionMetadataBuilder[MiniMaxM3SparseMetadata]):
    # Full cudagraphs for uniform decode batches, incl. spec-decode verify
    # batches with >1 query token/request.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    # Raised to 1 + num_speculative_tokens by _init_reorder_batch_threshold when
    # spec decode is on; must match the indexer builder so the splits agree.
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        # Stable context-length buffer for decode cudagraph replays.
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
            qsl_cpu = common_attn_metadata.query_start_loc_cpu
            query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
            decode_query_len = int(query_lens_cpu[0].item())
            assert decode_query_len > 0
            assert torch.all(
                (query_lens_cpu == decode_query_len) | (query_lens_cpu == 0)
            )
            assert num_decode_tokens == num_decodes * decode_query_len
            decode_metadata = MiniMaxM3SparseDecodeMetadata(
                seq_lens=seq_lens[:num_decodes],
                block_table=block_table[:num_decodes],
                decode_query_len=decode_query_len,
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
    """Abstract base for block-sparse GQA over the indexer-selected blocks.

    Inherits ``AttentionImplBase`` for a custom forward signature (the layer
    pre-inserts K/V and runs the indexer, so forward takes the queries +
    ``topk_idx``). The Triton and MSA subclasses each own a full ``forward`` --
    no shared forward code.
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
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.use_fp8_kv = is_quantized_kv_cache(kv_cache_dtype)
        if "e5m2" in kv_cache_dtype:
            self.kv_cache_fp8_dtype = (
                torch.float8_e5m2fnuz
                if current_platform.is_fp8_fnuz()
                else torch.float8_e5m2
            )
        else:
            self.kv_cache_fp8_dtype = current_platform.fp8_dtype()
        # Sparse selection parameters (block_size == page size == SPARSE_BLOCK_SIZE).
        self.topk_blocks = topk_blocks
        self.block_size = sparse_block_size

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_idx: tuple[torch.Tensor | None, torch.Tensor | None],
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Attend the queries to the indexer-selected blocks. Per kernel."""
        raise NotImplementedError


class MiniMaxM3SparseTritonImpl(MiniMaxM3SparseImpl):
    """Triton block-sparse attend (``minimax_m3_sparse_attn``) + Triton decode."""

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_idx: tuple[torch.Tensor | None, torch.Tensor | None],
        output: torch.Tensor,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output  # profiling run; caches unbound
        main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        assert isinstance(main_md, MiniMaxM3SparseMetadata)
        decode_topk, prefill_topk = topk_idx

        nd = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        hd = self.head_size
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)
        kv_cache = (
            kv_cache.view(self.kv_cache_fp8_dtype) if self.use_fp8_kv else kv_cache
        )

        # Decode [:nd]: split-K over the selected blocks (request-major chunks).
        if main_md.num_decodes > 0:
            d = main_md.decode
            assert d is not None and decode_topk is not None
            minimax_m3_sparse_attn_decode(
                q[:nd],
                kv_cache,
                decode_topk,
                d.block_table,
                d.seq_lens,
                self.num_kv_heads,
                self.scale,
                out[:nd],
                d.decode_query_len,
            )

        # Prefill [nd:]: cu_seqlens_q already rebased to 0.
        if main_md.num_prefills > 0:
            p = main_md.prefill
            assert p is not None and prefill_topk is not None
            minimax_m3_sparse_attn(
                q[nd:],
                kv_cache,
                prefill_topk,
                p.block_table,
                p.cu_seqlens_q,
                p.seq_lens,
                p.context_lens,
                p.max_query_len,
                self.num_kv_heads,
                self.scale,
                out[nd:],
            )
        return output


def select_main_impl_cls(
    *,
    topk_blocks: int,
    kv_cache_dtype: str,
) -> type[MiniMaxM3SparseImpl]:
    """Pick the main attend impl off the main KV-cache dtype.

    bf16 on Blackwell (SM100) uses the MSA attend; fp8 or non-Blackwell falls
    back to Triton. The MSA module is imported lazily so AMD/non-SM100 never
    import fmha_sm100.
    """
    if (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and topk_blocks in (4, 8, 16, 32)
        and not is_quantized_kv_cache(kv_cache_dtype)
    ):
        from vllm.models.minimax_m3.nvidia.sparse_attention_msa import (
            MiniMaxM3SparseMSAImpl,
        )

        return MiniMaxM3SparseMSAImpl
    return MiniMaxM3SparseTritonImpl
