# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER (ROCm) indexer impl for MiniMax M3.

Scores index blocks and selects the top-k with AITER's fp8 MFMA kernels on both
sides of the batch: ``pa_sparse_block_score_decode`` for the uniform-query-length
decode rows and ``pa_sparse_block_score_prefill`` for the ragged prefill rows,
then ``pa_sparse_block_topk`` for each. The two scoring passes share one tile
body in AITER, so they agree block for block, and both encode the forced
init/local blocks as the same sentinel scores the Triton indexer uses.

The top-k also emits the attend's page table. The winners are already in its
workgroup's LDS, so resolving them through the block table there costs one wave
and replaces the separate Triton pass in ``ops.sparse_pa``; the rows it writes
are one per (token, kv head) with the page ids folded head-minor, which is the
layout ``pa_decode_gluon`` reads after it flattens the cache.

The score kernels are built on ``v_mfma_f32_16x16x32_fp8_fp8``, so this impl
requires an fp8 (e4m3) index cache and an fp8 index query -- the fused
QK-norm/RoPE kernel emits both directly when the index cache is e4m3. See
``aiter_indexer_unsupported_reason`` for the full set of limits;
``select_indexer_impl_cls`` refuses to pick this impl unless they all hold.

AITER imports are function-local so this module stays import-safe off ROCm.
"""

import math
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.attention import IndexerKVDType
from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.amd.ops.sparse_pa import ASM_PAGE_SIZE
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerBackend,
    MiniMaxM3IndexerDecodeMetadata,
    MiniMaxM3IndexerImpl,
    MiniMaxM3IndexerMetadata,
    MiniMaxM3IndexerMetadataBuilder,
    MiniMaxM3IndexerPrefillMetadata,
)
from vllm.models.minimax_m3.common.sparse_attention import MiniMaxM3SparseMetadata
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec

# Page size == sparse block size == index-K block.
PAGE_SIZE = 128
# Wave width the top-k is written against: it gives one lane per output slot and
# reads the score row in wave-wide strips.
WAVE_SIZE = 64
# Per-lane register slots the top-k can hold, which is what caps the context: a
# row may span at most SLOTS_MAX * WAVE_SIZE blocks.
SLOTS_MAX = 128
MAX_SUPPORTED_BLOCKS = SLOTS_MAX * WAVE_SIZE
# MFMA columns the score pass has, one per (query token, index head) pair.
MFMA_COLS = 16
# The fp8 MFMA the score kernels are built on exists on these targets only.
SUPPORTED_ARCHS = ("gfx942", "gfx950")


def _pow2_ceil(n: int) -> int:
    return 1 << (n - 1).bit_length() if n >= 1 else 1


def score_block_width(max_seq_len: int, block_size: int) -> int:
    """Block-axis width the top-k requires of the score buffer.

    Lanes read whole wave-wide strips with no tail guard and each holds a
    power-of-two count of them, so the axis is padded past the block count.
    """
    max_blk = math.ceil(max(max_seq_len, 1) / block_size)
    return _pow2_ceil(math.ceil(max_blk / WAVE_SIZE)) * WAVE_SIZE


def aiter_indexer_max_decode_query_len(vllm_config: VllmConfig) -> int:
    """Longest query a decode row can carry, which spec decode is what sets.

    Mirrors ``_init_reorder_batch_threshold(1, supports_spec_as_decode=True)``,
    since that is what the builder splits the batch on and therefore what the
    decode kernel will actually be handed.
    """
    spec = vllm_config.speculative_config
    if spec is None or spec.num_speculative_tokens is None:
        return 1
    return 1 + (2 if spec.parallel_drafting else 1) * spec.num_speculative_tokens


def aiter_indexer_unsupported_reason(
    *,
    topk_blocks: int,
    sparse_block_size: int,
    num_index_heads: int,
    index_head_dim: int,
    indexer_kv_dtype: IndexerKVDType,
    max_model_len: int,
    max_decode_query_len: int = 1,
) -> str | None:
    """Why the AITER indexer cannot serve this configuration, or None if it can.

    Returning the reason rather than a bool keeps the fallback log actionable,
    since most of these limits are shape constants a deployment can change.
    """
    if not current_platform.is_rocm():
        return "not running on ROCm"
    if indexer_kv_dtype not in ("fp8", "fp8_e4m3"):
        # The score kernels are fp8 MFMA; there is no bf16 instantiation.
        return (
            f"needs an fp8 e4m3 index cache, got indexer_kv_dtype={indexer_kv_dtype!r}"
        )
    from vllm.platforms.rocm import on_gfx942, on_gfx950

    if not (on_gfx942() or on_gfx950()):
        return f"needs {' or '.join(SUPPORTED_ARCHS)} for the fp8 MFMA"
    if topk_blocks > WAVE_SIZE:
        return f"topk_blocks={topk_blocks} exceeds one wave ({WAVE_SIZE})"
    if num_index_heads > MFMA_COLS:
        return f"num_index_heads={num_index_heads} exceeds the {MFMA_COLS} MFMA columns"
    # A decode row's whole query shares one MFMA tile, one column per (token,
    # head) pair, so spec decode trades columns against the head count.
    if num_index_heads * max_decode_query_len > MFMA_COLS:
        return (
            f"num_index_heads={num_index_heads} x max_decode_query_len="
            f"{max_decode_query_len} exceeds the {MFMA_COLS} MFMA columns"
        )
    if index_head_dim % 64 != 0:
        return f"index_head_dim={index_head_dim} must be a multiple of 64"
    if sparse_block_size % MFMA_COLS != 0:
        return (
            f"sparse_block_size={sparse_block_size} must tile into "
            f"{MFMA_COLS}-token MFMA rows"
        )
    max_blocks = math.ceil(max_model_len / sparse_block_size)
    if max_blocks > MAX_SUPPORTED_BLOCKS:
        return (
            f"max_model_len={max_model_len} needs {max_blocks} blocks per row, "
            f"more than the top-k's {MAX_SUPPORTED_BLOCKS}"
        )
    return None


class MiniMaxM3IndexerAiterBackend(MiniMaxM3IndexerBackend):
    """Indexer side-cache backend selecting the AITER builder."""

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3IndexerAiterMetadataBuilder"]:
        return MiniMaxM3IndexerAiterMetadataBuilder


@dataclass
class MiniMaxM3IndexerAiterMetadata(MiniMaxM3IndexerMetadata):
    """Adds the per-row shape the ragged top-k needs.

    The uniform decode rows are recovered inside the kernel from ``seq_lens`` and
    the shared query length, which also clamps cudagraph padding rows to nothing.
    Prefill rows have no such shape, so it is materialized here once per forward
    and shared by every layer -- which is also where the emitted page table's
    tail block comes from, since a block count alone does not say how many tokens
    the last block holds.
    """

    # [num_prefill_tokens] int32, cdiv(position + 1, PAGE_SIZE) per prefill row.
    prefill_num_valid_pages: torch.Tensor | None = None
    # [num_prefill_tokens] int32, the request each prefill row belongs to.
    prefill_row_req_id: torch.Tensor | None = None
    # [num_prefill_tokens] int32, causal token count (position + 1) per row.
    prefill_kv_lens: torch.Tensor | None = None


class MiniMaxM3IndexerAiterMetadataBuilder(MiniMaxM3IndexerMetadataBuilder):
    """The Triton indexer's metadata plus the prefill rows' causal shape."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # Companions to the base's num_valid_pages_buffer, for the two vectors
        # only the emitted table needs.
        self.row_req_id_buffer = torch.empty(
            max_tokens, dtype=torch.int32, device=device
        )
        self.kv_lens_buffer = torch.empty(max_tokens, dtype=torch.int32, device=device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3IndexerAiterMetadata:
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
        prefill_num_valid_pages: torch.Tensor | None = None
        prefill_row_req_id: torch.Tensor | None = None
        prefill_kv_lens: torch.Tensor | None = None
        if num_prefills > 0:
            cu_seqlens_q = (query_start_loc[num_decodes:] - num_decode_tokens).to(
                torch.int32
            )
            prefill_metadata = MiniMaxM3IndexerPrefillMetadata(
                cu_seqlens_q=cu_seqlens_q,
                seq_lens=seq_lens[num_decodes:],
                context_lens=context_lens[num_decodes:],
                block_table=block_table[num_decodes:],
                max_query_len=common_attn_metadata.max_query_len,
                max_seq_len=common_attn_metadata.max_seq_len,
            )
            # A prefill row sees its own position, so its causal length and block
            # count both follow from that alone; the request it belongs to comes
            # from the query offsets. Prefill batches are never captured, so the
            # stable buffers are only being reused here, not required.
            positions = common_attn_metadata.positions
            assert positions is not None
            row_positions = positions[num_decode_tokens:num_tokens]
            prefill_num_valid_pages = self.num_valid_pages_buffer[
                num_decode_tokens:num_tokens
            ]
            prefill_num_valid_pages.copy_(
                row_positions // PAGE_SIZE + 1, non_blocking=True
            )
            prefill_kv_lens = self.kv_lens_buffer[num_decode_tokens:num_tokens]
            prefill_kv_lens.copy_(row_positions + 1, non_blocking=True)
            prefill_row_req_id = self.row_req_id_buffer[num_decode_tokens:num_tokens]
            prefill_row_req_id.copy_(
                torch.searchsorted(
                    cu_seqlens_q[1:].contiguous(),
                    torch.arange(
                        num_prefill_tokens,
                        dtype=torch.int32,
                        device=cu_seqlens_q.device,
                    ),
                    right=True,
                ),
                non_blocking=True,
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

        return MiniMaxM3IndexerAiterMetadata(
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
            prefill_num_valid_pages=prefill_num_valid_pages,
            prefill_row_req_id=prefill_row_req_id,
            prefill_kv_lens=prefill_kv_lens,
        )


class MiniMaxM3IndexerAiterImpl(MiniMaxM3IndexerImpl):
    """AITER fp8 score + top-k for both prefill and decode."""

    indexer_backend_cls: ClassVar[type[AttentionBackend]] = MiniMaxM3IndexerAiterBackend
    emits_sparse_block_table: ClassVar[bool] = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Both passes are v_mfma_f32_16x16x32_fp8_fp8 with no bf16 instantiation,
        # so an index cache of any other dtype would be read as e4m3 bytes.
        # select_indexer_impl_cls will not get here, but nothing else may either.
        if self.indexer_kv_dtype not in ("fp8", "fp8_e4m3"):
            raise ValueError(
                "The AITER indexer requires an fp8 e4m3 index cache, got "
                f"indexer_kv_dtype={self.indexer_kv_dtype!r}"
            )

    @property
    def pages_per_block(self) -> int:
        """Physical pages one selected block expands into for the attend."""
        return self.block_size // ASM_PAGE_SIZE

    def _table_rows(self, lo: int, hi: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The page table and context rows covering score rows ``[lo, hi)``.

        One table row per (token, kv head), head minor, which is the order
        ``pa_decode_gluon`` reads once it flattens the cache. Slices of the
        shared buffers rather than copies, so the attend sees the writes; a
        model that reserved no buffers gets throwaway ones, and the attend
        rebuilds the table itself in that case.
        """
        kv_heads = self.num_kv_heads
        if self.sparse_bt_buffer is None or self.sparse_ctx_buffer is None:
            rows = (hi - lo) * kv_heads
            device = self.index_cache.kv_cache.device
            return (
                torch.empty(
                    (rows, self.topk_blocks * self.pages_per_block),
                    dtype=torch.int32,
                    device=device,
                ),
                torch.empty(rows, dtype=torch.int32, device=device),
            )
        return (
            self.sparse_bt_buffer[lo * kv_heads : hi * kv_heads],
            self.sparse_ctx_buffer[lo * kv_heads : hi * kv_heads],
        )

    def _new_score(self, rows: int, max_seq_len: int) -> torch.Tensor:
        """Score buffer for ``rows`` query rows.

        Left uninitialized on purpose: the score pass writes every block up to
        the longest row it covers, and the top-k reads only the blocks its own
        row can see, so nothing downstream observes the padded tail. Filling it
        would cost a write over the whole [H, rows, width] extent, which at long
        context is the largest tensor in the indexer.
        """
        return torch.empty(
            (
                self.num_index_heads,
                rows,
                score_block_width(max_seq_len, self.block_size),
            ),
            dtype=torch.float32,
            device=self.index_cache.kv_cache.device,
        )

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        from aiter.ops.sparse_attention import (
            pa_sparse_block_score_decode,
            pa_sparse_block_score_prefill,
            pa_sparse_block_topk,
        )

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None  # profiling run; caches unbound
        md = attn_metadata[self.index_cache.prefix]
        assert isinstance(md, MiniMaxM3IndexerAiterMetadata)
        # The emitted page table addresses the main cache, whose blocks are a
        # different group's than the index cache's, so the top-k resolves the
        # selection through the attend's block table -- only the score pass reads
        # the index cache and takes the indexer's.
        attend_md = attn_metadata[self.attend_layer_name]
        assert isinstance(attend_md, MiniMaxM3SparseMetadata)

        num_tokens = md.num_actual_tokens
        nd = md.num_decode_tokens
        iq = index_query[:num_tokens].view(
            -1, self.num_index_heads, self.index_head_dim
        )
        kv = self.index_cache.kv_cache

        # Both sides write into the single shared persistent buffer (decode at
        # [:, :nd], prefill at [:, nd:]) and return views into it. The top-k
        # takes the head/row strides, so these slices need no copy back.
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
            score = self._new_score(nd, d.max_seq_len)
            pa_sparse_block_score_decode(
                iq[:nd],
                kv,
                score,
                d.block_table,
                d.seq_lens,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                query_len=d.decode_query_len,
                max_seq_len=d.max_seq_len,
            )
            decode_topk = buf[:, :nd, :]
            sparse_bt, sparse_ctx = self._table_rows(0, nd)
            # A decode row's causal length is seq_len - query_len + token + 1,
            # which the kernel derives itself, so the emitted table covers
            # speculative rows without any extra per-row shape.
            assert attend_md.decode is not None
            pa_sparse_block_topk(
                score,
                decode_topk,
                attend_md.decode.block_table,
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
            p = md.prefill
            assert p is not None
            assert md.prefill_num_valid_pages is not None
            assert md.prefill_row_req_id is not None
            assert md.prefill_kv_lens is not None
            score = self._new_score(num_tokens - nd, p.max_seq_len)
            pa_sparse_block_score_prefill(
                iq[nd:],
                kv,
                score,
                p.block_table,
                p.cu_seqlens_q,
                p.seq_lens,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                max_query_len=p.max_query_len,
                max_seq_len=p.max_seq_len,
            )
            prefill_topk = buf[:, nd:num_tokens, :]
            sparse_bt, sparse_ctx = self._table_rows(nd, num_tokens)
            # Ragged rows carry their request and causal length explicitly: the
            # block count alone cannot place the tail block the table ends on.
            assert attend_md.prefill is not None
            pa_sparse_block_topk(
                score,
                prefill_topk,
                attend_md.prefill.block_table,
                p.seq_lens,
                sparse_bt,
                sparse_ctx,
                max_seq_len=p.max_seq_len,
                block_size=self.block_size,
                num_valid_pages=md.prefill_num_valid_pages,
                row_req_id=md.prefill_row_req_id,
                kv_lens=md.prefill_kv_lens,
                num_kv_heads=self.num_kv_heads,
                pages_per_block=self.pages_per_block,
            )

        return decode_topk, prefill_topk
