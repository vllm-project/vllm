# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA (gfx950/CDNA4) indexer impl for MiniMax M3.

Scores index blocks and selects the top-k with AITER's fp8 MFMA kernels. Only
the scorer differs between the two sides of the batch: prefill's rows are
ragged, each with its own causal reach, so they take
``pa_sparse_block_score_prefill``, while decode's are uniform per request and
take ``pa_sparse_block_score_decode``. The two share one tile body in AITER, so
they agree block for block, and both hand a ``[heads, rows, blocks]`` fp32
score to the same ``pa_sparse_block_topk``.

That top-k also emits the attend's page table. The winners are already in its
workgroup's LDS, so resolving them through the block table there costs one wave
and saves the attend a second pass over the selection.

Indexer CP shards the block axis instead of replicating it, so its decode runs
a three-stage chain: score this rank's stride of the blocks, cut that to this
rank's own full top-k, exchange the candidates, merge them into the global
selection. AITER serves all three, over the same kernels as the unsharded pair
-- a ``world`` of 1 collapses the shard back to the whole axis, which is what
keeps the two paths from drifting. The shapes the MFMA cannot take are refused
by the CP gate rather than served another way, since the fallback for them is
simply to run unsharded.

Both phases emit the attend's page table in the same numbering, so the two
sides of a batch can share one buffer whichever wrote which rows.
"""

import math
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.attention import IndexerKVDType
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.amd.indexer_cp import IndexerCpPeers, get_indexer_cp_peers
from vllm.models.minimax_m3.amd.ops.sparse_pa import ASM_PAGE_SIZE
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerBackend,
    MiniMaxM3IndexerCache,
    MiniMaxM3IndexerDecodeMetadata,
    MiniMaxM3IndexerImpl,
    MiniMaxM3IndexerMetadata,
    MiniMaxM3IndexerMetadataBuilder,
    MiniMaxM3IndexerPrefillMetadata,
)

# The single source of truth for whether the AITER attend was asked for; this
# indexer is only usable alongside it.
from vllm.models.minimax_m3.common.sparse_attention import (
    _minimax_m3_aiter_sparse_pa_requested,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# MiniMax-M3 score/top-k shape contract
MSA_TOPK_BLOCKS = 16
MSA_SCORE_TYPE = "max"
MSA_SPARSE_BLOCK_SIZE = 128
MSA_INDEX_HEAD_DIM = 128
# The fp8 MFMA the score kernels are built on exists on these targets only.
SUPPORTED_ARCHS = ("gfx950",)

# Wave width the top-k is written against: it gives one lane per output slot
# and reads the score row in wave-wide strips.
WAVE_SIZE = 64
# Per-lane register slots the top-k can hold, which is what caps the context:
# a row may span at most SLOTS_MAX * WAVE_SIZE blocks.
SLOTS_MAX = 128
MAX_SUPPORTED_BLOCKS = SLOTS_MAX * WAVE_SIZE


def _score_width(max_seq_len: int, block_size: int, world: int = 1) -> int:
    """Blocks one score row spans, padded as the top-k requires.

    Lanes read whole wave-wide strips with no tail guard and each holds a
    power-of-two count of them, so the block axis is padded past the block
    count the context actually needs.

    ``world`` is the block shard the row holds: a context-parallel row covers
    only this rank's ``1/world`` of the blocks, so the padding is reached from
    that count and not from the context's.
    """
    blocks = cdiv(cdiv(max(max_seq_len, 1), block_size), world)
    return next_power_of_2(cdiv(blocks, WAVE_SIZE)) * WAVE_SIZE


def _score_buffer(
    heads: int,
    rows: int,
    max_seq_len: int,
    block_size: int,
    device: torch.device,
    world: int = 1,
) -> torch.Tensor:
    """Score buffer for ``rows`` query rows, padded as the top-k requires.

    Left uninitialized on purpose: the score pass writes every block up to the
    longest row it covers and the top-k reads only the blocks its own row can
    see, so nothing downstream observes the padded tail. Filling it would cost
    a write over what is, at long context, the largest tensor in the indexer.
    """
    width = _score_width(max_seq_len, block_size, world)
    return torch.empty((heads, rows, width), dtype=torch.float32, device=device)


class MiniMaxM3IndexerMSABackend(MiniMaxM3IndexerBackend):
    """Indexer side-cache backend selecting the MSA builder."""

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3IndexerMSAMetadataBuilder"]:
        return MiniMaxM3IndexerMSAMetadataBuilder


@dataclass
class MiniMaxM3IndexerMSAMetadata(MiniMaxM3IndexerMetadata):
    """Adds the per-row shape the ragged top-k needs.

    The uniform decode rows are recovered inside the kernel from ``seq_lens`` and
    the shared query length, which also clamps cudagraph padding rows to nothing.
    Prefill rows have no such shape, so it is materialized here once per forward
    and shared by every layer -- which is also where the emitted page table's
    tail block comes from, since a block count alone does not say how many tokens
    the last block holds.
    """

    # [num_prefill_tokens] int32, causal block count per row.
    prefill_num_valid_pages: torch.Tensor | None = None
    # [num_prefill_tokens] int32, the request each prefill row belongs to.
    prefill_row_req_id: torch.Tensor | None = None
    # [num_prefill_tokens] int32, causal token count (position + 1) per row.
    prefill_kv_lens: torch.Tensor | None = None


class MiniMaxM3IndexerMSAMetadataBuilder(MiniMaxM3IndexerMetadataBuilder):
    """The Triton indexer's metadata plus the prefill rows' causal shape."""

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
        self.sparse_block_size = int(
            text_config.sparse_attention_config["sparse_block_size"]
        )
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
    ) -> MiniMaxM3IndexerMSAMetadata:
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
                row_positions // self.sparse_block_size + 1, non_blocking=True
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

        return MiniMaxM3IndexerMSAMetadata(
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


class MiniMaxM3IndexerMSAImpl(MiniMaxM3IndexerImpl):
    """Fp8 score + top-k for both prefill and decode."""

    indexer_backend_cls: ClassVar[type[AttentionBackend]] = MiniMaxM3IndexerMSABackend

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Both passes are v_mfma_f32_16x16x32_fp8_fp8 with no bf16 instantiation,
        # so an index cache of any other dtype would be read as e4m3 bytes.
        # The selector will not get here, but nothing else may either.
        if self.indexer_kv_dtype not in ("fp8", "fp8_e4m3"):
            raise ValueError(
                "The MSA indexer requires an fp8 e4m3 index cache, got "
                f"indexer_kv_dtype={self.indexer_kv_dtype!r}"
            )
        # Shared, stable-address page table + per-row context bound the top-k
        # emits for the attend. Owned by the model so one allocation serves
        # every layer; left None when it reserved none, in which case the
        # attend rebuilds the table itself.
        self.sparse_bt_buffer: torch.Tensor | None = None
        self.sparse_ctx_buffer: torch.Tensor | None = None
        # Indexer CP, assigned by the wrapper for the same reason the buffers
        # are: the impl's base ``__init__`` lives in common and knows none of
        # this. ``cp_world == 1`` is the tensor-parallel path, where
        # ``num_index_heads`` is already only this rank's heads.
        self.indexer_cp = False
        self.cp_world = 1
        self.cp_rank = 0
        # The CP decode pass's bound, score buffer and peer mapping, filled in
        # by ``init_cp`` once the topology above is known.
        self.cp_max_seq_len = 0
        # Rows the candidate buffers are strided by, which has to stay fixed
        # across calls even as the decode batch varies.
        self.cp_cand_rows = 0
        self._cp_score: torch.Tensor | None = None
        self._cp_peers: IndexerCpPeers | None = None

    def init_cp(self) -> None:
        """Resolve the CP decode pass and allocate its buffers.

        Separate from ``__init__`` because the topology is assigned after it,
        and persistent because the decode path is captured: a replay reuses the
        addresses it recorded, so the chain has to hand back the same ones.

        Every shape is taken from ``max_model_len`` rather than the batch, for
        the same reason. Capture goes through a dummy batch one token long, so a
        width derived from that batch's own ``max_seq_len`` would bake a
        one-block shard into the graph and then replay it at full context. The
        per-request causal bound clips the real work at runtime instead.
        """
        config = get_current_vllm_config()
        self.cp_max_seq_len = config.model_config.max_model_len
        spec = getattr(config, "speculative_config", None)
        n_spec = int(getattr(spec, "num_speculative_tokens", 0) or 0) if spec else 0
        # A verification step's draft tokens are extra rows of one decode pass:
        # they score against the same blocks, so the indexer sees them together.
        max_query_len = n_spec + 1

        comp = config.compilation_config
        max_capture = getattr(comp, "max_cudagraph_capture_size", 0) or 0
        # Decode rows the chain may be handed, which a captured batch pads up to.
        max_rows = (
            max(config.scheduler_config.max_num_seqs, max_capture) * max_query_len
        )
        device = self.index_cache.kv_cache.device
        # Every rank scores every index head over its own shard, so the score
        # buffer carries the full head count; only what survives the merge is
        # this rank's own run.
        self._cp_score = _score_buffer(
            self.num_index_heads,
            max_rows,
            self.cp_max_seq_len,
            self.block_size,
            device,
            world=self.cp_world,
        )
        # The candidates never surface as a tensor here: the selector writes
        # them into a peer's buffer and reads its own back within one launch,
        # so what this layer needs is the mapping and not the storage. Shared
        # with every other layer, which the generation each candidate carries
        # is what makes safe.
        self.cp_cand_rows = max_rows
        self._cp_peers = get_indexer_cp_peers(
            self.num_owned_index_heads, max_rows, self.topk_blocks
        )
        logger.info_once(
            "MiniMax M3 indexer CP: fused decode selector over %d ranks "
            "[index_heads=%d, shard=%d blocks of %d]",
            self.cp_world,
            self.num_index_heads,
            cdiv(cdiv(self.cp_max_seq_len, self.block_size), self.cp_world),
            cdiv(self.cp_max_seq_len, self.block_size),
        )

    @property
    def num_owned_index_heads(self) -> int:
        """Index heads whose selection this rank ends up holding.

        Under CP the scoring pass runs every head on every rank, but the merge
        leaves each rank only the run it owns -- so this, not
        ``num_index_heads``, is the width of the top-k the attend reads and of
        everything downstream of the exchange. It equals this rank's KV head
        count, which is what the emitted page table is laid out against.
        """
        return self.num_index_heads // self.cp_world

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

    def _decode_cp_aiter(
        self,
        index_query: torch.Tensor,  # [num_decode_tokens, num_index_heads, D]
        kv: torch.Tensor,
        d: MiniMaxM3IndexerDecodeMetadata,
        decode_topk: torch.Tensor,
        decode_page16_block_table: torch.Tensor,
        sparse_bt: torch.Tensor,
        sparse_ctx: torch.Tensor,
    ) -> None:
        """Score this rank's block shard, then select across the shards.

        Two kernels, not four: the selector nominates its shard's candidates,
        writes them into the peer that owns those heads over the IPC mapping
        ``init_cp`` set up, merges what arrives and emits the attend's page
        table, all in one launch. There is no collective between the halves --
        see ``indexer_cp`` for why removing it is the point.

        Both passes have to agree on one context bound, since the score pass
        sizes the shard off it and the selector cuts its causal reach against
        it. ``cp_max_seq_len`` is that bound for both, and it is
        ``max_model_len`` rather than the batch's own longest row so that a
        captured graph cannot specialize a shard width.

        Every rank must reach this with the same row count, which the batch
        being replicated across the tensor-parallel group already gives: the
        ranks pair a block up with its own index on each peer, so a mismatch
        hangs rather than producing wrong output.
        """
        from aiter.ops.msa_attention import (
            pa_sparse_block_score_decode,
            pa_sparse_block_topk_cp,
        )

        assert self._cp_score is not None
        assert self._cp_peers is not None
        rows = index_query.shape[0]
        score = self._cp_score[:, :rows]

        pa_sparse_block_score_decode(
            index_query,
            kv,
            score,
            d.block_table,
            d.seq_lens,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            query_len=d.decode_query_len,
            max_seq_len=self.cp_max_seq_len,
            rank=self.cp_rank,
            world=self.cp_world,
        )
        # Each rank nominates the full top-k of its shard and never
        # topk/world, since the global winners may lie entirely inside one
        # shard. Forced blocks need no pinning on the way in -- the round-robin
        # puts each of them in exactly one rank's shard, and the score pass
        # already wrote its sentinel there -- but the merge re-pins them, so
        # the pin survives the round trip through a score.
        pa_sparse_block_topk_cp(
            score,
            self._cp_peers.cand_ptrs,
            self._cp_peers.cp_gen,
            decode_topk,
            d.seq_lens,
            decode_page16_block_table,
            sparse_bt,
            sparse_ctx,
            max_seq_len=self.cp_max_seq_len,
            block_size=self.block_size,
            cand_rows=self.cp_cand_rows,
            query_len=d.decode_query_len,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            num_kv_heads=self.num_kv_heads,
            pages_per_block=self.pages_per_block,
            rank=self.cp_rank,
            world=self.cp_world,
        )

    def _decode(
        self,
        index_query: torch.Tensor,  # [num_decode_tokens, num_index_heads, D]
        kv: torch.Tensor,
        d: MiniMaxM3IndexerDecodeMetadata,
        decode_topk: torch.Tensor,
        decode_page16_block_table: torch.Tensor,
        sparse_bt: torch.Tensor,
        sparse_ctx: torch.Tensor,
    ) -> None:
        """Score the blocks, take the top-k, emit the attend's page table.

        The unsharded pair does both in one rank's worth of work, which is the
        thing CP exists not to do, so the sharded case takes the fused
        cross-rank selector instead.
        """
        if self.indexer_cp:
            self._decode_cp_aiter(
                index_query,
                kv,
                d,
                decode_topk,
                decode_page16_block_table,
                sparse_bt,
                sparse_ctx,
            )
            return

        from aiter.ops.msa_attention import (
            pa_sparse_block_score_decode,
            pa_sparse_block_topk,
        )

        heads, tokens = decode_topk.shape[:2]
        score = _score_buffer(
            heads, tokens, d.max_seq_len, self.block_size, index_query.device
        )
        pa_sparse_block_score_decode(
            index_query,
            kv,
            score,
            d.block_table,
            d.seq_lens,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            query_len=d.decode_query_len,
            max_seq_len=d.max_seq_len,
        )
        pa_sparse_block_topk(
            score,
            decode_topk,
            decode_page16_block_table,
            d.seq_lens,
            sparse_bt=sparse_bt,
            sparse_ctx=sparse_ctx,
            max_seq_len=d.max_seq_len,
            block_size=self.block_size,
            query_len=d.decode_query_len,
            num_kv_heads=self.num_kv_heads,
            pages_per_block=self.pages_per_block,
        )

    def _prefill(
        self,
        index_query: torch.Tensor,  # [num_prefill_tokens, num_index_heads, D]
        kv: torch.Tensor,
        p: MiniMaxM3IndexerPrefillMetadata,
        prefill_topk: torch.Tensor,
        prefill_page16_block_table: torch.Tensor,
        sparse_bt: torch.Tensor,
        sparse_ctx: torch.Tensor,
        num_valid_pages: torch.Tensor,
        row_req_id: torch.Tensor,
        kv_lens: torch.Tensor,
    ) -> None:
        """Score the blocks, take the top-k, emit the attend's page table.

        AITER both sides, CP or not: prefill has a whole chunk of query rows to
        spread over the machine and nothing to gain from sharding the block
        axis on top of that, so the sharded chain the decode path takes buys it
        only an exchange it would otherwise not pay for.
        """
        from aiter.ops.msa_attention import (
            pa_sparse_block_score_prefill,
            pa_sparse_block_topk,
        )

        if self.indexer_cp:
            # The scorer indexes the head axis directly rather than taking its
            # stride, so this rank's run of the replicated projection has to be
            # compacted. Tensor-parallel rows arrive contiguous already.
            owned = self.num_owned_index_heads
            lo = self.cp_rank * owned
            index_query = index_query[:, lo : lo + owned, :].contiguous()

        heads, total_q = prefill_topk.shape[:2]
        score = _score_buffer(
            heads, total_q, p.max_seq_len, self.block_size, index_query.device
        )
        pa_sparse_block_score_prefill(
            index_query,
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
        pa_sparse_block_topk(
            score,
            prefill_topk,
            prefill_page16_block_table,
            p.seq_lens,
            sparse_bt=sparse_bt,
            sparse_ctx=sparse_ctx,
            max_seq_len=p.max_seq_len,
            block_size=self.block_size,
            num_valid_pages=num_valid_pages,
            row_req_id=row_req_id,
            kv_lens=kv_lens,
            num_kv_heads=self.num_kv_heads,
            pages_per_block=self.pages_per_block,
        )

    def forward(
        self,
        index_query: torch.Tensor,
        *,
        decode_page16_block_table: torch.Tensor | None = None,
        prefill_page16_block_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None  # profiling run; caches unbound
        md = attn_metadata[self.index_cache.prefix]
        assert isinstance(md, MiniMaxM3IndexerMSAMetadata)
        # The emitted page table addresses the main cache, whose blocks are a
        # different group's than the index cache's, so the top-k resolves the
        # selection through the attend's block table -- only the score pass reads
        # the index cache and takes the indexer's. It has to be the page-16
        # rebase of that table, which the attend's own metadata builder does
        # once per step; the indexer's metadata cannot reach another group's
        # blocks, so the layer reads it off the attend and hands it in.
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
                (self.num_owned_index_heads, num_tokens, self.topk_blocks),
                dtype=torch.int32,
                device=iq.device,
            )

        decode_topk: torch.Tensor | None = None
        prefill_topk: torch.Tensor | None = None

        if md.num_decodes > 0:
            d = md.decode
            assert d is not None
            assert decode_page16_block_table is not None, (
                "the MSA indexer's top-k emits the attend's page table and "
                "needs the page-16 rebase of the attend's decode block table"
            )
            decode_topk = buf[:, :nd, :]
            sparse_bt, sparse_ctx = self._table_rows(0, nd)
            # A decode row's causal length is seq_len - query_len + token + 1,
            # which every kernel in the chain derives itself, so speculative
            # rows need no extra per-row shape.
            self._decode(
                iq[:nd],
                kv,
                d,
                decode_topk,
                decode_page16_block_table,
                sparse_bt,
                sparse_ctx,
            )

        if md.num_prefills > 0:
            p = md.prefill
            assert p is not None
            assert prefill_page16_block_table is not None, (
                "the MSA indexer's top-k emits the attend's page table and "
                "needs the page-16 rebase of the attend's prefill block table"
            )
            assert md.prefill_num_valid_pages is not None
            assert md.prefill_row_req_id is not None
            assert md.prefill_kv_lens is not None
            prefill_topk = buf[:, nd:num_tokens, :]
            sparse_bt, sparse_ctx = self._table_rows(nd, num_tokens)
            self._prefill(
                iq[nd:],
                kv,
                p,
                prefill_topk,
                prefill_page16_block_table,
                sparse_bt,
                sparse_ctx,
                md.prefill_num_valid_pages,
                md.prefill_row_req_id,
                md.prefill_kv_lens,
            )

        return decode_topk, prefill_topk


class MiniMaxM3MSAIndexer(nn.Module):
    """``MiniMaxM3Indexer``'s surface over the MSA impl.
    Fp8 score + top-k for both prefill and decode.
    """

    def __init__(
        self,
        *,
        impl_cls: type[MiniMaxM3IndexerMSAImpl],
        sparse_bt_buffer: torch.Tensor | None = None,
        sparse_ctx_buffer: torch.Tensor | None = None,
        indexer_cp: bool = False,
        **impl_kwargs,
    ) -> None:
        super().__init__()
        self.impl = impl_cls(**impl_kwargs)
        # Assigned rather than passed: the impl's base ``__init__`` lives in
        # common and takes neither the table buffers nor the CP topology.
        self.impl.sparse_bt_buffer = sparse_bt_buffer
        self.impl.sparse_ctx_buffer = sparse_ctx_buffer
        self.impl.indexer_cp = indexer_cp
        if indexer_cp:
            self.impl.cp_world = get_tensor_model_parallel_world_size()
            self.impl.cp_rank = get_tensor_model_parallel_rank()
            # Resolves which decode chain runs and allocates its buffers, which
            # needs the topology above and so cannot happen in the impl's own
            # ``__init__``.
            self.impl.init_cp()

    @property
    def index_cache(self) -> MiniMaxM3IndexerCache:
        return self.impl.index_cache

    @property
    def num_index_heads(self) -> int:
        return self.impl.num_index_heads

    def forward(
        self,
        index_query: torch.Tensor,
        *,
        decode_page16_block_table: torch.Tensor | None = None,
        prefill_page16_block_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.impl(
            index_query,
            decode_page16_block_table=decode_page16_block_table,
            prefill_page16_block_table=prefill_page16_block_table,
        )


def msa_indexer_unsupported_reason(
    *,
    topk_blocks: int,
    sparse_block_size: int,
    index_head_dim: int,
    indexer_kv_dtype: IndexerKVDType,
    max_model_len: int,
    score_type: str = "max",
    cp_world: int = 1,
    num_index_heads: int = 1,
    num_kv_heads: int = 1,
    max_query_len: int = 1,
    dcp_size: int = 1,
) -> str | None:
    """Return why this config cannot use this indexer, or None if it can.

    Checks platform (ROCm/gfx950), the AITER sparse PA attend, index-cache
    dtype, the kernels' shape contract and the max context in blocks.
    ``select_msa_indexer_impl_cls`` logs the string and falls back when it is
    not None.

    The default ``cp_world=1`` asks about the ordinary indexer and ignores
    every argument below it. Above one the question becomes whether the same
    indexer can be run context-parallel, which is the same checks plus the
    ones sharding adds, and the extra arguments describe the shape sharding
    creates. One gate rather than two because CP is a way of running this
    indexer and not a different one: everything the unsharded pass needs, a
    sharded pass needs too, and a second gate would be free to disagree about
    it. Still a reason and not an error at either width -- a config refused
    with ``cp_world > 1`` runs unsharded, one refused at 1 runs on Triton.
    """
    if not current_platform.is_rocm():
        return (
            "needs ROCm for the fp8 MFMA score/top-k, "
            f"got platform={current_platform.device_type!r}"
        )
    if not _minimax_m3_aiter_sparse_pa_requested():
        # The top-k emits the attend's page table in page-16 numbering, which
        # only addresses the interleaved cache the AITER attend reads. Paired
        # with any other attend there is nowhere valid to write it, so this
        # indexer is not usable on its own.
        return (
            "needs the AITER sparse PA attend, whose page table its top-k "
            "emits (rocm_aiter_ops + shuffle KV cache layout)"
        )
    if indexer_kv_dtype not in ("fp8", "fp8_e4m3"):
        # The score kernels are fp8 MFMA; there is no bf16 instantiation.
        return (
            f"needs an fp8 e4m3 index cache, got indexer_kv_dtype={indexer_kv_dtype!r}"
        )
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return f"needs {' or '.join(SUPPORTED_ARCHS)} for the fp8 MFMA"
    if score_type != MSA_SCORE_TYPE:
        return f"needs score_type={MSA_SCORE_TYPE!r}, got score_type={score_type!r}"
    if topk_blocks != MSA_TOPK_BLOCKS:
        return f"needs topk_blocks={MSA_TOPK_BLOCKS}, got topk_blocks={topk_blocks}"
    if sparse_block_size != MSA_SPARSE_BLOCK_SIZE:
        return (
            f"needs sparse_block_size={MSA_SPARSE_BLOCK_SIZE}, "
            f"got sparse_block_size={sparse_block_size}"
        )
    if index_head_dim != MSA_INDEX_HEAD_DIM:
        return (
            f"needs index_head_dim={MSA_INDEX_HEAD_DIM}, "
            f"got index_head_dim={index_head_dim}"
        )

    max_blocks = math.ceil(max_model_len / sparse_block_size)
    if max_blocks > MAX_SUPPORTED_BLOCKS:
        return (
            f"max_model_len={max_model_len} needs {max_blocks} blocks per row, "
            f"more than the {MAX_SUPPORTED_BLOCKS} the top-k is compiled for"
        )

    if cp_world <= 1:
        return None

    # From here, only what sharding is what creates. The kernels' own limits
    # are read from AITER rather than restated: they are properties of how the
    # passes are built -- one MFMA tile's columns, one wave's lanes -- so a
    # copy would be a second answer to a question AITER already answers, free
    # to drift from the one that is enforced at the call.
    from aiter.ops.msa_block_select import SCORE_MFMA_COLS, TOPK_CP_MAX_CAND

    if dcp_size > 1:
        # Both features claim the context axis, but DCP claims it in the
        # cache: virtual block size, slot mappings that drop non-owned tokens,
        # an LSE-merged attend. Layering this on top would shard an
        # already-sharded context.
        return (
            f"decode_context_parallel_size={dcp_size} > 1 (KV-cache DCP owns "
            "the context axis; indexer CP replaces it, it does not extend it)"
        )
    if cp_world > num_index_heads or num_index_heads % cp_world:
        # Every rank scores every head, so what the exchange routes is a
        # contiguous run of heads per rank; that run has to be the same width
        # on all of them, and it has to be the same run the tensor-parallel
        # split already gave this rank its kv heads from.
        return (
            f"needs num_index_heads divisible by cp_world, got "
            f"cp_world={cp_world}, num_index_heads={num_index_heads}"
        )
    if num_index_heads * max_query_len > SCORE_MFMA_COLS:
        # The score pass packs one (query token, index head) pair per MFMA
        # column. Under CP the head count is the model's rather than this
        # rank's, so a wide enough draft is what runs the columns out.
        return (
            f"num_index_heads={num_index_heads} x query_len={max_query_len} "
            f"exceeds the {SCORE_MFMA_COLS} MFMA columns of the score pass"
        )
    candidates = cp_world * topk_blocks
    if candidates > TOPK_CP_MAX_CAND:
        # The merge reads every shard's nominations at once, which is what
        # lets it skip the whole-row selector's histogram narrowing, and it
        # holds them in one wave's lanes.
        return (
            f"cp_world={cp_world} x topk_blocks={topk_blocks} = {candidates} "
            f"exceeds the {TOPK_CP_MAX_CAND} candidates the merge holds"
        )
    if num_index_heads // cp_world != num_kv_heads:
        # The merge emits the page table off its own selection, one row per kv
        # head, so it needs the two 1:1.
        return (
            f"this rank would own {num_index_heads // cp_world} index head(s) "
            f"but {num_kv_heads} kv head(s); the merge's page table needs "
            "them 1:1"
        )
    return None


def select_msa_indexer_impl_cls(
    *,
    topk_blocks: int,
    sparse_block_size: int,
    index_head_dim: int,
    indexer_kv_dtype: IndexerKVDType,
    score_type: str = "max",
) -> type[MiniMaxM3IndexerMSAImpl] | None:
    """The MSA indexer impl if this config can use it, else None.

    ``None`` sends the caller to the platform-neutral ``MiniMaxM3Indexer``,
    which on ROCm means the Triton indexer -- and a bf16-only one, so an fp8
    index cache that lands here has nowhere to go.
    """
    reason = msa_indexer_unsupported_reason(
        topk_blocks=topk_blocks,
        sparse_block_size=sparse_block_size,
        index_head_dim=index_head_dim,
        indexer_kv_dtype=indexer_kv_dtype,
        max_model_len=get_current_vllm_config().model_config.max_model_len,
        score_type=score_type,
    )
    if reason is not None:
        logger.info_once("MiniMax M3 indexer: MSA unavailable (%s)", reason)
        return None
    logger.info_once(
        "MiniMax M3 indexer: selected MSA (Triton fp8 MFMA score + top-k) "
        "[topk_blocks=%d, indexer_kv_dtype=%s]",
        topk_blocks,
        indexer_kv_dtype,
    )
    return MiniMaxM3IndexerMSAImpl


def msa_indexer_cp_enabled(
    vllm_config: VllmConfig,
    *,
    msa_indexer_selected: bool,
) -> bool:
    """Whether to run the selected MSA indexer context-parallel.

    The same gate ``select_msa_indexer_impl_cls`` used, asked again at the CP
    world so that the conditions sharding adds come into it. Cheap to re-ask
    and deliberately the same function: the two answers cannot disagree about
    the platform or the kernels' contract, which is what would otherwise let
    a rank shard an indexer that was never selected.

    Recomputed rather than cached: every input is a config or topology value
    fixed for the process, so the answer is stable, and a module-level cache
    would only make it survive across the configs a test builds.

    Asked once per model and handed to every layer, since the projection's
    shard layout and the indexer's head count have to be decided off one
    answer.
    """
    requested = vllm_config.attention_config.indexer_cp
    sparse_cfg = getattr(
        vllm_config.model_config.hf_text_config, "sparse_attention_config", None
    )
    if not requested or sparse_cfg is None:
        return False

    if not msa_indexer_selected:
        # CP has no tensor-parallel fallback to offer: the qkv projection is
        # built with index_q replicated, so an indexer that did not take this
        # path would read its own heads out of a replicated tensor. The
        # selection logged its own reason on the way past.
        reason: str | None = "the MSA indexer was not selected, so nothing to shard"
    else:
        config = vllm_config.model_config.hf_text_config
        tp_size = get_tensor_model_parallel_world_size()
        spec = vllm_config.speculative_config
        n_spec = int(getattr(spec, "num_speculative_tokens", 0) or 0) if spec else 0
        reason = msa_indexer_unsupported_reason(
            topk_blocks=int(sparse_cfg["sparse_topk_blocks"]),
            sparse_block_size=int(sparse_cfg["sparse_block_size"]),
            index_head_dim=int(sparse_cfg["sparse_index_dim"]),
            indexer_kv_dtype=vllm_config.attention_config.resolve_indexer_kv_dtype(
                "bf16"
            ),
            max_model_len=vllm_config.model_config.max_model_len,
            score_type=sparse_cfg.get("sparse_score_type", "max"),
            cp_world=tp_size,
            num_index_heads=int(sparse_cfg["sparse_num_index_heads"]),
            # Mirrors the model's own split, which is where the emitted page
            # table's head axis comes from.
            num_kv_heads=max(1, int(config.num_key_value_heads) // tp_size),
            # A verification step's draft tokens are extra rows of one decode
            # pass: they score against the same blocks, so the indexer sees
            # them together.
            max_query_len=n_spec + 1,
            dcp_size=vllm_config.parallel_config.decode_context_parallel_size,
        )

    if reason is not None:
        logger.info_once("MiniMax M3 indexer CP: disabled (%s)", reason)
        return False
    world = get_tensor_model_parallel_world_size()
    logger.info_once(
        "MiniMax M3 indexer CP: enabled over %d ranks (each scores 1/%d of "
        "the blocks for every index head)",
        world,
        world,
    )
    return True
