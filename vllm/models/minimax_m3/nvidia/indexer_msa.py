# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA (SM100/Blackwell) indexer impl for MiniMax M3.

Both sides write block scores into one unified token-major buffer
`[total_q, H, max_k_tiles]`, then a single `fmha_sm100.sparse_topk_select`
selects the top-k blocks for the whole batch (decode `[:nd]` + prefill
`[nd:]`) into the shared `topk_indices_buffer`. It bounds each row by its
causal page count and force-includes the init/local blocks, so the unwritten
tail of the buffer is pre-filled with `-inf`.

Prefill scores with `fmha_sm100`'s score-only (`OnlyScore`) path (much faster
than Triton for the wide prefill score, benchmarked ~3-5x), writing its
`max_score` straight into the buffer's prefill region (stride-aware, no copy).

Decode scores with CuteDSL when the flattened query tile is supported and fall
back to Triton otherwise, writing into the decode region. Only the top-k is
shared with prefill.

`fmha_sm100` imports are function-local so this module is import-safe on
AMD / non-SM100.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerBackend,
    MiniMaxM3IndexerDecodeMetadata,
    MiniMaxM3IndexerImpl,
    MiniMaxM3IndexerMetadata,
    MiniMaxM3IndexerMetadataBuilder,
)
from vllm.models.minimax_m3.common.ops.index_topk import (
    minimax_m3_index_decode_score,
)
from vllm.models.minimax_m3.nvidia.ops import minimax_m3_index_decode_score_cutedsl
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    CommonAttentionMetadata,
    max_decode_query_len,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from flashinfer.experimental.cake_msa_icp_indexer.cake_backend import (
        MSAIcpScorePlan,
    )
    from flashinfer.experimental.cake_msa_icp_indexer.cake_indexer import (
        MSAIcpGraphStep,
        MSAIcpLayerGraphs,
    )

logger = init_logger(__name__)

# Page size == sparse block size == index-K block; fmha tile id == M3 block id.
PAGE_SIZE = 128

# Fill for unwritten score tiles: -inf so they never win the top-k (score kernels
# only write causally-valid blocks).
_SCORE_SENTINEL = float("-inf")

# Tile (KV-block) dim of the unified score buffer, hardcoded as a cudagraph
# capture-time constant so the decode score kernel's buffer shape is frozen
# across replays. 8192 tiles == 1M tokens of context; -inf padding +
# num_valid_pages bound each row to its causal range, so shorter replays reuse
# the same buffer safely.
MAX_K_TILES = 8192


SUPPORTED_TP = (2, 4)
# Receive slot sets the layers rotate over.
NUM_SLOT_SETS = 3
TOPK = 16
PREFILL_FMHA_CHUNK_SIZE = 8192
SPIN_LIMIT = 1 << 26
# Speculative methods whose drafter does not reuse the target's sparse layers.
_SPEC_METHODS = (None, "eagle", "eagle3", "ngram", "suffix", "draft_model")


class MiniMaxM3IndexerMSABackend(MiniMaxM3IndexerBackend):
    """Indexer side-cache backend selecting the MSA builder."""

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3IndexerMSAMetadataBuilder"]:
        return MiniMaxM3IndexerMSAMetadataBuilder


@dataclass
class MiniMaxM3IndexerMSAPrefillMetadata:
    """fmha score plan + Triton top-k inputs for the prefill side (eager)."""

    plan: dict  # fmha_sm100 PlanInfo
    cu_seqlens_q: torch.Tensor  # [num_prefills + 1] int32, rebased to 0
    prefix_lens: torch.Tensor  # [num_prefills] int32, context tokens
    max_query_len: int
    page_table: torch.Tensor  # flat physical page indices for the prefill side


@dataclass
class MiniMaxM3IndexerMSAMetadata(MiniMaxM3IndexerMetadata):
    """Decode reuses the inherited base `decode` field (the Triton decode
    metadata); `prefill_msa` carries the fmha score plan for the prefill side
    (the base `prefill` field is unused on this path)."""

    prefill_msa: MiniMaxM3IndexerMSAPrefillMetadata | None = None
    # Per-forward view (``[:num_tokens]``) of the builder's persistent unified
    # score buffer ``[total_q, H, MAX_K_TILES]``, shared by decode and prefill
    # and reused across all layers. Pre-filled with the -inf sentinel in
    # ``build()``; each layer overwrites its valid tiles before its own top-k.
    unified_scores: torch.Tensor | None = None
    # Tile (KV-block) dim of the unified score buffer (== ``MAX_K_TILES``).
    # Forced as the fmha plan's ``max_k_tiles`` so prefill writes its max_score
    # straight into the shared buffer.
    max_k_tiles: int = 0
    # Batch-wide inputs for the single top-k over the unified buffer (decode +
    # prefill in one call). ``cu_seqlens_q`` is the per-request query-start
    # offsets (== query_start_loc, a stable view for cudagraph); ``prefix_lens``
    # is the per-request context length (== context_lens).
    topk_cu_seqlens_q: torch.Tensor | None = None
    topk_prefix_lens: torch.Tensor | None = None
    topk_max_query_len: int = 0
    # Per-token causal page count cdiv(seq_pos+1, PAGE_SIZE), [total_q] int32:
    # drives sparse_topk_select force_end_blocks + -1 out-of-range clamp.
    topk_num_valid_pages: torch.Tensor | None = None


@dataclass
class MiniMaxM3IndexerMSAShardedPrefillMetadata:
    local_chunks: list[tuple[int, int, torch.Tensor, "MSAIcpScorePlan"]]
    candidate_pool: torch.Tensor
    graphs: "MSAIcpLayerGraphs"
    candidate_buffers: torch.Tensor
    graph_step: "MSAIcpGraphStep | None" = None


@dataclass(kw_only=True)
class MiniMaxM3IndexerMSAShardedMetadata(MiniMaxM3IndexerMetadata):
    num_valid_pages: torch.Tensor
    prefill_msa: MiniMaxM3IndexerMSAShardedPrefillMetadata | None = None


class MiniMaxM3IndexerMSAMetadataBuilder(MiniMaxM3IndexerMetadataBuilder):
    """Decode metadata is the cudagraph-safe Triton decode metadata; the prefill
    fmha plan is built eagerly (prefill batches are not captured)."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        content = kv_cache_spec.state_content_bytes
        full = kv_cache_spec.head_size * kv_cache_spec.dtype.itemsize
        self.index_k_tp = 1 if content is None else full // content
        if self.index_k_tp > 1:
            self._init_sharded_prefill(vllm_config, device)
            return
        # Persistent unified score buffer [T, H, MAX_K_TILES] shared by all
        # indexer layers and reused across forwards. Stable address (required
        # for the captured decode path) + fixed tile dim so the decode score
        # kernel's shape is frozen at capture. Filled with -inf per forward in
        # build(); the valid/padding partition is a per-forward constant, so
        # every layer overwrites the same valid tiles before its own top-k.
        self.unified_scores_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            self.num_index_heads,
            MAX_K_TILES,
            dtype=torch.float32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3IndexerMSAMetadata | MiniMaxM3IndexerMSAShardedMetadata:
        if self.index_k_tp > 1:
            return self._build_sharded(common_attn_metadata)
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor
        query_start_loc = common_attn_metadata.query_start_loc

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )
        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        # Context (prefix) lengths into the stable cudagraph buffer.
        context_lens = self.context_len_buffer[:num_reqs]
        context_lens.copy_(
            common_attn_metadata.compute_num_computed_tokens(), non_blocking=True
        )

        # Per-token causal page count for the top-k, into the stable cg buffer.
        positions = common_attn_metadata.positions
        assert positions is not None
        num_valid_pages = self.num_valid_pages_buffer[:num_tokens]
        num_valid_pages.copy_(positions[:num_tokens] // PAGE_SIZE + 1)

        # Unified score buffer: a per-forward view of the persistent buffer,
        # reset to the -inf sentinel once here and shared by every layer (the
        # tile dim is the capture-time constant MAX_K_TILES).
        max_k_tiles = MAX_K_TILES
        unified_scores = self.unified_scores_buffer[:num_tokens]
        unified_scores.fill_(_SCORE_SENTINEL)

        decode: MiniMaxM3IndexerDecodeMetadata | None = None
        if num_decodes > 0:
            qsl_cpu = common_attn_metadata.query_start_loc_cpu
            query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
            decode_query_len = int(query_lens_cpu[0].item())
            assert decode_query_len > 0
            assert torch.all(
                (query_lens_cpu == decode_query_len) | (query_lens_cpu == 0)
            )
            decode = MiniMaxM3IndexerDecodeMetadata(
                seq_lens=seq_lens[:num_decodes],
                block_table=block_table[:num_decodes],
                max_seq_len=common_attn_metadata.max_seq_len,
                decode_query_len=decode_query_len,
                max_decode_query_len=self.max_decode_query_len,
            )

        prefill: MiniMaxM3IndexerMSAPrefillMetadata | None = None
        if num_prefills > 0:
            # Prefill is eager (not captured); the host lengths it needs (and the
            # _fmha_sm100_plan .tolist() inside) make the D->H sync acceptable.
            from vllm.third_party.fmha_sm100.api import _fmha_sm100_plan

            lo, hi = num_decodes, num_reqs
            qsl_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
            qo_lens_cpu = (qsl_cpu[1:] - qsl_cpu[:-1]).to(torch.int32)
            kv_lens_cpu = seq_lens[:num_reqs].cpu().to(torch.int32)
            nvp = (kv_lens_cpu + PAGE_SIZE - 1) // PAGE_SIZE
            side_qo = qo_lens_cpu[lo:hi]
            side_kv = kv_lens_cpu[lo:hi]
            plan = _fmha_sm100_plan(
                side_qo,
                side_kv,
                self.num_index_heads,
                num_kv_heads=1,
                qo_offset=side_kv - side_qo,  # bottom-right causal
                page_size=PAGE_SIZE,
                output_maxscore=True,
                causal=True,
                num_kv_splits=1,
            )
            # Force the plan's tile dim to the unified buffer's so prefill writes
            # its max_score straight into unified[:, :, nd:] (the stride-aware
            # binding shape-matches the tile dim exactly). max_k_tiles >= the
            # plan's natural value, so the extra tiles are simply never written.
            plan["max_k_tiles"] = max_k_tiles
            cols = torch.arange(block_table.shape[1], device=block_table.device)
            valid = cols[None, :] < nvp[lo:hi].to(block_table.device)[:, None]
            prefill = MiniMaxM3IndexerMSAPrefillMetadata(
                plan=plan,
                cu_seqlens_q=(query_start_loc[lo : hi + 1] - query_start_loc[lo]).to(
                    torch.int32
                ),
                prefix_lens=context_lens[lo:hi],
                max_query_len=int(side_qo.max()),
                page_table=block_table[lo:hi][valid].to(torch.int32),
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
            decode=decode,
            prefill_msa=prefill,
            unified_scores=unified_scores,
            max_k_tiles=max_k_tiles,
            topk_cu_seqlens_q=query_start_loc[: num_reqs + 1],
            topk_prefix_lens=context_lens,
            topk_max_query_len=common_attn_metadata.max_query_len,
            topk_num_valid_pages=num_valid_pages,
        )

    def _init_sharded_prefill(self, vllm_config: VllmConfig, device: torch.device):
        from flashinfer.experimental.cake_msa_icp_indexer.cake_indexer import (
            MSAIcpIndexerPlan,
        )

        heads = self.num_index_heads * self.index_k_tp
        scheduler = vllm_config.scheduler_config
        text_config = vllm_config.model_config.hf_text_config
        plan = MSAIcpIndexerPlan(
            world_size=self.index_k_tp,
            rank=get_tensor_model_parallel_rank(),
            num_index_heads=heads,
            page_size=PAGE_SIZE,
            init_blocks=0,
            local_blocks=text_config.index_local_blocks,
            max_decode_tokens=scheduler.max_num_seqs * self.max_decode_query_len,
            max_decode_query_len=self.max_decode_query_len,
            max_num_batched_tokens=scheduler.max_num_batched_tokens,
            max_num_seqs=scheduler.max_num_seqs,
            max_model_len=vllm_config.model_config.max_model_len,
            num_sparse_layers=1,
            device=device,
        )
        self.prefill_graphs = plan.layer_graphs()
        self.prefill_candidate_pool = torch.empty(
            plan.prefill_pool_shape(), dtype=torch.uint8, device=device
        )
        self.prefill_candidates_buffer = torch.empty(
            2,
            heads * scheduler.max_num_batched_tokens * (2 * TOPK + 1),
            dtype=torch.int32,
            device=device,
        )
        windows = cdiv(scheduler.max_num_batched_tokens, PREFILL_FMHA_CHUNK_SIZE)
        self.prefill_window_metadata = torch.empty(
            windows, 5, scheduler.max_num_seqs + 1, dtype=torch.int32, device=device
        )

    def _build_sharded(
        self, common: CommonAttentionMetadata
    ) -> MiniMaxM3IndexerMSAShardedMetadata:
        num_reqs = common.num_reqs
        num_tokens = common.num_actual_tokens
        seq_lens = common.seq_lens
        block_table = common.block_table_tensor
        num_decodes, num_prefills, nd, np = split_decodes_and_prefills(
            common, decode_threshold=self.reorder_batch_threshold, require_uniform=True
        )
        context_lens = self.context_len_buffer[:num_reqs]
        context_lens.copy_(common.compute_num_computed_tokens(), non_blocking=True)
        positions = common.positions
        assert positions is not None
        slots = common.slot_mapping
        valid_pages = self.num_valid_pages_buffer[:num_tokens]
        valid_pages.zero_()
        active = min(num_tokens, positions.numel(), slots.numel())
        valid_pages[:active].copy_((positions[:active] // PAGE_SIZE + 1).clamp_min(0))
        valid_pages[:active].masked_fill_(slots[:active] < 0, 0)
        decode = None
        if num_decodes:
            query_lens = common.query_start_loc_cpu[: num_decodes + 1].diff()
            decode_query_len = int(query_lens[0])
            assert decode_query_len > 0
            assert torch.all((query_lens == decode_query_len) | (query_lens == 0))
            decode = MiniMaxM3IndexerDecodeMetadata(
                seq_lens=seq_lens[:num_decodes],
                block_table=block_table[:num_decodes],
                max_seq_len=common.max_seq_len,
                decode_query_len=decode_query_len,
                max_decode_query_len=self.max_decode_query_len,
            )
        prefill = self._build_sharded_prefill(common, num_decodes, np) if np else None
        return MiniMaxM3IndexerMSAShardedMetadata(
            seq_lens=seq_lens,
            max_seq_len=common.max_seq_len,
            slot_mapping=slots,
            num_actual_tokens=num_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=nd,
            num_prefills=num_prefills,
            num_prefill_tokens=np,
            decode=decode,
            num_valid_pages=valid_pages,
            prefill_msa=prefill,
        )

    def _build_sharded_prefill(
        self, common: CommonAttentionMetadata, num_decodes: int, num_tokens: int
    ) -> MiniMaxM3IndexerMSAShardedPrefillMetadata:
        lo, hi = num_decodes, common.num_reqs
        starts = common.query_start_loc[lo : hi + 1]
        starts = starts - starts[0]
        query_lens = starts.diff()
        prefixes = self.context_len_buffer[lo:hi]
        block_table = common.block_table_tensor[lo:hi]
        indexer = self.prefill_graphs.indexer
        chunks = []
        for window, start in enumerate(range(0, num_tokens, PREFILL_FMHA_CHUNK_SIZE)):
            end = min(start + PREFILL_FMHA_CHUNK_SIZE, num_tokens)
            cu_q, q_lens, k_lens, prefix, cu_k = self.prefill_window_metadata[
                window, :, : hi - lo + 1
            ].unbind(0)
            q_lens, k_lens, prefix = q_lens[:-1], k_lens[:-1], prefix[:-1]
            cu_q.copy_((starts - start).clamp(0, end - start))
            q_lens.copy_(cu_q.diff())
            prefix.copy_(
                prefixes + (start - starts[:-1]).clamp_min(0).minimum(query_lens)
            )
            k_lens.copy_(prefix + q_lens)
            cu_k[0].zero_()
            torch.cumsum(k_lens, 0, dtype=torch.int32, out=cu_k[1:])
            plan = indexer.prefill_window_plan(
                window,
                q_lens,
                k_lens,
                cu_q,
                cu_k,
                prefix,
                total_q=end - start,
                max_query_len=end - start,
                max_seq_len=min(
                    indexer.max_model_len, block_table.shape[1] * PAGE_SIZE
                ),
            )
            chunks.append((start, end, block_table, plan))
        heads = self.num_index_heads * self.index_k_tp
        candidate_elements = heads * num_tokens * (2 * TOPK + 1)
        return MiniMaxM3IndexerMSAShardedPrefillMetadata(
            local_chunks=chunks,
            candidate_pool=self.prefill_candidate_pool,
            graphs=self.prefill_graphs,
            candidate_buffers=self.prefill_candidates_buffer[
                :, :candidate_elements
            ].view(2, heads, num_tokens, 2 * TOPK + 1),
        )


class MiniMaxM3IndexerMSAImpl(MiniMaxM3IndexerImpl):
    """MSA score/top-k, with CAKE scoring and exchange for sharded caches."""

    indexer_backend_cls: ClassVar[type[AttentionBackend]] = MiniMaxM3IndexerMSABackend

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None  # profiling run; caches unbound
        md = attn_metadata[self.index_cache.prefix]
        if isinstance(md, MiniMaxM3IndexerMSAShardedMetadata):
            self._forward_sharded(index_query, md)
            return None, None
        assert isinstance(md, MiniMaxM3IndexerMSAMetadata)

        num_tokens = md.num_actual_tokens
        nd = md.num_decode_tokens
        index_q = index_query[:num_tokens].view(
            -1, self.num_index_heads, self.index_head_dim
        )
        kv = self.index_cache.kv_cache
        # Shared persistent top-k output buffer; the unified top-k below writes
        # the selected block ids into buf[:num_tokens].
        buf = self.topk_indices_buffer
        assert buf is not None

        # Unified token-major score buffer [total_q, H, MAX_K_TILES]: the tile
        # dim is innermost/contiguous, so both fmha writes (native [T,H,K]) and
        # the block-iterating top-k reads hit contiguous tiles. Each side gets a
        # contiguous slice on dim 0: decode [:nd], prefill [nd:]; the kernels
        # read/write by strides. The builder allocates it once (persistent,
        # shared by all layers) and resets it to the sentinel each forward, so
        # the top-k never picks an unwritten tile.
        unified_scores = md.unified_scores
        assert unified_scores is not None

        # Decode scores -> unified[:nd] (transposed [H, nd, MK] view; the kernel
        # writes by strides). Top-k is deferred to the single unified call below.
        if md.decode is not None:
            d = md.decode
            # max_decode_query_len avoids recompiles across runtime decode sizes.
            # Fall back when the flattened Q tile gets too wide for this kernel.
            decode_score = (
                minimax_m3_index_decode_score_cutedsl
                if self.num_index_heads * d.max_decode_query_len <= 32
                else minimax_m3_index_decode_score
            )
            decode_score(
                index_q[:nd],
                kv,
                d.block_table,
                d.seq_lens,
                d.max_seq_len,
                self.init_blocks,
                self.local_blocks,
                self.num_kv_heads,
                d.decode_query_len,
                d.max_decode_query_len,
                score_out=unified_scores[:nd].transpose(0, 1),
            )

        if md.prefill_msa is not None:
            from vllm.third_party.fmha_sm100.api import _fmha_sm100

            p = md.prefill_msa
            # Index-K cache (num_blocks, 128, D) -> paged MQA (num_blocks,1,128,D).
            k_pages = kv.view(kv.shape[0], 1, PAGE_SIZE, self.index_head_dim)
            # fmha writes its max_score natively as [nnz_p, H, max_k_tiles] into
            # the prefill region (stride-aware; plan max_k_tiles forced to
            # md.max_k_tiles so the shape matches exactly -> no copy).
            _fmha_sm100(
                index_q[nd:],
                k_pages,
                k_pages,  # V placeholder; not read in OnlyScore
                p.plan,
                kv_indices=p.page_table,
                output_o=False,
                output_maxscore=True,
                sm_scale=self.scale,
                max_score=unified_scores[nd:],
            )

        # Single top-k over the unified buffer via fmha_sm100 sparse_topk_select
        # (THK, no transpose) into ``buf``. num_valid_pages drives force_end_blocks
        # (always keep each token's local block; fmha OnlyScore won't) + the -1
        # out-of-range clamp.
        from vllm.third_party.fmha_sm100.api import sparse_topk_select

        sparse_topk_select(
            unified_scores,
            self.topk_blocks,
            num_valid_pages=md.topk_num_valid_pages,
            force_begin_blocks=self.init_blocks,
            force_end_blocks=self.local_blocks,
            output=buf[:num_tokens],
            max_score_layout="THK",
        )

        # The attend reads ``buf`` directly; this return is vestigial.
        return None, None

    def set_shard_comm(self, comm: "IndexKShardComm") -> None:
        assert self.index_cache.tp_shard == comm.world
        self.index_k_comm = comm
        self.index_k_ordinal = comm.num_layers
        comm.num_layers += 1

    def _forward_sharded(
        self, index_query: torch.Tensor, md: MiniMaxM3IndexerMSAShardedMetadata
    ) -> None:
        num_tokens, nd = md.num_actual_tokens, md.num_decode_tokens
        index_q = index_query[:num_tokens].view(
            -1, self.num_index_heads * self.index_cache.tp_shard, self.index_head_dim
        )
        kv = self.index_cache.kv_cache
        buf = self.topk_indices_buffer
        assert buf is not None
        num_valid_pages = md.num_valid_pages
        if md.decode is not None:
            d = md.decode
            from flashinfer.experimental.cake_msa_icp_indexer import cake_backend

            comm = self.index_k_comm
            assert nd <= comm.max_decode_tokens
            slot_set = comm.slot_set(self.index_k_ordinal)
            tagged = (
                comm.tagged_enabled and comm.num_layers > 1 and md.prefill_msa is None
            )
            scores = comm.scores[:nd]
            cake_backend.decode_index_score_sharded(
                index_q[:nd],
                kv,
                d.block_table,
                d.seq_lens,
                scores,
                rank=comm.rank,
                world_size=comm.world,
                page_size=PAGE_SIZE,
                decode_query_len=d.decode_query_len,
                max_decode_query_len=d.max_decode_query_len,
                enable_pdl=True,
            )
            cake_backend.topk_select_publish(
                scores,
                num_valid_pages[:nd],
                comm.tagged_recv_bases if tagged else comm.recv_bases,
                comm.errors,
                slot=slot_set,
                rank=comm.rank,
                capacity=comm.max_decode_tokens,
                world_size=comm.world,
                heads_per_rank=comm.hpr,
                force_begin_blocks=self.init_blocks,
                force_end_blocks=self.local_blocks,
                spin_limit=SPIN_LIMIT,
                enable_pdl=True,
                generations=comm.publish_generations if tagged else None,
            )
            cake_backend.merge_candidates(
                (comm.tagged_recv if tagged else comm.recv)[slot_set],
                num_valid_pages[:nd],
                buf[:nd],
                comm.errors,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                spin_limit=SPIN_LIMIT,
                enable_pdl=True,
                expected_generations=(
                    comm.expected_generations[slot_set] if tagged else None
                ),
            )
            if self.index_k_ordinal == comm.num_layers - 1:
                torch._assert_async(
                    comm.errors == 0, "MiniMax M3 indexer CP exchange timed out"
                )

        if md.prefill_msa is not None:
            prefill_local_score(
                index_q[nd:],
                kv,
                md.prefill_msa,
                self.index_k_comm,
                num_valid_pages[nd:],
                buf[nd:num_tokens],
            )


def prefill_local_score(
    index_q: torch.Tensor,
    kv: torch.Tensor,
    metadata: MiniMaxM3IndexerMSAShardedPrefillMetadata,
    comm: "IndexKShardComm",
    num_valid_pages: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Score bounded query chunks, then exchange their candidates once per layer."""
    graph_step = metadata.graph_step
    if graph_step is None:
        send, received = metadata.candidate_buffers.unbind(0)
        received = received.view(comm.world, comm.hpr, index_q.shape[0], 2 * TOPK + 1)
        prefill = metadata.graphs.indexer.prepare_prefill_step(
            metadata.local_chunks,
            None,
            num_valid_pages,
            send,
            received,
            output,
            pool=metadata.candidate_pool,
        )

        def exchange():
            torch.distributed.all_to_all_single(
                received, send.view_as(received), group=comm.group.device_group
            )

        graph_step = metadata.graphs.begin_step(
            prefill=prefill,
            exchange=exchange,
            exchange_key=id(comm.group.device_group),
        )
        metadata.graph_step = graph_step
    graph_step.run(kv.data_ptr(), index_q, kv)


def resolve_index_k_shard(
    vllm_config: VllmConfig,
    total_index_heads: int,
    indexer_kv_dtype: str,
    topk_blocks: int,
) -> int:
    """Return the shard factor, or 1 for unsupported/disabled deployments."""
    mode = vllm_config.attention_config.indexer_cache_sharding
    if not mode:
        return 1
    unsupported = _unsupported_reason(
        vllm_config, total_index_heads, indexer_kv_dtype, topk_blocks
    )
    if unsupported is not None:
        logger.warning_once(
            "MiniMax M3 index-K TP sharding (%s) not applied: %s; keeping the "
            "replicated index-K cache",
            mode,
            unsupported,
        )
        return 1
    tp = get_tensor_model_parallel_world_size()
    logger.info_once(
        "MiniMax M3 index-K cache sharded over TP%d (%d of %d tokens per block "
        "per rank)",
        tp,
        PAGE_SIZE // tp,
        PAGE_SIZE,
    )
    return tp


def _unsupported_reason(
    vllm_config: VllmConfig,
    total_index_heads: int,
    indexer_kv_dtype: str,
    topk_blocks: int,
) -> str | None:
    tp = get_tensor_model_parallel_world_size()
    if tp not in SUPPORTED_TP:
        return f"needs tensor_parallel_size in {SUPPORTED_TP} (got {tp})"
    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ):
        return "needs an SM100-family (Blackwell) GPU"
    if topk_blocks != 16:
        return f"needs the MSA indexer (topk_blocks=16, got {topk_blocks})"
    if indexer_kv_dtype not in ("fp8", "fp8_e4m3"):
        return f"needs indexer_kv_dtype='fp8' (got {indexer_kv_dtype!r})"
    if vllm_config.cache_config.block_size != PAGE_SIZE:
        return f"needs block_size={PAGE_SIZE}"
    if total_index_heads != 4:
        return f"CAKE indexer CP requires 4 index heads (got {total_index_heads})"
    transfer = vllm_config.kv_transfer_config
    if (
        transfer is not None
        and transfer.is_kv_transfer_instance
        and (
            transfer.kv_connector != "SimpleCPUOffloadConnector"
            or transfer.kv_connector_module_path is not None
            or transfer.get_from_extra_config("kv_offload_backend", "cpu") != "cpu"
        )
    ):
        return "sharded index caches only support rank-local SimpleCPUOffloadConnector"
    parallel = vllm_config.parallel_config
    if parallel.use_ubatching:
        return "microbatching needs separate indexer scratch and communication state"
    if (
        parallel.decode_context_parallel_size > 1
        or parallel.prefill_context_parallel_size > 1
    ):
        return "cannot combine indexer sharding with prefill/decode context parallelism"
    spec = vllm_config.speculative_config
    method = getattr(spec, "method", None)
    if method not in _SPEC_METHODS:
        return f"speculative method {method!r} is not supported"
    if total_index_heads * max_decode_query_len(vllm_config) > 32:
        return "index heads x decode query length exceed the score kernel's 32"
    return None


class IndexKShardComm:
    """Symmetric receive slots shared by every sharded indexer layer.

    Tagged rows carry 16 score/ID pairs. Mixed and single-layer calls retain
    ready/empty rows; a tagged slot needs an intervening exchange before reuse.
    """

    def __init__(self, vllm_config: VllmConfig, total_index_heads: int) -> None:
        import torch.distributed._symmetric_memory as symm_mem
        from flashinfer.experimental.cake_msa_icp_indexer import cake_backend
        from flashinfer.experimental.cake_msa_icp_indexer.cake_indexer import (
            MSAIcpIndexerPlan,
        )

        from vllm.distributed.parallel_state import get_tp_group, in_the_same_node_as

        if not callable(cake_backend.decode_index_score_sharded) or not callable(
            MSAIcpIndexerPlan.layer_graphs
        ):
            raise ImportError(
                "Indexer CP requires FlashInfer CAKE decode_index_score_sharded "
                "and MSAIcpIndexerPlan.layer_graphs"
            )
        group = get_tp_group()
        if not all(in_the_same_node_as(group.cpu_group)):
            raise ValueError("MiniMax M3 indexer CP requires a single-node TP group")
        self.group = group
        self.world = group.world_size
        self.rank = group.rank_in_group
        self.hpr = total_index_heads // self.world
        self.max_decode_tokens = (
            vllm_config.scheduler_config.max_num_seqs
            * max_decode_query_len(vllm_config)
        )
        self.device = torch.device("cuda", torch.accelerator.current_device_index())
        shape = (
            NUM_SLOT_SETS,
            self.world,
            self.hpr,
            self.max_decode_tokens,
            2 * TOPK + 1,
        )
        rows = NUM_SLOT_SETS * total_index_heads * self.max_decode_tokens
        ready_words = rows * (2 * TOPK + 1)
        self._recv_storage = symm_mem.empty(
            (ready_words + rows * 4 * TOPK,),
            dtype=torch.int32,
            device=self.device,
        )
        self._recv_storage.zero_()
        self.recv = self._recv_storage[:ready_words].view(shape)
        self.tagged_recv = (
            self._recv_storage[ready_words:]
            .view(torch.int64)
            .view(*shape[:-1], 2 * TOPK)
        )
        self.tagged_enabled = symm_mem.get_backend(self.device) in (None, "CUDA")
        self.publish_generations = torch.zeros(
            (NUM_SLOT_SETS, total_index_heads, self.max_decode_tokens),
            dtype=torch.int32,
            device=self.device,
        )
        self.expected_generations = torch.zeros(
            (NUM_SLOT_SETS, self.hpr, self.max_decode_tokens),
            dtype=torch.int32,
            device=self.device,
        )
        # Decode writes four pages per stage, including the final partial stage.
        score_cols = min(
            MAX_K_TILES,
            cdiv(vllm_config.model_config.max_model_len, 4 * PAGE_SIZE) * 4,
        )
        self.scores = torch.empty(
            (self.max_decode_tokens, total_index_heads, score_cols),
            dtype=torch.float32,
            device=self.device,
        )
        self.recv_bases = torch.empty(self.world, dtype=torch.int64, device=self.device)
        self.tagged_recv_bases = torch.empty_like(self.recv_bases)
        self.errors = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.num_layers = 0
        torch.accelerator.synchronize(self.device)
        # Rendezvous failures must terminate the worker; peers may already be
        # inside the collective, so local fallback is no longer safe.
        handle = symm_mem.rendezvous(self._recv_storage, group.device_group.group_name)
        assert handle is not None
        self.handle = handle
        bases = [int(p) for p in handle.buffer_ptrs]
        assert bases[self.rank] == self.recv.data_ptr()
        self.recv_bases.copy_(torch.tensor(bases, dtype=torch.int64, device="cpu"))
        self.tagged_recv_bases.copy_(
            torch.tensor(
                [
                    base + ready_words * self._recv_storage.element_size()
                    for base in bases
                ],
                dtype=torch.int64,
                device="cpu",
            )
        )
        torch.accelerator.synchronize(self.device)
        logger.info_once(
            "MiniMax M3 index-K shard exchange: %.1f MiB symmetric receive slots "
            "(max %d decode tokens)",
            self._recv_storage.nbytes / (1 << 20),
            self.max_decode_tokens,
        )

    def slot_set(self, ordinal: int) -> int:
        """Adjacent layers, including the step boundary, use different slots."""
        n = self.num_layers
        if n % 2 == 0:
            return ordinal % 2
        if n % 3 == 0:
            return ordinal % 3
        return 2 if ordinal == n - 1 else ordinal % 2
