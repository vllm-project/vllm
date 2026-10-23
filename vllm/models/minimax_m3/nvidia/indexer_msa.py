# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA (SM100/Blackwell) indexer impl for MiniMax M3.

Both sides write block scores into one unified token-major buffer
``[total_q, H, max_k_tiles]``, then a single ``fmha_sm100.sparse_topk_select``
selects the top-k blocks for the whole batch (decode ``[:nd]`` + prefill
``[nd:]``) into the shared ``topk_indices_buffer``. It bounds each row by its
causal page count and force-includes the init/local blocks, so the unwritten
tail of the buffer is pre-filled with ``-inf``.

Prefill scores with ``fmha_sm100``'s score-only (``OnlyScore``) path (much faster
than Triton for the wide prefill score, benchmarked ~3-5x), writing its
``max_score`` straight into the buffer's prefill region (stride-aware, no copy).

Decode scores with CuteDSL when the flattened query tile is supported and fall
back to Triton otherwise, writing into the decode region. Only the top-k is
shared with prefill.

Sharded decode selects local candidates and merges them at each head owner.

``fmha_sm100`` imports are function-local so this module is import-safe on
AMD / non-SM100.
"""

import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig, get_current_vllm_config
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
    minimax_m3_index_score,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec

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
PREFILL_CHUNK_SIZE = 1024
PREFILL_SCORE_BYTES = 32 << 20
PREFILL_FMHA_CHUNK_SIZE = 8192
PREFILL_FMHA_SCRATCH_BYTES = 256 << 20
SPIN_LIMIT = 1 << 26
# Speculative methods whose drafter does not reuse the target's sparse layers.
_SPEC_METHODS = (None, "eagle", "eagle3", "ngram", "suffix", "draft_model")
_COMM: "IndexKShardComm | None" = None
_COMM_FAILED = False
_COMM_OWNER: weakref.ReferenceType | None = None
_COMM_GROUP: weakref.ReferenceType | None = None
_COMM_SHAPE: tuple[int, ...] | None = None


class MiniMaxM3IndexerMSABackend(MiniMaxM3IndexerBackend):
    """Indexer side-cache backend selecting the MSA builder."""

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3IndexerMSAMetadataBuilder"]:
        return MiniMaxM3IndexerMSAMetadataBuilder


@dataclass
class MiniMaxM3IndexerMSAPrefillChunk:
    start: int
    end: int
    max_seq_len: int
    block_table: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seq_lens: torch.Tensor
    prefix_lens: torch.Tensor
    plan: dict | None = None


@dataclass
class MiniMaxM3IndexerMSAPrefillMetadata:
    """MSA gather plan or bounded local-score chunks for eager prefill."""

    plan: dict | None  # fmha_sm100 PlanInfo
    cu_seqlens_q: torch.Tensor  # [num_prefills + 1] int32, rebased to 0
    prefix_lens: torch.Tensor  # [num_prefills] int32, context tokens
    max_query_len: int
    page_table: torch.Tensor | None  # flat physical pages for gathered prefill
    # Physical pages to gather; page_table indexes the gathered result.
    gather_pages: torch.Tensor | None = None
    local_chunks: list[MiniMaxM3IndexerMSAPrefillChunk] | None = None
    local_scores: torch.Tensor | None = None
    candidate_buffers: torch.Tensor | None = None
    unified: bool = False


@dataclass
class MiniMaxM3IndexerMSAMetadata(MiniMaxM3IndexerMetadata):
    """Decode reuses the inherited base ``decode`` field (the Triton decode
    metadata); ``prefill_msa`` carries the fmha score plan for the prefill side
    (the base ``prefill`` field is unused on this path)."""

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
    # Stable insert slots for this rank's index-K rows; -1 skips other ranks.
    index_slot_mapping: torch.Tensor | None = None


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
        self.page_size = kv_cache_spec.block_size
        self.index_k_tp = 1 if content is None else full // content
        self.prefill_local_score = (
            self.index_k_tp > 1
            and vllm_config.attention_config.indexer_prefill_cp
            in ("local_score", "local_score_unified")
        )
        self.prefill_unified = (
            self.prefill_local_score
            and vllm_config.attention_config.indexer_prefill_cp == "local_score_unified"
        )
        self.index_k_rank = 0
        self.index_k_slots: torch.Tensor | None = None
        if self.index_k_tp > 1:
            self.index_k_rank = get_tensor_model_parallel_rank()
            max_capture = vllm_config.compilation_config.max_cudagraph_capture_size or 0
            self.index_k_slots = torch.full(
                (
                    max(
                        vllm_config.scheduler_config.max_num_batched_tokens, max_capture
                    ),
                ),
                -1,
                dtype=torch.int64,
                device=device,
            )
        # Persistent unified score buffer [T, H, MAX_K_TILES] shared by all
        # indexer layers and reused across forwards. Stable address (required
        # for the captured decode path) + fixed tile dim so the decode score
        # kernel's shape is frozen at capture. Filled with -inf per forward in
        # build(); the valid/padding partition is a per-forward constant, so
        # every layer overwrites the same valid tiles before its own top-k.
        self.unified_scores_buffer: torch.Tensor | None = None
        self.prefill_scores_buffer: torch.Tensor | None = None
        self.prefill_candidates_buffer: torch.Tensor | None = None
        self.prefill_window_metadata: torch.Tensor | None = None
        self.prefill_work_ranges: torch.Tensor | None = None
        self.prefill_work_info: torch.Tensor | None = None
        self.prefill_workspace: torch.Tensor | None = None
        if self.prefill_local_score:
            heads = self.num_index_heads * self.index_k_tp
            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            score_elements = (
                PREFILL_FMHA_SCRATCH_BYTES // 5
                if self.index_k_tp == 2
                else PREFILL_SCORE_BYTES // 4
            )
            max_seq_len = vllm_config.model_config.max_model_len
            max_score_cols = (
                ((max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE + 15) // 16 * 16
            )
            self.prefill_scores_buffer = torch.empty(
                min(max_tokens * heads * max_score_cols, score_elements),
                dtype=torch.float32,
                device=device,
            )
            # Each active arm is a contiguous prefix, independent of the
            # previous step's token count; NCCL never exchanges capacity padding.
            self.prefill_candidates_buffer = torch.empty(
                2,
                heads * max_tokens * (2 * TOPK + 1),
                dtype=torch.int32,
                device=device,
            )
            max_reqs = vllm_config.scheduler_config.max_num_seqs
            min_chunk = 128 if self.index_k_tp == 2 else 64
            windows = (max_tokens + min_chunk - 1) // min_chunk
            self.prefill_window_metadata = torch.empty(
                windows, 5, max_reqs + 1, dtype=torch.int32, device=device
            )
            if self.index_k_tp == 2:
                self.prefill_work_ranges = torch.empty(
                    windows,
                    num_compute_units(device.index or 0),
                    dtype=torch.int64,
                    device=device,
                )
                max_chunk = min(PREFILL_FMHA_CHUNK_SIZE, max_tokens)
                self.prefill_work_info = torch.empty(
                    windows,
                    heads * ((max_chunk + 127) // 128 + max_reqs - 1),
                    dtype=torch.int64,
                    device=device,
                )
                self.prefill_workspace = torch.empty(
                    32 << 20, dtype=torch.uint8, device=device
                )
        if not self.prefill_local_score:
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
    ) -> MiniMaxM3IndexerMSAMetadata:
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

        index_slot_mapping = None
        context_lens = self.context_len_buffer[:num_reqs]
        positions = common_attn_metadata.positions
        assert positions is not None
        num_valid_pages = self.num_valid_pages_buffer[:num_tokens]
        if self.index_k_slots is not None:
            from vllm.models.minimax_m3.nvidia.ops.index_prefill import (
                prepare_indexer_metadata,
            )

            slots = common_attn_metadata.slot_mapping
            index_slot_mapping = self.index_k_slots[: slots.numel()]
            prepare_indexer_metadata(
                query_start_loc[: num_reqs + 1],
                seq_lens[:num_reqs],
                positions,
                slots,
                context_lens,
                num_valid_pages,
                index_slot_mapping,
                self.index_k_rank,
                self.index_k_tp,
                self.page_size,
            )
        else:
            context_lens.copy_(
                common_attn_metadata.compute_num_computed_tokens(), non_blocking=True
            )
            num_valid_pages.copy_(positions[:num_tokens] // PAGE_SIZE + 1)

        # Unified score buffer: a per-forward view of the persistent buffer,
        # reset to the -inf sentinel once here and shared by every layer (the
        # tile dim is the capture-time constant MAX_K_TILES).
        max_k_tiles = MAX_K_TILES
        unified_scores = None
        if self.unified_scores_buffer is not None:
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
            lo = 0 if self.prefill_unified else num_decodes
            hi = num_reqs
            side_tokens = num_tokens if self.prefill_unified else num_prefill_tokens
            qsl_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
            qo_lens_cpu = (qsl_cpu[1:] - qsl_cpu[:-1]).to(torch.int32)
            side_qo = qo_lens_cpu[lo:hi]
            plan = None
            if not self.prefill_local_score:
                from vllm.third_party.fmha_sm100.api import _fmha_sm100_plan

                kv_lens_cpu = seq_lens[:num_reqs].cpu().to(torch.int32)
                nvp = (kv_lens_cpu + PAGE_SIZE - 1) // PAGE_SIZE
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
            page_table = None
            gather_pages = None
            if not self.prefill_local_score:
                cols = torch.arange(block_table.shape[1], device=block_table.device)
                valid = cols[None, :] < nvp[lo:hi].to(block_table.device)[:, None]
                page_table = block_table[lo:hi][valid].to(torch.int32)
                if self.index_k_tp > 1:
                    gather_pages = page_table.to(torch.int64)
                    page_table = torch.arange(
                        page_table.numel(), dtype=torch.int32, device=page_table.device
                    )
            local_chunks = None
            local_scores = None
            candidate_buffers = None
            if self.prefill_local_score:
                from vllm.models.minimax_m3.nvidia.ops.index_prefill import (
                    prepare_prefill_window,
                )

                local_chunks = []
                heads = self.num_index_heads * self.index_k_tp
                max_seq_len = common_attn_metadata.max_seq_len
                blocks = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
                score_cols = (blocks + 15) // 16 * 16
                chunk_size = min(
                    PREFILL_CHUNK_SIZE,
                    max(64, PREFILL_SCORE_BYTES // (heads * score_cols * 4) // 64 * 64),
                )
                if self.index_k_tp == 2:
                    from vllm.third_party.fmha_sm100.api import (
                        fmha_index_score_sharded_plan_device,
                    )

                    chunk_size = min(
                        PREFILL_FMHA_CHUNK_SIZE,
                        max(
                            128,
                            PREFILL_FMHA_SCRATCH_BYTES
                            // (heads * score_cols * 5)
                            // 128
                            * 128,
                        ),
                    )
                assert self.prefill_window_metadata is not None
                for window, start in enumerate(range(0, side_tokens, chunk_size)):
                    end = min(start + chunk_size, side_tokens)
                    window_metadata = self.prefill_window_metadata[window]
                    prepare_prefill_window(
                        query_start_loc[lo : hi + 1],
                        seq_lens[lo:hi],
                        start,
                        end,
                        window_metadata,
                    )
                    cu_q = window_metadata[0, : hi - lo + 1]
                    q_lens = window_metadata[1, : hi - lo]
                    k_lens = window_metadata[2, : hi - lo]
                    prefixes = window_metadata[3, : hi - lo]
                    chunk_plan = None
                    if self.index_k_tp == 2:
                        assert self.prefill_work_ranges is not None
                        assert self.prefill_work_info is not None
                        assert self.prefill_workspace is not None
                        chunk_plan = fmha_index_score_sharded_plan_device(
                            q_lens,
                            k_lens,
                            cu_q,
                            window_metadata[4, : hi - lo + 1],
                            prefixes,
                            heads,
                            total_q=end - start,
                            max_query_len=end - start,
                            max_seq_len=max_seq_len,
                            packed_work_range=self.prefill_work_ranges[window],
                            packed_work_info=self.prefill_work_info[window],
                            workspace=self.prefill_workspace,
                        )
                    local_chunks.append(
                        MiniMaxM3IndexerMSAPrefillChunk(
                            start=start,
                            end=end,
                            max_seq_len=max_seq_len,
                            block_table=block_table[lo:hi],
                            cu_seqlens_q=cu_q,
                            seq_lens=k_lens,
                            prefix_lens=prefixes,
                            plan=chunk_plan,
                        )
                    )
                assert self.prefill_scores_buffer is not None
                assert self.prefill_candidates_buffer is not None
                score_rows = min(chunk_size, side_tokens)
                score_elements = score_rows * heads * score_cols
                assert score_elements <= self.prefill_scores_buffer.numel()
                local_scores = self.prefill_scores_buffer[:score_elements].view(
                    score_rows, heads, score_cols
                )
                candidate_elements = heads * side_tokens * (2 * TOPK + 1)
                candidate_buffers = self.prefill_candidates_buffer[
                    :, :candidate_elements
                ].view(2, heads, side_tokens, 2 * TOPK + 1)
            prefill = MiniMaxM3IndexerMSAPrefillMetadata(
                plan=plan,
                cu_seqlens_q=(query_start_loc[lo : hi + 1] - query_start_loc[lo]).to(
                    torch.int32
                ),
                prefix_lens=context_lens[lo:hi],
                max_query_len=int(side_qo.max()),
                page_table=page_table,
                gather_pages=gather_pages,
                local_chunks=local_chunks,
                local_scores=local_scores,
                candidate_buffers=candidate_buffers,
                unified=self.prefill_unified,
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
            index_slot_mapping=index_slot_mapping,
        )


class MiniMaxM3IndexerMSAImpl(MiniMaxM3IndexerImpl):
    """Decode: CuteDSL/Triton score. Prefill: fmha_sm100 OnlyScore + top-k."""

    indexer_backend_cls: ClassVar[type[AttentionBackend]] = MiniMaxM3IndexerMSABackend

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.index_k_comm: IndexKShardComm | None = None
        self.index_k_ordinal = -1
        self.index_k_rank = 0
        if self.index_cache.tp_shard > 1:
            comm = get_comm(
                get_current_vllm_config(),
                self.num_index_heads * self.index_cache.tp_shard,
            )
            assert comm is not None
            self.index_k_comm = comm
            self.index_k_ordinal = comm.register_layer()
            self.index_k_rank = comm.rank

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None  # profiling run; caches unbound
        md = attn_metadata[self.index_cache.prefix]
        assert isinstance(md, MiniMaxM3IndexerMSAMetadata)

        num_tokens = md.num_actual_tokens
        nd = md.num_decode_tokens
        sharded = self.index_cache.tp_shard > 1
        index_q = index_query[:num_tokens].view(
            -1, self.num_index_heads * self.index_cache.tp_shard, self.index_head_dim
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
        num_valid_pages = md.topk_num_valid_pages
        assert num_valid_pages is not None
        from vllm.third_party.fmha_sm100.api import sparse_topk_select

        unified_prefill = md.prefill_msa is not None and md.prefill_msa.unified
        if md.decode is not None and not unified_prefill:
            d = md.decode
            # max_decode_query_len avoids recompiles across runtime decode sizes.
            # Fall back when the flattened Q tile gets too wide for this kernel.
            if sharded:
                from vllm.models.minimax_m3.nvidia.ops import (
                    index_decode_score_sharded as sharded_score,
                )

                comm = self.index_k_comm
                assert comm is not None
                assert nd <= comm.max_decode_tokens
                slot_set = comm.slot_set(self.index_k_ordinal)
                tagged = (
                    comm.tagged_enabled
                    and comm.num_layers > 1
                    and md.prefill_msa is None
                )
                scores = comm.scores[:nd]
                sharded_score.minimax_m3_index_decode_score_sharded_cutedsl(
                    index_q[:nd],
                    kv,
                    d.block_table,
                    d.seq_lens,
                    comm.rank,
                    d.decode_query_len,
                    d.max_decode_query_len,
                    scores,
                    use_pdl=True,
                    prefetch_history=True,
                )
                sharded_score.minimax_m3_index_select_publish_candidates(
                    scores,
                    num_valid_pages[:nd],
                    comm.tagged_recv_bases if tagged else comm.recv_bases,
                    slot_set,
                    comm.rank,
                    comm.max_decode_tokens,
                    comm.errors,
                    world_size=comm.world,
                    heads_per_rank=comm.hpr,
                    init_blocks=self.init_blocks,
                    local_blocks=self.local_blocks,
                    spin_limit=SPIN_LIMIT,
                    use_pdl=True,
                    generations=comm.publish_generations if tagged else None,
                )
                sharded_score.minimax_m3_index_merge_candidates(
                    (comm.tagged_recv if tagged else comm.recv)[slot_set],
                    num_valid_pages[:nd],
                    buf[:nd],
                    comm.errors,
                    init_blocks=self.init_blocks,
                    local_blocks=self.local_blocks,
                    spin_limit=SPIN_LIMIT,
                    use_pdl=True,
                    expected_generations=(
                        comm.expected_generations[slot_set] if tagged else None
                    ),
                )
            else:
                from vllm.models.minimax_m3.nvidia.ops import (
                    minimax_m3_index_decode_score_cutedsl,
                )

                assert unified_scores is not None
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

        local_prefill = (
            md.prefill_msa is not None and md.prefill_msa.local_chunks is not None
        )
        if local_prefill:
            assert md.prefill_msa is not None and self.index_k_comm is not None
            start = 0 if unified_prefill else nd
            prefill_local_score(
                index_q[start:],
                kv,
                md.prefill_msa,
                self.index_k_comm,
                num_valid_pages[start:],
                buf[start:num_tokens],
                self.init_blocks,
                self.local_blocks,
            )
        elif md.prefill_msa is not None:
            from vllm.third_party.fmha_sm100.api import _fmha_sm100

            assert unified_scores is not None
            p = md.prefill_msa
            prefill_q = index_q[nd:]
            if p.gather_pages is not None:
                assert self.index_k_comm is not None
                kv = gather_prefill_pages(kv, p.gather_pages, self.index_k_comm)
                h = self.num_index_heads
                prefill_q = prefill_q[
                    :, self.index_k_rank * h : (self.index_k_rank + 1) * h
                ]
                prefill_q = prefill_q.contiguous()
            # Index-K cache (num_blocks, 128, D) -> paged MQA (num_blocks,1,128,D).
            k_pages = kv.view(kv.shape[0], 1, PAGE_SIZE, self.index_head_dim)
            # fmha writes its max_score natively as [nnz_p, H, max_k_tiles] into
            # the prefill region (stride-aware; plan max_k_tiles forced to
            # md.max_k_tiles so the shape matches exactly -> no copy).
            _fmha_sm100(
                prefill_q,
                k_pages,
                k_pages,  # V placeholder; not read in OnlyScore
                p.plan,
                kv_indices=p.page_table,
                output_o=False,
                output_maxscore=True,
                sm_scale=self.scale,
                max_score=unified_scores[nd:],
            )

        start = nd if sharded else 0
        if start < num_tokens and not local_prefill:
            assert unified_scores is not None
            sparse_topk_select(
                unified_scores[start:],
                self.topk_blocks,
                num_valid_pages=num_valid_pages[start:],
                force_begin_blocks=self.init_blocks,
                force_end_blocks=self.local_blocks,
                output=buf[start:num_tokens],
                max_score_layout="THK",
            )

        # The attend reads ``buf`` directly; this return is vestigial.
        return None, None


def prefill_local_score(
    index_q: torch.Tensor,
    kv: torch.Tensor,
    metadata: MiniMaxM3IndexerMSAPrefillMetadata,
    comm: "IndexKShardComm",
    num_valid_pages: torch.Tensor,
    output: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
) -> None:
    """Score bounded query chunks, then exchange their candidates once per layer."""
    from vllm.models.minimax_m3.nvidia.ops.index_decode_score_sharded import (
        minimax_m3_index_merge_candidates,
    )
    from vllm.third_party.fmha_sm100.api import sparse_topk_select_prefill

    assert metadata.local_chunks is not None and metadata.local_scores is not None
    assert metadata.candidate_buffers is not None
    total_tokens, heads, _ = index_q.shape
    send, received = metadata.candidate_buffers.unbind(0)
    for chunk in metadata.local_chunks:
        query = index_q[chunk.start : chunk.end]
        tokens = query.shape[0]
        scores = metadata.local_scores[:tokens]
        if chunk.plan is not None:
            from vllm.third_party.fmha_sm100.api import fmha_index_score_sharded

            fmha_index_score_sharded(
                query,
                kv,
                chunk.block_table,
                chunk.plan,
                rank=comm.rank,
                max_score=scores,
                sm_scale=1.0,
                initialize=False,
            )
        else:
            minimax_m3_index_score(
                query,
                kv,
                chunk.block_table,
                chunk.cu_seqlens_q,
                chunk.seq_lens,
                chunk.prefix_lens,
                tokens,
                chunk.max_seq_len,
                heads,
                rank=comm.rank,
                tp=comm.world,
                score_out=scores.transpose(0, 1),
            )
        chunk_send = send[:, chunk.start : chunk.end]
        valid_pages = num_valid_pages[chunk.start : chunk.end]
        sparse_topk_select_prefill(
            scores,
            valid_pages,
            chunk_send,
            force_begin_blocks=init_blocks,
            force_end_blocks=local_blocks,
        )
    send = send.view(comm.world, comm.hpr, total_tokens, 2 * TOPK + 1)
    received = received.view_as(send)
    torch.distributed.all_to_all_single(received, send, group=comm.group.device_group)
    minimax_m3_index_merge_candidates(
        received,
        num_valid_pages,
        output,
        comm.errors,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        prefill=True,
    )


def _max_decode_query_len(vllm_config: VllmConfig) -> int:
    spec = vllm_config.speculative_config
    n = getattr(spec, "num_speculative_tokens", None) if spec is not None else None
    return 1 + (n or 0)


def resolve_index_k_shard(
    vllm_config: VllmConfig,
    total_index_heads: int,
    indexer_kv_dtype: str,
    topk_blocks: int,
) -> int:
    """Return the shard factor, or 1 for unsupported/disabled deployments."""
    mode = vllm_config.attention_config.indexer_cache_sharding
    if mode == "off":
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
    if get_comm(vllm_config, total_index_heads) is None:
        return 1
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
    if total_index_heads % tp:
        return f"{total_index_heads} index heads do not split over TP{tp}"
    if vllm_config.kv_transfer_config is not None:
        return "KV transfer with sharded index caches is not supported"
    parallel = vllm_config.parallel_config
    if parallel.use_ubatching:
        return "microbatching needs separate indexer scratch and communication state"
    if (
        getattr(parallel, "decode_context_parallel_size", 1) > 1
        or getattr(parallel, "prefill_context_parallel_size", 1) > 1
    ):
        return "context parallelism is not supported"
    spec = vllm_config.speculative_config
    method = getattr(spec, "method", None)
    if method not in _SPEC_METHODS:
        return f"speculative method {method!r} is not supported"
    if total_index_heads * _max_decode_query_len(vllm_config) > 32:
        return "index heads x decode query length exceed the score kernel's 32"
    return None


class _IndexKInitError(RuntimeError):
    pass


@contextmanager
def _local_init() -> Iterator[None]:
    """Agree on local setup failures before entering peer communication."""
    from vllm.distributed.parallel_state import get_tp_group

    error = None
    try:
        yield
    except Exception as e:  # noqa: BLE001 - coordinate local setup failures
        error = e
    ok = torch.tensor([int(error is None)], dtype=torch.int32, device="cpu")
    torch.distributed.all_reduce(
        ok, op=torch.distributed.ReduceOp.MIN, group=get_tp_group().cpu_group
    )
    if not int(ok.item()):
        raise _IndexKInitError(f"Local initialization failed: {error!r}") from error


def get_comm(
    vllm_config: VllmConfig | None = None, total_index_heads: int = 0
) -> "IndexKShardComm | None":
    """Reuse initialization only within one model configuration and TP group."""
    from vllm.distributed.parallel_state import get_tp_group, in_the_same_node_as

    global _COMM, _COMM_FAILED, _COMM_OWNER, _COMM_GROUP, _COMM_SHAPE
    if vllm_config is None:
        return _COMM
    owner = vllm_config.compilation_config
    group = get_tp_group()
    shape = (
        total_index_heads,
        vllm_config.model_config.max_model_len,
        vllm_config.cache_config.block_size,
        vllm_config.scheduler_config.max_num_seqs,
        _max_decode_query_len(vllm_config),
        int(
            vllm_config.model_config.hf_text_config.sparse_attention_config[
                "sparse_index_dim"
            ]
        ),
    )
    if (
        _COMM_OWNER is None
        or _COMM_OWNER() is not owner
        or _COMM_GROUP is None
        or _COMM_GROUP() is not group
        or shape != _COMM_SHAPE
    ):
        _COMM = None
        _COMM_FAILED = False
        _COMM_OWNER, _COMM_GROUP = weakref.ref(owner), weakref.ref(group)
        _COMM_SHAPE = shape
    if _COMM is not None or _COMM_FAILED:
        return _COMM
    # Check the TP group collectively before entering a GPU IPC rendezvous.
    if not all(in_the_same_node_as(group.cpu_group)):
        logger.warning_once(
            "MiniMax M3 index-K sharding requires a single-node TP group"
        )
        _COMM_FAILED = True
        return None
    try:
        with _local_init():
            physical_id = current_platform.visible_device_id_to_physical_device_id(
                torch.accelerator.current_device_index()
            )
        physical_ids = [None] * group.world_size
        torch.distributed.all_gather_object(
            physical_ids, physical_id, group=group.cpu_group
        )
        with _local_init():
            if len(
                set(physical_ids)
            ) != group.world_size or not current_platform.is_fully_connected(
                physical_ids
            ):
                raise RuntimeError(
                    "index-K sharding requires distinct, fully NVLink-connected GPUs"
                )
        comm = IndexKShardComm(vllm_config, total_index_heads)
    except _IndexKInitError as error:
        logger.warning(
            "MiniMax M3 index-K sharding disabled: symmetric memory set-up failed (%r)",
            error,
        )
        _COMM_FAILED = True
        return None
    _COMM = comm
    return _COMM


class IndexKShardComm:
    """Symmetric receive slots shared by every sharded indexer layer.

    Tagged rows carry 16 score/ID pairs. Mixed and single-layer calls retain
    ready/empty rows; a tagged slot needs an intervening exchange before reuse.
    """

    def __init__(self, vllm_config: VllmConfig, total_index_heads: int) -> None:
        from vllm.distributed.parallel_state import get_tp_group

        with _local_init():
            import torch.distributed._symmetric_memory as symm_mem

            group = get_tp_group()
            self.group = group
            self.world = group.world_size
            self.rank = group.rank_in_group
            self.hpr = total_index_heads // self.world
            self.max_decode_tokens = (
                vllm_config.scheduler_config.max_num_seqs
                * _max_decode_query_len(vllm_config)
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
            legacy_words = rows * (2 * TOPK + 1)
            self._recv_storage = symm_mem.empty(
                (legacy_words + rows * 4 * TOPK,),
                dtype=torch.int32,
                device=self.device,
            )
            self._recv_storage.zero_()
            self.recv = self._recv_storage[:legacy_words].view(shape)
            self.tagged_recv = (
                self._recv_storage[legacy_words:]
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
            self.recv_bases = torch.empty(
                self.world, dtype=torch.int64, device=self.device
            )
            self.tagged_recv_bases = torch.empty_like(self.recv_bases)
            self.errors = torch.zeros(1, dtype=torch.int32, device=self.device)
            self.num_layers = 0
            torch.accelerator.synchronize(self.device)
        handle = symm_mem.rendezvous(self._recv_storage, group.device_group.group_name)
        with _local_init():
            assert handle is not None
            self.handle = handle
            bases = [int(p) for p in handle.buffer_ptrs]
            assert bases[self.rank] == self.recv.data_ptr()
            self.recv_bases.copy_(torch.tensor(bases, dtype=torch.int64, device="cpu"))
            self.tagged_recv_bases.copy_(
                torch.tensor(
                    [base + legacy_words * 4 for base in bases],
                    dtype=torch.int64,
                    device="cpu",
                )
            )
            torch.accelerator.synchronize(self.device)
        logger.info_once(
            "MiniMax M3 index-K shard exchange: %.1f MiB symmetric receive slots "
            "(max %d decode tokens)",
            self._recv_storage.numel() * 4 / (1 << 20),
            self.max_decode_tokens,
        )

    def register_layer(self) -> int:
        """Ordinal of a sharded indexer layer (layers run in ordinal order)."""
        self.num_layers += 1
        return self.num_layers - 1

    def slot_set(self, ordinal: int) -> int:
        """Adjacent layers, including the step boundary, use different slots."""
        n = self.num_layers
        if n % 2 == 0:
            return ordinal % 2
        if n % 3 == 0:
            return ordinal % 3
        return 2 if ordinal == n - 1 else ordinal % 2


def gather_prefill_pages(
    kv_shard: torch.Tensor, pages: torch.Tensor, comm: IndexKShardComm | None = None
) -> torch.Tensor:
    """Full ``[P, 128, D]`` index-K pages of ``pages`` from every rank's shard
    (token ``i * tp + r`` of a block comes from rank ``r``'s row ``i``)."""
    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group() if comm is None else comm.group
    tp = group.world_size
    num_pages = pages.numel()
    _, rows, dim = kv_shard.shape
    local = kv_shard.view(torch.uint8).index_select(0, pages)
    gathered = group.all_gather(local, dim=0)
    full = (
        gathered.view(tp, num_pages, rows, dim)
        .permute(1, 2, 0, 3)
        .reshape(num_pages, rows * tp, dim)
    )
    return full.view(kv_shard.dtype)
