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

``fmha_sm100`` imports are function-local so this module is import-safe on
AMD / non-SM100.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
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
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec

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


class MiniMaxM3IndexerMSAImpl(MiniMaxM3IndexerImpl):
    """Decode: CuteDSL/Triton score. Prefill: fmha_sm100 OnlyScore + top-k."""

    indexer_backend_cls: ClassVar[type[AttentionBackend]] = MiniMaxM3IndexerMSABackend

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
