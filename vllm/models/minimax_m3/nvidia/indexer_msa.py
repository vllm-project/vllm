# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA (SM100/Blackwell) indexer impl for MiniMax M3.

The lightning indexer's per-128-block QK max-score is computed with
``fmha_sm100``'s score-only (``OnlyScore``) path; the top-k block selection
reuses the existing Triton ``minimax_m3_index_topk`` kernel (it handles the
per-token causal window + forced init/local blocks for any ``topk``). This
mirrors how ``MiniMaxM3SparseMSAImpl`` pairs the SM100 attend with Triton.

Decode and prefill requests are split manually (the batch is decode-first) and
each side gets its own fmha score plan + ``_fmha_sm100`` call -- the public
``fmha_sm100`` wrapper's mixed-batch split + max-score merge is bypassed.

The decode side is cudagraph-replay-safe: ``__init__`` reserves persistent plan
buffers (sized for any decode batch up to ``max_num_seqs``), ``build`` fills them
in place and launches the plan kernel directly with a fixed, batch-size-only
``num_kv_splits`` (``estimate_num_kv_splits``, no device->host sync), and the run
writes a persistent ``max_score`` + the shared ``topk_indices_buffer``. Prefill
keeps the eager ``_fmha_sm100_plan`` path (prefill batches are not captured).
``fmha_sm100`` imports are function-local so this module is import-safe on
AMD / non-SM100.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerBackend,
    MiniMaxM3IndexerImpl,
    MiniMaxM3IndexerMetadata,
    MiniMaxM3IndexerMetadataBuilder,
)
from vllm.models.minimax_m3.common.ops.index_topk import minimax_m3_index_topk
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec

# Page size == sparse block size == index-K block; fmha tile id == M3 block id.
PAGE_SIZE = 128


def estimate_num_kv_splits(
    num_tokens: int,
    *,
    num_qo_heads: int,
    num_sms: int,
    context_len: int,
    kv_tile_size: int = 256,
) -> int:
    """Estimate a fixed ``num_kv_splits`` for the score-only decode plan.

    Splits the KV scan ~``num_sms / work_rows`` ways to fill the GPU, then CAPS
    it. Measured on the fp8 decode score (``benchmark_m3/indexer_score_fp8.py``,
    cudagraph timing): >16 splits with more than one work row hit a sharp
    split-KV latency cliff (up to ~9x slower); the single-row (bs=1) case has no
    cliff and benefits from more splits. The optimum is ~context-independent --
    more context just means more KV tiles per split -- so ``context_len`` only
    bounds the split count for short contexts (via the KV-tile count). The split
    depends only on the (fixed-per-cudagraph) batch size, so it is stable across
    replays. ``num_tokens`` is the number of independent Q rows for planning --
    for uniform decode (q_len <= 128) that is the request count, not
    ``requests * q_len``; ``num_qo_heads`` is the packed head count.
    """
    if num_tokens <= 0:
        return 1
    work_rows = num_tokens * num_qo_heads
    kv_iters = (context_len + kv_tile_size - 1) // kv_tile_size
    if work_rows == 1:
        # bs=1 has no split-KV cliff; fill with many splits, bounded by KV tiles.
        return max(1, min(64, kv_iters))
    # bs>=2: split ~ num_sms/work_rows to fill the GPU, capped to stay off the
    # split-KV decode-score cliff. The cliff ceiling rises with context, so the
    # cap is 16 through the ~60-100k target and 32 for longer context.
    cap = 32 if context_len > 98304 else 16
    return max(1, min(num_sms // work_rows, cap, kv_iters))


class MiniMaxM3IndexerMSABackend(MiniMaxM3IndexerBackend):
    """Indexer side-cache backend selecting the MSA builder."""

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3IndexerMSAMetadataBuilder"]:
        return MiniMaxM3IndexerMSAMetadataBuilder


@dataclass
class MiniMaxM3IndexerMSASubMetadata:
    """Per-side (decode or prefill) fmha score plan + Triton top-k inputs.

    For decode all tensors are persistent (stable-address) views; for prefill
    they are freshly allocated each step (eager).
    """

    plan: dict  # fmha_sm100 PlanInfo
    cu_seqlens_q: torch.Tensor  # [n + 1] int32, rebased to 0
    prefix_lens: torch.Tensor  # [n] int32, context tokens before this side
    max_query_len: int
    page_table: torch.Tensor | None  # flat physical page indices for this side
    max_score: torch.Tensor | None  # persistent (decode); None -> run allocates


@dataclass
class MiniMaxM3IndexerMSAMetadata(MiniMaxM3IndexerMetadata):
    """Indexer metadata with separate decode/prefill fmha score plans."""

    decode_metadata: MiniMaxM3IndexerMSASubMetadata | None = None
    prefill_metadata: MiniMaxM3IndexerMSASubMetadata | None = None


class MiniMaxM3IndexerMSAMetadataBuilder(MiniMaxM3IndexerMetadataBuilder):
    """Decode plan is cudagraph-replay-safe (persistent buffers + a fixed,
    batch-size-only ``num_kv_splits``); the prefill plan stays eager."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        from vllm.third_party.fmha_sm100.api import (
            _compute_pack_factor,
            _get_num_cta,
        )

        H = self.num_index_heads
        dql = self.reorder_batch_threshold  # uniform decode query length
        self._pack_factor = _compute_pack_factor(dql, H, 1)
        self._packed_heads = max(1, H // self._pack_factor)
        self._num_ctas = _get_num_cta(device)
        self._ctx_len = envs.VLLM_M3_INDEXER_CONTEXT_LEN

        max_model_len = vllm_config.model_config.max_model_len
        pages_per_req = (max_model_len + PAGE_SIZE - 1) // PAGE_SIZE
        self._max_k_tiles = ((pages_per_req + 127) // 128) * 128

        max_reqs = vllm_config.scheduler_config.max_num_seqs

        def splits(reqs: int) -> int:
            return estimate_num_kv_splits(
                reqs,
                num_qo_heads=self._packed_heads,
                num_sms=self._num_ctas,
                context_len=self._ctx_len,
            )

        # build() fills these buffers for ANY decode size in [1, max_reqs] (eager
        # too, not just captured sizes), and num_kv_splits is largest at small
        # batch -- so size over a scan that covers both endpoints and the cliff
        # where work_rows exceeds 4096 and splits drops. Decode emits 1 qo tile/
        # req: plan work items = reqs*packed_heads*splits; workspaces scale with
        # the packed total qo length = reqs*dql*pack_factor.
        cand = {1, max_reqs}
        r = 1
        while r < max_reqs:
            cand.add(r)
            r *= 2
        work_bound = 2 * max(r * self._packed_heads * splits(r) for r in cand) + 4096
        wo_size = max(
            (r * dql * self._pack_factor) * splits(r) * self._packed_heads * 128
            for r in cand
        )
        wl_size = max(
            splits(r) * (r * dql * self._pack_factor) * self._packed_heads for r in cand
        )
        max_total_qo = max_reqs * dql * self._pack_factor

        def seg(n: int) -> torch.Tensor:
            return torch.empty(n, dtype=torch.int32, device=device)

        # Segment buffers, computed on-GPU per build() (no host staging needed).
        self._qo_seg_off = seg(max_reqs + 1)
        self._kv_seg_off = seg(max_reqs + 1)
        self._kv_page_indptr = seg(max_reqs + 1)
        self._qo_offset = seg(max_reqs)
        self._kv_seg_lens = seg(max_reqs)
        self._qo_seg_lens = seg(max_reqs)
        # Worklist + workspaces written by the plan/run kernels.
        self._packed_work_range = torch.empty(
            self._num_ctas, dtype=torch.int64, device=device
        )
        self._packed_work_info = torch.empty(
            work_bound, dtype=torch.int64, device=device
        )
        self._kv_tile_begin = torch.empty(work_bound, dtype=torch.int32, device=device)
        self._kv_tile_end = torch.empty(work_bound, dtype=torch.int32, device=device)
        self._kv_split = torch.empty(work_bound, dtype=torch.int32, device=device)
        self._num_kv_splits_per_row = torch.empty(
            max_total_qo, dtype=torch.int32, device=device
        )
        self._workspace_o = torch.empty(wo_size, dtype=torch.bfloat16, device=device)
        self._workspace_lse = torch.empty(wl_size, dtype=torch.float32, device=device)
        # Dedicated (not the global _alloc_workspace_buf cache): a captured decode
        # graph bakes this address, and a larger fmha call elsewhere (prefill /
        # main sparse attention) would otherwise realloc the shared buffer and
        # move it out from under the graph -> IMA.
        self._cute_workspace = torch.empty(
            32 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        # Run outputs (decode). max_score is 1-D backing so each build takes a
        # CONTIGUOUS [H, max_k_tiles, nnz_qo] view (the fmha kernel assumes a
        # contiguous layout; slicing the token dim of a 3-D buffer would not be).
        self._max_score = torch.empty(
            H * self._max_k_tiles * max_reqs * dql, dtype=torch.float32, device=device
        )
        self._decode_kv_indices = torch.empty(
            max_reqs * pages_per_req, dtype=torch.int32, device=device
        )
        self._cu_seqlens_q = torch.empty(max_reqs + 1, dtype=torch.int32, device=device)

    def _plan_decode(
        self,
        qsl_dec: torch.Tensor,
        seq_lens_dec: torch.Tensor,
        num_decode_tokens: int,
        num_kv_splits: int,
    ) -> dict:
        """Fill the persistent plan buffers on-GPU and launch the plan kernel
        directly, returning a PlanInfo over those (stable-address) buffers.

        ``qsl_dec`` is the decode slice of ``query_start_loc`` ([n+1]); the
        per-request q/kv lengths and their cumsums are derived with torch (no
        host sync) and written into the persistent segment buffers in place.
        """
        from vllm.third_party.fmha_sm100.api import _call_plan, _make_plan_info

        n = seq_lens_dec.shape[0]
        pf = self._pack_factor
        hq = self._packed_heads
        # Uniform decode -> packed total/max known from host scalars (no sync).
        total_qo_len = num_decode_tokens * pf
        max_qo_len = (num_decode_tokens // n) * pf

        qo = (qsl_dec[1:] - qsl_dec[:-1]).to(torch.int32)
        kv = seq_lens_dec.to(torch.int32)
        packed_qo = qo * pf
        nvp = (kv + PAGE_SIZE - 1) // PAGE_SIZE
        self._qo_seg_lens[:n] = packed_qo
        self._kv_seg_lens[:n] = kv
        self._qo_offset[:n] = kv - qo  # bottom-right causal
        self._qo_seg_off[0] = 0
        self._kv_seg_off[0] = 0
        self._kv_page_indptr[0] = 0
        torch.cumsum(packed_qo, 0, out=self._qo_seg_off[1 : n + 1])
        torch.cumsum(kv, 0, out=self._kv_seg_off[1 : n + 1])
        torch.cumsum(nvp, 0, out=self._kv_page_indptr[1 : n + 1])

        qo_seg_off = self._qo_seg_off[: n + 1]
        kv_seg_off = self._kv_seg_off[: n + 1]
        kv_page_indptr = self._kv_page_indptr[: n + 1]
        qo_offset_gpu = self._qo_offset[:n]
        kv_seg_lens = self._kv_seg_lens[:n]
        qo_seg_lens = self._qo_seg_lens[:n]

        split = num_kv_splits > 1
        kv_tile_begin = self._kv_tile_begin if split else None
        kv_tile_end = self._kv_tile_end if split else None
        kv_split = self._kv_split if split else None
        num_kv_splits_per_row = self._num_kv_splits_per_row if split else None
        workspace_o = self._workspace_o if split else None
        workspace_lse = self._workspace_lse if split else None
        lse_total_size = num_kv_splits * total_qo_len * hq if split else 0

        _call_plan(
            qo_seg_off,
            qo_seg_lens,
            kv_seg_lens,
            self._packed_work_range,
            self._packed_work_info,
            128,  # qo_tile_size (packed qo len <= 128)
            256,  # kv_tile_size
            hq,
            self._num_ctas,
            True,  # causal
            qo_offset_gpu,
            num_kv_splits,
            kv_tile_begin,
            kv_tile_end,
            kv_split,
            0,  # chunk_size (deterministic: no auto split estimate)
            None,  # out_max_sm_cost
            num_kv_splits_per_row,
            workspace_lse,
            lse_total_size,
            pf,
            torch.cuda.current_stream().cuda_stream,
        )
        return _make_plan_info(
            packed_work_range=self._packed_work_range,
            packed_work_info=self._packed_work_info,
            kv_tile_begin_indices=kv_tile_begin,
            kv_tile_end_indices=kv_tile_end,
            kv_split_indices=kv_split,
            num_kv_splits=num_kv_splits,
            workspace_o=workspace_o,
            workspace_lse=workspace_lse,
            max_qo_len=max_qo_len,
            predicted_speedup=1.0,
            num_kv_splits_per_row=num_kv_splits_per_row,
            qo_segment_offsets=qo_seg_off,
            kv_segment_offsets=kv_seg_off,
            kv_page_indptr=kv_page_indptr,
            max_k_tiles=self._max_k_tiles,
            qo_segment_lens=qo_seg_lens,
            kv_segment_lens=kv_seg_lens,
            qo_offset=qo_offset_gpu,
            pack_factor=pf,
            orig_num_qo_heads=self.num_index_heads,
            qo_len_uniform=True,
            cute_workspace_buffer=self._cute_workspace,
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

        # Context (prefix) lengths into the stable buffer; sliced per side below.
        context_lens = self.context_len_buffer[:num_reqs]
        context_lens.copy_(
            common_attn_metadata.compute_num_computed_tokens(), non_blocking=True
        )

        cols = torch.arange(block_table.shape[1], device=block_table.device)

        decode_metadata: MiniMaxM3IndexerMSASubMetadata | None = None
        if num_decodes > 0:
            num_kv_splits = estimate_num_kv_splits(
                num_decodes,
                num_qo_heads=self._packed_heads,
                num_sms=self._num_ctas,
                context_len=self._ctx_len,
            )
            # Fills self._kv_page_indptr (per-request page offsets) on-GPU.
            plan = self._plan_decode(
                query_start_loc[: num_decodes + 1],
                seq_lens[:num_decodes],
                num_decode_tokens,
                num_kv_splits,
            )
            # Flat request-major decode page table, built on-GPU (no host sync):
            # scatter block_table[b, :nvp[b]] to _decode_kv_indices[indptr[b] + j].
            # The full buffer is passed to the run; it bounds reads via the page
            # indptr, so no host page count is needed.
            nvp = (seq_lens[:num_decodes] + PAGE_SIZE - 1) // PAGE_SIZE
            valid = cols[None, :] < nvp[:, None]
            dest = self._kv_page_indptr[:num_decodes, None] + cols[None, :]
            self._decode_kv_indices[dest[valid]] = block_table[:num_decodes][valid].to(
                torch.int32
            )
            self._cu_seqlens_q[: num_decodes + 1].copy_(
                query_start_loc[: num_decodes + 1] - query_start_loc[0],
                non_blocking=True,
            )
            decode_metadata = MiniMaxM3IndexerMSASubMetadata(
                plan=plan,
                cu_seqlens_q=self._cu_seqlens_q[: num_decodes + 1],
                prefix_lens=context_lens[:num_decodes],
                max_query_len=num_decode_tokens // num_decodes,  # uniform decode qlen
                page_table=self._decode_kv_indices,
                max_score=self._max_score[
                    : self.num_index_heads * self._max_k_tiles * num_decode_tokens
                ].view(self.num_index_heads, self._max_k_tiles, num_decode_tokens),
            )

        prefill_metadata: MiniMaxM3IndexerMSASubMetadata | None = None
        if num_prefills > 0:
            # Prefill is eager (not captured); the host lengths it needs (and the
            # _fmha_sm100_plan .tolist() inside) make the D->H sync unavoidable
            # here, but it never runs on the captured decode path.
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
            valid = cols[None, :] < nvp[lo:hi].to(block_table.device)[:, None]
            prefill_metadata = MiniMaxM3IndexerMSASubMetadata(
                plan=plan,
                cu_seqlens_q=(query_start_loc[lo : hi + 1] - query_start_loc[lo]).to(
                    torch.int32
                ),
                prefix_lens=context_lens[lo:hi],
                max_query_len=int(side_qo.max()),
                page_table=block_table[lo:hi][valid].to(torch.int32),
                max_score=None,
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
            decode_metadata=decode_metadata,
            prefill_metadata=prefill_metadata,
        )


class MiniMaxM3IndexerMSAImpl(MiniMaxM3IndexerImpl):
    """fmha_sm100 OnlyScore for the per-block scores; Triton top-k selection."""

    indexer_backend_cls: ClassVar[type[AttentionBackend]] = MiniMaxM3IndexerMSABackend

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        from vllm.third_party.fmha_sm100.api import _fmha_sm100

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
        # Index-K cache (num_blocks, 128, D) -> paged MQA (num_blocks, 1, 128, D).
        kv = self.index_cache.kv_cache
        k_pages = kv.view(kv.shape[0], 1, PAGE_SIZE, self.index_head_dim)

        def score_topk(
            meta: MiniMaxM3IndexerMSASubMetadata,
            query: torch.Tensor,
            topk_out: torch.Tensor | None,
        ) -> torch.Tensor:
            # Persistent max_score (decode) holds stale tiles; reset before the
            # run writes only the in-range ones (top-k masks the rest anyway).
            if meta.max_score is not None:
                meta.max_score.fill_(-float("inf"))
            # OnlyScore -> max_score [num_index_heads, max_k_tiles, num_tokens].
            _, max_score = _fmha_sm100(
                query,
                k_pages,
                k_pages,  # V placeholder; not read in OnlyScore
                meta.plan,
                kv_indices=meta.page_table,
                max_score=meta.max_score,
                output_o=False,
                output_maxscore=True,
                sm_scale=self.scale,
            )
            # Triton top-k wants [num_index_heads, num_tokens, max_block]; the
            # transpose is a strided view (the kernel reads via strides). One
            # 128-token KV tile == one M3 sparse block. ``topk_out`` is the slice
            # of the shared persistent buffer this side writes into (stable
            # address); None -> the kernel allocates fresh.
            return minimax_m3_index_topk(
                max_score.transpose(1, 2),
                meta.cu_seqlens_q,
                meta.prefix_lens,
                meta.max_query_len,
                self.topk_blocks,
                self.init_blocks,
                self.local_blocks,
                out=topk_out,
            )

        # Both sides write into the single persistent topk_indices_buffer (no
        # fresh allocations): decode tokens at [:, :nd], prefill at [:, nd:]
        # (the index_topk out= writes out[:, :total_q] for each side).
        buf = self.topk_indices_buffer

        def run_decode() -> torch.Tensor | None:
            if md.decode_metadata is None:
                return None
            return score_topk(md.decode_metadata, index_q[:nd], buf)

        def run_prefill() -> torch.Tensor | None:
            if md.prefill_metadata is None:
                return None
            out = buf[:, nd:, :] if buf is not None else None
            return score_topk(md.prefill_metadata, index_q[nd:], out)

        return run_decode(), run_prefill()
