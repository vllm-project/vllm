# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA (SM100/Blackwell) indexer impl for MiniMax M3.

The lightning indexer's per-128-block QK max-score is computed with
``fmha_sm100``'s score-only (``OnlyScore``) path; the top-k block selection
reuses the existing Triton ``minimax_m3_index_topk`` kernel (it handles the
per-token causal window + forced init/local blocks for any ``topk``). This
mirrors how ``MiniMaxM3SparseMSAImpl`` pairs the SM100 attend with Triton.

Decode and prefill requests are split manually (the batch is decode-first) and
each side gets its own ``_fmha_sm100_plan`` / ``_fmha_sm100`` call -- the public
``fmha_sm100`` wrapper's mixed-batch split + max-score merge is bypassed.

The plan is built eagerly at metadata-build time (it ``.tolist()``s the segment
lengths and allocates fresh workspaces), so this builder is not
cudagraph-replay-safe -- it declares ``AttentionCGSupport.NEVER`` and the
attention runs eager (broken out of the graph by ``_run_attention``).
``fmha_sm100`` imports are function-local so this module is import-safe on
AMD / non-SM100.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

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

# Page size == sparse block size == index-K block; fmha tile id == M3 block id.
PAGE_SIZE = 128


class MiniMaxM3IndexerMSABackend(MiniMaxM3IndexerBackend):
    """Indexer side-cache backend selecting the MSA builder."""

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3IndexerMSAMetadataBuilder"]:
        return MiniMaxM3IndexerMSAMetadataBuilder


@dataclass
class MiniMaxM3IndexerMSASubMetadata:
    """Per-side (decode or prefill) fmha score plan + Triton top-k inputs."""

    plan: dict  # _fmha_sm100_plan PlanInfo
    cu_seqlens_q: torch.Tensor  # [n + 1] int32, rebased to 0
    prefix_lens: torch.Tensor  # [n] int32, context tokens before this side
    max_query_len: int


@dataclass
class MiniMaxM3IndexerMSAMetadata(MiniMaxM3IndexerMetadata):
    """Indexer metadata with separate decode/prefill fmha score plans."""

    # Whole-batch flattened physical page table [sum_pages] int32, request-major
    # (decode reqs first); split at ``decode_pages`` for the two _fmha_sm100 runs.
    kv_indices: torch.Tensor | None = None
    decode_pages: int = 0
    decode_metadata: MiniMaxM3IndexerMSASubMetadata | None = None
    prefill_metadata: MiniMaxM3IndexerMSASubMetadata | None = None


class MiniMaxM3IndexerMSAMetadataBuilder(MiniMaxM3IndexerMetadataBuilder):
    """Builds separate decode/prefill fmha_sm100 score plans (eager only)."""

    # Plans/workspaces are allocated fresh each build() and the run reads
    # per-request metadata off the plan, so this is not cudagraph-replay-safe.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3IndexerMSAMetadata:
        from vllm.third_party.fmha_sm100.api import _fmha_sm100_plan

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

        # Exact per-request lengths (host): the plan .tolist()s these and uses the
        # exact KV lengths for run-time causal masking.
        qsl_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
        qo_lens_cpu = (qsl_cpu[1:] - qsl_cpu[:-1]).to(torch.int32)
        kv_lens_cpu = seq_lens[:num_reqs].cpu().to(torch.int32)
        nvp = (kv_lens_cpu + PAGE_SIZE - 1) // PAGE_SIZE

        # Whole-batch request-major flat page table; decode pages come first.
        max_blocks = block_table.shape[1]
        cols = torch.arange(max_blocks, device=block_table.device)
        valid = cols[None, :] < nvp.to(block_table.device)[:, None]
        kv_indices = block_table[valid].to(torch.int32)
        decode_pages = int(nvp[:num_decodes].sum())

        def build_side_metadata(
            lo: int, hi: int
        ) -> MiniMaxM3IndexerMSASubMetadata | None:
            if hi <= lo:
                return None
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
            cu = query_start_loc[lo : hi + 1] - query_start_loc[lo]
            return MiniMaxM3IndexerMSASubMetadata(
                plan=plan,
                cu_seqlens_q=cu.to(torch.int32),
                prefix_lens=context_lens[lo:hi],
                max_query_len=int(side_qo.max()),
            )

        # Decode requests precede prefill requests in the reordered batch.
        decode_metadata = build_side_metadata(0, num_decodes)
        prefill_metadata = build_side_metadata(num_decodes, num_reqs)

        return MiniMaxM3IndexerMSAMetadata(
            seq_lens=seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_actual_tokens=num_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            kv_indices=kv_indices,
            decode_pages=decode_pages,
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
        kv_indices = md.kv_indices

        def score_topk(
            meta: MiniMaxM3IndexerMSASubMetadata,
            query: torch.Tensor,
            page_table: torch.Tensor | None,
        ) -> torch.Tensor:
            # OnlyScore -> max_score [num_index_heads, max_k_tiles, num_tokens].
            _, max_score = _fmha_sm100(
                query,
                k_pages,
                k_pages,  # V placeholder; not read in OnlyScore
                meta.plan,
                kv_indices=page_table,
                output_o=False,
                output_maxscore=True,
                sm_scale=self.scale,
            )
            # Triton top-k wants [num_index_heads, num_tokens, max_block]; the
            # transpose is a strided view (the kernel reads via strides). One
            # 128-token KV tile == one M3 sparse block.
            return minimax_m3_index_topk(
                max_score.transpose(1, 2),
                meta.cu_seqlens_q,
                meta.prefix_lens,
                meta.max_query_len,
                self.topk_blocks,
                self.init_blocks,
                self.local_blocks,
            )

        def run_decode() -> torch.Tensor | None:
            decode_metadata = md.decode_metadata
            if decode_metadata is None:
                return None
            decode_pages = (
                kv_indices[: md.decode_pages] if kv_indices is not None else None
            )
            return score_topk(decode_metadata, index_q[:nd], decode_pages)

        def run_prefill() -> torch.Tensor | None:
            prefill_metadata = md.prefill_metadata
            if prefill_metadata is None:
                return None
            prefill_pages = (
                kv_indices[md.decode_pages :] if kv_indices is not None else None
            )
            return score_topk(prefill_metadata, index_q[nd:], prefill_pages)

        return run_decode(), run_prefill()
