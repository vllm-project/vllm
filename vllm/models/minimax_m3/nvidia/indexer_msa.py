# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA (SM100/Blackwell) indexer impl for MiniMax M3.

Prefill scores with ``fmha_sm100``'s score-only (``OnlyScore``) path then selects
top-k blocks with the Triton ``minimax_m3_index_topk`` kernel -- fmha is much
faster than Triton for the wide prefill score (benchmarked ~3-5x).

Decode uses the Triton fused ``minimax_m3_index_decode`` (the same kernel the
Triton indexer impl uses): for q_len==1 it is a purpose-built vector x matrix
score (no wasted tensor-core tiles) with a 256-way split-K and a fused split-K
top-k, which beats fmha's OnlyScore (wasted MMA on a single query, 64-split cap)
by ~1.1-3.7x. It is cudagraph-safe by construction (shape-constant split grids)
and writes the shared ``topk_indices_buffer`` via ``out=``.

``fmha_sm100`` imports are function-local so this module is import-safe on
AMD / non-SM100.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerBackend,
    MiniMaxM3IndexerDecodeMetadata,
    MiniMaxM3IndexerImpl,
    MiniMaxM3IndexerMetadata,
    MiniMaxM3IndexerMetadataBuilder,
)
from vllm.models.minimax_m3.common.ops.index_topk import (
    minimax_m3_index_decode,
    minimax_m3_index_topk,
)
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


class MiniMaxM3IndexerMSAMetadataBuilder(MiniMaxM3IndexerMetadataBuilder):
    """Decode metadata is the cudagraph-safe Triton decode metadata; the prefill
    fmha plan is built eagerly (prefill batches are not captured)."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

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
        )


class MiniMaxM3IndexerMSAImpl(MiniMaxM3IndexerImpl):
    """Decode: Triton fused score+top-k. Prefill: fmha_sm100 OnlyScore + top-k."""

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
        # Both sides write into the single shared persistent topk_indices_buffer:
        # decode at [:, :nd], prefill at [:, nd:] (each kernel writes [:, :total_q]).
        buf = self.topk_indices_buffer

        decode_topk: torch.Tensor | None = None
        if md.decode is not None:
            d = md.decode
            decode_topk = minimax_m3_index_decode(
                index_q[:nd],
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

        prefill_topk: torch.Tensor | None = None
        if md.prefill_msa is not None:
            from vllm.third_party.fmha_sm100.api import _fmha_sm100

            p = md.prefill_msa
            # Index-K cache (num_blocks, 128, D) -> paged MQA (num_blocks,1,128,D).
            k_pages = kv.view(kv.shape[0], 1, PAGE_SIZE, self.index_head_dim)
            _, max_score = _fmha_sm100(
                index_q[nd:],
                k_pages,
                k_pages,  # V placeholder; not read in OnlyScore
                p.plan,
                kv_indices=p.page_table,
                output_o=False,
                output_maxscore=True,
                sm_scale=self.scale,
            )
            # Triton top-k wants [num_index_heads, num_tokens, max_block]; the
            # transpose is a strided view (the kernel reads via strides).
            out = buf[:, nd:, :] if buf is not None else None
            prefill_topk = minimax_m3_index_topk(
                max_score.transpose(1, 2),
                p.cu_seqlens_q,
                p.prefix_lens,
                p.max_query_len,
                self.topk_blocks,
                self.init_blocks,
                self.local_blocks,
                out=out,
            )

        return decode_topk, prefill_topk
