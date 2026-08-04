# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer indexer impl for MiniMax M3 on SM120/SM121.

One whole-batch ``msa_proxy_score`` call plus one ``msa_topk_select`` call, no
decode/prefill split. flashinfer's top-k takes only batch-wide scalars, so the
per-token local window and causal-range clamp are recovered in Python (see
``forward``). Imported only when ``minimax_m3_use_flashinfer_msa`` passes.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from flashinfer.msa_ops import msa_proxy_score, msa_topk_select

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerBackend,
    MiniMaxM3IndexerImpl,
    MiniMaxM3IndexerMetadata,
    MiniMaxM3IndexerMetadataBuilder,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec

# KV page size == sparse block size == flashinfer's MSA block size.
PAGE_SIZE = 128


class MiniMaxM3IndexerFlashInferBackend(MiniMaxM3IndexerBackend):
    """Indexer side-cache backend selecting the FlashInfer builder."""

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3IndexerFlashInferMetadataBuilder"]:
        return MiniMaxM3IndexerFlashInferMetadataBuilder


@dataclass
class MiniMaxM3IndexerFlashInferMetadata(MiniMaxM3IndexerMetadata):
    """Whole-batch proxy/top-k inputs; ``prefill``/``decode`` stay ``None``."""

    proxy_cu_seqlens_q: torch.Tensor | None = None  # [num_reqs + 1] int32
    proxy_seqused_k: torch.Tensor | None = None  # [num_reqs] int32
    proxy_page_table: torch.Tensor | None = None  # [num_reqs, max_pages] int32
    proxy_max_seqlen_q: int = 0
    max_k_tiles: int = 0
    num_valid_pages: torch.Tensor | None = None  # [total_q] int32
    # Trailing local window, forced into every token's own selection.
    local_blocks: int = 0


class MiniMaxM3IndexerFlashInferMetadataBuilder(MiniMaxM3IndexerMetadataBuilder):
    """Whole-batch metadata; the impl runs under the eager cudagraph break,
    so nothing here needs a stable address."""

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
        self.local_blocks = sparse_cfg.get("sparse_local_block", 0)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3IndexerFlashInferMetadata:
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

        positions = common_attn_metadata.positions
        assert positions is not None
        num_valid_pages = self.num_valid_pages_buffer[:num_tokens]
        num_valid_pages.copy_(positions[:num_tokens] // PAGE_SIZE + 1)
        # msa_topk_select wants int32. .to() is a no-op when the shared buffer
        # already is, and this runs once per step rather than once per layer.
        num_valid_pages = num_valid_pages.to(torch.int32)

        return MiniMaxM3IndexerFlashInferMetadata(
            seq_lens=seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_actual_tokens=num_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            proxy_cu_seqlens_q=query_start_loc[: num_reqs + 1].to(torch.int32),
            proxy_seqused_k=seq_lens[:num_reqs].to(torch.int32),
            proxy_page_table=block_table[:num_reqs],
            proxy_max_seqlen_q=common_attn_metadata.max_query_len,
            max_k_tiles=(common_attn_metadata.max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE,
            num_valid_pages=num_valid_pages,
            local_blocks=self.local_blocks,
        )


class MiniMaxM3IndexerFlashInferImpl(MiniMaxM3IndexerImpl):
    """One flashinfer proxy-score + top-k pair over the whole batch."""

    indexer_backend_cls: ClassVar[type[AttentionBackend]] = (
        MiniMaxM3IndexerFlashInferBackend
    )

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None  # profiling run; caches unbound
        md = attn_metadata[self.index_cache.prefix]
        assert isinstance(md, MiniMaxM3IndexerFlashInferMetadata)

        num_tokens = md.num_actual_tokens
        index_q = index_query[:num_tokens].view(
            -1, self.num_index_heads, self.index_head_dim
        )
        # The proxy takes bf16/fp16 queries only.
        if index_q.dtype == torch.float8_e4m3fn:
            index_q = index_q.to(torch.bfloat16)
        # Index-K cache viewed as a paged MQA cache.
        kv = self.index_cache.kv_cache
        k_pages = kv.view(kv.shape[0], 1, PAGE_SIZE, self.index_head_dim)
        buf = self.topk_indices_buffer
        assert buf is not None

        # The default q_offset right-aligns tokens, matching vLLM's positions.
        # Scores are unscaled; a positive scale cannot change the ranking.
        scores = msa_proxy_score(
            index_q,
            k_pages,
            md.proxy_cu_seqlens_q,
            page_table=md.proxy_page_table,
            seqused_k=md.proxy_seqused_k,
            causal=True,
            max_seqlen_q=md.proxy_max_seqlen_q,
            max_k_tiles=md.max_k_tiles,
        )

        # Per-token valid-page counts carry both the causal clamp and the
        # trailing local window into the kernel, so nothing needs repairing
        # afterwards: no selected index can exceed its own token's extent, and
        # the ascending, -1-tail-padded contract holds by construction.
        assert md.num_valid_pages is not None
        # Both clamped so msa_topk_select's force-region checks cannot raise.
        force_begin = min(self.init_blocks, md.max_k_tiles, self.topk_blocks)
        force_end = min(md.local_blocks, self.topk_blocks - force_begin)
        msa_topk_select(
            scores,
            self.topk_blocks,
            num_valid_pages=md.num_valid_pages,
            force_begin_blocks=force_begin,
            force_end_blocks=force_end,
            output=buf[:num_tokens],
        )

        # The attend reads the shared buffer directly.
        return None, None
