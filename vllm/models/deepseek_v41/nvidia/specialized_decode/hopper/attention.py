# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 attention layer with SM90 decode attention.

It is the FlashMLA layer with one change: decode tokens attend through the
CuTe DSL sparse MLA (`ampere/sparse_mla.py`) instead of FlashMLA's decode
kernel. Each decode token attends to its window keys (the SWA cache) and to
the compressed keys its indexer selected (the compressed cache), both stored
as 584-byte V4 records, which the kernel reads in place
(`v4_records=True`). The kernel leaves the output's RoPE dims rotated, as
FlashMLA does, because the stock FP8 output projection applies the inverse
RoPE.

FlashMLA's decode kernel was the only reason to pad the queries to 64 heads:
its sparse prefill takes 8, 16 or 32 heads on SM90 as they are. For those
head counts the queries are no longer padded; otherwise the decode kernel
reads the padded queries' first `n_local_heads` heads. The SWA metadata
builder skips FlashMLA's decode tile scheduler, which nothing here reads.
The layer's indexer is `indexer.py`'s, whose candidate consumers score only
their candidate keys on decode steps. Prefill context parallelism keeps the
stock layer (the model selects it).
"""

from typing import TYPE_CHECKING, Any

import torch

from vllm.models.deepseek_v41.attention import (
    DeepseekV4Indexer,
    DeepseekV4IndexerCache,
)
from vllm.models.deepseek_v41.common.ops import compute_global_topk_indices_and_lens
from vllm.models.deepseek_v41.nvidia.flashmla import (
    DeepseekSparseSWAFlashMLABackend,
    DeepseekSparseSWAFlashMLAMetadataBuilder,
    DeepseekV4FlashMLAAttention,
)
from vllm.models.deepseek_v41.sparse_mla import DeepseekV4FlashMLAMetadata

from ..ampere.sparse_mla import sparse_mla
from .indexer import HopperIndexer, HopperIndexerCache

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

# Query heads per rank that FlashMLA's SM90 sparse prefill takes unpadded.
UNPADDED_PREFILL_HEADS = (8, 16, 32)


class HopperSWAMetadataBuilder(DeepseekSparseSWAFlashMLAMetadataBuilder):
    """SWA metadata without FlashMLA's decode tile-scheduler plans."""

    def build_tile_scheduler(self, num_decode_tokens: int) -> dict[str, Any]:
        """No plans: the base builds none for a step without decode tokens."""
        return super().build_tile_scheduler(0)


class HopperSWABackend(DeepseekSparseSWAFlashMLABackend):
    """The SWA cache backend of the SM90 layer."""

    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type[HopperSWAMetadataBuilder]:
        return HopperSWAMetadataBuilder


class DeepseekV41HopperAttention(DeepseekV4FlashMLAAttention):
    """DeepSeek-V4.1 attention on SM90 with CuTe DSL decode attention."""

    swa_backend_cls = HopperSWABackend

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        """No padding where FlashMLA's sparse prefill takes the heads as they
        are; otherwise FlashMLA's padding, which the prefill still needs."""
        if num_heads in UNPADDED_PREFILL_HEADS:
            return num_heads
        return super().get_padded_num_q_heads(num_heads)

    def _indexer_cls(
        self, k_cache: DeepseekV4IndexerCache | None
    ) -> type[DeepseekV4Indexer]:
        """The indexer with compact candidate consumers on decode steps."""
        return HopperIndexer

    def _indexer_cache_cls(self) -> type[DeepseekV4IndexerCache]:
        """The index K cache with the SM90 indexer metadata."""
        return HopperIndexerCache

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        """Sparse MLA of the decode tokens over their window and selected
        keys, the output's RoPE dims left rotated."""
        num_tokens = swa_metadata.num_decode_tokens
        heads = self.n_local_heads
        assert swa_metadata.decode_swa_indices is not None
        assert swa_metadata.decode_swa_lens is not None
        compressed_indices = compressed_lens = None
        if not swa_only:
            # Local indices filled by the index-source layer's indexer.
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            assert swa_metadata.token_to_req_indices is not None
            assert self.topk_indices_buffer is not None
            compressed_indices, compressed_lens = compute_global_topk_indices_and_lens(
                self.topk_indices_buffer[:num_tokens],
                swa_metadata.token_to_req_indices,
                attn_metadata.block_table[: swa_metadata.num_decodes],
                attn_metadata.block_size // self.compress_ratio,
                swa_metadata.is_valid_token[:num_tokens],
            )
        sparse_mla(
            q[:, :heads],
            self.swa_cache_layer.kv_cache,
            swa_metadata.decode_swa_indices.view(num_tokens, -1),
            swa_metadata.decode_swa_lens,
            kv_cache,
            compressed_indices,
            compressed_lens,
            self.attn_sink[:heads],
            self.scale,
            None,
            None,
            output[:, :heads],
            v4_records=True,
        )
