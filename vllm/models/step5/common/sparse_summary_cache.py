# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CSA region summaries, kept as a compressed KV cache layer."""

from typing import Any, ClassVar

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from .csa_tail_ring import Step5CSATailRing


class Step5SparseSummaryCache:
    """FP8 summaries of each block's closed eight-token regions.

    ``mean`` is ``[blocks, block_size // 8, 256]``; blocks may be strided. A
    region is summarized once, by its eighth row, from the rows waiting in
    ``Step5CSATailRing``; this cache holds no per-request state.
    """

    def __init__(self, mean: torch.Tensor):
        self.mean = mean

    def update_decode(self, key, z, slots, positions, ring: Step5CSATailRing):
        from .ops.summary_update import update_decode

        n = slots.shape[0]
        update_decode(
            key,
            z,
            slots,
            positions,
            ring.metadata().base[:n],
            self.mean,
            ring.rows,
            ring.capacity,
        )

    def update_verify(
        self, key, z, slots, query_start_loc, seq_lens, ring: Step5CSATailRing
    ):
        from .ops.summary_update import update_verify

        update_verify(
            key,
            z,
            slots,
            query_start_loc,
            seq_lens,
            ring.metadata().base[: slots.shape[0]],
            self.mean,
            ring.rows,
            ring.capacity,
        )

    def update_prefill(self, key, z, metadata, positions, ring: Step5CSATailRing):
        from .ops.summary_update import update_prefill

        n = metadata.shape[0]
        ring_metadata = ring.metadata()
        update_prefill(
            key,
            z,
            metadata,
            positions,
            ring_metadata.base[:n],
            ring_metadata.slot_mapping[:n],
            self.mean,
            ring.rows,
            ring.capacity,
        )


class Step5SparseSummaryMetadataBuilder(AttentionMetadataBuilder):
    """Summaries are addressed through the DSA layer's own metadata."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> None:
        return None


class Step5SparseSummaryBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "STEP5_CSA_SUMMARY"

    @staticmethod
    def get_builder_cls() -> type[Step5SparseSummaryMetadataBuilder]:
        return Step5SparseSummaryMetadataBuilder


class Step5SparseSummaryCacheLayer(nn.Module, AttentionLayerBase):
    """One 256-byte FP8 summary per eight tokens, beside the DSA layer's KV.

    It is a full-attention layer with ``tokens_per_state=8`` and the main block
    size, so it shares the DSA layer's KV cache group and block IDs, and is
    prefix cached and transferred like any other page.
    """

    def __init__(
        self,
        *,
        prefix: str,
        target_impl: Any,
        static_forward_context: dict[str, Any],
    ):
        super().__init__()
        if prefix in static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_forward_context[prefix] = self
        self.layer_name = prefix
        self._target_impl = target_impl
        self.kv_cache = torch.tensor([])

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Step5SparseSummaryBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        block_size = vllm_config.cache_config.block_size
        if block_size % 8:
            raise ValueError("Step5 KV blocks must align to eight-token regions")
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=256,
            head_size_v=0,
            dtype=torch.uint8,
            tokens_per_state=8,
        )

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [blocks, 1, block_size // 8, 256] -> [blocks, block_size // 8, 256]
        self.kv_cache = kv_cache
        self._target_impl.summary_cache = Step5SparseSummaryCache(kv_cache[:, 0])
