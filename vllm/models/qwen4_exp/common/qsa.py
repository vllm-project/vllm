# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Main-attention metadata shared by the QSA platform implementations."""

from dataclasses import dataclass, replace
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec


@dataclass
class Qwen4ExpQSAMetadata(AttentionMetadata):
    num_actual_tokens: int
    max_query_len: int
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


class Qwen4ExpQSAMetadataBuilder(AttentionMetadataBuilder[Qwen4ExpQSAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    supports_update_block_table = True
    supports_draft_decode_metadata_update = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Qwen4ExpQSAMetadata:
        """Keep cache addresses shared with the current runner batch."""
        return Qwen4ExpQSAMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
        )

    def update_block_table(
        self,
        metadata: Qwen4ExpQSAMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> Qwen4ExpQSAMetadata:
        """Reuse batch bounds while replacing cache-group-specific addresses."""
        return replace(metadata, block_table=blk_table, slot_mapping=slot_mapping)

    def update_draft_decode_metadata(self, metadata: Qwen4ExpQSAMetadata) -> None:
        """Decode batch bounds stay fixed and addresses reference runner buffers."""
        pass
