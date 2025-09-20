# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import (AttentionMetadataBuilder,
                                              CommonAttentionMetadata,
                                              split_decodes_and_prefills)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


class ShortConvAttentionBackend(AttentionBackend):

    @staticmethod
    def get_builder_cls() -> type["ShortConvAttentionMetadataBuilder"]:
        return ShortConvAttentionMetadataBuilder


@dataclass
class ShortConvAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int

    query_start_loc: torch.Tensor
    has_initial_states: torch.Tensor
    state_indices_tensor: torch.Tensor  # shape: [batch,]

    # For causal_conv1d
    nums_dict: Optional[dict] = None
    cu_seqlen: Optional[int] = None
    batch_ptr: Optional[torch.Tensor] = None
    token_chunk_offset_ptr: Optional[torch.Tensor] = None


class ShortConvAttentionMetadataBuilder(
        AttentionMetadataBuilder[ShortConvAttentionMetadata]):

    reorder_batch_threshold: ClassVar[int] = 1

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert isinstance(kv_cache_spec, MambaSpec)

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> ShortConvAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc

        state_indices_tensor = common_attn_metadata.block_table_tensor[:, 0]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold))
        has_initial_states = None
        if num_prefills > 0:
            #[batch,]
            has_initial_states_cpu = (
                common_attn_metadata.
                num_computed_tokens_cpu[num_reqs - num_prefills:num_reqs] > 0)
            has_initial_states = has_initial_states_cpu.to(
                query_start_loc.device)

        attn_metadata = ShortConvAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc=query_start_loc,
            has_initial_states=has_initial_states,
            state_indices_tensor=state_indices_tensor,
        )
        return attn_metadata
