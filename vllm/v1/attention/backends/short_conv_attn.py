# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadataBuilder)
from vllm.v1.attention.backends.utils import (CommonAttentionMetadata,
                                              split_decodes_and_prefills)


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
    state_indices_tensor: torch.Tensor
    has_initial_states: Optional[torch.Tensor]

    # For causal_conv1d
    nums_dict: Optional[dict] = None
    cu_seqlen: Optional[int] = None
    batch_ptr: Optional[torch.Tensor] = None
    token_chunk_offset_ptr: Optional[torch.Tensor] = None


class ShortConvAttentionMetadataBuilder(
        BaseMambaAttentionMetadataBuilder[ShortConvAttentionMetadata]):

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> ShortConvAttentionMetadata:
        query_start_loc = common_attn_metadata.query_start_loc
        state_indices_tensor = common_attn_metadata.block_table_tensor[:, 0]
        context_lens_tensor = common_attn_metadata.num_computed_tokens_cpu.to(
            query_start_loc.device)

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold))

        has_initial_states = None
        if num_prefills > 0:
            has_initial_states = context_lens_tensor > 0
        elif (num_decodes > 0 and num_decodes <= self.decode_cudagraph_max_bs
              and self.compilation_config.full_cuda_graph):
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_decodes)
            self.state_indices_tensor[:num_decodes].copy_(state_indices_tensor,
                                                          non_blocking=True)
            state_indices_tensor = self.state_indices_tensor[:num_input_tokens]
            state_indices_tensor[num_decodes:] = PAD_SLOT_ID

        attn_metadata = ShortConvAttentionMetadata(
            query_start_loc=query_start_loc,
            state_indices_tensor=state_indices_tensor,
            has_initial_states=has_initial_states,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
        )
        return attn_metadata
