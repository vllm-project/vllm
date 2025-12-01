# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID,
    CommonAttentionMetadata,
    compute_causal_conv1d_metadata,
    split_decodes_and_prefills,
)


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
    has_initial_states_p: torch.Tensor | None

    # For causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


class ShortConvAttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[ShortConvAttentionMetadata]
):
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> ShortConvAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        state_indices_tensor = common_attn_metadata.block_table_tensor[:, 0]

        # for causal_conv1d
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.reorder_batch_threshold
            )
        )

        has_initial_states_p = None
        if num_prefills > 0:
            has_initial_states_cpu = (
                common_attn_metadata.num_computed_tokens_cpu[
                    num_reqs - num_prefills : num_reqs
                ]
                > 0
            )
            has_initial_states_p = has_initial_states_cpu.to(query_start_loc.device)

            query_start_loc_p = (
                common_attn_metadata.query_start_loc[-num_prefills - 1 :]
                - num_decode_tokens
            )

            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(query_start_loc_p)
            )

        elif (
            num_decodes > 0
            and num_decodes <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_decodes)
            self.state_indices_tensor[:num_decodes].copy_(
                state_indices_tensor, non_blocking=True
            )
            state_indices_tensor = self.state_indices_tensor[:num_input_tokens]
            state_indices_tensor[num_decodes:] = PAD_SLOT_ID

        attn_metadata = ShortConvAttentionMetadata(
            query_start_loc=query_start_loc,
            state_indices_tensor=state_indices_tensor,
            has_initial_states_p=has_initial_states_p,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        return attn_metadata
