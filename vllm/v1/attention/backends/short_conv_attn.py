# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID,
    CommonAttentionMetadata,
    compute_causal_conv1d_metadata,
    split_decodes_and_prefills,
)

from vllm.config import VllmConfig
from vllm.utils import cdiv
from vllm.v1.kv_cache_interface import AttentionSpec

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
    has_initial_states_p: Optional[torch.Tensor]

    # For prefix caching
    block_idx_last_scheduled_token: torch.Tensor  # shape: [batch,]
    block_idx_first_scheduled_token_p: torch.Tensor  # shape: [batch,]
    block_idx_last_computed_token: torch.Tensor  # shape: [batch,]
    num_computed_tokens_p: torch.Tensor  # shape: [batch,]
    query_start_loc_p: torch.Tensor
    seq_lens: torch.Tensor
        
    #WILL Check its necessity
    prep_initial_states: bool
    chunk_size: int
    seq_idx_p: Optional[torch.Tensor]
    cu_chunk_seqlen_p: Optional[torch.Tensor]
    last_chunk_indices_p: Optional[torch.Tensor]
    

    # For causal_conv1d
    nums_dict: Optional[dict] = None
    batch_ptr: Optional[torch.Tensor] = None
    token_chunk_offset_ptr: Optional[torch.Tensor] = None


class ShortConvAttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[ShortConvAttentionMetadata]
):
    
    # For prefix caching    
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.chunk_size = vllm_config.model_config.get_mamba_chunk_size()
        # assert self.chunk_size is not None, (
        #     "chunk_size needs to be set in the model config"
        # )
        if self.vllm_config.cache_config.enable_prefix_caching:
            self.state_indices_tensor = torch.empty(
                (
                    self.decode_cudagraph_max_bs,
                    cdiv(
                        vllm_config.model_config.max_model_len, kv_cache_spec.block_size
                    ),
                ),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_last_scheduled_token = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_last_computed_token = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> ShortConvAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc

        # for causal_conv1d
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None

        seq_lens = common_attn_metadata.seq_lens 
        query_start_loc_p = None
        seq_idx_p = None
        cu_chunk_seqlen_p = None
        last_chunk_indices_p = None

        # Need flags to indicate if there are initial states
        has_initial_states_p = None
        prep_initial_states = False   

        num_computed_tokens, num_computed_tokens_p = None, None
        block_idx_first_scheduled_token = None
        block_idx_first_scheduled_token_p = None             

        if self.vllm_config.cache_config.enable_prefix_caching:
            # Return a tensor of shape (#requests, #max blocks)
            state_indices_tensor = common_attn_metadata.block_table_tensor

            mamba_block_size = self.kv_cache_spec.block_size
            num_computed_tokens = common_attn_metadata.num_computed_tokens_cpu.to(
                self.device
            )
           
            block_idx_last_computed_token = (
                cdiv(num_computed_tokens, mamba_block_size) - 1
            )

            block_idx_first_scheduled_token = (
                cdiv(num_computed_tokens, mamba_block_size)
            )

            block_idx_last_scheduled_token = (
                cdiv(common_attn_metadata.seq_lens, mamba_block_size) - 1
            )
 
            block_idx_last_computed_token = block_idx_last_computed_token.clamp(min=0)

        else:
            # Always return just a single block per each request:
            state_indices_tensor = common_attn_metadata.block_table_tensor[:, 0]
            # Additional cache-related varaiables:
            block_idx_last_scheduled_token = None
            block_idx_last_computed_token = None

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.reorder_batch_threshold
            )
        )

        if num_prefills > 0:
            has_initial_states_cpu = (
                common_attn_metadata.num_computed_tokens_cpu[
                    num_reqs - num_prefills : num_reqs
                ]
                > 0
            )
            prep_initial_states = torch.any(has_initial_states_cpu).item()
            has_initial_states_p = has_initial_states_cpu.to(
                common_attn_metadata.query_start_loc.device
            )

            query_start_loc_p = (
                common_attn_metadata.query_start_loc[-num_prefills - 1 :]
                - num_decode_tokens
            )

            if self.vllm_config.cache_config.enable_prefix_caching:
                assert num_computed_tokens is not None
                num_computed_tokens_p = num_computed_tokens[
                    num_reqs - num_prefills : num_reqs
                ]
                assert block_idx_first_scheduled_token is not None
                block_idx_first_scheduled_token_p = block_idx_first_scheduled_token[
                    num_reqs - num_prefills : num_reqs
                ]
            num_computed_tokens_p_cpu = common_attn_metadata.num_computed_tokens_cpu[
                num_reqs - num_prefills : num_reqs
            ]
            query_start_loc_p_cpu = (
                common_attn_metadata.query_start_loc_cpu[-num_prefills - 1 :]
                - num_decode_tokens
            )
            
            #WILL - need this?
            cu_chunk_seqlen = []
            seq_idx = []
            last_chunk_indices = []
            seqlen_pos = 0
            for req_idx in range(num_prefills):
                this_num_computed = num_computed_tokens_p_cpu[req_idx].item()
                this_new_tokens = (
                    query_start_loc_p_cpu[req_idx + 1].item()
                    - query_start_loc_p_cpu[req_idx].item()
                )

            cu_chunk_seqlen.append(seqlen_pos)

            seq_idx_p = torch.as_tensor(
                seq_idx, device=query_start_loc_p.device, dtype=torch.int32
            )
            cu_chunk_seqlen_p = torch.as_tensor(
                cu_chunk_seqlen, device=query_start_loc_p.device, dtype=torch.int32
            )
            last_chunk_indices_p = torch.as_tensor(
                last_chunk_indices, device=query_start_loc_p.device, dtype=torch.int32
            )

            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(query_start_loc_p)
            )

        elif (
            num_decodes <= self.decode_cudagraph_max_bs
            and self.compilation_config.full_cuda_graph
        ):
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_decodes)
            self.state_indices_tensor[:num_decodes].copy_(
                state_indices_tensor, non_blocking=True
            )
            state_indices_tensor = self.state_indices_tensor[:num_input_tokens]
            state_indices_tensor[num_decodes:] = PAD_SLOT_ID

            if self.vllm_config.cache_config.enable_prefix_caching:
                self.block_idx_last_scheduled_token[:num_decodes].copy_(
                    block_idx_last_scheduled_token, non_blocking=True
                )
                block_idx_last_scheduled_token = self.block_idx_last_scheduled_token[
                    :num_input_tokens
                ]
                block_idx_last_scheduled_token[num_decodes:] = 0

                self.block_idx_last_computed_token[:num_decodes].copy_(
                    block_idx_last_computed_token, non_blocking=True
                )
                block_idx_last_computed_token = self.block_idx_last_computed_token[
                    :num_input_tokens
                ]
                block_idx_last_computed_token[num_decodes:] = 0


        attn_metadata = ShortConvAttentionMetadata(
            query_start_loc=query_start_loc,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc_p=query_start_loc_p,
            seq_lens=seq_lens,
            prep_initial_states=prep_initial_states,
            chunk_size=self.chunk_size,
            has_initial_states_p=has_initial_states_p,
            seq_idx_p=seq_idx_p,
            state_indices_tensor=state_indices_tensor,
            cu_chunk_seqlen_p=cu_chunk_seqlen_p,
            last_chunk_indices_p=last_chunk_indices_p,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            block_idx_first_scheduled_token_p=block_idx_first_scheduled_token_p,
            block_idx_last_computed_token=block_idx_last_computed_token,
            num_computed_tokens_p=num_computed_tokens_p,
        )
        return attn_metadata
