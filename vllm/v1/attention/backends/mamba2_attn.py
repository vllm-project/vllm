# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import VllmConfig
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadataBuilder)
from vllm.v1.attention.backends.utils import (CommonAttentionMetadata,
                                              split_decodes_and_prefills)
from vllm.v1.kv_cache_interface import AttentionSpec


class Mamba2AttentionBackend(AttentionBackend):

    @staticmethod
    def get_builder_cls() -> type["Mamba2AttentionMetadataBuilder"]:
        return Mamba2AttentionMetadataBuilder


@dataclass
class Mamba2AttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    query_start_loc_p: torch.Tensor
    seq_lens: torch.Tensor

    prep_initial_states: bool
    chunk_size: int

    # The following tensors only contain prefill requests and will be None if
    # the batch has no prefill request.
    has_initial_states_p: Optional[torch.Tensor]
    seq_idx_p: Optional[torch.Tensor]
    chunk_indices_p: Optional[torch.Tensor]
    chunk_offsets_p: Optional[torch.Tensor]
    chunk_inv_start_p: Optional[torch.Tensor]

    state_indices_tensor: torch.Tensor  # shape: [batch,]

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: Optional[dict] = None
    batch_ptr: Optional[torch.tensor] = None
    token_chunk_offset_ptr: Optional[torch.tensor] = None


class Mamba2AttentionMetadataBuilder(
        BaseMambaAttentionMetadataBuilder[Mamba2AttentionMetadata]):

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.chunk_size = vllm_config.model_config.get_mamba_chunk_size()
        assert self.chunk_size is not None, (
            "chunk_size needs to be set in the model config for Mamba2 models")

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> Mamba2AttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc_p = None
        seq_lens = common_attn_metadata.seq_lens

        seq_idx_p = None
        chunk_indices_p, chunk_offsets_p, chunk_inv_start = None, None, None
        # Need flags to indicate if there are initial states
        # currently we really only support the FlashAttention backend
        has_initial_states_p = None
        prep_initial_states = False

        state_indices_tensor = common_attn_metadata.block_table_tensor[:, 0]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(common_attn_metadata,
                                       decode_threshold=1))

        # Compute seq_idx, chunk_indices and chunk_offsets for prefill only
        if num_prefills > 0:
            #[batch,]
            has_initial_states_cpu = (
                common_attn_metadata.
                num_computed_tokens_cpu[num_reqs - num_prefills:num_reqs] > 0)
            prep_initial_states = torch.any(has_initial_states_cpu).item()
            has_initial_states_p = has_initial_states_cpu.to(
                common_attn_metadata.query_start_loc.device)

            query_start_loc_p = common_attn_metadata.query_start_loc[
                -num_prefills - 1:] - num_decode_tokens

            seq_idx_p = torch.repeat_interleave(torch.arange(
                num_prefills,
                dtype=torch.int32,
                device=query_start_loc_p.device),
                                                query_start_loc_p.diff(),
                                                output_size=num_prefill_tokens)

            # We compute metadata for chunked prefill once at the top level
            # model forward and reuse them in mamba layers. If not needed,
            # they will be ignored inside mamba kernels.
            if prep_initial_states:
                chunk_indices_p, chunk_offsets_p, chunk_inv_start = (
                    _query_start_loc_to_chunk_indices_offsets(
                        query_start_loc_p, self.chunk_size,
                        num_prefill_tokens))

        elif num_decodes <= self.decode_cudagraph_max_bs:
            # Pad state tensor for CUDA graph
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_decodes)
            self.state_indices_tensor[:num_decodes].copy_(state_indices_tensor,
                                                          non_blocking=True)
            state_indices_tensor = self.state_indices_tensor[:num_input_tokens]
            state_indices_tensor[num_decodes:] = PAD_SLOT_ID

        attn_metadata = Mamba2AttentionMetadata(
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
            chunk_indices_p=chunk_indices_p,
            chunk_offsets_p=chunk_offsets_p,
            chunk_inv_start_p=chunk_inv_start,
            state_indices_tensor=state_indices_tensor,
        )
        return update_metadata(
            attn_metadata) if num_prefills > 0 else attn_metadata


def _query_start_loc_to_chunk_indices_offsets(query_start_loc: torch.Tensor,
                                              chunk_size: int,
                                              total_seqlens: int):

    cu_seqlens = query_start_loc[1:]  # remove prepended 0

    # outputs will have length expansion of chunks that do not divide
    # chunk_size
    N = math.ceil(total_seqlens / chunk_size) + (cu_seqlens[:-1] % chunk_size
                                                 > 0).sum()
    chunk_indices = torch.arange(N,
                                 dtype=torch.int,
                                 device=query_start_loc.device)
    chunk_offsets = torch.zeros((N, ),
                                dtype=torch.int,
                                device=query_start_loc.device)

    p = 0  # num of insertions
    for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:]):

        # if does not divide chunk_size, then there is one chunk insertion
        p += (s % chunk_size > 0)

        # get the dimensions
        # - the + 1 for _e is to shift the boundary by one chunk
        # - this shifting is not needed if chunk_size divides e
        _s, _e = s // chunk_size + p, e // chunk_size + p + (e % chunk_size
                                                             > 0)

        # adjust indices and offsets
        chunk_indices[_s:_e] -= p
        chunk_offsets[_s] = s % chunk_size

    # TODO: optimize, could be a Triton kernel with atomic add
    nchunks = math.ceil(total_seqlens / chunk_size)
    chunk_indices_cpu = chunk_indices.to('cpu').numpy()
    # need offset by 1 because a logical chunk corresponding to a
    # physical chunk should push the next physical chunk boundry,
    # not the current
    chunk_inv_start = torch.zeros((nchunks + 1, ),
                                  dtype=torch.int32,
                                  device='cpu')
    for chunk_idx in chunk_indices_cpu:
        chunk_inv_start[chunk_idx + 1] += 1
    # now we have a map from physical chunk index to how many logical
    # chunk indices cumsum gives us the start logical chunk for each
    # physical chunk
    chunk_inv_start = chunk_inv_start.to('cuda')
    chunk_inv_start = chunk_inv_start.cumsum(dim=0)

    return chunk_indices, chunk_offsets, chunk_inv_start


# update_metadata computes metadata required by triton conv1d kernel
# NOTE: argument type is removed for now to prevent circular dependency
#       V0 metadata will eventually be obsolete and we can add back
#       the correct type by then
def update_metadata(mamba2_metadata):
    # TODO: add has_attr assertions

    seqlens = mamba2_metadata.query_start_loc_p.diff().to('cpu')
    nums_dict = {}  # type: ignore
    for BLOCK_M in [8]:  # cover all BLOCK_M values
        nums = -(-seqlens // BLOCK_M)
        nums_dict[BLOCK_M] = {}
        nums_dict[BLOCK_M]['nums'] = nums
        nums_dict[BLOCK_M]['tot'] = nums.sum().item()
        mlist = torch.from_numpy(np.repeat(np.arange(len(nums)), nums))
        nums_dict[BLOCK_M]['mlist'] = mlist
        mlist_len = len(nums_dict[BLOCK_M]['mlist'])
        nums_dict[BLOCK_M]['mlist_len'] = mlist_len
        MAX_NUM_PROGRAMS = max(1024, mlist_len) * 2
        offsetlist = []  # type: ignore
        for idx, num in enumerate(nums):
            offsetlist.extend(range(num))
        offsetlist = torch.tensor(offsetlist, dtype=torch.int32)
        nums_dict[BLOCK_M]['offsetlist'] = offsetlist

        if mamba2_metadata.batch_ptr is None:
            # Update default value after class definition
            #mamba2_metadata.MAX_NUM_PROGRAMS *= 2
            mamba2_metadata.batch_ptr = torch.full((MAX_NUM_PROGRAMS, ),
                                                   PAD_SLOT_ID,
                                                   dtype=torch.int32,
                                                   device='cuda')
            mamba2_metadata.token_chunk_offset_ptr = torch.full(
                (MAX_NUM_PROGRAMS, ),
                PAD_SLOT_ID,
                dtype=torch.int32,
                device='cuda')
        else:
            if mamba2_metadata.batch_ptr.nelement() < MAX_NUM_PROGRAMS:
                mamba2_metadata.batch_ptr.resize_(MAX_NUM_PROGRAMS).fill_(
                    PAD_SLOT_ID)
                mamba2_metadata.token_chunk_offset_ptr.resize_(  # type: ignore
                    MAX_NUM_PROGRAMS).fill_(PAD_SLOT_ID)

        mamba2_metadata.batch_ptr[0:mlist_len].copy_(mlist)
        mamba2_metadata.token_chunk_offset_ptr[  # type: ignore
            0:mlist_len].copy_(offsetlist)
        nums_dict[BLOCK_M]['batch_ptr'] = mamba2_metadata.batch_ptr
        nums_dict[BLOCK_M]['token_chunk_offset_ptr'] = (
            mamba2_metadata.token_chunk_offset_ptr)  # type: ignore
    mamba2_metadata.nums_dict = nums_dict
    return mamba2_metadata
