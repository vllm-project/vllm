# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.backends.placeholder_attn import (
    PlaceholderAttentionMetadata)
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.platforms import current_platform
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.mamba2_attn import (
    Mamba2AttentionMetadata, _query_start_loc_to_chunk_indices_offsets)


@dataclass
class Mamba2Metadata:
    prep_initial_states: bool
    chunk_size: int

    has_initial_states_p: torch.Tensor
    seq_idx_p: torch.Tensor
    chunk_indices_p: torch.Tensor
    chunk_offsets_p: torch.Tensor
    """
    With continuous batching layout of `x` in vLLM, to enable a Triton program
    to handle a request in parallel, two supporting tensors are used
    (batch_ptr, token_chunk_offset_ptr)
    BLOCK_M = the # tokens to be handled by a Triton program
              (can be customized for different hardware)

    nums_dict:
       tracks the data associated with a given value of BLOCK_M
       BLOCK_M = #tokens handled by a Triton program
    cu_seqlen: total tokens per batch
           (used as flag to update other data at each new input)
    batch_ptr: tracks batch-id handled by the Triton program
    token_chunk_offset_ptr: tracks token group_idx handled by the Triton program
           (Triton implementation of causal_conv1d handles parallelism in 3-axes
           - feature-axis
           - batch-axis
           - sequence-axis)
    """
    nums_dict: Optional[dict] = None
    cu_seqlen: Optional[int] = None
    batch_ptr: Optional[torch.Tensor] = None
    token_chunk_offset_ptr: Optional[torch.Tensor] = None


def get_platform_metadata_classes() -> tuple[type[AttentionMetadata], ...]:
    """Returns the appropriate metadata classes for the current platform."""
    if current_platform.is_rocm():
        from vllm.v1.attention.backends.rocm_aiter_fa import (
            AiterFlashAttentionMetadata)
        from vllm.v1.attention.backends.triton_attn import (
            TritonAttentionMetadata)
        return (AiterFlashAttentionMetadata, TritonAttentionMetadata,
                PlaceholderAttentionMetadata)
    if current_platform.is_cuda():
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionMetadata)
        from vllm.v1.attention.backends.xformers import (
            XFormersAttentionMetadata)
        return (FlashAttentionMetadata, XFormersAttentionMetadata,
                PlaceholderAttentionMetadata)
    raise ValueError(
        f"Unsupported platform for Mamba2: {current_platform.device_type}")


def prepare_mamba2_metadata(
    chunk_size: int,
    attn_metadata: AttentionMetadata,
) -> Mamba2Metadata:

    # compute number of prefill and decode requests
    # NOTE: in V0 we assume prefills are before decodes
    num_prefills = attn_metadata.num_prefills
    num_prefill_tokens = attn_metadata.num_prefill_tokens

    seq_idx_p = None
    chunk_indices_p, chunk_offsets_p = None, None
    # Need flags to indicate if there are initial states
    # currently we really only support the FlashAttention backend
    has_initial_states_p = None
    prep_initial_states = False

    # Compute seq_idx, chunk_indices and chunk_offsets for prefill only
    if num_prefills > 0:
        attn_metadata_instances = get_platform_metadata_classes()
        if (isinstance(attn_metadata, attn_metadata_instances)
                and attn_metadata.context_lens_tensor is not None):
            # precompute flag to avoid device syncs later in mamba2 layer
            # forwards
            # prep is only needed for mamba2 ssd prefill processing
            has_initial_states_p = (
                attn_metadata.context_lens_tensor[:num_prefills] > 0)
            prep_initial_states = torch.any(has_initial_states_p).item()
        query_start_loc_p = attn_metadata.query_start_loc[:num_prefills + 1]
        seq_idx_p = torch.repeat_interleave(torch.arange(
            num_prefills, dtype=torch.int32, device=query_start_loc_p.device),
                                            query_start_loc_p.diff(),
                                            output_size=num_prefill_tokens)
        seq_idx_p.unsqueeze_(0)

        # We compute metadata for chunked prefill once at the top level model
        # forward and reuse them in mamba layers. If not needed, they will be
        # ignored inside mamba kernels.
        if prep_initial_states:
            chunk_indices_p, chunk_offsets_p = \
                _query_start_loc_to_chunk_indices_offsets(
                query_start_loc_p, chunk_size, num_prefill_tokens)

    return Mamba2Metadata(has_initial_states_p=has_initial_states_p,
                          prep_initial_states=prep_initial_states,
                          chunk_size=chunk_size,
                          seq_idx_p=seq_idx_p,
                          chunk_indices_p=chunk_indices_p,
                          chunk_offsets_p=chunk_offsets_p)


def update_metadata(x: torch.Tensor, query_start_loc: torch.Tensor,
                    mamba2_metadata: Union[Mamba2Metadata,
                                           Mamba2AttentionMetadata,
                                           GDNAttentionMetadata]):
    """
    this is triggered upon handling a new input at the first layer
    """
    dim, cu_seqlen = x.shape
    mamba2_metadata.cu_seqlen = cu_seqlen
    seqlens = np.diff(query_start_loc.to('cpu'))
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
