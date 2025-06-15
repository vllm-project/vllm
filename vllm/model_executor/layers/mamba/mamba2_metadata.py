# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.backends.placeholder_attn import (
    PlaceholderAttentionMetadata)
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.platforms import current_platform


@dataclass
class Mamba2Metadata:

    has_initial_states: torch.Tensor
    prep_initial_states: bool

    chunk_size: int
    seq_idx: torch.Tensor
    chunk_indices: torch.Tensor
    chunk_offsets: torch.Tensor
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
    MAX_NUM_PROGRAMS: tracks current size of batch_ptr, which is also
           the maximum number of Triton program along .program_id(0)
    """
    nums_dict: Optional[dict] = None
    cu_seqlen: Optional[int] = None
    MAX_NUM_PROGRAMS: int = 1024
    batch_ptr: Optional[torch.tensor] = None
    token_chunk_offset_ptr: Optional[torch.tensor] = None


def get_platform_metadata_classes() -> tuple[type[AttentionMetadata], ...]:
    """Returns the appropriate metadata classes for the current platform."""
    if current_platform.is_rocm():
        from vllm.attention.backends.rocm_flash_attn import (
            ROCmFlashAttentionMetadata)
        return (ROCmFlashAttentionMetadata, PlaceholderAttentionMetadata)
    elif current_platform.is_cuda():
        from vllm.attention.backends.flash_attn import FlashAttentionMetadata
        from vllm.attention.backends.xformers import XFormersMetadata
        return (FlashAttentionMetadata, XFormersMetadata,
                PlaceholderAttentionMetadata)
    raise ValueError(
        f"Unsupported platform for Mamba2: {current_platform.device_type}")


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

        # adjust inidces and offsets
        chunk_indices[_s:_e] -= p
        chunk_offsets[_s] = s % chunk_size

    return chunk_indices, chunk_offsets


def prepare_mamba2_metadata(
    chunk_size: int,
    attn_metadata: AttentionMetadata,
    mamba2_metadata=None,
) -> Mamba2Metadata:

    # compute number of prefill and decode requests
    # NOTE: in V0 we assume prefills are before decodes
    num_prefills = attn_metadata.num_prefills
    num_prefill_tokens = attn_metadata.num_prefill_tokens

    seq_idx = None
    chunk_indices, chunk_offsets = None, None
    # Need flags to indicate if there are initial states
    # currently we really only support the FlashAttention backend
    has_initial_states = None
    prep_initial_states = False

    # Compute seq_idx, chunk_indices and chunk_offsets for prefill only
    if num_prefills > 0:
        attn_metadata_instances = get_platform_metadata_classes()
        if (isinstance(attn_metadata, attn_metadata_instances)
                and attn_metadata.context_lens_tensor is not None):
            # precompute flag to avoid device syncs later in mamba2 layer
            # forwards
            # prep is only needed for mamba2 ssd prefill processing
            has_initial_states = attn_metadata.context_lens_tensor > 0
            prep_initial_states = torch.any(
                has_initial_states[:num_prefills]).item()
        query_start_loc = attn_metadata.query_start_loc[:num_prefills + 1]
        seq_idx = torch.repeat_interleave(torch.arange(
            num_prefills, dtype=torch.int32, device=query_start_loc.device),
                                          query_start_loc.diff(),
                                          output_size=num_prefill_tokens)
        seq_idx.unsqueeze_(0)

        # We compute metadata for chunked prefill once at the top level model
        # forward and reuse them in mamba layers. If not needed, they will be
        # ignored inside mamba kernels.
        if prep_initial_states:
            chunk_indices, chunk_offsets = \
                _query_start_loc_to_chunk_indices_offsets(
                query_start_loc, chunk_size, num_prefill_tokens)

    if mamba2_metadata is not None:
        mamba2_metadata.has_initial_states = has_initial_states
        mamba2_metadata.prep_initial_states = prep_initial_states
        mamba2_metadata.chunk_size = chunk_size
        mamba2_metadata.seq_idx = seq_idx
        mamba2_metadata.chunk_indices = chunk_indices
        mamba2_metadata.chunk_offsets = chunk_offsets
        # We use 1 reset flag:
        #  * mamba2_metadata.cu_seqlen is None
        #      update config specific to (each input)
        #      (become available at first layer, e.g. conv_weights)
        mamba2_metadata.cu_seqlen = None  # suppose to be updated at each input

        return mamba2_metadata
    return Mamba2Metadata(has_initial_states=has_initial_states,
                          prep_initial_states=prep_initial_states,
                          chunk_size=chunk_size,
                          seq_idx=seq_idx,
                          chunk_indices=chunk_indices,
                          chunk_offsets=chunk_offsets)


def update_metadata(x: torch.Tensor, query_start_loc: torch.Tensor,
                    mamba2_metadata: Mamba2Metadata):
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
        # type: ignore
        mamba2_metadata.MAX_NUM_PROGRAMS = max(
            mamba2_metadata.MAX_NUM_PROGRAMS, mlist_len)
        offsetlist = []  # type: ignore
        for idx, num in enumerate(nums):
            offsetlist.extend(range(num))
        offsetlist = torch.tensor(offsetlist, dtype=torch.int32)
        nums_dict[BLOCK_M]['offsetlist'] = offsetlist

        if mamba2_metadata.batch_ptr is None:
            # Update default value after class definition
            mamba2_metadata.MAX_NUM_PROGRAMS *= 2
            mamba2_metadata.batch_ptr = torch.full(
                (mamba2_metadata.MAX_NUM_PROGRAMS, ),
                PAD_SLOT_ID,
                dtype=torch.int32,
                device='cuda')
            mamba2_metadata.token_chunk_offset_ptr = torch.full(
                (mamba2_metadata.MAX_NUM_PROGRAMS, ),
                PAD_SLOT_ID,
                dtype=torch.int32,
                device='cuda')
        else:
            if mamba2_metadata.batch_ptr.nelement(
            ) < mamba2_metadata.MAX_NUM_PROGRAMS:
                mamba2_metadata.batch_ptr.resize_(
                    mamba2_metadata.MAX_NUM_PROGRAMS).fill_(
                        PAD_SLOT_ID)  # type: ignore
                mamba2_metadata.token_chunk_offset_ptr.resize_(  # type: ignore
                    mamba2_metadata.MAX_NUM_PROGRAMS).fill_(
                        PAD_SLOT_ID)  # type: ignore

        mamba2_metadata.batch_ptr[0:mlist_len].copy_(mlist)
        mamba2_metadata.token_chunk_offset_ptr[  # type: ignore
            0:mlist_len].copy_(offsetlist)
        nums_dict[BLOCK_M]['batch_ptr'] = mamba2_metadata.batch_ptr
        nums_dict[BLOCK_M]['token_chunk_offset_ptr'] = (
            mamba2_metadata.token_chunk_offset_ptr)  # type: ignore
    mamba2_metadata.nums_dict = nums_dict
    return mamba2_metadata
