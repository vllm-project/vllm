# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional

import torch

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.backends.placeholder_attn import (
    PlaceholderAttentionMetadata)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mamba2_attn import (
    _query_start_loc_to_chunk_indices_offsets, update_metadata)


@dataclass
class Mamba2Metadata:

    has_initial_states: torch.Tensor
    prep_initial_states: bool

    chunk_size: int
    seq_idx: torch.Tensor
    chunk_indices: torch.Tensor
    chunk_offsets: torch.Tensor
    chunk_inv_start_p: torch.Tensor
    query_start_loc_p: torch.Tensor
    """
    With continuous batching layout of `x` in vLLM, to enable a Triton program
    to handle a request in parallel, two supporting tensors are used
    (batch_ptr, token_chunk_offset_ptr)
    BLOCK_M = the # tokens to be handled by a Triton program
              (can be customized for different hardware)

    nums_dict:
       tracks the data associated with a given value of BLOCK_M
       BLOCK_M = #tokens handled by a Triton program
    batch_ptr: tracks batch-id handled by the Triton program
    token_chunk_offset_ptr: tracks token group_idx handled by the Triton program
           (Triton implementation of causal_conv1d handles parallelism in 3-axes
           - feature-axis
           - batch-axis
           - sequence-axis)
    """
    nums_dict: Optional[dict] = None
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


def prepare_mamba2_metadata(
    chunk_size: int,
    attn_metadata: AttentionMetadata,
) -> Mamba2Metadata:

    # compute number of prefill and decode requests
    # NOTE: in V0 we assume prefills are before decodes
    num_prefills = attn_metadata.num_prefills
    num_prefill_tokens = attn_metadata.num_prefill_tokens

    seq_idx = None
    chunk_indices, chunk_offsets, chunk_inv_start = None, None, None
    # Need flags to indicate if there are initial states
    # currently we really only support the FlashAttention backend
    has_initial_states = None
    prep_initial_states = False
    query_start_loc_p = None
    # Compute seq_idx, chunk_indices and chunk_offsets for prefill only
    if num_prefills > 0:
        attn_metadata_instances = get_platform_metadata_classes()
        if (isinstance(attn_metadata, attn_metadata_instances)
                and attn_metadata.context_lens_tensor is not None):
            # precompute flag to avoid device syncs later in mamba2 layer
            # forwards
            # prep is only needed for mamba2 ssd prefill processing
            has_initial_states = \
                attn_metadata.context_lens_tensor[:num_prefills] > 0
            prep_initial_states = torch.any(has_initial_states).item()
        query_start_loc_p = attn_metadata.query_start_loc[:num_prefills + 1]
        seq_idx = torch.repeat_interleave(torch.arange(
            num_prefills, dtype=torch.int32, device=query_start_loc_p.device),
                                          query_start_loc_p.diff(),
                                          output_size=num_prefill_tokens)

        # We compute metadata for chunked prefill once at the top level model
        # forward and reuse them in mamba layers. If not needed, they will be
        # ignored inside mamba kernels.
        if prep_initial_states:
            chunk_indices, chunk_offsets, chunk_inv_start = \
                _query_start_loc_to_chunk_indices_offsets(
                query_start_loc_p, chunk_size, num_prefill_tokens)

    metadata = Mamba2Metadata(has_initial_states=has_initial_states,
                              prep_initial_states=prep_initial_states,
                              chunk_size=chunk_size,
                              query_start_loc_p=query_start_loc_p,
                              seq_idx=seq_idx,
                              chunk_indices=chunk_indices,
                              chunk_offsets=chunk_offsets,
                              chunk_inv_start_p=chunk_inv_start)

    return update_metadata(metadata) if num_prefills > 0 else metadata
