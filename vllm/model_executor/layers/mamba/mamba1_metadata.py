# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch

from vllm import envs
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.backends.flash_attn import FlashAttentionMetadata


@dataclass
class Mamba1Metadata:
    """Metadata for Mamba1 (original Mamba) implementation.
    
    This class contains metadata needed for the MambaMixer to operate in continuous 
    batching and prefill modes. The metadata is computed at top-level model forward 
    since it stays the same and is reused for all mamba layers in the same iteration.
    """
    # Tensor indicating which sequences have initial states (context_lens > 0)
    has_initial_states: Optional[torch.Tensor]
    context_lens_tensor: Optional[torch.Tensor]
    
    # Tensor containing the starting location of each query in the sequence
    query_start_loc: Optional[torch.Tensor]
    
    # Tensor containing indices for accessing the state cache
    state_indices_tensor: Optional[torch.Tensor]
    
    # Number of prefill requests (request count)
    num_prefills: int
    
    # Number of decode tokens (token count = request)
    num_decode_tokens: int
    
    # Number of prefill tokens (token count)
    num_prefill_tokens: int


def prepare_mamba1_metadata(
    attn_metadata: Union[AttentionMetadata, Dict[str, AttentionMetadata]],
    mamba1_metadata: Optional[Mamba1Metadata] = None,
) -> Mamba1Metadata:
    """Prepare metadata for Mamba1 from attention metadata.
    
    Args:
        attn_metadata: Attention metadata containing sequence information.
                      Can be either AttentionMetadata or a dict mapping layer prefix to AttentionMetadata.
        mamba1_metadata: Optional existing metadata to update
        
    Returns:
        Mamba1Metadata object with required fields populated
    """
    # Handle dict case
    if isinstance(attn_metadata, dict):
        # Take the first value since all layers should have same metadata
        attn_metadata = next(iter(attn_metadata.values()))
    
    # Get counts from attention metadata
    num_prefills = attn_metadata.num_prefills
    num_decode_tokens = attn_metadata.num_decode_tokens
    num_prefill_tokens = attn_metadata.num_prefill_tokens
    
    # Get initial states info
    if envs.VLLM_USE_V1:
        has_initial_states = attn_metadata.context_lens_tensor > 0
    
    # Get query start locations and state indices
    query_start_loc = getattr(attn_metadata, 'query_start_loc', None)
    state_indices_tensor = getattr(attn_metadata, 'state_indices_tensor', None)
    
    if mamba1_metadata is not None:
        # Update existing metadata
        mamba1_metadata.has_initial_states = has_initial_states
        mamba1_metadata.query_start_loc = query_start_loc
        mamba1_metadata.state_indices_tensor = state_indices_tensor
        mamba1_metadata.num_prefills = num_prefills
        mamba1_metadata.num_decode_tokens = num_decode_tokens
        mamba1_metadata.num_prefill_tokens = num_prefill_tokens
        return mamba1_metadata
        
    # Create new metadata
    return Mamba1Metadata(
        has_initial_states=has_initial_states,
        query_start_loc=query_start_loc,
        state_indices_tensor=state_indices_tensor,
        num_prefills=num_prefills,
        num_decode_tokens=num_decode_tokens,
        num_prefill_tokens=num_prefill_tokens,
        context_lens_tensor=attn_metadata.context_lens_tensor,
    ) 