# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from dataclasses import dataclass
from typing import Optional

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import VllmConfig
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadataBuilder)
from vllm.v1.attention.backends.utils import (CommonAttentionMetadata,
                                              split_decodes_and_prefills)
from vllm.v1.kv_cache_interface import AttentionSpec


def _query_start_loc_to_chunk_indices_offsets(
        query_start_loc: torch.Tensor, chunk_size: int,
        total_seqlens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        query_start_loc (torch.Tensor): 1D tensor of cumulative sequence 
            lengths, shape (num_seqs + 1,).
            The first element should be 0. Each entry represents the starting
            index of a sequence in the flattened token array.
        chunk_size (int): The size of each physical mamba chunk
            (number of tokens per chunk).
        total_seqlens (int): The total number of tokens in the batch.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - chunk_indices (torch.Tensor): 1D tensor of indices 
                indicating the physical chunk for each logical chunk.
            - chunk_offsets (torch.Tensor): 1D tensor of offsets
                indicating the starting index of each logical chunk within
                its physical chunk.

    This function computes the chunk indices and offsets for the given
    query_start_loc and chunk_size. Both are tensors of integers with length N,
    where N is the number of logical (pseudo) chunks.
    A logical chunk is a sequence of tokens that are all part of the same
    sequence and are all in the same physical mamba chunk.
    In other words, a logical chunk changes every time we cross a sequence
    boundary or a physical mamba chunk boundary.
    Logical chunks are needed to handle batched requests with initial states
    (see _state_passing_fwd and _chunk_scan_fwd).
    The chunk_indices tensor contains the index of the physical chunk for each
    logical chunk.
    The chunk_offsets tensor contains the offset (AKA starting index) of the
    logical chunk in the physical chunk.

    Example:
    query_start_loc = [0, 5, 10]
    chunk_size = 8
    total_seqlens = 10
    -> chunk_indices = [0, 0, 1]
    -> chunk_offsets = [0, 5, 0]

    In this example, we have 2 sequences, each with 5 tokens. The physical
    chunk size is 8 tokens.
    We have three logical chunks:
    - the first logical chunk starts at token 0 in the first physical chunk
        and contains all 5 tokens from the first sequence
    - the second logical chunk starts at token 5 in the first physical chunk
        and contains first 3 tokens from the second sequence
    - the third logical chunk starts at token 0 in the second physical chunk
        and contains the remaining 2 tokens from the second sequence
    """

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

    return chunk_indices, chunk_offsets


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
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor

    prep_initial_states: bool
    chunk_size: int

    # The following tensors only contain prefill requests and will be None if
    # the batch has no prefill request.
    has_initial_states_p: Optional[torch.Tensor]
    seq_idx_p: Optional[torch.Tensor]
    chunk_indices_p: Optional[torch.Tensor]
    chunk_offsets_p: Optional[torch.Tensor]

    state_indices_tensor: torch.Tensor  # shape: [batch,]

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: Optional[dict] = None
    cu_seqlen: Optional[int] = None
    batch_ptr: Optional[torch.Tensor] = None
    token_chunk_offset_ptr: Optional[torch.Tensor] = None


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
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens

        seq_idx_p = None
        chunk_indices_p, chunk_offsets_p = None, None
        # Need flags to indicate if there are initial states
        # currently we really only support the FlashAttention backend
        has_initial_states_p = None
        prep_initial_states = False

        state_indices_tensor = common_attn_metadata.block_table_tensor[:, 0]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold))

        # Compute seq_idx, chunk_indices and chunk_offsets for prefill only
        if num_prefills > 0:
            #[batch,]
            has_initial_states_cpu = (
                common_attn_metadata.
                num_computed_tokens_cpu[num_reqs - num_prefills:num_reqs] > 0)
            prep_initial_states = torch.any(has_initial_states_cpu).item()
            has_initial_states_p = has_initial_states_cpu.to(
                query_start_loc.device)

            query_start_loc_p = common_attn_metadata.query_start_loc[
                -num_prefills - 1:] - num_decode_tokens

            seq_idx_p = torch.repeat_interleave(torch.arange(
                num_prefills,
                dtype=torch.int32,
                device=query_start_loc_p.device),
                                                query_start_loc_p.diff(),
                                                output_size=num_prefill_tokens)
            seq_idx_p.unsqueeze_(0)

            # We compute metadata for chunked prefill once at the top level
            # model forward and reuse them in mamba layers. If not needed,
            # they will be ignored inside mamba kernels.
            if prep_initial_states:
                chunk_indices_p, chunk_offsets_p = (
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
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            prep_initial_states=prep_initial_states,
            chunk_size=self.chunk_size,
            has_initial_states_p=has_initial_states_p,
            seq_idx_p=seq_idx_p,
            chunk_indices_p=chunk_indices_p,
            chunk_offsets_p=chunk_offsets_p,
            state_indices_tensor=state_indices_tensor,
        )
        return attn_metadata
