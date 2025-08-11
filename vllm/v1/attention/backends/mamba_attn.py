# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import (AttentionCGSupport,
                                              AttentionMetadataBuilder,
                                              CommonAttentionMetadata,
                                              split_decodes_and_prefills)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


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

    has_initial_states: torch.Tensor
    prep_initial_states: bool
    chunk_size: int
    seq_idx: torch.Tensor
    chunk_indices: torch.Tensor
    chunk_offsets: torch.Tensor

    state_indices_tensor: torch.Tensor  # shape: [batch,]
    nums_dict: Optional[dict] = None
    cu_seqlen: Optional[int] = None
    batch_ptr: Optional[torch.tensor] = None
    token_chunk_offset_ptr: Optional[torch.tensor] = None


class Mamba2AttentionMetadataBuilder(
        AttentionMetadataBuilder[Mamba2AttentionMetadata]):
    attn_cudagraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.PURE_DECODE_ONLY

    reorder_batch_threshold: ClassVar[int] = 1

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        assert isinstance(kv_cache_spec, MambaSpec)
        self.kv_cache_spec = kv_cache_spec
        self.chunk_size = vllm_config.model_config.get_mamba_chunk_size()
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        assert self.chunk_size is not None, (
            "chunk_size needs to be set in the model config for Mamba2 models")
        self.decode_cudagraph_max_bs = min(
            self.vllm_config.scheduler_config.max_num_seqs,
            self.compilation_config.max_capture_size)
        self.state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs, ),
            dtype=torch.int32,
            device=device,
        )

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> Mamba2AttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens

        seq_idx = None
        chunk_indices, chunk_offsets = None, None
        # Need flags to indicate if there are initial states
        # currently we really only support the FlashAttention backend
        has_initial_states = None
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
            has_initial_states = has_initial_states_cpu.to(
                query_start_loc.device)

            query_start_loc_p = common_attn_metadata.query_start_loc[
                -num_prefills - 1:] - num_decode_tokens

            seq_idx = torch.repeat_interleave(torch.arange(
                num_prefills,
                dtype=torch.int32,
                device=query_start_loc_p.device),
                                              query_start_loc_p.diff(),
                                              output_size=num_prefill_tokens)
            seq_idx.unsqueeze_(0)

            # We compute metadata for chunked prefill once at the top level
            # model forward and reuse them in mamba layers. If not needed,
            # they will be ignored inside mamba kernels.
            if prep_initial_states:
                chunk_indices, chunk_offsets = (
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
            has_initial_states=has_initial_states,
            prep_initial_states=prep_initial_states,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            state_indices_tensor=state_indices_tensor,
        )
        return attn_metadata

    def build_for_cudagraph_capture(
            self, common_attn_metadata: CommonAttentionMetadata):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert m.num_reqs == m.num_actual_tokens, \
            "Mamba only supports decode-only full CUDAGraph capture. " \
            "Make sure all cudagraph capture sizes <= max_num_seq."

        m.max_query_len = 1  # decode-only

        return self.build(0, m)

    def can_run_in_cudagraph(
            self, common_attn_metadata: CommonAttentionMetadata) -> bool:
        return common_attn_metadata.max_query_len == 1
