# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.v1.attention.backends.utils import (AttentionMetadataBuilder,
                                              CommonAttentionMetadata)
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.block_table import BlockTable

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


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

    def __init__(self, runner: "GPUModelRunner", kv_cache_spec: MambaSpec,
                 block_table: BlockTable):
        self.runner = runner
        self.kv_cache_spec = kv_cache_spec
        self.block_table = block_table
        self.chunk_size = runner.vllm_config.model_config.get_mamba_chunk_size(
        )
        assert self.chunk_size is not None, (
            "chunk_size needs to be set in the model config for Mamba2 models")

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        # NOTE (Chen): Copied from MLACommonMetadataBuilder and
        # FlashInferMetadataBuilder. Should be refactored later to avoid code
        # duplication of these 3 functions.
        # We now want to reorder the batch so that the "decode" requests are and
        # the front and the "prefill" requests are at the using the least amount
        # swaps possible. (NOTE for now we loosely use "decode" to mean requests
        # where attention is likely memory-bound and "prefill" to mean requests
        # where attention is likely compute-bound, TODO(lucas): figure out a
        # better naming here)
        decodes = []
        prefills = []
        num_decode_tokens = 0
        num_prefill_tokens = 0

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # for now treat 1 scheduled token as "decode" even if its not,
            # we should update this to something like < 8 in the future but
            # currently the decode run only supports num_tokens = 1
            if num_tokens == 1:
                decodes.append(i)
                num_decode_tokens += num_tokens
            else:
                prefills.append(i)
                num_prefill_tokens += num_tokens

        # We hope that this is fairly minimal since decodes
        # should be around for a number of iterations so hopefully they are
        # relatively stationary (and new request are generally appended to the
        # persistent batch so already should be at the back)
        # To achieve this we loop over the decodes in descending order and
        # the prefills in ascending order. We swap decodes from the  "back"
        # i.e. past where the last decode should be in the reodorered with
        # prefills from the front of the batch.
        # `decodes` and `prefills` are already in ascending order just based on
        # the above loop
        num_decodes = len(decodes)
        num_prefills = len(prefills)
        modified_batch = False

        for i in range(1, min(num_decodes, num_prefills) + 1):
            # If the decode is at the "back" of the batch, i, we can swap it
            # with the prefill closest to the front of the batch
            decode_idx = decodes[num_decodes - i]
            if decode_idx < num_decodes:
                break

            input_batch.swap_states(prefills[i - 1], decode_idx)
            modified_batch = True

        # Save for next `build` call
        # TODO(lucas): this is a bit of a hack, we should probably have a
        # better way of doing this
        self._num_decodes = num_decodes
        self._num_prefills = num_prefills
        self._num_decode_tokens = num_decode_tokens
        self._num_prefill_tokens = num_prefill_tokens

        return modified_batch

    def build(self, common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata):
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens

        seq_idx = None
        chunk_indices, chunk_offsets = None, None
        # Need flags to indicate if there are initial states
        # currently we really only support the FlashAttention backend
        has_initial_states = None
        prep_initial_states = False

        state_indices_tensor = self.block_table.block_table[:num_reqs, 0]

        # Compute seq_idx, chunk_indices and chunk_offsets for prefill only
        if self._num_prefills > 0:
            #[batch,]
            has_initial_states_cpu = (
                self.runner.input_batch.
                num_computed_tokens_cpu_tensor[num_reqs -
                                               self._num_prefills:num_reqs]
                > 0)
            prep_initial_states = torch.any(has_initial_states_cpu).item()
            has_initial_states = has_initial_states_cpu.to(
                query_start_loc.device)

            query_start_loc_p = common_attn_metadata.query_start_loc[
                -self._num_prefills - 1:] - self._num_decode_tokens

            seq_idx = torch.repeat_interleave(
                torch.arange(self._num_prefills,
                             dtype=torch.int32,
                             device=query_start_loc_p.device),
                query_start_loc_p.diff(),
                output_size=self._num_prefill_tokens)
            seq_idx.unsqueeze_(0)

            # We compute metadata for chunked prefill once at the top level
            # model forward and reuse them in mamba layers. If not needed,
            # they will be ignored inside mamba kernels.
            if prep_initial_states:
                chunk_indices, chunk_offsets = (
                    _query_start_loc_to_chunk_indices_offsets(
                        query_start_loc_p, self.chunk_size,
                        self._num_prefill_tokens))

        attn_metadata = Mamba2AttentionMetadata(
            num_prefills=self._num_prefills,
            num_prefill_tokens=self._num_prefill_tokens,
            num_decodes=self._num_decodes,
            num_decode_tokens=self._num_decode_tokens,
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
