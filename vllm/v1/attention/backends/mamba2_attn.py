# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import itertools
from dataclasses import dataclass

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    compute_causal_conv1d_metadata,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec


def compute_varlen_chunk_metadata(
    query_start_loc: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build chunk-aligned, variable-length metadata used by Mamba2 SSD kernels.

    Given per-sequence cumulative token starts `query_start_loc` of shape [B+1]
    and a physical `chunk_size`, returns three tensors on the same device:
      - cu_chunk_seqlens:  (nchunks+1,) int32   exclusive prefix-sum of
        logical-chunk lengths (each logical chunk never crosses a sequence or
        physical-chunk boundary).
      - last_chunk_indices: (B,)       int32   index of the last logical chunk
        for each sequence (=-1 for empty sequences).
      - seq_idx_chunks:     (nchunks,) int32   sequence index for each logical
        chunk in order.

    This is intentionally lightweight and CPU-side; it mirrors the metadata
    produced by the V1 Mamba2 meta-data builder and is exported so tests
    (and other callers) can avoid duplicating the logic.
    """
    assert query_start_loc.ndim == 1, "query_start_loc must be 1-D [B+1]"
    assert int(query_start_loc[0].item()) == 0, "query_start_loc[0] must be 0"
    device = query_start_loc.device

    qsl64 = query_start_loc.to(torch.int64)
    starts = qsl64[:-1].tolist()
    ends = qsl64[1:].tolist()
    total = int(qsl64[-1].item())

    chunk_lens: list[int] = []
    seq_idx_chunks: list[int] = []
    last_chunk_indices: list[int] = [-1] * len(starts)

    for b, (s, e) in enumerate(zip(starts, ends)):
        if e <= s:
            # empty sequence
            continue
        pos = s
        while pos < e:
            # split at both sequence boundaries and physical chunk boundaries
            room = chunk_size - (pos % chunk_size)
            take = min(room, e - pos)
            chunk_lens.append(int(take))
            seq_idx_chunks.append(b)
            last_chunk_indices[b] = len(chunk_lens) - 1
            pos += take

    # Exclusive prefix sum over logical-chunk lengths
    if chunk_lens:
        cu_chunk_seqlens = torch.tensor(
            [0] + list(itertools.accumulate(chunk_lens)),
            device=device,
            dtype=torch.int32,
        )
        # Final boundary must equal total tokens
        assert int(cu_chunk_seqlens[-1].item()) == total
    else:
        cu_chunk_seqlens = torch.tensor([0], device=device, dtype=torch.int32)

    last_chunk_indices_t = (
        torch.tensor(last_chunk_indices, device=device, dtype=torch.int32)
        if len(starts) > 0
        else torch.empty((0,), device=device, dtype=torch.int32)
    )
    seq_idx_chunks_t = torch.tensor(seq_idx_chunks, device=device, dtype=torch.int32)
    return cu_chunk_seqlens, last_chunk_indices_t, seq_idx_chunks_t


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
    has_initial_states_p: torch.Tensor | None
    seq_idx_p: torch.Tensor | None

    # cu_chunk_seqlen_p is a tensor of shape (nchunks+1,) that contains, for
    # each chunk, its offests into the varlen sequence dimension. It is defined
    # such that the i-th chunk contains tokens from cu_chunk_seqlen_p[i] to
    # cu_chunk_seqlen_p[i+1].
    cu_chunk_seqlen_p: torch.Tensor | None

    # last_chunk_indices_p is a tensor of shape (batch,) that contains the
    # index of the last chunk for every sequence in the (prefill) batch.
    last_chunk_indices_p: torch.Tensor | None

    state_indices_tensor: torch.Tensor  # shape: [batch,]
    block_idx_last_scheduled_token: torch.Tensor  # shape: [batch,]
    block_idx_first_scheduled_token_p: torch.Tensor  # shape: [batch,]
    block_idx_last_computed_token: torch.Tensor  # shape: [batch,]
    num_computed_tokens_p: torch.Tensor  # shape: [batch,]

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


class Mamba2AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba2AttentionMetadata]
):
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.chunk_size = vllm_config.model_config.get_mamba_chunk_size()
        assert self.chunk_size is not None, (
            "chunk_size needs to be set in the model config for Mamba2 models"
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Mamba2AttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        seq_lens = common_attn_metadata.seq_lens

        query_start_loc_p = None
        seq_idx_p = None
        cu_chunk_seqlen_p = None
        last_chunk_indices_p = None

        # Need flags to indicate if there are initial states
        has_initial_states_p = None
        prep_initial_states = False

        # for causal_conv1d
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None

        num_computed_tokens, num_computed_tokens_p = None, None
        block_idx_first_scheduled_token = None
        block_idx_first_scheduled_token_p = None

        if self.vllm_config.cache_config.enable_prefix_caching:
            # Return a tensor of shape (#requests, #max blocks)
            state_indices_tensor = common_attn_metadata.block_table_tensor
            # Additional cache-related varaiables:
            mamba_block_size = self.kv_cache_spec.block_size
            num_computed_tokens = common_attn_metadata.num_computed_tokens_cpu.to(
                self.device
            )
            (
                block_idx_last_computed_token,
                block_idx_first_scheduled_token,
                block_idx_last_scheduled_token,
            ) = self._compute_prefix_caching_block_indices(
                common_attn_metadata, mamba_block_size
            )
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

        # Compute seq_idx for prefill only
        if num_prefills > 0:
            # [batch,]
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

            # The code below carefully constructs the chunks such that:
            # 1. Chunks contain tokens from a *single* sequence only.
            # 2. For every sequence, we are guaranteed that we can
            #    retrieve the mamba state *every* chunk_size tokens.
            # Constraint (1) dramatically simplifies the mamba2 kernels.
            # Constraint (2) dramatically simplifies the implementation
            # of prefix caching for mamba2 (wip). We need to take care
            # of the interaction with chunked prefill in order to
            # satisfy constraint (2).
            # TODO (tdoublep): This code could probably be optimized.
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

                # if computed tokens are not chunk-aligned, use the first
                # chunk to finish it off
                if this_num_computed % self.chunk_size != 0:
                    seq_idx.append(req_idx)
                    cu_chunk_seqlen.append(seqlen_pos)
                    # how many tokens to finish the chunk?
                    chunk_len = (
                        cdiv(this_num_computed, self.chunk_size) * self.chunk_size
                        - this_num_computed
                    )
                    # we can only use at most this_new_tokens
                    chunk_len = min(chunk_len, this_new_tokens)
                    seqlen_pos += chunk_len
                    this_new_tokens -= chunk_len

                n_chunks = cdiv(this_new_tokens, self.chunk_size)
                for chunk in range(n_chunks):
                    seq_idx.append(req_idx)
                    cu_chunk_seqlen.append(seqlen_pos)
                    chunk_len = min(self.chunk_size, this_new_tokens)
                    seqlen_pos += chunk_len
                    this_new_tokens -= chunk_len

                assert this_new_tokens == 0
                last_chunk_indices.append(len(cu_chunk_seqlen) - 1)

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
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            self.state_indices_tensor[:num_decodes].copy_(
                state_indices_tensor, non_blocking=True
            )
            state_indices_tensor = self.state_indices_tensor[:num_decode_tokens]

            if self.vllm_config.cache_config.enable_prefix_caching:
                self.block_idx_last_scheduled_token[:num_decodes].copy_(
                    block_idx_last_scheduled_token, non_blocking=True
                )
                block_idx_last_scheduled_token = self.block_idx_last_scheduled_token[
                    :num_decode_tokens
                ]

                self.block_idx_last_computed_token[:num_decodes].copy_(
                    block_idx_last_computed_token, non_blocking=True
                )
                block_idx_last_computed_token = self.block_idx_last_computed_token[
                    :num_decode_tokens
                ]

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

    def update_block_table(
        self,
        metadata: Mamba2AttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> Mamba2AttentionMetadata:
        new_metadata = copy.copy(metadata)
        prefix_caching = self.vllm_config.cache_config.enable_prefix_caching
        state_indices_t = blk_table if prefix_caching else blk_table[:, 0]
        num_reqs = blk_table.shape[0]

        # For CUDA graphs, copy to persistent buffer
        if (
            metadata.num_prefills == 0
            and num_reqs <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            persistent_state_indices_t = self.state_indices_tensor[:num_reqs]
            persistent_state_indices_t.copy_(state_indices_t, non_blocking=True)
            state_indices_t = persistent_state_indices_t

        new_metadata.state_indices_tensor = state_indices_t
        return new_metadata
