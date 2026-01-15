# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import itertools
from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID,
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
    def get_name() -> str:
        return "MAMBA2_ATTN"

    @staticmethod
    def get_builder_cls() -> type["Mamba2AttentionMetadataBuilder"]:
        return Mamba2AttentionMetadataBuilder


@dataclass
class Mamba2AttentionMetadata(BaseMambaAttentionMetadata):
    prep_initial_states: bool = False
    chunk_size: int = 0

    # Chunk-related metadata (only for prefill)
    seq_idx_p: torch.Tensor | None = None
    # cu_chunk_seqlen_p is a tensor of shape (nchunks+1,) that contains, for
    # each chunk, its offests into the varlen sequence dimension. It is defined
    # such that the i-th chunk contains tokens from cu_chunk_seqlen_p[i] to
    # cu_chunk_seqlen_p[i+1].
    cu_chunk_seqlen_p: torch.Tensor | None = None
    # last_chunk_indices_p is a tensor of shape (batch,) that contains the
    # index of the last chunk for every sequence in the (prefill) batch.
    last_chunk_indices_p: torch.Tensor | None = None

    # Speculative decoding support
    num_spec_decodes: int = 0
    num_spec_decode_tokens: int = 0

    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    # Token-level state indices for speculative decode
    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec+1]
    non_spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch]

    # Which sequences are doing speculative decode
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]

    # Query locations split by spec vs non-spec
    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    # Number of accepted tokens for each spec sequence (for loading correct checkpoint)
    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # Number of computed tokens per request
    num_computed_tokens: torch.Tensor | None = None  # shape: [batch,]


class Mamba2AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba2AttentionMetadata]
):
    metadata_cls = Mamba2AttentionMetadata

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        chunk_size = vllm_config.model_config.get_mamba_chunk_size()
        assert chunk_size is not None, (
            "chunk_size needs to be set in the model config for Mamba2 models"
        )
        self.chunk_size: int = chunk_size

        self.speculative_config = vllm_config.speculative_config
        self.num_spec: int = 0
        if (
            self.speculative_config is not None
            and self.speculative_config.num_speculative_tokens is not None
        ):
            self.use_spec_decode = True
            self.num_spec = self.speculative_config.num_speculative_tokens
        else:
            self.use_spec_decode = False

        if self.use_spec_decode:
            # TODO smor- a lot of code duplication here, should be refactored,
            # depends on if we want to support spec decode only here on base class
            self.decode_cudagraph_max_bs = (
                self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
            )
            if self.compilation_config.max_cudagraph_capture_size is not None:
                self.decode_cudagraph_max_bs = min(
                    self.decode_cudagraph_max_bs,
                    self.compilation_config.max_cudagraph_capture_size,
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

            # Spec decode state indices: [batch, num_spec+1]
            self.spec_state_indices_tensor_buffer = torch.empty(
                (self.decode_cudagraph_max_bs, self.num_spec + 1),
                dtype=torch.int32,
                device=device,
            )
            # Non-spec state indices: [batch]
            self.non_spec_state_indices_tensor_buffer = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            # Spec sequence masks: [batch]
            self.spec_sequence_masks_buffer = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.bool,
                device=device,
            )
            # Spec query start locations: [batch+1]
            self.spec_query_start_loc_buffer = torch.empty(
                (self.decode_cudagraph_max_bs + 1,),
                dtype=torch.int32,
                device=device,
            )
            # Non-spec query start locations: [batch+1]
            self.non_spec_query_start_loc_buffer = torch.empty(
                (self.decode_cudagraph_max_bs + 1,),
                dtype=torch.int32,
                device=device,
            )
            # Number of accepted tokens: [batch]
            self.num_accepted_tokens_buffer = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.spec_token_indx = torch.empty(
                (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
                dtype=torch.int32,
                device=device,
            )
            self.non_spec_token_indx = torch.empty(
                (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
                dtype=torch.int32,
                device=device,
            )

    def stable_boolean_sort(self, mask: torch.Tensor):
        idx = torch.arange(mask.numel(), device=mask.device)
        key = mask.to(torch.int32) * (mask.numel() + 1) + idx
        return torch.argsort(key, stable=True)

    def _compute_chunk_metadata(
        self,
        num_prefills: int,
        num_computed_tokens_p_cpu: torch.Tensor,
        query_start_loc_p_cpu: torch.Tensor,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Compute chunk-specific metadata for Mamba2.

        The code below carefully constructs the chunks such that:
        1. Chunks contain tokens from a *single* sequence only.
        2. For every sequence, we are guaranteed that we can
           retrieve the mamba state *every* chunk_size tokens.
        Constraint (1) dramatically simplifies the mamba2 kernels.
        Constraint (2) dramatically simplifies the implementation
        of prefix caching for mamba2 (wip). We need to take care
        of the interaction with chunked prefill in order to
        satisfy constraint (2).
        """
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

        return cu_chunk_seqlen, seq_idx, last_chunk_indices

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> Mamba2AttentionMetadata:
        num_accepted_tokens: torch.Tensor | None = kwargs.get("num_accepted_tokens")
        num_decode_draft_tokens_cpu: torch.Tensor | None = kwargs.get(
            "num_decode_draft_tokens_cpu"
        )

        query_start_loc_p = None
        seq_idx_p = None
        cu_chunk_seqlen_p = None
        last_chunk_indices_p = None

        has_initial_states_p = None
        prep_initial_states = False

        # for causal_conv1d
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None

        num_computed_tokens = None
        num_computed_tokens_p = None
        block_idx_first_scheduled_token = None
        block_idx_first_scheduled_token_p = None

        num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()

        if self.vllm_config.cache_config.enable_prefix_caching:
            # Return a tensor of shape (#requests, #max blocks)
            # Additional cache-related varaiables:
            mamba_block_size = self.kv_cache_spec.block_size
            (
                block_idx_last_computed_token,
                block_idx_first_scheduled_token,
                block_idx_last_scheduled_token,
            ) = self._compute_prefix_caching_block_indices(
                common_attn_metadata, mamba_block_size
            )
        else:
            # Additional cache-related variables:
            block_idx_last_scheduled_token = None
            block_idx_last_computed_token = None

        # Initialize spec decode variables
        spec_state_indices_tensor = None
        non_spec_state_indices_tensor = None
        spec_sequence_masks = None
        spec_query_start_loc = None
        non_spec_query_start_loc = None
        num_spec_decodes = 0
        num_spec_decode_tokens = 0
        num_accepted_tokens_filtered = None

        # Check if we have spec decode sequences
        if (
            not self.use_spec_decode
            or num_decode_draft_tokens_cpu is None
            or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
            .sum()
            .item()
            == 0
        ):
            # No spec decode sequences
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            # We have spec decode sequences
            spec_sequence_masks = num_decode_draft_tokens_cpu >= 0
            num_spec_decodes = spec_sequence_masks.sum().item()
            if num_spec_decodes == 0:
                spec_sequence_masks = None
            else:
                spec_sequence_masks = spec_sequence_masks.to(
                    common_attn_metadata.query_start_loc.device, non_blocking=True
                )

        # Compute decode/prefill split
        if spec_sequence_masks is None:
            # No spec decode - use standard split
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(
                    common_attn_metadata, decode_threshold=self.reorder_batch_threshold
                )
            )
            num_spec_decode_tokens = 0
            if self.vllm_config.cache_config.enable_prefix_caching:
                non_spec_state_indices_tensor = common_attn_metadata.block_table_tensor
            else:
                non_spec_state_indices_tensor = common_attn_metadata.block_table_tensor[
                    :, 0
                ]
            non_spec_query_start_loc = common_attn_metadata.query_start_loc
            spec_token_indx = None
            non_spec_token_indx = None
        else:
            # Have spec decode - compute counts EXCLUDING spec sequences
            query_lens = (
                common_attn_metadata.query_start_loc[1:]
                - common_attn_metadata.query_start_loc[:-1]
            )

            non_spec_query_lens = query_lens[~spec_sequence_masks]
            num_decodes = (non_spec_query_lens == 1).sum().item()
            num_prefills = non_spec_query_lens.size(0) - num_decodes
            num_decode_tokens = num_decodes
            num_prefill_tokens = non_spec_query_lens.sum().item() - num_decode_tokens
            num_spec_decode_tokens = (
                query_lens.sum().item() - num_prefill_tokens - num_decode_tokens
            )

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    common_attn_metadata.query_start_loc[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=common_attn_metadata.query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0,
                    dtype=torch.int32,
                    device=common_attn_metadata.query_start_loc.device,
                )

                if self.vllm_config.cache_config.enable_prefix_caching:
                    num_cacheable_blocks = num_computed_tokens // mamba_block_size
                    block_indices = num_cacheable_blocks.unsqueeze(1) + torch.arange(
                        self.num_spec + 1, device=self.device
                    ).unsqueeze(0)
                    batch_indices = torch.arange(
                        common_attn_metadata.block_table_tensor.size(0),
                        device=self.device,
                    ).unsqueeze(1)
                    spec_state_indices_tensor = common_attn_metadata.block_table_tensor[
                        batch_indices, block_indices
                    ]
                else:
                    spec_state_indices_tensor = common_attn_metadata.block_table_tensor[
                        :, : self.num_spec + 1
                    ]
                non_spec_state_indices_tensor = None
                spec_query_start_loc = common_attn_metadata.query_start_loc
                non_spec_query_start_loc = None
            else:
                if self.vllm_config.cache_config.enable_prefix_caching:
                    assert block_idx_first_scheduled_token is not None
                    block_idx_first_scheduled_token = block_idx_first_scheduled_token[
                        ~spec_sequence_masks
                    ]
                    block_idx_last_scheduled_token = block_idx_last_scheduled_token[
                        ~spec_sequence_masks
                    ]
                    block_idx_last_computed_token = block_idx_last_computed_token[
                        ~spec_sequence_masks
                    ]
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks, query_lens
                )
                index = self.stable_boolean_sort(spec_token_masks)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                if self.vllm_config.cache_config.enable_prefix_caching:
                    num_cacheable_blocks = num_computed_tokens // mamba_block_size
                    block_indices = num_cacheable_blocks[spec_sequence_masks].unsqueeze(
                        1
                    ) + torch.arange(self.num_spec + 1, device=self.device).unsqueeze(0)
                    batch_indices = torch.arange(
                        spec_sequence_masks.sum().item(), device=self.device
                    ).unsqueeze(1)
                    spec_state_indices_tensor = common_attn_metadata.block_table_tensor[
                        spec_sequence_masks
                    ][batch_indices, block_indices]
                    non_spec_state_indices_tensor = (
                        common_attn_metadata.block_table_tensor[~spec_sequence_masks]
                    )
                else:
                    spec_state_indices_tensor = common_attn_metadata.block_table_tensor[
                        spec_sequence_masks, : self.num_spec + 1
                    ]
                    non_spec_state_indices_tensor = (
                        common_attn_metadata.block_table_tensor[~spec_sequence_masks, 0]
                    )

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=common_attn_metadata.query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[spec_sequence_masks], dim=0, out=spec_query_start_loc[1:]
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=common_attn_metadata.query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[~spec_sequence_masks],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )

            # Filter num_accepted_tokens to only spec sequences
            if num_accepted_tokens is not None:
                num_accepted_tokens_filtered = num_accepted_tokens[spec_sequence_masks]

        # Compute seq_idx for prefill only
        if num_prefills > 0:
            # [batch,]
            if spec_sequence_masks is not None:
                assert num_decode_draft_tokens_cpu is not None
                num_computed_tokens_cpu_non_spec = (
                    common_attn_metadata.num_computed_tokens_cpu[
                        ~(num_decode_draft_tokens_cpu >= 0)
                    ]
                )
            else:
                num_computed_tokens_cpu_non_spec = (
                    common_attn_metadata.num_computed_tokens_cpu
                )
            has_initial_states_cpu = num_computed_tokens_cpu_non_spec[num_decodes:] > 0
            prep_initial_states = torch.any(has_initial_states_cpu).item()
            has_initial_states_p = has_initial_states_cpu.to(
                common_attn_metadata.query_start_loc.device
            )

            # Subtract ALL decode tokens (spec + non-spec)
            # to get prefill-only coordinates
            total_decode_tokens = num_decode_tokens + num_spec_decode_tokens

            if spec_sequence_masks is not None:
                query_lens_all = (
                    common_attn_metadata.query_start_loc[1:]
                    - common_attn_metadata.query_start_loc[:-1]
                )
                query_lens_non_spec = query_lens_all[~spec_sequence_masks]
                query_lens_prefills = query_lens_non_spec[num_decodes:]
                query_start_loc_p = torch.zeros(
                    num_prefills + 1,
                    dtype=common_attn_metadata.query_start_loc.dtype,
                    device=common_attn_metadata.query_start_loc.device,
                )
                torch.cumsum(query_lens_prefills, dim=0, out=query_start_loc_p[1:])

                query_lens_cpu_all = (
                    common_attn_metadata.query_start_loc_cpu[1:]
                    - common_attn_metadata.query_start_loc_cpu[:-1]
                )
                assert num_decode_draft_tokens_cpu is not None
                query_lens_cpu_non_spec = query_lens_cpu_all[
                    ~(num_decode_draft_tokens_cpu >= 0)
                ]
                query_lens_cpu_prefills = query_lens_cpu_non_spec[num_decodes:]
                query_start_loc_p_cpu = torch.zeros(
                    num_prefills + 1,
                    dtype=common_attn_metadata.query_start_loc_cpu.dtype,
                )
                torch.cumsum(
                    query_lens_cpu_prefills, dim=0, out=query_start_loc_p_cpu[1:]
                )
            else:
                query_start_loc_p = (
                    common_attn_metadata.query_start_loc[-num_prefills - 1 :]
                    - total_decode_tokens
                )
                query_start_loc_p_cpu = (
                    common_attn_metadata.query_start_loc_cpu[-num_prefills - 1 :]
                    - total_decode_tokens
                )

            if self.vllm_config.cache_config.enable_prefix_caching:
                assert num_computed_tokens is not None
                if spec_sequence_masks is not None:
                    num_computed_tokens_non_spec = num_computed_tokens[
                        ~spec_sequence_masks
                    ]
                else:
                    num_computed_tokens_non_spec = num_computed_tokens
                num_computed_tokens_p = num_computed_tokens_non_spec[num_decodes:]
                assert block_idx_first_scheduled_token is not None
                block_idx_first_scheduled_token_p = block_idx_first_scheduled_token[
                    num_decodes:
                ]
            num_computed_tokens_p_cpu = num_computed_tokens_cpu_non_spec[num_decodes:]

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

        # CUDA graph padding for spec decode
        elif (
            self.use_spec_decode
            and spec_sequence_masks is not None
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            # Pad for CUDA graph (pure spec decode batch)
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                num_spec_decode_tokens
            )
            batch_size = min(self.decode_cudagraph_max_bs, num_input_tokens)

            # Copy and pad spec_state_indices_tensor
            self.spec_state_indices_tensor_buffer[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor_buffer[
                :batch_size
            ]
            spec_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)

            # Copy and pad spec_sequence_masks
            self.spec_sequence_masks_buffer[:num_spec_decodes].copy_(
                spec_sequence_masks, non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks_buffer[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            # Copy and pad spec_query_start_loc
            assert spec_query_start_loc is not None
            self.spec_query_start_loc_buffer[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]
            spec_query_start_loc = self.spec_query_start_loc_buffer[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            # Copy and pad num_accepted_tokens
            if num_accepted_tokens_filtered is not None:
                self.num_accepted_tokens_buffer[:num_spec_decodes].copy_(
                    num_accepted_tokens_filtered, non_blocking=True
                )
                num_accepted_tokens_filtered = self.num_accepted_tokens_buffer[
                    :batch_size
                ]
                num_accepted_tokens_filtered[num_spec_decodes:].fill_(1)

        # CUDA graph padding for non-spec decode
        elif (
            0 < num_decodes <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            # Pad state tensor for CUDA graph (regular decode only)
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_decodes)
            self.state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.state_indices_tensor[:num_input_tokens]
            non_spec_state_indices_tensor[num_decodes:] = PAD_SLOT_ID

            if self.vllm_config.cache_config.enable_prefix_caching:
                assert block_idx_last_scheduled_token is not None
                assert block_idx_last_computed_token is not None
                self.block_idx_last_scheduled_token[:num_decodes].copy_(
                    block_idx_last_scheduled_token, non_blocking=True
                )
                block_idx_last_scheduled_token = self.block_idx_last_scheduled_token[
                    :num_decode_tokens
                ]
                block_idx_last_scheduled_token[num_decodes:] = 0

                self.block_idx_last_computed_token[:num_decodes].copy_(
                    block_idx_last_computed_token, non_blocking=True
                )
                block_idx_last_computed_token = self.block_idx_last_computed_token[
                    :num_decode_tokens
                ]

        attn_metadata = Mamba2AttentionMetadata(
            num_reqs=common_attn_metadata.num_reqs,
            state_indices_tensor=non_spec_state_indices_tensor,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc_p=query_start_loc_p,
            prep_initial_states=prep_initial_states,
            chunk_size=self.chunk_size,
            has_initial_states_p=has_initial_states_p,
            seq_idx_p=seq_idx_p,
            cu_chunk_seqlen_p=cu_chunk_seqlen_p,
            last_chunk_indices_p=last_chunk_indices_p,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            block_idx_first_scheduled_token_p=block_idx_first_scheduled_token_p,
            block_idx_last_computed_token=block_idx_last_computed_token,
            num_computed_tokens_p=num_computed_tokens_p,
            # Speculative decode fields
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens_filtered,
            num_computed_tokens=num_computed_tokens,
        )

        return attn_metadata

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> Mamba2AttentionMetadata:
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba2.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"Mamba2 only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()
        m._num_computed_tokens_cpu = m.seq_lens_cpu - num_accepted_tokens.cpu()

        return self.build(
            0,
            m,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        )

    def update_block_table(
        self,
        metadata: Mamba2AttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
    ) -> Mamba2AttentionMetadata:
        new_metadata = copy.copy(metadata)
        spec_sequence_masks = new_metadata.spec_sequence_masks
        prefix_caching = self.vllm_config.cache_config.enable_prefix_caching
        num_reqs = blk_table.shape[0]
        non_spec_state_indices_tensor = blk_table if prefix_caching else blk_table[:, 0]

        if (
            metadata.num_prefills == 0
            and num_reqs <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            assert self.use_spec_decode is False, (
                "Mamba2AttentionBackend metadata caching isn't supported for "
                "full CUDA graphs with spec decode"
            )

            persistent_state_indices_t = self.state_indices_tensor[:num_reqs]
            persistent_state_indices_t.copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = persistent_state_indices_t

        if spec_sequence_masks is None:
            new_metadata.non_spec_state_indices_tensor = non_spec_state_indices_tensor
            new_metadata.state_indices_tensor = (
                new_metadata.non_spec_state_indices_tensor
            )

            return new_metadata

        num_cacheable_blocks = None
        spec_token_range = None

        if prefix_caching:
            num_computed_tokens = new_metadata.num_computed_tokens
            assert num_computed_tokens is not None

            num_cacheable_blocks = num_computed_tokens // self.kv_cache_spec.block_size

            spec_token_range = torch.arange(self.num_spec + 1, device=self.device)

        is_decode_only = new_metadata.num_prefills == 0

        if is_decode_only:
            if prefix_caching:
                assert num_cacheable_blocks is not None
                block_indices = num_cacheable_blocks.unsqueeze(1) + spec_token_range
                batch_indices = torch.arange(
                    blk_table.size(0),
                    device=self.device,
                ).unsqueeze(1)
                spec_state_indices_tensor = blk_table[batch_indices, block_indices]
            else:
                spec_state_indices_tensor = blk_table[:, : self.num_spec + 1]

        else:  # mixed batch
            if prefix_caching:
                assert num_cacheable_blocks is not None
                block_indices = (
                    num_cacheable_blocks[spec_sequence_masks].unsqueeze(1)
                    + spec_token_range
                )
                batch_indices = torch.arange(
                    spec_sequence_masks.sum().item(), device=self.device
                ).unsqueeze(1)
                spec_state_indices_tensor = blk_table[spec_sequence_masks][
                    batch_indices, block_indices
                ]
                non_spec_state_indices_tensor = blk_table[~spec_sequence_masks]
            else:
                spec_state_indices_tensor = blk_table[
                    spec_sequence_masks, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = blk_table[~spec_sequence_masks, 0]

            new_metadata.non_spec_state_indices_tensor = non_spec_state_indices_tensor

        new_metadata.spec_state_indices_tensor = spec_state_indices_tensor
        return new_metadata
