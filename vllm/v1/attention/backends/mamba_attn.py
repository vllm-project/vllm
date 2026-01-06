# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import abc
import copy
from dataclasses import dataclass
from typing import ClassVar, TypeVar

import torch

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    compute_causal_conv1d_metadata,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

M = TypeVar("M", bound="BaseMambaAttentionMetadata")


@dataclass
class BaseMambaAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_reqs: int

    # The following tensors only contain prefill requests and will be None if
    # the batch has no prefill request.
    has_initial_states_p: torch.Tensor | None
    query_start_loc_p: torch.Tensor | None
    num_computed_tokens_p: torch.Tensor | None

    state_indices_tensor: torch.Tensor

    # The following tensors are only used for prefix caching and are None if disabled
    block_idx_last_scheduled_token: torch.Tensor | None
    block_idx_first_scheduled_token_p: torch.Tensor | None
    block_idx_last_computed_token: torch.Tensor | None

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


class BaseMambaAttentionMetadataBuilder(AttentionMetadataBuilder[M], abc.ABC):
    metadata_cls: type[M]
    reorder_batch_threshold: int = 1
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        assert isinstance(kv_cache_spec, MambaSpec)
        self.compilation_config = vllm_config.compilation_config
        self.decode_cudagraph_max_bs = min(
            self.vllm_config.scheduler_config.max_num_seqs,
            self.compilation_config.max_cudagraph_capture_size,
        )

        if self.vllm_config.cache_config.enable_prefix_caching:
            self.state_indices_tensor = torch.empty(
                (
                    self.decode_cudagraph_max_bs,
                    cdiv(
                        self.vllm_config.model_config.max_model_len,
                        self.kv_cache_spec.block_size,
                    ),
                ),
                dtype=torch.int32,
                device=device,
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
        else:
            self.state_indices_tensor = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> M:
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert m.num_reqs == m.num_actual_tokens, (
            "Mamba only supports decode-only full CUDAGraph capture. "
            "Make sure all cudagraph capture sizes <= max_num_seq."
        )

        m.max_query_len = 1  # decode-only

        return self.build(0, m)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> M:
        """
        Default build implementation for Mamba-like attention backends.
        Subclasses (e.g., Mamba2) can override to add additional metadata.
        """
        return self._compute_common_metadata(common_attn_metadata)

    def _compute_prefix_caching_block_indices(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        mamba_block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()
        # Block index of the last computed token
        block_idx_last_computed_token = cdiv(num_computed_tokens, mamba_block_size) - 1
        # which is <= block index for the first scheduled token
        block_idx_first_scheduled_token = (
            cdiv(num_computed_tokens + 1, mamba_block_size) - 1
        )
        # which is <= block index of the last scheduled token
        block_idx_last_scheduled_token = (
            cdiv(common_attn_metadata.seq_lens, mamba_block_size) - 1
        )
        # -1 in case it's non-computed and causes later issues with indexing
        block_idx_last_computed_token = block_idx_last_computed_token.clamp(min=0)
        # -1 in the case we have a padded request (0 seq-len)
        block_idx_last_scheduled_token = block_idx_last_scheduled_token.clamp(min=0)

        return (
            block_idx_last_computed_token,
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
        )

    def _compute_common_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> M:
        """
        Compute metadata common to both Mamba1 and Mamba2.
        """
        num_reqs = common_attn_metadata.num_reqs

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.reorder_batch_threshold
            )
        )

        # Need flags to indicate if there are initial states
        has_initial_states_p = None
        query_start_loc_p = None
        num_computed_tokens = None
        num_computed_tokens_p = None

        # for prefix caching
        block_idx_first_scheduled_token = None
        block_idx_first_scheduled_token_p = None
        block_idx_last_computed_token = None
        block_idx_last_scheduled_token = None

        # for causal_conv1d
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None

        if self.vllm_config.cache_config.enable_prefix_caching:
            num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()

            # Return a tensor of shape (#requests, #max blocks)
            state_indices_tensor = common_attn_metadata.block_table_tensor
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
            # Always return just a single block per each request:
            state_indices_tensor = common_attn_metadata.block_table_tensor[:, 0]

        if num_prefills > 0:
            if num_computed_tokens is None:
                num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()
            num_computed_tokens_cpu = num_computed_tokens.cpu()

            query_start_loc_p = (
                common_attn_metadata.query_start_loc[-num_prefills - 1 :]
                - num_decode_tokens
            )
            has_initial_states_cpu = (
                num_computed_tokens_cpu[num_reqs - num_prefills : num_reqs] > 0
            )
            has_initial_states_p = has_initial_states_cpu.to(
                common_attn_metadata.query_start_loc.device
            )

            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(query_start_loc_p)
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
        elif (
            num_decodes <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            self.state_indices_tensor[:num_decodes].copy_(
                state_indices_tensor, non_blocking=True
            )
            state_indices_tensor = self.state_indices_tensor[:num_decode_tokens]
            state_indices_tensor[num_decodes:] = PAD_SLOT_ID

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

        return self.metadata_cls(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc_p=query_start_loc_p,
            has_initial_states_p=has_initial_states_p,
            state_indices_tensor=state_indices_tensor,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            block_idx_first_scheduled_token_p=block_idx_first_scheduled_token_p,
            block_idx_last_computed_token=block_idx_last_computed_token,
            num_computed_tokens_p=num_computed_tokens_p,
            num_reqs=num_reqs,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )

    def update_block_table(
        self,
        metadata: M,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> M:
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
