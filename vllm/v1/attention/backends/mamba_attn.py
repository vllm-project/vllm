# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import abc
from dataclasses import dataclass, replace
from typing import Any, ClassVar, TypeVar

import torch

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
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
    # the batch has no prefill requests.
    has_initial_states_p: torch.Tensor | None
    query_start_loc_p: torch.Tensor | None
    num_computed_tokens_p: torch.Tensor | None
    state_indices_tensor_p: torch.Tensor | None

    # The following tensors are used for decode requests and
    # speculative decoding compatibility, and will be None if the batch
    # has no decode requests.
    state_indices_tensor_d: torch.Tensor | None
    query_start_loc_d: torch.Tensor | None  # shape: [num_decodes + 1,]

    # Number of accepted tokens for each spec sequence (for loading correct checkpoint)
    # Includes the bonus token (so minimum is 1)
    num_accepted_tokens: torch.Tensor | None  # shape: [batch,]

    # The following tensors are only used for prefix caching in all mode and
    # are None if disabled
    block_idx_last_scheduled_token: torch.Tensor | None
    block_idx_first_scheduled_token_p: torch.Tensor | None
    block_idx_last_computed_token: torch.Tensor | None

    # The following tensor is only used for prefix caching in align mode
    seq_lens: torch.Tensor

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


class BaseMambaAttentionMetadataBuilder(AttentionMetadataBuilder[M], abc.ABC):
    metadata_cls: type[M]
    reorder_batch_threshold: int = 1
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    # Will be disabled if speculative decoding is used
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # Enable speculative decoding support
        self.speculative_config = vllm_config.speculative_config
        self.compilation_config = vllm_config.compilation_config
        self.num_spec_tokens: int = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config is not None
            else 0
        )
        if self.num_spec_tokens is None:
            self.num_spec_tokens = 0
        self.use_spec_decode = self.num_spec_tokens > 0

        assert isinstance(kv_cache_spec, MambaSpec)
        self.compilation_config = vllm_config.compilation_config
        self.decode_cudagraph_max_bs = self.vllm_config.scheduler_config.max_num_seqs
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        if self.vllm_config.cache_config.mamba_cache_mode == "all":
            max_num_blocks = cdiv(
                self.vllm_config.model_config.max_model_len,
                self.kv_cache_spec.block_size,
            )
            # Speculative decoding not supported with prefix caching,
            # so keep shape consistent with prefill buffer
            # TODO: reduce this size as needed for decode-only cudagraph capture
            self.state_indices_tensor_d = torch.empty(
                (
                    self.decode_cudagraph_max_bs,
                    max_num_blocks,
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
            self.state_indices_tensor_d = torch.empty(
                (self.decode_cudagraph_max_bs, 1 + self.num_spec_tokens),
                dtype=torch.int32,
                device=device,
            )

        # For speculative decoding, we need to store the following buffers
        # for CUDA graph capture during decode
        if self.num_spec_tokens > 0:
            self.decode_num_accepted_tokens = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )

        self._init_reorder_batch_threshold(1, self.use_spec_decode)
        if self.use_spec_decode:
            self.supports_update_block_table = False

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> M:
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.max_query_len <= 1 + self.num_spec_tokens
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            "Mamba only supports decode-only full CUDAGraph capture. "
            "Make sure all cudagraph capture sizes <= max_num_seq."
        )

        assert m.max_query_len == 1 + self.num_spec_tokens  # decode-only

        num_accepted_tokens = None
        if self.num_spec_tokens > 0:
            num_accepted_tokens = torch.diff(m.query_start_loc)

        return self.build(0, m, num_accepted_tokens=num_accepted_tokens)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> M:
        """
        Default build implementation for Mamba-like attention backends.
        Subclasses (e.g., Mamba2) can override to add additional metadata.
        """
        return self._compute_common_metadata(
            common_attn_metadata, num_accepted_tokens=num_accepted_tokens
        )

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
        block_idx_last_computed_token = torch.clamp(
            block_idx_last_computed_token, min=0
        )
        # -1 in the case we have a padded request (0 seq-len)
        block_idx_last_scheduled_token = torch.clamp(
            block_idx_last_scheduled_token, min=0
        )

        return (
            block_idx_last_computed_token,
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
        )

    def _compute_common_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
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
        query_start_loc_d = None
        num_computed_tokens = None
        num_computed_tokens_p = None

        # for prefix caching
        block_idx_first_scheduled_token = None
        block_idx_first_scheduled_token_p = None
        block_idx_last_computed_token = None
        block_idx_last_scheduled_token = None

        # for causal_conv1d
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None

        if self.vllm_config.cache_config.mamba_cache_mode == "all":
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
            state_indices_tensor = mamba_get_block_table_tensor(
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.seq_lens,
                self.kv_cache_spec,
                self.vllm_config.cache_config.mamba_cache_mode,
            )

        if state_indices_tensor.dim() == 1:
            state_indices_tensor = state_indices_tensor.unsqueeze(-1)

        state_indices_tensor_d, state_indices_tensor_p = torch.split(
            state_indices_tensor,
            [num_decodes, num_prefills],
            dim=0,
        )
        state_indices_tensor_d = state_indices_tensor_d[:, : 1 + self.num_spec_tokens]
        state_indices_tensor_p = state_indices_tensor_p[:, 0]

        if num_decodes > 0 and self.use_spec_decode:
            assert num_accepted_tokens is not None
            query_start_loc_d = common_attn_metadata.query_start_loc[: num_decodes + 1]
            num_accepted_tokens = num_accepted_tokens[:num_decodes]

        if num_prefills > 0:
            if num_computed_tokens is None:
                num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()

            query_start_loc_p_cpu = (
                common_attn_metadata.query_start_loc_cpu[-num_prefills - 1 :]
                - num_decode_tokens
            )
            query_start_loc_p = (
                common_attn_metadata.query_start_loc[-num_prefills - 1 :]
                - num_decode_tokens
            )
            has_initial_states_p = (
                num_computed_tokens[num_reqs - num_prefills : num_reqs] > 0
            )

            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    query_start_loc_p_cpu,
                    device=common_attn_metadata.query_start_loc.device,
                )
            )

            if self.vllm_config.cache_config.mamba_cache_mode == "all":
                assert num_computed_tokens is not None
                num_computed_tokens_p = num_computed_tokens[
                    num_reqs - num_prefills : num_reqs
                ]
                assert block_idx_first_scheduled_token is not None
                block_idx_first_scheduled_token_p = block_idx_first_scheduled_token[
                    num_reqs - num_prefills : num_reqs
                ]

        metadata = self.metadata_cls(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc_p=query_start_loc_p,
            has_initial_states_p=has_initial_states_p,
            state_indices_tensor_p=state_indices_tensor_p,
            state_indices_tensor_d=state_indices_tensor_d,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc_d=query_start_loc_d,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            block_idx_first_scheduled_token_p=block_idx_first_scheduled_token_p,
            block_idx_last_computed_token=block_idx_last_computed_token,
            num_computed_tokens_p=num_computed_tokens_p,
            num_reqs=num_reqs,
            seq_lens=common_attn_metadata.seq_lens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )

        return self._update_metadata_for_cudagraph_capture(metadata)

    def _update_metadata_for_cudagraph_capture(
        self,
        metadata: M,
    ) -> M:
        """
        Update the metadata for cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        state_indices_tensor_d = metadata.state_indices_tensor_d
        query_start_loc_d = metadata.query_start_loc_d
        num_accepted_tokens = metadata.num_accepted_tokens
        block_idx_last_scheduled_token = metadata.block_idx_last_scheduled_token
        block_idx_last_computed_token = metadata.block_idx_last_computed_token
        if (
            metadata.num_prefills == 0
            and metadata.num_decodes <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            padded_bs = metadata.num_reqs
            self.state_indices_tensor_d[: metadata.num_decodes].copy_(
                state_indices_tensor_d, non_blocking=True
            )
            state_indices_tensor_d = self.state_indices_tensor_d[:padded_bs]
            state_indices_tensor_d[metadata.num_decodes :] = PAD_SLOT_ID

            if self.use_spec_decode:
                assert query_start_loc_d is not None
                assert num_accepted_tokens is not None
                query_start_loc_d = query_start_loc_d[: padded_bs + 1]
                self.decode_num_accepted_tokens[: metadata.num_decodes].copy_(
                    num_accepted_tokens, non_blocking=True
                )
                num_accepted_tokens = self.decode_num_accepted_tokens[:padded_bs]
                num_accepted_tokens[metadata.num_decodes :] = (
                    1  # pad with 1st slot index
                )

            if self.vllm_config.cache_config.mamba_cache_mode == "all":
                assert block_idx_last_scheduled_token is not None
                assert block_idx_last_computed_token is not None
                self.block_idx_last_scheduled_token[: metadata.num_decodes].copy_(
                    block_idx_last_scheduled_token, non_blocking=True
                )
                block_idx_last_scheduled_token = self.block_idx_last_scheduled_token[
                    :padded_bs
                ]

                self.block_idx_last_computed_token[: metadata.num_decodes].copy_(
                    block_idx_last_computed_token, non_blocking=True
                )
                block_idx_last_computed_token = self.block_idx_last_computed_token[
                    :padded_bs
                ]

        return replace(
            metadata,
            state_indices_tensor_d=state_indices_tensor_d,
            query_start_loc_d=query_start_loc_d,
            num_accepted_tokens=num_accepted_tokens,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            block_idx_last_computed_token=block_idx_last_computed_token,
        )

    def update_block_table(
        self,
        metadata: M,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> M:
        state_indices_tensor = mamba_get_block_table_tensor(
            blk_table,
            metadata.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
        if state_indices_tensor.dim() == 1:
            state_indices_tensor = state_indices_tensor.unsqueeze(-1)

        assert (
            metadata.num_prefills + metadata.num_decodes
            == state_indices_tensor.shape[0]
        ), (
            "Mismatch in number of requests when updating block table."
            f" Expected {metadata.num_prefills + metadata.num_decodes}, "
            f"got {state_indices_tensor.shape[0]}."
        )

        state_indices_tensor_d, state_indices_tensor_p = torch.split(
            state_indices_tensor,
            [metadata.num_decodes, metadata.num_prefills],
            dim=0,
        )
        state_indices_tensor_d = state_indices_tensor_d[:, : 1 + self.num_spec_tokens]
        state_indices_tensor_p = state_indices_tensor_p[:, 0]

        new_metadata = replace(
            metadata,
            state_indices_tensor_d=state_indices_tensor_d,
            state_indices_tensor_p=state_indices_tensor_p,
        )

        return self._update_metadata_for_cudagraph_capture(new_metadata)
