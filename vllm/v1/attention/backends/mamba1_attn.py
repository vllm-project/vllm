# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import VllmConfig
from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


class Mamba1AttentionBackend(AttentionBackend):
    @staticmethod
    def get_builder_cls() -> type["Mamba1AttentionMetadataBuilder"]:
        return Mamba1AttentionMetadataBuilder


@dataclass
class Mamba1AttentionMetadata:
    query_start_loc_p: torch.Tensor
    state_indices_tensor: torch.Tensor
    has_initial_states_p: torch.Tensor | None
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_padded_decodes: int

    block_idx_last_scheduled_token: torch.Tensor  # shape: [batch,]
    block_idx_first_scheduled_token_p: torch.Tensor  # shape: [batch,]
    block_idx_last_computed_token: torch.Tensor  # shape: [batch,]
    num_computed_tokens_p: torch.Tensor  # shape: [batch,]


class Mamba1AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba1AttentionMetadata]
):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert isinstance(kv_cache_spec, MambaSpec)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Mamba1AttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.reorder_batch_threshold
            )
        )

        has_initial_states_p = None
        query_start_loc_p = None
        padded_decodes = num_decodes
        num_computed_tokens, num_computed_tokens_p = None, None
        block_idx_first_scheduled_token = None
        block_idx_first_scheduled_token_p = None

        # TODO(@Josephasafg) Mamba1 and Mamba2 have a lot of code in common here.
        # We should consolidate this code
        if self.vllm_config.cache_config.enable_prefix_caching:
            # Return a tensor of shape (#requests, #max blocks)
            state_indices_tensor = common_attn_metadata.block_table_tensor
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
            block_idx_last_scheduled_token = None
            block_idx_last_computed_token = None

        if num_prefills > 0:
            query_start_loc_p = (
                common_attn_metadata.query_start_loc[-num_prefills - 1 :]
                - num_decode_tokens
            )
            has_initial_states_cpu = (
                common_attn_metadata.num_computed_tokens_cpu[
                    num_reqs - num_prefills : num_reqs
                ]
                > 0
            )
            has_initial_states_p = has_initial_states_cpu.to(
                common_attn_metadata.query_start_loc.device
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
            num_decodes > 0
            and num_decodes <= self.decode_cudagraph_max_bs
            and self.compilation_config.full_cuda_graph
        ):
            padded_decodes = self.vllm_config.pad_for_cudagraph(num_decodes)
            self.state_indices_tensor[:num_decodes].copy_(
                state_indices_tensor, non_blocking=True
            )
            state_indices_tensor = self.state_indices_tensor[:padded_decodes]
            state_indices_tensor[num_decodes:] = PAD_SLOT_ID

            if self.vllm_config.cache_config.enable_prefix_caching:
                self.block_idx_last_scheduled_token[:num_decodes].copy_(
                    block_idx_last_scheduled_token, non_blocking=True
                )
                block_idx_last_scheduled_token = self.block_idx_last_scheduled_token[
                    :padded_decodes
                ]
                block_idx_last_scheduled_token[num_decodes:] = 0

                self.block_idx_last_computed_token[:num_decodes].copy_(
                    block_idx_last_computed_token, non_blocking=True
                )
                block_idx_last_computed_token = self.block_idx_last_computed_token[
                    :padded_decodes
                ]
                block_idx_last_computed_token[num_decodes:] = 0

        return Mamba1AttentionMetadata(
            query_start_loc_p=query_start_loc_p,
            has_initial_states_p=has_initial_states_p,
            state_indices_tensor=state_indices_tensor,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_padded_decodes=padded_decodes,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            block_idx_first_scheduled_token_p=block_idx_first_scheduled_token_p,
            block_idx_last_computed_token=block_idx_last_computed_token,
            num_computed_tokens_p=num_computed_tokens_p,
        )
