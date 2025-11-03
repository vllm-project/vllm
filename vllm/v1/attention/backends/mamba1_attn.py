# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import VllmConfig
from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


class Mamba1AttentionBackend(AttentionBackend):
    @staticmethod
    def get_builder_cls() -> type["Mamba1AttentionMetadataBuilder"]:
        return Mamba1AttentionMetadataBuilder


@dataclass
class Mamba1AttentionMetadata:
    query_start_loc_p: torch.Tensor | None
    state_indices_tensor: torch.Tensor
    has_initial_states_p: torch.Tensor | None
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_padded_decodes: int

    block_idx_last_scheduled_token: torch.Tensor | None  # shape: [batch,]
    block_idx_first_scheduled_token_p: torch.Tensor | None  # shape: [batch,]
    block_idx_last_computed_token: torch.Tensor | None  # shape: [batch,]
    num_computed_tokens_p: torch.Tensor | None  # shape: [batch,]


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
        common = self._compute_common_metadata(common_attn_metadata)

        return Mamba1AttentionMetadata(
            query_start_loc_p=common.query_start_loc_p,
            has_initial_states_p=common.has_initial_states_p,
            state_indices_tensor=common.state_indices_tensor,
            num_prefills=common.num_prefills,
            num_prefill_tokens=common.num_prefill_tokens,
            num_decodes=common.num_decodes,
            num_decode_tokens=common.num_decode_tokens,
            num_padded_decodes=common.num_padded_decodes,
            block_idx_last_scheduled_token=common.block_idx_last_scheduled_token,
            block_idx_first_scheduled_token_p=common.block_idx_first_scheduled_token_p,
            block_idx_last_computed_token=common.block_idx_last_computed_token,
            num_computed_tokens_p=common.num_computed_tokens_p,
        )
