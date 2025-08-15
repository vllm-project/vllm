# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import (AttentionMetadataBuilder,
                                              CommonAttentionMetadata,
                                              split_decodes_and_prefills)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


class Mamba1AttentionBackend(AttentionBackend):

    @staticmethod
    def get_builder_cls() -> type["Mamba1AttentionMetadataBuilder"]:
        return Mamba1AttentionMetadataBuilder


@dataclass
class Mamba1AttentionMetadata:
    query_start_loc: torch.Tensor
    context_lens_tensor: torch.Tensor
    state_indices_tensor: torch.Tensor
    has_initial_states: Optional[torch.Tensor]
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int


class Mamba1AttentionMetadataBuilder(
        AttentionMetadataBuilder[Mamba1AttentionMetadata]):
    reorder_batch_threshold: ClassVar[int] = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        vllm_config: VllmConfig,
        device: torch.device,
        layer_names: list[str],
    ):
        assert isinstance(kv_cache_spec, MambaSpec)
        self.kv_cache_spec = kv_cache_spec
        self.device = device
        self.vllm_config = vllm_config
        self.layer_names = layer_names

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Mamba1AttentionMetadata:
        query_start_loc = common_attn_metadata.query_start_loc

        state_indices_tensor = common_attn_metadata.block_table_tensor[:, 0]
        context_lens_tensor = common_attn_metadata.num_computed_tokens_cpu.to(
            query_start_loc.device)

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(common_attn_metadata,
                                       decode_threshold=1))

        has_initial_states = None

        if num_prefills > 0:
            has_initial_states = context_lens_tensor > 0

        return Mamba1AttentionMetadata(
            query_start_loc=query_start_loc,
            context_lens_tensor=context_lens_tensor,
            has_initial_states=has_initial_states,
            state_indices_tensor=state_indices_tensor,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
        )
