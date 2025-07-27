# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder, CommonAttentionMetadata,
    reorder_batch_to_split_decodes_and_prefills, split_decodes_and_prefills)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch


class Mamba1AttentionBackend(AttentionBackend):

    @staticmethod
    def get_builder_cls() -> type["Mamba1AttentionMetadataBuilder"]:
        return Mamba1AttentionMetadataBuilder


@dataclass
class Mamba1AttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    context_lens_tensor: torch.Tensor
    state_indices_tensor: torch.Tensor
    has_initial_states: torch.Tensor


class Mamba1AttentionMetadataBuilder(
        AttentionMetadataBuilder[Mamba1AttentionMetadata]):

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        assert isinstance(kv_cache_spec, MambaSpec)
        self.kv_cache_spec = kv_cache_spec
        self.device = device
        self.vllm_config = vllm_config

    def reorder_batch(
        self,
        input_batch: "InputBatch",
        scheduler_output: "SchedulerOutput",
    ) -> bool:
        return reorder_batch_to_split_decodes_and_prefills(
            input_batch,
            scheduler_output,
            decode_threshold=1,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Mamba1AttentionMetadata:
        query_start_loc = common_attn_metadata.query_start_loc

        seq_lens = (common_attn_metadata.seq_lens.to(
            query_start_loc.device).to(torch.int32))

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = \
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=1,
            )

        state_indices_tensor = common_attn_metadata.block_table_tensor[:, 0]
        context_lens_tensor = common_attn_metadata.num_computed_tokens_cpu.to(
            query_start_loc.device)
        has_initial_states = (context_lens_tensor > 0)

        return Mamba1AttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc=query_start_loc,
            context_lens_tensor=context_lens_tensor,
            seq_lens=seq_lens,
            has_initial_states=has_initial_states,
            state_indices_tensor=state_indices_tensor,
        )
