# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma 4 slot-readout model with a PrefixLM mask on its full-attention layers."""

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.core.single_type_kv_cache_manager import get_kv_cache_kind
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpecKind
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.utils import AttentionGroup

from .gemma4 import Gemma4Attention
from .gemma4_mm import Gemma4ForConditionalGeneration
from .slot_decision import PrefixLMBlocksMixin


def full_attention_groups(kv_cache_config: KVCacheConfig) -> list[bool]:
    # Sliding-window groups also hold an AttentionSpec, so read the kind.
    return [
        get_kv_cache_kind(group.manager_config, group.distinct_specs)
        == KVCacheSpecKind.FULL_ATTENTION
        for group in kv_cache_config.kv_cache_groups
    ]


class Gemma4SlotDecisionModelState(PrefixLMBlocksMixin, DefaultModelState):
    """Makes a PrefixLM request's prefix and question bidirectional in full layers."""

    def get_additional_cg_support(self) -> tuple[AttentionCGSupport, str | None]:
        # A FULL graph applies no ranges, so only one-token decodes may run in one.
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE, type(self).__name__

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, block_tables: BlockTables
    ) -> None:
        super().initialize_kv_cache(kv_cache_config, block_tables)
        self.full_groups = full_attention_groups(kv_cache_config)

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        ranges = self._block_ranges(input_batch)
        # A FULL graph runs one-token queries, which the ranges leave causal.
        if cudagraph_mode == CUDAGraphMode.FULL:
            ranges = None
        has_mm_ranges = self.mm_prefix_ranges is not None and any(
            self.mm_prefix_ranges.get(req_id) for req_id in input_batch.req_ids
        )
        own = [
            (ranges is not None or has_mm_ranges) and full for full in self.full_groups
        ]
        attn_metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            [[] if o else g for o, g in zip(own, attn_groups)],
            kv_cache_config,
            for_capture,
            ubatch_idx,
            model_specific_attn_metadata,
        )
        if not any(own):
            return attn_metadata
        # As DefaultModelState builds them, but with the PrefixLM ranges alone.
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        max_query_len = input_batch.max_query_len
        if max_query_len is None:
            max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            max_seq_len = self.max_model_len
        else:
            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()
        pcp = input_batch.pcp
        return attn_metadata | build_attn_metadata(
            attn_groups=[g if o else [] for o, g in zip(own, attn_groups)],
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc[: num_reqs + 1],
            query_start_loc_cpu=torch.from_numpy(
                input_batch.query_start_loc_np[: num_reqs + 1]
            ),
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            dcp_local_seq_lens_cpu_upper_bound=(
                None if pcp is None else pcp.dcp_local_seq_lens_cpu_upper_bound
            ),
            positions=input_batch.positions,
            is_prefilling=torch.from_numpy(input_batch.is_prefilling_np),
            mm_req_doc_ranges=ranges,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
            req_idx=input_batch.idx_mapping_np,
            ubatch_idx=ubatch_idx,
            fast_prefill=input_batch.fast_prefill,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )


class Gemma4ForSlotDecision(Gemma4ForConditionalGeneration):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        for module in self.modules():
            if isinstance(module, Gemma4Attention) and not module.is_sliding:
                module.attn.doc_range_segments = True

    @staticmethod
    def get_model_state_cls() -> type[Gemma4SlotDecisionModelState]:
        return Gemma4SlotDecisionModelState

    def _clear_mm_prefix_for_full_attn_layers(self) -> None:
        """No-op: the model state never gives full-attention groups image ranges."""
