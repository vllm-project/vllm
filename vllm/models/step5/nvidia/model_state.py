# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRv2 model state that exposes persistent request indices to Step5 DSA."""

from typing import Any

import torch

from vllm.config import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.utils import AttentionGroup


class Step5ModelState(DefaultModelState):
    """DefaultModelState plus request state indices for DSA speculative undo.

    DSA summaries keep a per-request undo journal so a rejected draft window
    can be rolled back on device. Journal rows are addressed by the persistent
    request state index, which must stay at a fixed address for FULL CUDA
    graph replay; padded batch rows are -1.
    """

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        self.request_state = torch.full(
            (self.max_num_reqs,), -1, dtype=torch.int64, device=device
        )

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
        from .sparse_attention import Step5DSAMetadataBuilder

        num_reqs = input_batch.num_reqs
        self.request_state[num_reqs:].fill_(-1)
        self.request_state[:num_reqs].copy_(input_batch.idx_mapping)
        for groups in attn_groups:
            for group in groups:
                builder = group.get_metadata_builder(ubatch_idx)
                if isinstance(builder, Step5DSAMetadataBuilder):
                    builder.step5_request_state = self.request_state
        return super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )
