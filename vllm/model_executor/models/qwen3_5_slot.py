# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5 slot-readout model with a prefix/question PrefixLM mask."""

from typing import Any

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.core.sched.prefix_lm_scheduler import prefix_lm_len
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.utils import AttentionGroup

from .qwen3_5 import Qwen3_5ForConditionalGeneration


class Qwen3_5SlotDecisionModelState(MambaHybridModelState):
    """Makes a PrefixLM request's prefix and question its bidirectional ranges."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.blocks: dict[str, list[tuple[int, int]]] = {}

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        if new_req_data.mm_features:
            # The encoder budget can end a step inside the prompt.
            return
        ids = new_req_data.prompt_token_ids
        prefix_len = prefix_lm_len(new_req_data.sampling_params, ids)
        if prefix_len is not None:
            assert ids is not None
            blocks = [(0, prefix_len - 1)]
            if prefix_len < len(ids):
                blocks.append((prefix_len, len(ids) - 1))
            self.blocks[new_req_data.req_id] = blocks

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        self.blocks.pop(req_id, None)

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        ranges = self._block_ranges(input_batch)
        # A FULL graph runs one-token queries, which the ranges leave causal.
        ranged = ranges is not None and cudagraph_mode != CUDAGraphMode.FULL
        # With ranges, this state builds the full-attention groups.
        own = [
            ranged and isinstance(group.kv_cache_spec, AttentionSpec)
            for group in kv_cache_config.kv_cache_groups
        ]
        attn_metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            [[] if o else g for o, g in zip(own, attn_groups)],
            kv_cache_config,
            *args,
            **kwargs,
        )
        if not ranged:
            return attn_metadata
        num_reqs = input_batch.num_reqs
        return attn_metadata | build_attn_metadata(
            attn_groups=[g if o else [] for o, g in zip(own, attn_groups)],
            num_reqs=num_reqs,
            num_tokens=input_batch.num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np),
            max_query_len=input_batch.max_query_len
            or int(input_batch.num_scheduled_tokens.max()),
            seq_lens=input_batch.seq_lens,
            max_seq_len=int(input_batch.seq_lens_cpu_upper_bound[:num_reqs].max()),
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
            mm_req_doc_ranges=ranges,
            rswa_prefix_lens=input_batch.prompt_lens,
        )

    def _block_ranges(
        self, input_batch: InputBatch
    ) -> dict[int, list[tuple[int, int]]] | None:
        if not self.blocks:
            return None
        ranges: dict[int, list[tuple[int, int]]] = {}
        for i, req_id in enumerate(input_batch.req_ids):
            # Decode queries follow every block, where the mask is causal.
            if (
                not input_batch.is_prefilling_np[i]
                or (blocks := self.blocks.get(req_id)) is None
            ):
                continue
            # A step that ends inside a block computes the KV of its earlier
            # tokens without the later ones.
            end = int(
                input_batch.num_computed_prefill_tokens_np[i]
                + input_batch.num_scheduled_tokens[i]
            )
            if end <= blocks[-1][1] and end != blocks[0][1] + 1:
                raise RuntimeError(
                    f"PrefixLM request {req_id} is scheduled up to token "
                    f"{end}, inside one of its blocks {blocks}"
                )
            ranges[i] = blocks
        return ranges or None


class Qwen3_5ForSlotDecision(Qwen3_5ForConditionalGeneration):
    @staticmethod
    def get_model_state_cls() -> type[Qwen3_5SlotDecisionModelState]:
        return Qwen3_5SlotDecisionModelState
