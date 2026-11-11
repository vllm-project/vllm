# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.multimodal.inputs import PlaceholderRange
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState


class Ernie4_5_VLModelState(DefaultModelState):
    """Mirror the encoder gather's presence gate on every pipeline stage."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        # Keep positions, not image/video payloads or cached encoder tensors.
        self._mm_positions: dict[str, tuple[PlaceholderRange, ...]] = {}

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        positions = tuple(feature.mm_position for feature in new_req_data.mm_features)
        for position in positions:
            if position.is_embed is not None and not position.is_embed.is_cpu:
                raise ValueError("ERNIE multimodal placeholder metadata must be on CPU")
            # Resolve optional sparse-mask prefix counts at admission, not forward.
            _ = position.embeds_cumsum
        super().add_request(req_index, new_req_data)
        self._mm_positions[new_req_data.req_id] = positions

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        self._mm_positions.pop(req_id, None)

    def _has_multimodal_embeds(self, input_batch: InputBatch) -> bool:
        for req_id, start, count, prefill_len in zip(
            input_batch.req_ids,
            input_batch.num_computed_tokens_np,
            input_batch.num_scheduled_tokens,
            input_batch.prefill_len_np,
            strict=True,
        ):
            # Like EncoderRunner.gather_mm_embeddings, exclude decode requests.
            if start >= prefill_len:
                continue
            end = int(start + count)
            for position in self._mm_positions.get(req_id, ()):
                first = max(int(start) - position.offset, 0)
                last = min(end - position.offset, position.length)
                if first >= last:
                    continue
                embed_start, embed_end = position.get_embeds_indices_in_range(
                    first, last
                )
                if embed_start != embed_end:
                    return True
        return False

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        inputs: dict[str, Any] = super().prepare_inputs(input_batch, req_states)
        # This is a whole-step gate, not a replacement for ERNIE's token-ID mask.
        inputs["has_multimodal_embeds"] = self._has_multimodal_embeds(input_batch)
        return inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        inputs["has_multimodal_embeds"] = False
        return inputs
