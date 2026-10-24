# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5 with a slot readout, whose full attention layers read a request's
prefix and question as two bidirectional PrefixLM blocks."""

from vllm.config import VllmConfig
from vllm.sampling_params import PREFIX_LM_LEN
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

from .qwen3_5 import Qwen3_5ForConditionalGeneration


class Qwen3_5SlotDecisionModelState(MambaHybridModelState):
    """Makes a request's prefix and question its PrefixLM ranges. A request
    without a valid prefix length is read causally."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.blocks: dict[str, list[tuple[int, int]]] = {}

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        params = new_req_data.sampling_params
        prefix_len = (params.extra_args or {}).get(PREFIX_LM_LEN) if params else None
        length = len(new_req_data.prompt_token_ids or ())
        if type(prefix_len) is int and 0 < prefix_len <= length:
            blocks = [(0, prefix_len - 1)]
            if prefix_len < length:
                blocks.append((prefix_len, length - 1))
            self.blocks[new_req_data.req_id] = blocks

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        self.blocks.pop(req_id, None)

    def get_mm_prefix_ranges(
        self, input_batch: InputBatch
    ) -> dict[int, list[tuple[int, int]]] | None:
        ranges = {
            i: self.blocks[req_id]
            for i, req_id in enumerate(input_batch.req_ids)
            if req_id in self.blocks
        }
        return ranges or None


class Qwen3_5ForSlotDecision(Qwen3_5ForConditionalGeneration):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Model Runner V1 does not pass the PrefixLM ranges to the attention.
        if not vllm_config.use_v2_model_runner:
            raise ValueError(f"{type(self).__name__} needs Model Runner V2")

    def get_model_state_cls(self) -> type[Qwen3_5SlotDecisionModelState]:
        return Qwen3_5SlotDecisionModelState
