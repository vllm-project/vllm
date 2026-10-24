# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5 with a slot readout, whose full attention layers read a request's
prefix and question as two bidirectional PrefixLM blocks."""

from vllm.config import VllmConfig
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.core.sched.prefix_lm_scheduler import prefix_lm_len
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

    def get_mm_prefix_ranges(
        self, input_batch: InputBatch
    ) -> dict[int, list[tuple[int, int]]] | None:
        ranges: dict[int, list[tuple[int, int]]] = {}
        for i, req_id in enumerate(input_batch.req_ids):
            if (blocks := self.blocks.get(req_id)) is None:
                continue
            ranges[i] = blocks
            if input_batch.is_prefilling_np[i]:
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
        return ranges or None


class Qwen3_5ForSlotDecision(Qwen3_5ForConditionalGeneration):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Model Runner V1 does not pass the PrefixLM ranges to the attention.
        if not vllm_config.use_v2_model_runner:
            raise ValueError(f"{type(self).__name__} needs Model Runner V2")

    def get_model_state_cls(self) -> type[Qwen3_5SlotDecisionModelState]:
        return Qwen3_5SlotDecisionModelState
