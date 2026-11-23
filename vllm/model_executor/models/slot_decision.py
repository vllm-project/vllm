# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PrefixLM block bookkeeping of the slot-decision model states."""

from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.core.sched.prefix_lm_scheduler import prefix_lm_len
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState


class PrefixLMBlocksMixin(ModelState):
    """Keeps each PrefixLM request's prefix and question blocks."""

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
