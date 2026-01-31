# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np

from vllm.lora.request import LoRARequest

NO_LORA_ID = 0


class LoraState:
    def __init__(self, max_num_reqs: int):
        self.lora_ids = np.zeros(max_num_reqs, dtype=np.int32)
        self.lora_ids.fill(NO_LORA_ID)
        # req_id -> lora_request
        self.lora_requests: dict[str, LoRARequest] = {}

    def add_request(
        self, req_id: str, req_index: int, lora_request: LoRARequest | None
    ) -> None:
        if lora_request is not None:
            self.lora_requests[req_id] = lora_request
            self.lora_ids[req_index] = lora_request.lora_int_id
        else:
            self.lora_ids[req_index] = NO_LORA_ID

    def remove_request(self, req_id: str) -> None:
        self.lora_requests.pop(req_id, None)

    def make_lora_inputs(
        self,
        req_ids: list[str],
        idx_mapping: np.ndarray,
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]:
        lora_ids = self.lora_ids[idx_mapping]
        prompt_lora_mapping = tuple(lora_ids)
        token_lora_mapping = tuple(lora_ids.repeat(num_scheduled_tokens))

        active_lora_requests: set[LoRARequest] = set()
        for req_id in req_ids:
            if (lora_request := self.lora_requests.get(req_id)) is not None:
                active_lora_requests.add(lora_request)
        return prompt_lora_mapping, token_lora_mapping, active_lora_requests
