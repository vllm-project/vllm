# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from vllm.lora.layers import LoRAMapping, LoRAMappingType
from vllm.lora.request import LoRARequest

NO_LORA_ID = 0


class LoraState:
    def __init__(self, max_num_reqs: int, max_loras_per_request: int = 1):
        if max_loras_per_request < 1:
            raise ValueError(
                f"max_loras_per_request must be >= 1, got {max_loras_per_request}"
            )
        self.max_loras_per_request = max_loras_per_request
        self.slot_lora_ids = np.full(
            (max_num_reqs, max_loras_per_request), NO_LORA_ID, dtype=np.int32
        )
        self.slot_lora_scales = np.zeros(
            (max_num_reqs, max_loras_per_request), dtype=np.float32
        )
        # Slot-0 view keeps single-adapter callers and tests working directly.
        self.lora_ids = self.slot_lora_ids[:, 0]
        # req_id -> lora_request
        self.lora_requests: dict[str, LoRARequest] = {}

    def add_request(
        self, req_id: str, req_index: int, lora_request: LoRARequest | None
    ) -> None:
        self.slot_lora_ids[req_index] = NO_LORA_ID
        self.slot_lora_scales[req_index] = 0.0
        if lora_request is not None:
            adapters = lora_request.get_adapters()
            if len(adapters) > self.max_loras_per_request:
                raise ValueError(
                    f"Request {req_id!r} specifies {len(adapters)} LoRA adapters, "
                    f"exceeding max_loras_per_request ({self.max_loras_per_request})."
                )
            self.lora_requests[req_id] = lora_request
            for slot_idx, (adapter, scale) in enumerate(adapters):
                self.slot_lora_ids[req_index, slot_idx] = adapter.lora_int_id
                self.slot_lora_scales[req_index, slot_idx] = scale
        else:
            self.lora_requests.pop(req_id, None)

    def remove_request(self, req_id: str) -> None:
        self.lora_requests.pop(req_id, None)

    def make_lora_inputs(
        self,
        req_ids: list[str],
        idx_mapping: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        num_logits: np.ndarray,
    ) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]:
        lora_ids = self.lora_ids[idx_mapping]
        # There is one lm_head LoRA ID per logits row.
        prompt_lora_mapping = tuple(lora_ids.repeat(num_logits))
        token_lora_mapping = tuple(lora_ids.repeat(num_scheduled_tokens))
        active_lora_requests: set[LoRARequest] = self.get_activate_loras(req_ids)
        return prompt_lora_mapping, token_lora_mapping, active_lora_requests

    def make_lora_mapping(
        self,
        req_ids: list[str],
        idx_mapping: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        num_logits: np.ndarray,
        mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
    ) -> tuple[LoRAMapping, set[LoRARequest]]:
        slot_ids = self.slot_lora_ids[idx_mapping]
        slot_scales = self.slot_lora_scales[idx_mapping]
        prompt_lora_mapping = tuple(slot_ids[:, 0].repeat(num_logits).tolist())
        token_lora_mapping = tuple(slot_ids[:, 0].repeat(num_scheduled_tokens).tolist())
        active_lora_requests = self.get_activate_loras(req_ids)

        active_slots = 1
        for slot_idx in range(self.max_loras_per_request - 1, 0, -1):
            if np.any(slot_ids[:, slot_idx] != NO_LORA_ID):
                active_slots = slot_idx + 1
                break

        has_custom_scales = bool(
            np.any(
                (slot_ids[:, :active_slots] != NO_LORA_ID)
                & (slot_scales[:, :active_slots] != 1.0)
            )
        )

        slot_index_mappings: tuple[tuple[int, ...], ...] | None = None
        slot_prompt_mappings: tuple[tuple[int, ...], ...] | None = None
        slot_index_scales: tuple[tuple[float, ...], ...] | None = None
        slot_prompt_scales: tuple[tuple[float, ...], ...] | None = None

        if active_slots > 1:
            slot_index_mappings = tuple(
                token_lora_mapping
                if s == 0
                else tuple(slot_ids[:, s].repeat(num_scheduled_tokens).tolist())
                for s in range(active_slots)
            )
            slot_prompt_mappings = tuple(
                prompt_lora_mapping
                if s == 0
                else tuple(slot_ids[:, s].repeat(num_logits).tolist())
                for s in range(active_slots)
            )

        if active_slots > 1 or has_custom_scales:
            slot_index_scales = tuple(
                tuple(slot_scales[:, s].repeat(num_scheduled_tokens).tolist())
                for s in range(active_slots)
            )
            slot_prompt_scales = tuple(
                tuple(slot_scales[:, s].repeat(num_logits).tolist())
                for s in range(active_slots)
            )

        lora_mapping = LoRAMapping(
            index_mapping=token_lora_mapping,
            prompt_mapping=prompt_lora_mapping,
            is_prefill=True,
            type=mapping_type,
            slot_index_mappings=slot_index_mappings,
            slot_prompt_mappings=slot_prompt_mappings,
            slot_index_scales=slot_index_scales,
            slot_prompt_scales=slot_prompt_scales,
        )
        return lora_mapping, active_lora_requests

    def get_activate_loras(self, req_ids: list[str]) -> set[LoRARequest]:
        active_lora_requests: set[LoRARequest] = set()
        for req_id in req_ids:
            lora_request = self.lora_requests.get(req_id)
            if lora_request is not None:
                active_lora_requests.update(lora_request.get_adapter_requests())
        return active_lora_requests
