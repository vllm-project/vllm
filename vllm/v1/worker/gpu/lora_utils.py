# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LoRA utilities for the Model Runner V2 and cudagraph.

"""

import bisect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_captured_lora_counts

if TYPE_CHECKING:
    from vllm.config.compilation import CompilationConfig
    from vllm.config.lora import LoRAConfig

NO_LORA_ID = 0


def make_graph_key(num_tokens: int, num_active_loras: int = 0) -> tuple[int, int]:
    """Create a unique key for CUDA graph storage (num_tokens, num_active_loras)."""
    return (num_tokens, num_active_loras)


def get_lora_capture_cases(
    lora_config: "LoRAConfig | None",
    compilation_config: "CompilationConfig",
) -> list[int]:
    """
    Return num_active_loras values for cudagraph capture.

    When cudagraph_specialize_lora=True: powers of 2 up to max_loras, plus
    max_loras+1. When False: [0, max_loras+1]. When LoRA disabled: [0].
    """
    if lora_config is None:
        return [0]
    if compilation_config.cudagraph_specialize_lora:
        specialize = getattr(lora_config, "specialize_active_lora", False)
        captured = get_captured_lora_counts(lora_config.max_loras, specialize)
        return [0] + [c for c in captured if c > 0]
    return [0, lora_config.max_loras + 1]


def resolve_effective_num_active_loras(
    num_active_loras: int, lora_capture_cases: list[int]
) -> int:
    """
    Resolve effective num_active_loras for graph lookup.
    Maps actual count to the nearest captured case.
    """
    if num_active_loras <= 0 or not lora_capture_cases:
        return num_active_loras
    captured_with_lora = [c for c in lora_capture_cases if c > 0]
    if not captured_with_lora:
        return num_active_loras
    idx = bisect.bisect_left(captured_with_lora, num_active_loras)
    if idx < len(captured_with_lora):
        return captured_with_lora[idx]
    return captured_with_lora[-1]


def get_num_active_loras_for_dispatch(
    lora_config: "LoRAConfig | None",
    lora_state: "LoraState",
    req_ids: list[str],
    dummy_run: bool,
) -> int:
    """Compute num_active_loras for cudagraph dispatch."""
    if lora_config and not dummy_run:
        return len(lora_state.get_activate_loras(req_ids))
    if dummy_run and lora_config:
        return lora_config.max_loras + 1
    return 0


def create_lora_capture_hook(
    lora_config: "LoRAConfig | None",
    runner: Any,
) -> Callable[[int, int, int], None] | None:
    """Create a hook to set up LoRA state before each cudagraph capture."""
    if lora_config is None:
        return
    def hook(num_active_loras: int, num_reqs: int, num_tokens: int) -> None:
   
        num_scheduled = np.full(num_reqs, num_tokens // num_reqs, dtype=np.int32)
        num_scheduled[-1] += num_tokens % num_reqs
        with runner.maybe_select_dummy_loras(
            lora_config, num_scheduled, num_active_loras=num_active_loras
        ):
            pass

    return hook 


def activate_loras_for_batch(
    lora_config: "LoRAConfig | None",
    lora_state: "LoraState",
    req_ids: list[str],
    idx_mapping_np: np.ndarray,
    num_scheduled_tokens: np.ndarray,
    set_active_loras_fn: Callable[
        [tuple[int, ...], tuple[int, ...], set[LoRARequest]], None
    ],
) -> None:
    """Activate LoRA adapters for the current batch if LoRA is enabled."""
    if lora_config is None:
        return
    prompt_mapping, token_mapping, lora_requests = lora_state.make_lora_inputs(
        req_ids, idx_mapping_np, num_scheduled_tokens
    )
    set_active_loras_fn(prompt_mapping, token_mapping, lora_requests)


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
        active_lora_requests: set[LoRARequest] = self.get_activate_loras(req_ids)
        return prompt_lora_mapping, token_lora_mapping, active_lora_requests

    def get_activate_loras(self, req_ids: list[str]) -> set[LoRARequest]:
        active_lora_requests: set[LoRARequest] = set()
        for req_id in req_ids:
            lora_request = self.lora_requests.get(req_id)
            if lora_request is not None:
                active_lora_requests.add(lora_request)
        return active_lora_requests
