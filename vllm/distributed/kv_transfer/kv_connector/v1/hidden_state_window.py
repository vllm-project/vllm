# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Position and layout contracts for the example hidden-state connector."""

from dataclasses import dataclass
from typing import Any


MAX_WINDOW_END = 2**31 - 1


@dataclass(frozen=True)
class HiddenStateWindow:
    start: int
    end: int
    min_rows: int = 1

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "HiddenStateWindow | None":
        value = params.get("hidden_state_window")
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) - {"start", "end", "min_rows"}:
            raise ValueError("hidden_state_window expects start, end and min_rows")
        start, end = value.get("start"), value.get("end")
        minimum = value.get("min_rows", 1)
        if type(start) is not int or type(end) is not int or type(minimum) is not int:
            raise ValueError("Window bounds and min_rows must be integers")
        if not 0 <= start < end or not 1 <= minimum <= end - start:
            raise ValueError("Require 0 <= start < end and 1 <= min_rows <= end-start")
        if end > MAX_WINDOW_END:
            raise ValueError(f"Window end must not exceed {MAX_WINDOW_END}")
        return cls(start, end, minimum)

    def absolute_bounds(self, prompt_len: int) -> tuple[int, int]:
        if prompt_len < 1:
            raise ValueError("Window capture requires a nonempty token prompt")
        return prompt_len - 1 + self.start, prompt_len - 1 + self.end

    def accepted_bounds(self, prompt_len: int, output_len: int) -> tuple[int, int]:
        start, end = self.absolute_bounds(prompt_len)
        return start, max(start, min(end, prompt_len - 1 + output_len))


def block_window_ranges(
    blocks: list[int],
    block_size: int,
    bounds: tuple[int, int] | None,
    first_block: int = 0,
) -> list[tuple[int, int, int]]:
    """Map logical window bounds to offsets within physical cache blocks."""
    result = []
    for index, block in enumerate(blocks, first_block):
        if block == 0:
            continue
        if bounds is None:
            start, end = 0, block_size
        else:
            base = index * block_size
            start = min(block_size, max(0, bounds[0] - base))
            end = min(block_size, max(0, bounds[1] - base))
        result.append((block, start, end))
    return result


def extraction_layout(vllm_config) -> tuple[str, list[int]]:
    spec = vllm_config.speculative_config
    hf = spec.draft_model_config.hf_config
    ids = list(getattr(hf, "eagle_aux_hidden_state_layer_ids", []))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Extraction layer ids must be nonempty and unique")
    depth = vllm_config.model_config.get_total_num_hidden_layers()
    if any(type(i) is not int or i < 0 or i > depth for i in ids):
        raise ValueError("Extraction layer ids must be in [0, num_hidden_layers]")
    layout = getattr(hf, "hidden_states_layout", "auxiliary")
    if layout == "auxiliary":
        return layout, ids
    if layout == "final":
        return layout, [-1]
    if layout == "auxiliary_plus_final":
        return layout, ids + [-1]
    raise ValueError(
        "hidden_states_layout must be auxiliary, final or auxiliary_plus_final"
    )


def validate_window_request(
    config, params, prompt_token_ids, prompt_embeds, resumable: bool
) -> None:
    """Reject invalid capture in the frontend before the scheduler receives it."""
    transfer = (params.extra_args or {}).get("kv_transfer_params") or {}
    window = HiddenStateWindow.from_params(transfer)
    if window is None:
        return
    connector = config.kv_transfer_config
    if (
        connector is None
        or connector.kv_connector != "ExampleHiddenStatesConnector"
        or not connector.get_from_extra_config("enable_window_capture", False)
    ):
        raise ValueError(
            "Window capture requires an enabled ExampleHiddenStatesConnector"
        )
    if params.stop:
        raise ValueError("Window capture does not support stop strings")
    if (
        params.n != 1
        or resumable
        or prompt_embeds is not None
        or prompt_token_ids is None
    ):
        raise ValueError(
            "Window capture requires one completion and a fixed token prompt"
        )
    window.absolute_bounds(len(prompt_token_ids))
    max_rows = connector.get_from_extra_config(
        "max_window_rows", config.model_config.max_model_len
    )
    if window.end > max_rows:
        raise ValueError(f"Window end must not exceed {max_rows}")
