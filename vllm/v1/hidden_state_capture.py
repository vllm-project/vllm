# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request-local, position-based capture of model output hidden states."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class HiddenStateCaptureState(Enum):
    DISABLED = "disabled"
    ARMED = "armed"
    CAPTURING = "capturing"
    FINISHED = "finished"


@dataclass(frozen=True)
class HiddenStateCapturePlan:
    request_id: str
    prompt_len: int
    window_start_abs: int
    window_end_abs: int
    collection_id: str
    min_rows: int = 1
    aux_layer_ids: tuple[int, ...] = ()
    hidden_layout: str = "final"

    def __post_init__(self) -> None:
        if (
            self.prompt_len < 1
            or self.window_start_abs < self.prompt_len - 1
            or self.window_start_abs >= self.window_end_abs
            or self.min_rows < 1
            or self.min_rows > self.window_end_abs - self.window_start_abs
        ):
            raise ValueError("Invalid hidden-state capture plan")

    @classmethod
    def from_window(
        cls,
        request_id: str,
        prompt_len: int,
        start: int,
        end: int,
        collection_id: str,
        *,
        coordinate: Literal["response", "absolute"] = "response",
        min_rows: int = 1,
        aux_layer_ids: tuple[int, ...] = (),
        hidden_layout: str | None = None,
    ) -> HiddenStateCapturePlan:
        if prompt_len < 1 or start < 0 or start >= end or min_rows < 1:
            raise ValueError("Invalid hidden-state capture window or prompt length")
        if coordinate == "response":
            offset = prompt_len - 1
        elif coordinate == "absolute":
            offset = 0
        else:
            raise ValueError(f"Unknown hidden-state coordinate: {coordinate}")
        if start + offset < prompt_len - 1:
            raise ValueError("Hidden-state window cannot start before the response")
        return cls(
            request_id=request_id,
            prompt_len=prompt_len,
            window_start_abs=start + offset,
            window_end_abs=end + offset,
            collection_id=collection_id,
            min_rows=min_rows,
            aux_layer_ids=aux_layer_ids,
            hidden_layout=hidden_layout or ("aux_final" if aux_layer_ids else "final"),
        )


def hidden_state_capture_capability(vllm_config: VllmConfig) -> str | None:
    """Return a reason when row-to-position mapping is not supported."""
    if vllm_config.device_config.device_type != "cuda":
        return "only the CUDA GPU model runners have a capture adapter"
    if vllm_config.scheduler_config.async_scheduling:
        return "async scheduling can reorder in-flight hidden rows"
    if vllm_config.parallel_config.prefill_context_parallel_size > 1:
        return "prefill context parallelism changes hidden-row ownership"
    if vllm_config.parallel_config.decode_context_parallel_size > 1:
        return "decode context parallelism changes hidden-row ownership"
    if vllm_config.model_config.is_encoder_decoder:
        return "encoder-decoder hidden positions need a separate mapping"
    if (
        vllm_config.speculative_config is not None
        and vllm_config.speculative_config.enable_adaptive_verification
    ):
        return "adaptive verification changes GPU-side token row allocation"
    return None


@dataclass
class HiddenStateCaptureChunk:
    positions: np.ndarray
    hidden_states: torch.Tensor
    copied_bytes: int = 0
    copy_ms: float = 0.0

    def accepted(self, start: int, end: int) -> HiddenStateCaptureChunk:
        keep = (self.positions >= start) & (self.positions < end)
        return HiddenStateCaptureChunk(
            self.positions[keep],
            self.hidden_states[torch.from_numpy(keep)],
            self.copied_bytes,
            self.copy_ms,
        )


def accepted_hidden_range(
    scheduled_end: int,
    num_scheduled: int,
    num_generated: int,
    speculative: bool,
) -> tuple[int, int]:
    """Map emitted tokens to their model-input hidden positions."""
    start = (
        scheduled_end - num_scheduled if speculative else scheduled_end - num_generated
    )
    return start, start + num_generated


def drop_incompatible_aux_plans(
    plans: dict[str, HiddenStateCapturePlan],
    req_ids: list[str],
    available_layer_ids: tuple[int, ...],
    aux_hidden_states: list[torch.Tensor] | None,
) -> dict[str, str]:
    """Disable captures whose requested auxiliary layout cannot be produced."""
    errors: dict[str, str] = {}
    for req_id in req_ids:
        plan = plans.get(req_id)
        if plan is None or not plan.aux_layer_ids:
            continue
        if (
            aux_hidden_states is None
            or plan.aux_layer_ids != available_layer_ids
            or len(aux_hidden_states) != len(available_layer_ids)
        ):
            plans.pop(req_id)
            errors[req_id] = "aux_layers_unavailable"
    return errors


def capture_scheduled_hidden_states(
    plans: dict[str, HiddenStateCapturePlan],
    req_ids: list[str],
    num_scheduled_tokens: dict[str, int],
    num_computed_tokens: dict[str, int],
    hidden_states: torch.Tensor,
    aux_hidden_states: list[torch.Tensor] | None = None,
) -> dict[str, HiddenStateCaptureChunk]:
    """Copy only scheduled rows inside each request's capture window."""
    chunks: dict[str, HiddenStateCaptureChunk] = {}
    row_offset = 0
    for req_id in req_ids:
        num_rows = num_scheduled_tokens[req_id]
        plan = plans.get(req_id)
        if plan is not None:
            start = num_computed_tokens[req_id]
            first = max(start, plan.window_start_abs)
            end = min(start + num_rows, plan.window_end_abs)
            if first < end:
                first_row = row_offset + first - start
                end_row = row_offset + end - start
                if plan.aux_layer_ids:
                    if aux_hidden_states is None or len(aux_hidden_states) != len(
                        plan.aux_layer_ids
                    ):
                        raise ValueError(
                            "Requested auxiliary hidden layers are unavailable"
                        )
                    parts = [state[first_row:end_row] for state in aux_hidden_states]
                    if plan.hidden_layout != "dflash_aux":
                        parts.append(hidden_states[first_row:end_row])
                    states = torch.cat(parts, dim=-1)
                else:
                    states = hidden_states[first_row:end_row]
                copy_started = perf_counter()
                cpu_states = states.detach().to("cpu")
                chunks[req_id] = HiddenStateCaptureChunk(
                    positions=np.arange(first, end, dtype=np.int64),
                    hidden_states=cpu_states,
                    copied_bytes=(
                        states.numel() * states.element_size()
                        if states.device.type != "cpu"
                        else 0
                    ),
                    copy_ms=(
                        (perf_counter() - copy_started) * 1000
                        if states.device.type != "cpu"
                        else 0.0
                    ),
                )
        row_offset += num_rows
    return chunks


@dataclass
class HiddenStateCaptureResult:
    hidden_positions: np.ndarray
    hidden_states: torch.Tensor
    hidden_position_start: int
    hidden_position_end: int
    hidden_window_start: int
    hidden_window_end: int
    hidden_layout: str
    request_id: str
    collection_id: str
    copied_bytes: int = 0
    copy_ms: float = 0.0


@dataclass
class HiddenStateCaptureBuffer:
    plan: HiddenStateCapturePlan
    chunks: list[HiddenStateCaptureChunk] = field(default_factory=list)
    state: HiddenStateCaptureState = HiddenStateCaptureState.ARMED
    copied_bytes: int = 0
    copy_ms: float = 0.0

    def add(self, chunk: HiddenStateCaptureChunk) -> None:
        self.copied_bytes += chunk.copied_bytes
        self.copy_ms += chunk.copy_ms
        if chunk.positions.size:
            self.state = HiddenStateCaptureState.CAPTURING
            self.chunks.append(chunk)

    @property
    def num_rows(self) -> int:
        if not self.chunks:
            return 0
        return len(np.unique(np.concatenate([c.positions for c in self.chunks])))

    def finish(self) -> HiddenStateCaptureResult | None:
        self.state = HiddenStateCaptureState.FINISHED
        if not self.chunks:
            return None
        positions = np.concatenate([chunk.positions for chunk in self.chunks])
        states = torch.cat([chunk.hidden_states for chunk in self.chunks])
        positions, unique_indices = np.unique(positions, return_index=True)
        states = states[torch.from_numpy(unique_indices)]
        if len(positions) < self.plan.min_rows:
            return None
        return HiddenStateCaptureResult(
            hidden_positions=positions,
            hidden_states=states,
            hidden_position_start=int(positions.min()),
            hidden_position_end=int(positions.max()) + 1,
            hidden_window_start=self.plan.window_start_abs,
            hidden_window_end=self.plan.window_end_abs,
            hidden_layout=self.plan.hidden_layout,
            request_id=self.plan.request_id,
            collection_id=self.plan.collection_id,
            copied_bytes=self.copied_bytes,
            copy_ms=self.copy_ms,
        )
