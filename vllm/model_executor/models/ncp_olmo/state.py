# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request-scoped incremental state for ConceptLM's chunk-rate HLM."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class RequestStateError(RuntimeError):
    """Raised when vLLM scheduling would make ConceptLM state ambiguous."""


@dataclass
class HLMKVState:
    """Dense K/V history for one HLM layer and one request."""

    key: Any | None = None
    value: Any | None = None
    length: int = 0


@dataclass
class TensorBuffer:
    """Geometrically growing device tensor with an explicit active length."""

    data: Any | None = None
    length: int = 0


def active_tensor_buffer(buffer: TensorBuffer) -> Any | None:
    """Return the active prefix without exposing unused capacity."""

    if buffer.length == 0:
        return None
    if buffer.data is None or buffer.length > int(buffer.data.shape[0]):
        raise ValueError("tensor buffer length exceeds storage capacity")
    return buffer.data[: buffer.length]


def append_tensor_buffer(
    buffer: TensorBuffer,
    values: Any,
    *,
    minimum_capacity: int = 1,
) -> Any:
    """Append one or more leading-dimension values without Python tensor lists."""

    if values.ndim == 0:
        raise ValueError("tensor buffer values require a leading dimension")
    num_values = int(values.shape[0])
    if num_values <= 0:
        raise ValueError("cannot append an empty tensor buffer segment")
    old_length = int(buffer.length)
    required_length = old_length + num_values
    if buffer.data is None:
        if old_length != 0:
            raise ValueError("empty tensor buffer has non-zero length")
        target = max(int(minimum_capacity), required_length)
        capacity = 1 << (target - 1).bit_length()
        buffer.data = values.new_empty((capacity, *values.shape[1:]))
    else:
        if (
            buffer.data.shape[1:] != values.shape[1:]
            or buffer.data.dtype != values.dtype
            or buffer.data.device != values.device
        ):
            raise ValueError("tensor buffer append does not match storage")
        capacity = int(buffer.data.shape[0])
        if old_length < 0 or old_length > capacity:
            raise ValueError("tensor buffer length exceeds storage capacity")
        if required_length > capacity:
            target = max(required_length, capacity * 2)
            new_capacity = 1 << (target - 1).bit_length()
            new_data = values.new_empty((new_capacity, *values.shape[1:]))
            new_data[:old_length].copy_(buffer.data[:old_length])
            buffer.data = new_data

    buffer.data[old_length:required_length].copy_(values)
    buffer.length = required_length
    return buffer.data[:required_length]


def clear_tensor_buffer(buffer: TensorBuffer) -> None:
    """Clear the active prefix while retaining device storage."""

    buffer.length = 0


@dataclass
class ConceptRequestState:
    """All non-PagedAttention state owned by one request."""

    req_id: str
    next_token_position: int = 0
    pending_encoder_final: TensorBuffer = field(default_factory=TensorBuffer)
    pending_encoder_layers: list[TensorBuffer] = field(default_factory=list)
    hlm_kv: list[HLMKVState] = field(default_factory=list)
    hlm_raw_layer_states: list[TensorBuffer] = field(default_factory=list)
    predicted_concepts: TensorBuffer = field(default_factory=TensorBuffer)

    @classmethod
    def empty(
        cls,
        req_id: str,
        *,
        encoder_layers: int,
        hlm_layers: int,
    ) -> ConceptRequestState:
        """Allocate empty per-layer containers without allocating tensors."""

        return cls(
            req_id=req_id,
            pending_encoder_layers=[TensorBuffer() for _ in range(encoder_layers)],
            hlm_kv=[HLMKVState() for _ in range(hlm_layers)],
            hlm_raw_layer_states=[TensorBuffer() for _ in range(hlm_layers)],
        )


@dataclass(frozen=True)
class ScheduledRequestSegment:
    """One contiguous flattened token segment scheduled for a request."""

    req_id: str
    flat_start: int
    flat_end: int
    position_start: int
    position_end: int
    state: ConceptRequestState


class ConceptRequestStateStore:
    """Own persistent ConceptLM state for requests admitted by the V2 runner."""

    def __init__(self, *, encoder_layers: int, hlm_layers: int) -> None:
        self.encoder_layers = int(encoder_layers)
        self.hlm_layers = int(hlm_layers)
        self._states: dict[str, ConceptRequestState] = {}

    @property
    def states(self) -> Mapping[str, ConceptRequestState]:
        """Expose a read-only mapping view for diagnostics."""

        return self._states

    def _new_state(self, req_id: str) -> ConceptRequestState:
        state = ConceptRequestState.empty(
            req_id,
            encoder_layers=self.encoder_layers,
            hlm_layers=self.hlm_layers,
        )
        self._states[req_id] = state
        return state

    def add_request(self, req_id: str, *, computed_tokens: int) -> None:
        """Initialize state when the V2 runner admits or re-admits a request.

        V2 removes preempted requests from the runner state and later re-admits
        them with a full replay from token position zero. Replacing an existing
        entry here is therefore intentional.
        """

        req_id = str(req_id)
        if computed_tokens != 0:
            raise RequestStateError(
                f"request {req_id!r} was admitted with {computed_tokens} cached "
                "tokens, but no matching HLM state snapshot exists; prefix "
                "caching and partial replay must be disabled"
            )
        self._new_state(req_id)

    def remove_request(self, req_id: str) -> None:
        """Release all dense HLM tensors owned by a finished request."""

        self._states.pop(str(req_id), None)

    def resolve_segments(self, input_batch: Any) -> tuple[ScheduledRequestSegment, ...]:
        """Resolve request segments from a V2 runner ``InputBatch``."""

        req_ids = tuple(str(req_id) for req_id in input_batch.req_ids)
        scheduled_counts = input_batch.num_scheduled_tokens
        computed_positions = input_batch.num_computed_tokens_np
        if len(scheduled_counts) < len(req_ids):
            raise RequestStateError(
                "vLLM scheduled-token metadata is shorter than the request batch"
            )
        expected_tokens = sum(
            int(scheduled_counts[req_index]) for req_index in range(len(req_ids))
        )
        total_scheduled_tokens = int(input_batch.num_tokens)
        if expected_tokens != total_scheduled_tokens:
            raise RequestStateError(
                "per-request scheduled token count does not match total: "
                f"{expected_tokens} != {total_scheduled_tokens}"
            )
        if len(computed_positions) < len(req_ids):
            raise RequestStateError(
                "vLLM computed-position metadata is shorter than the request batch"
            )

        segments = []
        flat_start = 0
        for req_index, req_id in enumerate(req_ids):
            token_count = int(scheduled_counts[req_index])
            if token_count <= 0:
                raise RequestStateError(f"request {req_id!r} has an empty segment")
            flat_end = flat_start + token_count
            position_start = int(computed_positions[req_index])
            if position_start < 0:
                raise RequestStateError(
                    f"request {req_id!r} has negative computed position "
                    f"{position_start}"
                )

            state = self._states.get(req_id)
            if state is None:
                raise RequestStateError(
                    f"request {req_id!r} was not admitted into the HLM state store"
                )
            if position_start < state.next_token_position:
                if position_start != 0:
                    raise RequestStateError(
                        f"request {req_id!r} rewound from "
                        f"{state.next_token_position} to {position_start}; only a "
                        "full replay from position 0 is supported"
                    )
                state = self._new_state(req_id)
            if position_start > state.next_token_position:
                raise RequestStateError(
                    f"request {req_id!r} starts at {position_start}, but its "
                    f"ConceptLM state ends at {state.next_token_position}; prefix "
                    "caching and partial replay must be disabled"
                )
            segments.append(
                ScheduledRequestSegment(
                    req_id=req_id,
                    flat_start=flat_start,
                    flat_end=flat_end,
                    position_start=position_start,
                    position_end=position_start + token_count,
                    state=state,
                )
            )
            flat_start = flat_end
        return tuple(segments)

    @staticmethod
    def commit(segment: ScheduledRequestSegment) -> None:
        """Commit one successfully processed request segment."""

        if segment.state.next_token_position != segment.position_start:
            raise RequestStateError(
                f"request {segment.req_id!r} state changed before commit"
            )
        segment.state.next_token_position = segment.position_end
