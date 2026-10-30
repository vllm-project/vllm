# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client-side state machines and internal request/result types.

The session layer never hands runner internals (InputBatch etc.) to the
transport; it converts them into the placement-neutral types here first.
"""

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.spec_decode.remote.protocol import SequenceKey, SpeculatorStatusCode

if TYPE_CHECKING:
    import torch

    from vllm.v1.spec_decode.proposal import SpeculatorOutput


class InvalidStateTransition(Exception):
    """A lifecycle transition that the state machine forbids."""


class HandleState(enum.Enum):
    """Lifecycle of one proposal handle; collect is one-shot."""

    CREATED = "created"
    DISPATCHED = "dispatched"
    COMPLETED = "completed"
    COLLECTED = "collected"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


_HANDLE_TRANSITIONS: dict[HandleState, frozenset[HandleState]] = {
    HandleState.CREATED: frozenset(
        {HandleState.DISPATCHED, HandleState.COMPLETED, HandleState.CANCELLED}
    ),
    HandleState.DISPATCHED: frozenset(
        {HandleState.COMPLETED, HandleState.CANCELLED, HandleState.TIMED_OUT}
    ),
    HandleState.COMPLETED: frozenset({HandleState.COLLECTED}),
    HandleState.COLLECTED: frozenset(),
    HandleState.CANCELLED: frozenset(),
    HandleState.TIMED_OUT: frozenset({HandleState.COLLECTED}),
}


class SequenceState(enum.Enum):
    """Client-side lifecycle of one remote draft sequence.

    DESYNCED sequences may recover by re-prefilling under a new
    generation; TARGET_ONLY is latched until the request finishes, and
    INVALID means the session epoch changed so the sequence must be
    re-opened on a new session.
    """

    OPENING = "opening"
    PREFILLING = "prefilling"
    READY = "ready"
    IN_FLIGHT = "in_flight"
    CLOSING = "closing"
    CLOSED = "closed"
    DESYNCED = "desynced"
    TARGET_ONLY = "target_only"
    INVALID = "invalid"


_ALWAYS_ALLOWED = frozenset(
    {SequenceState.CLOSING, SequenceState.INVALID, SequenceState.TARGET_ONLY}
)

_SEQUENCE_TRANSITIONS: dict[SequenceState, frozenset[SequenceState]] = {
    SequenceState.OPENING: frozenset({SequenceState.PREFILLING}) | _ALWAYS_ALLOWED,
    SequenceState.PREFILLING: (
        frozenset({SequenceState.READY, SequenceState.DESYNCED}) | _ALWAYS_ALLOWED
    ),
    SequenceState.READY: (
        frozenset({SequenceState.IN_FLIGHT, SequenceState.DESYNCED}) | _ALWAYS_ALLOWED
    ),
    SequenceState.IN_FLIGHT: (
        frozenset({SequenceState.READY, SequenceState.DESYNCED}) | _ALWAYS_ALLOWED
    ),
    SequenceState.DESYNCED: frozenset({SequenceState.PREFILLING}) | _ALWAYS_ALLOWED,
    SequenceState.TARGET_ONLY: frozenset({SequenceState.CLOSING}),
    SequenceState.INVALID: frozenset({SequenceState.OPENING, SequenceState.CLOSING}),
    SequenceState.CLOSING: frozenset({SequenceState.CLOSED}),
    SequenceState.CLOSED: frozenset(),
}


def transition_handle(current: HandleState, new: HandleState) -> HandleState:
    """Validate and return a handle state transition."""
    if new not in _HANDLE_TRANSITIONS[current]:
        raise InvalidStateTransition(
            f"proposal handle cannot go {current.value} -> {new.value}"
        )
    return new


def can_transition_sequence(current: SequenceState, new: SequenceState) -> bool:
    """Whether the sequence state machine allows ``current -> new``."""
    return new in _SEQUENCE_TRANSITIONS[current]


def transition_sequence(current: SequenceState, new: SequenceState) -> SequenceState:
    """Validate and return a sequence state transition."""
    if not can_transition_sequence(current, new):
        raise InvalidStateTransition(
            f"remote sequence cannot go {current.value} -> {new.value}"
        )
    return new


@dataclass(frozen=True)
class FeatureBatch:
    """Target features for one round, ordered by the negotiated schema."""

    schema_id: int
    slots: "tuple[torch.Tensor | None, ...]"


@dataclass(frozen=True)
class RemoteProposalBatch:
    """One advance-and-propose round in placement-neutral form."""

    batch_id: int
    keys: tuple[SequenceKey, ...]
    accepted_counts: "torch.Tensor"
    recovery_tokens: "torch.Tensor"
    features: FeatureBatch
    query_lens: "torch.Tensor | None" = None
    """[batch] CPU tensor: rows each key owns in token-level feature slots,
    which then hold ``sum(query_lens)`` rows in key order. None means one
    row per key."""


@dataclass
class RemoteProposalResult:
    """Runner-facing output plus per-row diagnostics for the session."""

    output: "SpeculatorOutput"
    row_statuses: tuple[SpeculatorStatusCode, ...]
