# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client-side lifecycle state machines: one-shot proposal handles and the
recover/latch rules for remote sequences."""

import pytest

from vllm.v1.spec_decode.remote.state import (
    HandleState,
    InvalidStateTransition,
    SequenceState,
    can_transition_sequence,
    transition_handle,
    transition_sequence,
)


def _walk_handle(*states: HandleState) -> HandleState:
    current = states[0]
    for new in states[1:]:
        current = transition_handle(current, new)
    return current


def _walk_sequence(*states: SequenceState) -> SequenceState:
    current = states[0]
    for new in states[1:]:
        current = transition_sequence(current, new)
    return current


def test_handle_happy_path():
    assert (
        _walk_handle(
            HandleState.CREATED,
            HandleState.DISPATCHED,
            HandleState.COMPLETED,
            HandleState.COLLECTED,
        )
        is HandleState.COLLECTED
    )


def test_handle_collect_is_one_shot():
    with pytest.raises(InvalidStateTransition, match="collected -> collected"):
        transition_handle(HandleState.COLLECTED, HandleState.COLLECTED)


def test_handle_cannot_skip_dispatch():
    with pytest.raises(InvalidStateTransition):
        transition_handle(HandleState.CREATED, HandleState.COLLECTED)


def test_timed_out_handle_is_still_collected_once():
    state = _walk_handle(
        HandleState.CREATED, HandleState.DISPATCHED, HandleState.TIMED_OUT
    )
    assert transition_handle(state, HandleState.COLLECTED) is HandleState.COLLECTED
    with pytest.raises(InvalidStateTransition):
        transition_handle(HandleState.TIMED_OUT, HandleState.COMPLETED)


@pytest.mark.parametrize("terminal", [HandleState.CANCELLED, HandleState.COLLECTED])
def test_terminal_handle_states_reject_everything(terminal):
    for new in HandleState:
        with pytest.raises(InvalidStateTransition):
            transition_handle(terminal, new)


def test_sequence_main_path():
    assert (
        _walk_sequence(
            SequenceState.OPENING,
            SequenceState.PREFILLING,
            SequenceState.READY,
            SequenceState.IN_FLIGHT,
            SequenceState.READY,
            SequenceState.CLOSING,
            SequenceState.CLOSED,
        )
        is SequenceState.CLOSED
    )


@pytest.mark.parametrize(
    "state",
    [
        SequenceState.OPENING,
        SequenceState.PREFILLING,
        SequenceState.READY,
        SequenceState.IN_FLIGHT,
        SequenceState.DESYNCED,
    ],
)
def test_live_sequence_can_always_close_invalidate_or_latch(state):
    for new in (
        SequenceState.CLOSING,
        SequenceState.INVALID,
        SequenceState.TARGET_ONLY,
    ):
        assert can_transition_sequence(state, new)


def test_target_only_is_latched_until_close():
    # The MVP never resumes remote proposals for a latched request.
    for new in (
        SequenceState.READY,
        SequenceState.PREFILLING,
        SequenceState.IN_FLIGHT,
        SequenceState.DESYNCED,
    ):
        assert not can_transition_sequence(SequenceState.TARGET_ONLY, new)
    assert can_transition_sequence(SequenceState.TARGET_ONLY, SequenceState.CLOSING)


def test_desynced_recovers_only_through_prefill():
    assert can_transition_sequence(SequenceState.DESYNCED, SequenceState.PREFILLING)
    for new in (SequenceState.READY, SequenceState.IN_FLIGHT):
        with pytest.raises(InvalidStateTransition):
            transition_sequence(SequenceState.DESYNCED, new)


def test_invalid_requires_reopen():
    assert can_transition_sequence(SequenceState.INVALID, SequenceState.OPENING)
    assert can_transition_sequence(SequenceState.INVALID, SequenceState.CLOSING)
    for new in (SequenceState.PREFILLING, SequenceState.READY):
        assert not can_transition_sequence(SequenceState.INVALID, new)


def test_closed_is_terminal():
    for new in SequenceState:
        assert not can_transition_sequence(SequenceState.CLOSED, new)
