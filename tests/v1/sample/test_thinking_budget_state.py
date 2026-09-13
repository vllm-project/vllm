# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ThinkingBudgetStateHolder batch index moves."""

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    MoveDirectionality,
)
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder


class _MockReasoningConfig:
    reasoning_start_token_ids = [151667]
    reasoning_end_token_ids = [151668]


def _make_holder() -> ThinkingBudgetStateHolder:
    return ThinkingBudgetStateHolder(
        _MockReasoningConfig(),
        8,
        0,
        torch.device("cpu"),
        False,
    )


def test_swap_budgeted_with_unbudgeted_clears_empty_side():
    """Asymmetric SWAP must not leave the empty index sharing state."""
    h = _make_holder()
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=[
                (0, SamplingParams(thinking_token_budget=5), None, []),
                (1, SamplingParams(), None, []),
            ],
            moved=(),
        )
    )
    assert list(h._state.keys()) == [0]
    budget_state = h._state[0]

    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )
    assert list(h._state.keys()) == [1]
    assert h._state[1] is budget_state
    assert h._state[1]["thinking_token_budget"] == 5

    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )
    assert list(h._state.keys()) == [0]
    assert h._state[0] is budget_state


def test_swap_exchanges_two_budgeted_states():
    h = _make_holder()
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=[
                (0, SamplingParams(thinking_token_budget=3), None, []),
                (1, SamplingParams(thinking_token_budget=7), None, []),
            ],
            moved=(),
        )
    )
    b0 = h._state[0]["thinking_token_budget"]
    b1 = h._state[1]["thinking_token_budget"]
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )
    assert h._state[0]["thinking_token_budget"] == b1
    assert h._state[1]["thinking_token_budget"] == b0


class _MultiTokenEndConfig:
    """Custom reasoning_end_str: transition phrase + the model's end tag."""

    reasoning_start_token_ids = [151667]
    reasoning_end_token_ids = [100, 200, 300, 151668]


def _make_multi_end_holder() -> ThinkingBudgetStateHolder:
    return ThinkingBudgetStateHolder(
        _MultiTokenEndConfig(),
        8,
        0,
        torch.device("cpu"),
        False,
    )


def _add_budgeted_request(h: ThinkingBudgetStateHolder, budget: int) -> None:
    h.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[(0, SamplingParams(thinking_token_budget=budget), None, [])],
            moved=(),
        )
    )


def test_natural_end_stops_counting_with_multi_token_end_str():
    """Regression for #39697: the model closing with its own bare end tag
    must stop budget counting even when a multi-token reasoning_end_str
    is configured. Otherwise the counter keeps running into the answer
    and force-injects the end sequence mid-content."""
    h = _make_multi_end_holder()
    _add_budgeted_request(h, budget=8)
    # think for 3 tokens, close with the model's own bare end tag,
    # then answer far past the 8-token budget
    out = [151667, 1, 2, 3, 151668, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    h.update_state([out], None)
    st = h._state[0]
    assert st["in_think"] is False, "natural end tag not recognized"
    assert st["in_end"] is False, "forcing engaged after thinking ended"
    assert st["force_index"] == [], "end sequence forced into the answer"


def test_full_custom_end_sequence_still_detected():
    h = _make_multi_end_holder()
    _add_budgeted_request(h, budget=50)
    out = [151667, 1, 2, 100, 200, 300, 151668, 10, 11, 12]
    h.update_state([out], None)
    st = h._state[0]
    assert st["in_think"] is False
    assert st["force_index"] == []


def test_budget_still_forces_while_thinking():
    """The budget must keep working: exceeding it while the model is
    still thinking has to schedule forcing of the end sequence."""
    h = _make_multi_end_holder()
    _add_budgeted_request(h, budget=5)
    out = [151667, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    h.update_state([out], None)
    st = h._state[0]
    assert st["in_end"] is True, "budget exceeded while thinking must force"
    assert st["force_index"], "forcing must be scheduled"
