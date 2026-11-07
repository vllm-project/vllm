# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ThinkingBudgetStateHolder batch index moves."""

import torch

from tests.v1.sample.utils import create_mock_reasoning_config
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    MoveDirectionality,
)
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder


def _make_holder() -> ThinkingBudgetStateHolder:
    return ThinkingBudgetStateHolder(
        create_mock_reasoning_config([151667], [151668]),
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


THINK_START = 151667
THINK_END = 151668
TURN_END = 151645


def _forced_tokens(prompt_tok_ids: list[int], budget: int) -> list[int]:
    """Add one request and return the token ids forced at its first step."""
    h = ThinkingBudgetStateHolder(
        create_mock_reasoning_config(
            [THINK_START], [THINK_END], turn_end_token_ids=[TURN_END]
        ),
        8,
        0,
        torch.device("cpu"),
        False,
    )
    output_tok_ids: list[int] = []
    h.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[
                (
                    0,
                    SamplingParams(thinking_token_budget=budget),
                    prompt_tok_ids,
                    output_tok_ids,
                )
            ],
            moved=(),
        )
    )
    h.update_state([output_tok_ids], None)
    logits = h.apply_to_logits(
        torch.zeros((1, THINK_END + 1)), predict_bonus_token=False, spec_token_ids=None
    )
    return (logits[0] == 1e9).nonzero().flatten().tolist()


def test_start_marker_in_earlier_prompt_turn_is_not_counted():
    """A start marker quoted in an earlier message, closed by an end-of-turn
    token, must not count the rest of the prompt against the budget."""
    prompt = [1, THINK_START, 10, 11, 12, TURN_END, 13, 14]
    assert _forced_tokens(prompt, budget=3) == []


def test_start_marker_after_prompt_turn_end_is_counted():
    """Reasoning left open in the final turn of the prompt, such as a
    resumed partial assistant message, still counts against the budget."""
    prompt = [1, TURN_END, 2, THINK_START, 10, 11, 12]
    assert _forced_tokens(prompt, budget=3) == [THINK_END]
