# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ThinkingBudgetStateHolder batch index moves."""

import pytest
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


THINK_START, THINK_END, FILLER, ANSWER = 100, 101, 5, 7
VOCAB = 128
NUM_SPEC = 3


def _natural(prefix: list[int], think_len: int) -> int:
    """Greedy target: ``think_len`` FILLER tokens, THINK_END, then ANSWER."""
    if THINK_END in prefix:
        return ANSWER
    return THINK_END if len(prefix) - 1 >= think_len else FILLER


def _step(holder, output, drafts, think_len):
    """One V1 verification step with greedy acceptance: ``update_state``, the
    bonus-row call in ``Sampler``, then the target-row call in
    ``RejectionSampler``."""
    holder.update_state([output], [drafts])
    bonus = torch.zeros(1, VOCAB)
    bonus[0, _natural(output + drafts, think_len)] = 1.0
    bonus = holder.apply_to_logits(bonus, True, [drafts])
    target = torch.zeros(len(drafts), VOCAB)
    for j in range(len(drafts)):
        target[j, _natural(output + drafts[:j], think_len)] = 1.0
    target = holder.apply_to_logits(target, False, [drafts])
    for j, d in enumerate(drafts):
        output.append(int(torch.argmax(target[j])))
        if output[-1] != d:
            return
    output.append(int(torch.argmax(bonus[0])))


@pytest.mark.parametrize(
    "think_len,budget",
    [
        (2, 3),  # natural end in drafts, budget expires in the same window
        (5, 6),  # same, natural end at a later draft position
        (2, 8),  # natural end well within budget
        (7, 3),  # budget reached first: end is forced
    ],
)
def test_spec_decode_natural_end_matches_non_spec(think_len, budget):
    """With a drafter proposing the target's own tokens, the output must match
    non-spec decoding: one THINK_END after min(think_len, budget) tokens."""
    holder = ThinkingBudgetStateHolder(
        create_mock_reasoning_config([THINK_START], [THINK_END]),
        8,
        NUM_SPEC,
        torch.device("cpu"),
        False,
    )
    output: list[int] = []
    params = SamplingParams(thinking_token_budget=budget)
    holder.sync_batch(
        BatchUpdate(
            batch_size=1, removed=(), added=[(0, params, None, output)], moved=()
        )
    )
    output.append(THINK_START)
    while len(output) < think_len + 6:
        drafts: list[int] = []
        for _ in range(NUM_SPEC):
            drafts.append(_natural(output + drafts, think_len))
        _step(holder, output, drafts, think_len)

    n = min(think_len, budget)
    assert output[: n + 2] == [THINK_START] + [FILLER] * n + [THINK_END], output
    assert set(output[n + 2 :]) == {ANSWER}, output
