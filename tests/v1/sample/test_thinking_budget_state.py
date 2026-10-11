# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ThinkingBudgetStateHolder."""

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


THINK_START = 151667
THINK_END = 151668
FILLER = 5
ANSWER = 7
VOCAB = 151680


def _start_spec(budget: int, num_spec: int) -> tuple[ThinkingBudgetStateHolder, list]:
    holder = ThinkingBudgetStateHolder(
        create_mock_reasoning_config([THINK_START], [THINK_END]),
        8,
        num_spec,
        torch.device("cpu"),
        False,
    )
    output: list[int] = []
    holder.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[(0, SamplingParams(thinking_token_budget=budget), None, output)],
            moved=(),
        )
    )
    output.append(THINK_START)
    return holder, output


def _natural(think_len: int):
    """Target argmax: ``think_len`` FILLER tokens, then THINK_END, then ANSWER."""

    def natural(prefix: list[int]) -> int:
        if THINK_END in prefix:
            return ANSWER
        return THINK_END if len(prefix) - 1 >= think_len else FILLER

    return natural


def _step(holder, output, drafts, natural):
    """One V1 verification step with greedy acceptance: ``update_state``, the
    bonus-row call in ``Sampler``, then the target-row call in
    ``RejectionSampler``."""
    holder.update_state([output], [drafts])
    bonus = torch.zeros(1, VOCAB)
    bonus[0, natural(output + drafts)] = 1.0
    bonus = holder.apply_to_logits(bonus, True, [drafts])
    target = torch.zeros(len(drafts), VOCAB)
    for j in range(len(drafts)):
        target[j, natural(output + drafts[:j])] = 1.0
    target = holder.apply_to_logits(target, False, [drafts])
    for j, d in enumerate(drafts):
        t = int(torch.argmax(target[j]))
        output.append(t)
        if t != d:
            return
    output.append(int(torch.argmax(bonus[0])))


@pytest.mark.parametrize("think_len", [2, 5, 6, 7])
@pytest.mark.parametrize("budget", [2, 3, 6, 7, 8])
def test_natural_end_inside_accepted_drafts_is_not_forced_again(think_len, budget):
    """Regression for #61116: a natural end in the draft window must not be
    followed by a second, forced end when the budget runs out in that window."""
    num_spec = 3
    natural = _natural(think_len)
    holder, output = _start_spec(budget, num_spec)
    while len(output) < think_len + 6:
        # Perfect drafter: proposes exactly what the target will pick.
        drafts: list[int] = []
        for _ in range(num_spec):
            drafts.append(natural(output + drafts))
        _step(holder, output, drafts, natural)

    assert output.count(THINK_END) == 1, output
    end = output.index(THINK_END)
    # Thinking tokens before the end: the model's own length, capped by budget.
    assert end - 1 == min(think_len, budget), output
    assert all(t == ANSWER for t in output[end + 1 :]), output
