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


def _make_holder(num_spec_tokens: int = 0) -> ThinkingBudgetStateHolder:
    return ThinkingBudgetStateHolder(
        create_mock_reasoning_config([151667], [151668]),
        8,
        num_spec_tokens,
        torch.device("cpu"),
        False,
    )


START, END, FILLER, VOCAB = 151667, 151668, 7, 160000


def _sync_two_budgeted(holder: ThinkingBudgetStateHolder, budget: int = 2) -> None:
    holder.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=[
                (0, SamplingParams(thinking_token_budget=budget), None, []),
                (1, SamplingParams(thinking_token_budget=budget), None, []),
            ],
            moved=(),
        )
    )


def _force_end(
    holder: ThinkingBudgetStateHolder, index: int, force_index: list[int]
) -> None:
    holder._state[index]["in_end"] = True
    holder._state[index]["force_index"] = force_index
    holder._state[index]["end_count"] = 0


def _boosted_positions(logits: torch.Tensor):
    return [tuple(p) for p in (logits == 1e9).nonzero().tolist()]


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


def test_draftless_step_uses_one_row_per_request():
    """A draft-less step under spec decoding must use the plain layout.

    When spec decoding is enabled but a step proposes no drafts, the step
    goes through the plain Sampler whose logits have one row per request.
    Mapping every request to row 0 would inject other requests' end tokens
    into request 0 and silently skip budget enforcement for the rest.
    """
    h = _make_holder(num_spec_tokens=3)
    _sync_two_budgeted(h)
    _force_end(h, 0, [0])
    _force_end(h, 1, [0])

    logits = torch.zeros(2, VOCAB)
    h.apply_to_logits(logits, False, [[], []])

    assert h.cu_num_tokens == {0: 0, 1: 1}
    assert _boosted_positions(logits) == [(0, END), (1, END)]


def test_draftless_step_enforces_budget_for_every_request():
    """End-to-end loop from issue #59272: every request must be forced."""
    holder = ThinkingBudgetStateHolder(
        _MockReasoningConfig(), 8, 3, torch.device("cpu"), False
    )
    outs = [[], []]
    holder.sync_batch(
        BatchUpdate(
            2,
            [],
            [
                (i, SamplingParams(thinking_token_budget=2), [], outs[i])
                for i in range(2)
            ],
            [],
        )
    )
    for step in range(6):
        holder.update_state(outs, None)
        logits = torch.zeros(2, VOCAB)
        for i in range(2):
            logits[i, START if step == 0 else FILLER] = 1.0
        holder.apply_to_logits(logits, False, [[], []])
        for i in range(2):
            outs[i].append(int(logits[i].argmax()))

    expected = [START, FILLER, FILLER, END, FILLER, FILLER]
    assert outs == [expected, expected]


def test_spec_layout_pins_draft_rows():
    """With drafts present, each request keeps its draft-token rows."""
    h = _make_holder(num_spec_tokens=3)
    _sync_two_budgeted(h)
    _force_end(h, 0, [1])
    _force_end(h, 1, [0])

    logits = torch.zeros(3, VOCAB)
    h.apply_to_logits(logits, False, [[11, 12], [33]])

    assert h.cu_num_tokens == {0: 0, 1: 2}
    # Request 0's second draft row and request 1's first (only) draft row.
    assert _boosted_positions(logits) == [(1, END), (2, END)]


def test_draftless_request_gets_no_rows_on_draft_step():
    """A request with no drafts owns no target-logits rows (gh #59272).

    Its forcing is deferred to the bonus-token pass; writing nothing here
    avoids corrupting another request's rows or indexing past the tensor.
    """
    h = _make_holder(num_spec_tokens=3)
    _sync_two_budgeted(h)
    _force_end(h, 0, [0])
    _force_end(h, 1, [0])

    logits = torch.zeros(1, VOCAB)
    h.apply_to_logits(logits, False, [[11], []])

    assert h.cu_num_tokens == {0: 0}
    assert 1 not in h.cu_num_tokens
    assert _boosted_positions(logits) == [(0, END)]


def test_bonus_token_pass_uses_one_row_per_request():
    """The bonus-token pass always lays out one row per request."""
    h = _make_holder(num_spec_tokens=3)
    _sync_two_budgeted(h)
    _force_end(h, 0, [2])
    _force_end(h, 1, [2])

    logits = torch.zeros(2, VOCAB)
    h.apply_to_logits(logits, True, [[11], [22]])

    assert h.cu_num_tokens == {0: 0, 1: 1}
    assert _boosted_positions(logits) == [(0, END), (1, END)]
