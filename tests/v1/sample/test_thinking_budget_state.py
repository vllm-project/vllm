# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ThinkingBudgetStateHolder batch index moves."""

import pytest
import torch

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams, validate_reasoning_eos_policy
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    MoveDirectionality,
)
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder

THINK_START = 20
THINK_END = 21
TOOL_CALL = 22
EOS = 2
VOCAB = 32


@pytest.fixture(autouse=True)
def _cpu_async_h2d(monkeypatch: pytest.MonkeyPatch) -> None:
    # async_tensor_h2d pins by default; CPU-only torch has no pinned allocator.
    monkeypatch.setattr("vllm.utils.torch_utils.PIN_MEMORY", False)


class _MockReasoningConfig:
    reasoning_start_token_ids = [THINK_START]
    reasoning_end_token_ids = [THINK_END]
    implicit_reasoning_end_token_ids = None


class _MockReasoningConfigWithToolCall(_MockReasoningConfig):
    implicit_reasoning_end_token_ids = [TOOL_CALL]


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


def _holder_with_tool_call() -> ThinkingBudgetStateHolder:
    return ThinkingBudgetStateHolder(
        _MockReasoningConfigWithToolCall(),
        8,
        0,
        torch.device("cpu"),
        False,
    )


def _add_in_think(
    holder: ThinkingBudgetStateHolder,
    params: SamplingParams,
    output: list[int] | None = None,
) -> None:
    tokens = output if output is not None else [THINK_START, 10, 11]
    holder.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[(0, params, None, tokens)],
            moved=(),
        )
    )
    holder.update_state([tokens], None)


def test_default_policy_does_not_track_without_budget():
    h = _make_holder()
    h.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[(0, SamplingParams(), None, [])],
            moved=(),
        )
    )
    assert not h.has_tracked_requests()


def test_force_end_tracks_without_budget():
    h = _make_holder()
    h.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[
                (0, SamplingParams(reasoning_eos_policy="force_end"), None, []),
            ],
            moved=(),
        )
    )
    assert h.has_tracked_requests()
    assert h._state[0]["thinking_token_budget"] is None
    assert h._state[0]["reasoning_eos_policy"] == "force_end"


def test_force_end_replaces_eos_argmax_with_reasoning_end():
    h = _make_holder()
    _add_in_think(
        h,
        SamplingParams(
            reasoning_eos_policy="force_end",
            stop_token_ids=[EOS],
        ),
    )
    assert h._state[0]["in_think"] is True

    logits = torch.zeros((1, VOCAB), dtype=torch.float32)
    logits[0, EOS] = 5.0
    logits[0, 7] = 1.0
    out = h.apply_to_logits(logits, False, None)
    assert float(out[0, THINK_END]) >= 1.0e8
    assert torch.isneginf(out[0, EOS])
    assert h._state[0]["in_end"] is True


def test_force_end_masks_eos_when_it_is_not_argmax():
    h = _make_holder()
    _add_in_think(
        h,
        SamplingParams(
            reasoning_eos_policy="force_end",
            stop_token_ids=[EOS],
        ),
    )
    logits = torch.zeros((1, VOCAB), dtype=torch.float32)
    logits[0, EOS] = 1.0
    logits[0, 7] = 5.0
    out = h.apply_to_logits(logits, False, None)
    assert torch.isneginf(out[0, EOS])
    assert float(out[0, THINK_END]) == 0.0
    assert h._state[0]["in_end"] is False


def test_stop_policy_leaves_eos_argmax_unchanged():
    h = _make_holder()
    _add_in_think(
        h,
        SamplingParams(
            thinking_token_budget=10_000,
            stop_token_ids=[EOS],
        ),
    )
    logits = torch.zeros((1, VOCAB), dtype=torch.float32)
    logits[0, EOS] = 5.0
    out = h.apply_to_logits(logits, False, None)
    assert float(out[0, EOS]) == 5.0
    assert float(out[0, THINK_END]) == 0.0


def test_force_end_does_not_apply_outside_think_block():
    h = _make_holder()
    tokens = [THINK_START, 10, THINK_END, 15]
    h.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[
                (
                    0,
                    SamplingParams(
                        reasoning_eos_policy="force_end",
                        stop_token_ids=[EOS],
                    ),
                    None,
                    tokens,
                )
            ],
            moved=(),
        )
    )
    h.update_state([tokens], None)
    assert h._state[0]["in_think"] is False

    logits = torch.zeros((1, VOCAB), dtype=torch.float32)
    logits[0, EOS] = 5.0
    out = h.apply_to_logits(logits, False, None)
    assert float(out[0, EOS]) == 5.0
    assert float(out[0, THINK_END]) == 0.0


def test_force_end_does_not_protect_tool_call_inside_think():
    h = _holder_with_tool_call()
    tokens = [THINK_START, 10, 11, TOOL_CALL, 30]
    h.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[
                (
                    0,
                    SamplingParams(
                        reasoning_eos_policy="force_end",
                        stop_token_ids=[EOS],
                    ),
                    None,
                    tokens,
                )
            ],
            moved=(),
        )
    )
    h.update_state([tokens], None)
    assert h._state[0]["in_think"] is False

    logits = torch.zeros((1, VOCAB), dtype=torch.float32)
    logits[0, EOS] = 5.0
    out = h.apply_to_logits(logits, False, None)
    assert float(out[0, EOS]) == 5.0
    assert float(out[0, THINK_END]) == 0.0


def test_force_end_does_not_rewrite_eos_after_speculative_tool_call():
    h = ThinkingBudgetStateHolder(
        _MockReasoningConfigWithToolCall(),
        8,
        2,
        torch.device("cpu"),
        False,
    )
    tokens = [THINK_START, 10, 11]
    h.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[
                (
                    0,
                    SamplingParams(
                        reasoning_eos_policy="force_end",
                        thinking_token_budget=10_000,
                        stop_token_ids=[EOS],
                    ),
                    None,
                    tokens,
                )
            ],
            moved=(),
        )
    )
    h.update_state([tokens], [[TOOL_CALL, EOS]])
    assert h._state[0]["in_end"] is False
    assert h._state[0]["force_index"] == []

    logits = torch.zeros((2, VOCAB), dtype=torch.float32)
    logits[1, EOS] = 5.0
    out = h.apply_to_logits(logits, False, [[TOOL_CALL, EOS]])
    assert float(out[1, EOS]) == 5.0
    assert float(out[0, THINK_END]) == 0.0
    assert float(out[1, THINK_END]) == 0.0


def test_force_end_empty_spec_does_not_mask_next_request_row():
    h = ThinkingBudgetStateHolder(
        _MockReasoningConfig(),
        8,
        2,
        torch.device("cpu"),
        False,
    )
    tokens = [THINK_START, 10, 11]
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=[
                (
                    0,
                    SamplingParams(
                        reasoning_eos_policy="force_end",
                        stop_token_ids=[EOS],
                    ),
                    None,
                    tokens,
                ),
            ],
            moved=(),
        )
    )
    h.update_state([tokens, [1]], [[], [7]])
    logits = torch.zeros((1, VOCAB), dtype=torch.float32)
    logits[0, EOS] = 5.0
    out = h.apply_to_logits(logits, False, [[], [7]])
    assert float(out[0, EOS]) == 5.0
    assert float(out[0, THINK_END]) == 0.0


def test_force_end_forces_at_speculative_eos_index():
    h = ThinkingBudgetStateHolder(
        _MockReasoningConfig(),
        8,
        2,
        torch.device("cpu"),
        False,
    )
    tokens = [THINK_START, 10, 11]
    h.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[
                (
                    0,
                    SamplingParams(
                        reasoning_eos_policy="force_end",
                        thinking_token_budget=10_000,
                        stop_token_ids=[EOS],
                    ),
                    None,
                    tokens,
                )
            ],
            moved=(),
        )
    )
    h.update_state([tokens], [[EOS, 7]])
    assert h._state[0]["in_end"] is True
    assert h._state[0]["force_index"] == [0]

    logits = torch.zeros((2, VOCAB), dtype=torch.float32)
    out = h.apply_to_logits(logits, False, [[EOS, 7]])
    assert float(out[0, THINK_END]) >= 1.0e8


def test_ignore_eos_does_not_treat_eos_as_stop():
    h = _make_holder()
    _add_in_think(
        h,
        SamplingParams(
            reasoning_eos_policy="force_end",
            ignore_eos=True,
            stop_token_ids=[],
        ),
    )
    logits = torch.zeros((1, VOCAB), dtype=torch.float32)
    logits[0, EOS] = 5.0
    out = h.apply_to_logits(logits, False, None)
    assert float(out[0, EOS]) == 5.0
    assert h._state[0]["in_end"] is False


def test_validate_reasoning_eos_policy_defaults_and_rejects():
    assert validate_reasoning_eos_policy(None) == "stop"
    assert validate_reasoning_eos_policy("stop") == "stop"
    assert validate_reasoning_eos_policy("force_end") == "force_end"
    with pytest.raises(VLLMValidationError, match="reasoning_eos_policy"):
        validate_reasoning_eos_policy("drop")
    with pytest.raises(VLLMValidationError, match="reasoning_eos_policy"):
        SamplingParams(reasoning_eos_policy="drop")  # type: ignore[arg-type]


def test_from_optional_forwards_reasoning_eos_policy():
    assert SamplingParams.from_optional().reasoning_eos_policy == "stop"
    params = SamplingParams.from_optional(reasoning_eos_policy="force_end")
    assert params.reasoning_eos_policy == "force_end"


def test_from_optional_preserves_positional_include_stop_str_in_output():
    # thinking_token_budget is the 13th positional arg; include_stop_str_in_output
    # must still bind at the 14th, not to reasoning_eos_policy.
    params = SamplingParams.from_optional(
        1,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0,
        0.0,
        None,
        None,
        None,
        None,
        None,
        True,
    )
    assert params.include_stop_str_in_output is True
    assert params.reasoning_eos_policy == "stop"


def test_stop_token_ids_that_finish_request_respects_ignore_eos():
    params = SamplingParams(stop_token_ids=[5], ignore_eos=False)
    params.update_from_generation_config({}, eos_token_id=EOS)
    assert set(params.stop_token_ids_that_finish_request()) == {5, EOS}

    ignored = SamplingParams(stop_token_ids=[5], ignore_eos=True)
    ignored.update_from_generation_config({}, eos_token_id=EOS)
    assert ignored.stop_token_ids_that_finish_request() == [5]
