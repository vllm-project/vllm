# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for reasoning-scoped loop breaking in ThinkingBudgetStateHolder."""

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    MoveDirectionality,
)
from vllm.v1.sample.thinking_budget_state import (
    _LOOP_BREAK_ONLY_BUDGET,
    ThinkingBudgetStateHolder,
)

THINK_START = 100
THINK_END = 200
TRANSITION = 201
VOCAB_SIZE = 256


class _LoopBreakReasoningConfig:
    reasoning_start_token_ids = [THINK_START]
    reasoning_end_token_ids = [THINK_END]
    loop_break_max_pattern_size = 8
    loop_break_min_pattern_size = 2
    loop_break_min_count = 3
    loop_break_min_reasoning_tokens = 32
    loop_break_check_interval = 4


class _NoLoopBreakReasoningConfig:
    reasoning_start_token_ids = [THINK_START]
    reasoning_end_token_ids = [THINK_END]


class _TransitionEndReasoningConfig(_LoopBreakReasoningConfig):
    # ``reasoning_end_str`` may prepend a transition phrase to the parser's own
    # end marker, so forcing writes a longer sequence than a natural exit.
    reasoning_end_token_ids = [TRANSITION, THINK_END]
    natural_reasoning_end_token_ids = [THINK_END]


def _make_holder(config=None) -> ThinkingBudgetStateHolder:
    return ThinkingBudgetStateHolder(
        config if config is not None else _LoopBreakReasoningConfig(),
        8,
        0,
        torch.device("cpu"),
        False,
    )


def _add_request(
    holder: ThinkingBudgetStateHolder,
    params: SamplingParams,
    index: int = 0,
) -> list[int]:
    output: list[int] = []
    holder.sync_batch(
        BatchUpdate(
            batch_size=index + 1,
            removed=(),
            added=[(index, params, None, output)],
            moved=(),
        )
    )
    return output


def _feed(
    holder: ThinkingBudgetStateHolder,
    output: list[int],
    tokens: list[int],
    chunk: int = 4,
) -> None:
    """Append tokens in chunks, refreshing holder state per engine step."""
    for i in range(0, len(tokens), chunk):
        output.extend(tokens[i : i + chunk])
        holder.update_state([output], None)


def _non_periodic(n: int, base: int = 10) -> list[int]:
    # Strictly increasing tokens can never form a repeating pattern.
    return [base + i for i in range(n)]


def test_loop_break_state_created_without_budget():
    h = _make_holder()
    _add_request(h, SamplingParams())
    assert 0 in h._state
    state = h._state[0]
    assert state["loop_break"] is True
    assert state["thinking_token_budget"] == _LOOP_BREAK_ONLY_BUDGET


def test_opt_out_creates_no_state():
    h = _make_holder()
    _add_request(h, SamplingParams(thinking_loop_break=False))
    assert 0 not in h._state


def test_true_override_without_server_config_is_inert():
    h = _make_holder(_NoLoopBreakReasoningConfig())
    _add_request(h, SamplingParams(thinking_loop_break=True))
    assert 0 not in h._state


def test_fires_on_exact_loop_and_forces_end_token():
    h = _make_holder()
    output = _add_request(h, SamplingParams())

    cycle = [7, 8, 9]
    _feed(h, output, [THINK_START] + _non_periodic(40) + cycle * 12)

    state = h._state[0]
    assert state["lb_fired"] is True
    assert state["in_end"] is True
    assert state["force_index"] == [0]

    logits = torch.zeros(1, VOCAB_SIZE)
    logits = h.apply_to_logits(logits, predict_bonus_token=False, spec_token_ids=None)
    assert logits[0, THINK_END].item() >= 1e9
    assert int(torch.argmax(logits[0]).item()) == THINK_END


def test_no_fire_on_non_periodic_output():
    h = _make_holder()
    output = _add_request(h, SamplingParams())
    _feed(h, output, [THINK_START] + _non_periodic(200))
    state = h._state[0]
    assert state["lb_fired"] is False
    assert state["in_end"] is False


def test_no_fire_below_reasoning_floor():
    h = _make_holder()
    output = _add_request(h, SamplingParams())
    cycle = [7, 8, 9]
    # 24 reasoning tokens of pure loop: below the 32-token floor.
    _feed(h, output, [THINK_START] + cycle * 8)
    state = h._state[0]
    assert state["lb_fired"] is False
    assert state["in_end"] is False
    # Crossing the floor with the loop still running must fire.
    _feed(h, output, cycle * 4)
    assert state["lb_fired"] is True
    assert state["in_end"] is True


def test_no_fire_outside_reasoning_section():
    h = _make_holder()
    output = _add_request(h, SamplingParams())
    cycle = [7, 8, 9]
    # Loop occurs after the reasoning section has closed.
    _feed(h, output, [THINK_START, 11, THINK_END] + cycle * 30)
    state = h._state[0]
    assert state["lb_fired"] is False
    assert state["in_end"] is False


def test_section_end_rearms_detection():
    h = _make_holder()
    output = _add_request(h, SamplingParams())
    cycle = [7, 8, 9]
    _feed(h, output, [THINK_START] + _non_periodic(40) + cycle * 12)
    state = h._state[0]
    assert state["lb_fired"] is True

    # The forced end token is accepted, closing the section.
    _feed(h, output, [THINK_END], chunk=1)
    assert state["lb_in_think"] is False
    assert state["lb_fired"] is False

    # A second reasoning section can fire again.
    _feed(h, output, [THINK_START] + _non_periodic(40, base=1000) + cycle * 12)
    assert state["lb_fired"] is True
    assert state["in_end"] is True


def test_natural_section_end_stops_loop_tracking():
    """A natural exit emits only the parser's end marker. Track just the forced
    sequence and answer tokens keep counting as reasoning, so a repetitive
    answer forces a second end sequence mid-answer."""
    h = _make_holder(_TransitionEndReasoningConfig())
    output = _add_request(h, SamplingParams())
    _feed(h, output, [THINK_START] + _non_periodic(40) + [THINK_END])
    state = h._state[0]
    assert state["lb_in_think"] is False

    _feed(h, output, [7, 8, 9] * 30)
    assert state["lb_fired"] is False
    assert state["in_end"] is False


def _add_request_with_prompt(
    holder: ThinkingBudgetStateHolder,
    params: SamplingParams,
    prompt: list[int],
) -> list[int]:
    output: list[int] = []
    holder.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[(0, params, prompt, output)],
            moved=(),
        )
    )
    return output


def test_prompt_closed_by_natural_end_is_not_in_think():
    """A prior turn that ended with the parser's own marker, without the
    transition phrase forcing writes, must not classify the next request as
    still reasoning: that would charge answer tokens to the budget and scope
    loop detection over them."""
    h = _make_holder(_TransitionEndReasoningConfig())
    prompt = [THINK_START] + _non_periodic(10) + [THINK_END] + [7, 8, 9]
    _add_request_with_prompt(h, SamplingParams(thinking_token_budget=5), prompt)
    state = h._state[0]
    assert state["in_think"] is False
    assert state["think_count"] == 0
    assert state["check_count_down"] == 5


def test_prompt_closed_by_forced_end_is_not_in_think():
    h = _make_holder(_TransitionEndReasoningConfig())
    prompt = [THINK_START] + _non_periodic(10) + [TRANSITION, THINK_END] + [7]
    _add_request_with_prompt(h, SamplingParams(thinking_token_budget=5), prompt)
    assert h._state[0]["in_think"] is False


def test_prompt_still_open_after_natural_end_is_in_think():
    h = _make_holder(_TransitionEndReasoningConfig())
    prompt = [THINK_START, 1, THINK_END, 2, THINK_START] + _non_periodic(3)
    _add_request_with_prompt(h, SamplingParams(thinking_token_budget=5), prompt)
    state = h._state[0]
    assert state["in_think"] is True
    assert state["think_count"] == 3


def test_budget_path_sees_a_natural_section_end():
    """The budget tracker's marker scan must also recognise the parser's own
    end marker, or a section the model closed itself stays open and answer
    tokens keep drawing down the budget."""
    h = _make_holder(_TransitionEndReasoningConfig())
    output = _add_request(h, SamplingParams(thinking_token_budget=64))
    _feed(h, output, [THINK_START] + _non_periodic(12) + [THINK_END])
    state = h._state[0]
    assert state["in_think"] is False
    _feed(h, output, _non_periodic(16, base=500))
    assert state["in_think"] is False
    assert state["check_count_down"] == 64


def test_budget_and_loop_break_coexist():
    h = _make_holder()
    output = _add_request(h, SamplingParams(thinking_token_budget=5))
    state = h._state[0]
    assert state["loop_break"] is True
    assert state["thinking_token_budget"] == 5
    # The budget fires long before the loop-break floor is reached.
    _feed(h, output, [THINK_START] + _non_periodic(12), chunk=1)
    assert state["in_end"] is True
    assert state["lb_fired"] is False


def test_swap_preserves_loop_break_state():
    h = _make_holder()
    output = _add_request(h, SamplingParams())
    _feed(h, output, [THINK_START] + _non_periodic(10))
    lb_state = h._state[0]

    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )
    assert list(h._state.keys()) == [1]
    assert h._state[1] is lb_state
    assert h._state[1]["loop_break"] is True
