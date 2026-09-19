# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for Model Runner V2 thinking budget tests",
        allow_module_level=True,
    )

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
START = 90
END = 91
END_A = 92
END_B = 93
VOCAB_SIZE = 128


class MockReasoningConfig:
    reasoning_start_token_ids = [START]
    reasoning_end_token_ids = [END]
    natural_reasoning_end_token_ids = [END]


class MockMultiTokenEndReasoningConfig:
    reasoning_start_token_ids = [START]
    reasoning_end_token_ids = [END_A, END_B]
    natural_reasoning_end_token_ids = [END_A, END_B]


class MockDistinctEndReasoningConfig:
    reasoning_start_token_ids = [START]
    reasoning_end_token_ids = [END_A, END_B]
    natural_reasoning_end_token_ids = [END]


class MockLoopBreakReasoningConfig(MockReasoningConfig):
    loop_break_max_pattern_size = 8
    loop_break_min_pattern_size = 2
    loop_break_min_count = 3
    loop_break_min_reasoning_tokens = 16
    loop_break_check_interval = 1


def _make_req_states(tokens: list[int], prompt_len: int = 1) -> RequestState:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=max(64, len(tokens) + 1),
        max_num_batched_tokens=16,
        num_speculative_steps=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    req_states.add_request(
        req_id="req",
        prompt_len=prompt_len,
        all_token_ids=tokens,
        num_computed_tokens=len(tokens),
        max_tokens=32,
    )
    req_states.apply_staged_writes()
    return req_states


def _apply(
    state: ThinkingBudgetState,
    logits: torch.Tensor,
    input_ids: list[int],
    local_pos: list[int],
) -> torch.Tensor:
    idx_mapping = torch.tensor([3], dtype=torch.int32, device=DEVICE)
    expanded_idx_mapping = torch.tensor(
        [3] * len(input_ids), dtype=torch.int32, device=DEVICE
    )
    idx_mapping_np = idx_mapping.cpu().numpy()
    state.apply(
        logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping_np,
        torch.tensor(input_ids, dtype=torch.int32, device=DEVICE),
        torch.tensor(local_pos, dtype=torch.int32, device=DEVICE),
    )
    return logits.cpu()


def test_v2_thinking_budget_forces_end_after_budget_reached():
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.arange(VOCAB_SIZE, dtype=torch.float32, device=DEVICE).view(1, -1)
    expected = logits.cpu()
    out = _apply(state, logits, input_ids=[12], local_pos=[0])

    expected[0, END] = 1.0e9
    torch.testing.assert_close(out, expected)


def test_v2_thinking_budget_restores_masked_end_token():
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    logits[0, END] = -float("inf")
    out = _apply(state, logits, input_ids=[12], local_pos=[0])

    assert out[0, END] == pytest.approx(1.0e9)


def test_v2_thinking_budget_allows_tokens_before_budget():
    req_states = _make_req_states([1, START, 10, 11], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[11], local_pos=[0])

    assert torch.all(out == 0)


def test_v2_thinking_budget_continues_multi_token_end_marker():
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockMultiTokenEndReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((2, VOCAB_SIZE), device=DEVICE)
    out = _apply(
        state,
        logits,
        input_ids=[12, END_A],
        local_pos=[0, 1],
    )

    assert out[0, END_A] == pytest.approx(1.0e9)
    assert out[1, END_B] == pytest.approx(1.0e9)


def test_v2_thinking_budget_uses_distinct_forced_end_marker():
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockDistinctEndReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((2, VOCAB_SIZE), device=DEVICE)
    out = _apply(
        state,
        logits,
        input_ids=[12, END_A],
        local_pos=[0, 1],
    )

    assert out[0, END_A] == pytest.approx(1.0e9)
    assert out[1, END_B] == pytest.approx(1.0e9)


def test_v2_thinking_budget_stops_after_natural_end_marker():
    req_states = _make_req_states(
        [1, START, 10, END, 20, 21, 22],
        prompt_len=1,
    )
    state = ThinkingBudgetState(req_states, MockDistinctEndReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[22], local_pos=[0])

    assert torch.all(out == 0)


def test_v2_thinking_budget_ignores_plain_request():
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams())
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[12], local_pos=[0])

    assert torch.all(out == 0)


def test_v2_greedy_sampling_applies_thinking_budget():
    """Greedy-only requests must not bypass thinking-budget processing."""
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    sampler = Sampler(
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
        reasoning_config=MockReasoningConfig(),
    )
    sampler.add_request(
        req_idx=3,
        prompt_len=1,
        sampling_params=SamplingParams(
            temperature=0.0,
            thinking_token_budget=3,
        ),
    )
    sampler.apply_staged_writes()

    idx_mapping = torch.tensor([3], dtype=torch.int32, device=DEVICE)
    idx_mapping_np = idx_mapping.cpu().numpy()
    expanded_idx_mapping = idx_mapping.clone()
    input_ids = torch.tensor([12], dtype=torch.int32, device=DEVICE)
    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = sampler.apply_sampling_params(
        logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping_np,
        torch.tensor([4], dtype=torch.int32, device=DEVICE),
        input_ids,
        torch.tensor([0], dtype=torch.int32, device=DEVICE),
    )

    assert out[0, END].item() == pytest.approx(1.0e9)


def test_v2_thinking_budget_latest_prefill_end_disables_forcing():
    req_states = _make_req_states(
        [1, START, 10, 11, 12, END, 13],
        prompt_len=1,
    )
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[13], local_pos=[0])

    assert torch.all(out == 0)


def test_v2_thinking_budget_uses_latest_prefill_start_boundary():
    req_states = _make_req_states(
        [1, START, 10, 11, 12, END, 13, START, 14, 15, 16],
        prompt_len=1,
    )
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[16], local_pos=[0])

    assert out[0, END] == pytest.approx(1.0e9)


def test_v2_thinking_budget_incrementally_scans_long_generation():
    """Guard against rescanning the full token history on every decode step."""
    tokens = [1, START, *([10] * 16382)]
    req_states = _make_req_states(tokens)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=32768))
    state.apply_staged_writes()

    _apply(state, torch.zeros((1, VOCAB_SIZE), device=DEVICE), [10], [0])
    assert state.cached_scan_pos[3].item() == len(tokens)

    req_states.all_token_ids.stage_write(3, len(tokens), [10])
    req_states.total_len.stage_write_elem(3, len(tokens) + 1)
    req_states.apply_staged_writes()
    _apply(state, torch.zeros((1, VOCAB_SIZE), device=DEVICE), [10], [0])

    assert state.cached_scan_pos[3].item() == len(tokens) + 1


def test_v2_thinking_budget_clamps_oversized_budget():
    """Budgets beyond int32 must not crash and behave as unlimited."""
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=2**40))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[12], local_pos=[0])

    assert torch.all(out == 0)


def test_v2_thinking_budget_continues_end_prefix_from_prompt():
    """A resumed prompt ending with a partial forced-end marker must not
    restart the marker sequence and duplicate its first token."""
    req_states = _make_req_states([1, START, 10, 11, END_A], prompt_len=5)
    state = ThinkingBudgetState(req_states, MockMultiTokenEndReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[END_A], local_pos=[0])

    assert out[0, END_B] == pytest.approx(1.0e9)
    assert out[0, END_A] == 0


# --- Reasoning loop breaking -------------------------------------------------


class MockShortPatternReasoningConfig(MockReasoningConfig):
    loop_break_max_pattern_size = 2
    loop_break_min_pattern_size = 2
    loop_break_min_count = 3
    loop_break_min_reasoning_tokens = 4
    loop_break_check_interval = 1


class MockIntervalReasoningConfig(MockReasoningConfig):
    loop_break_max_pattern_size = 2
    loop_break_min_pattern_size = 2
    loop_break_min_count = 2
    loop_break_min_reasoning_tokens = 4
    loop_break_check_interval = 8


def _filler(n: int, base: int = 10) -> list[int]:
    """``n`` distinct non-repeating tokens, clear of the marker token ids."""
    assert base + n <= START
    return list(range(base, base + n))


def _append_committed(req_states: RequestState, at: int, tokens: list[int]) -> None:
    """Commit ``tokens`` the way ``post_update`` does after a sampling step."""
    req_states.all_token_ids.stage_write(3, at, tokens)
    req_states.total_len.stage_write_elem(3, at + len(tokens))
    req_states.apply_staged_writes()


def _loop_break_state(
    tokens: list[int],
    config=None,
    params: SamplingParams | None = None,
) -> tuple[RequestState, ThinkingBudgetState]:
    req_states = _make_req_states(tokens, prompt_len=1)
    state = ThinkingBudgetState(
        req_states, config if config is not None else MockLoopBreakReasoningConfig()
    )
    state.add_request(3, params if params is not None else SamplingParams())
    state.apply_staged_writes()
    return req_states, state


def test_v2_loop_break_forces_end_on_repeating_reasoning_tail():
    """The whole point: a request with no thinking budget at all is forced out
    of reasoning once its open section ends in an exact repeat."""
    tokens = [1, START, *_filler(20), 7, 8, 7, 8, 7, 8]
    _, state = _loop_break_state(tokens)

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[tokens[-1]], local_pos=[0])

    assert out[0, END] == pytest.approx(1.0e9)


def test_v2_loop_break_ignores_non_periodic_reasoning():
    tokens = [1, START, *_filler(26)]
    _, state = _loop_break_state(tokens)

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[tokens[-1]], local_pos=[0])

    assert torch.all(out == 0)


def test_v2_loop_break_waits_for_the_reasoning_floor():
    """A short section repeats constantly at the start of generation; the floor
    is what keeps that from ending reasoning immediately."""
    tokens = [1, START, 7, 8, 7, 8, 7, 8]
    _, state = _loop_break_state(tokens)

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[8], local_pos=[0])

    assert torch.all(out == 0)


def test_v2_loop_break_ignores_repetition_outside_reasoning():
    """A repeating answer must not force a second end sequence."""
    tokens = [1, START, *_filler(20), END, *_filler(20), 7, 8, 7, 8, 7, 8]
    _, state = _loop_break_state(tokens)

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[8], local_pos=[0])

    assert torch.all(out == 0)


def test_v2_loop_break_pattern_may_not_span_the_section_start():
    """The tail is clamped to the section start, so a repeat that only closes
    by reaching back past ``<think>`` is not a reasoning loop."""
    spanning = [1, 7, 8, 7, 8, START, 7, 8, 7, 8]
    _, state = _loop_break_state(spanning, config=MockShortPatternReasoningConfig())
    out = _apply(
        state,
        torch.zeros((1, VOCAB_SIZE), device=DEVICE),
        input_ids=[8],
        local_pos=[0],
    )
    assert torch.all(out == 0)

    # Same tail, entirely inside the section: the control that proves the
    # negative above comes from the clamp and not from the detector.
    contained = [1, START, 7, 8, 7, 8, 7, 8]
    _, state = _loop_break_state(contained, config=MockShortPatternReasoningConfig())
    out = _apply(
        state,
        torch.zeros((1, VOCAB_SIZE), device=DEVICE),
        input_ids=[8],
        local_pos=[0],
    )
    assert out[0, END] == pytest.approx(1.0e9)


def test_v2_loop_break_honours_the_check_interval():
    tokens = [1, START, 7, 8, 7, 8]
    req_states, state = _loop_break_state(tokens, config=MockIntervalReasoningConfig())

    out = _apply(
        state,
        torch.zeros((1, VOCAB_SIZE), device=DEVICE),
        input_ids=[8],
        local_pos=[0],
    )
    assert torch.all(out == 0), "detection ran before the interval elapsed"

    _append_committed(req_states, len(tokens), [7, 8, 7, 8])
    out = _apply(
        state,
        torch.zeros((1, VOCAB_SIZE), device=DEVICE),
        input_ids=[8],
        local_pos=[0],
    )
    assert out[0, END] == pytest.approx(1.0e9)


def test_v2_loop_break_keeps_forcing_until_the_end_lands():
    """A forced end token can be rejected under speculative decoding. The flag
    is sticky, so the next step forces it again instead of resuming the loop."""
    tokens = [1, START, *_filler(20), 7, 8, 7, 8, 7, 8]
    req_states, state = _loop_break_state(tokens)

    out = _apply(
        state,
        torch.zeros((1, VOCAB_SIZE), device=DEVICE),
        input_ids=[8],
        local_pos=[0],
    )
    assert out[0, END] == pytest.approx(1.0e9)

    # The forced end was not accepted; a plain token was committed instead.
    _append_committed(req_states, len(tokens), [9])
    out = _apply(
        state,
        torch.zeros((1, VOCAB_SIZE), device=DEVICE),
        input_ids=[9],
        local_pos=[0],
    )
    assert out[0, END] == pytest.approx(1.0e9)


def test_v2_loop_break_rearms_after_the_section_closes():
    tokens = [1, START, *_filler(20), 7, 8, 7, 8, 7, 8]
    req_states, state = _loop_break_state(tokens)

    _apply(
        state,
        torch.zeros((1, VOCAB_SIZE), device=DEVICE),
        input_ids=[8],
        local_pos=[0],
    )
    assert state.loop_break_fired[3].item() == 1

    _append_committed(req_states, len(tokens), [END])
    out = _apply(
        state,
        torch.zeros((1, VOCAB_SIZE), device=DEVICE),
        input_ids=[END],
        local_pos=[0],
    )
    assert torch.all(out == 0)
    assert state.loop_break_fired[3].item() == 0


def test_v2_loop_break_per_request_opt_out():
    tokens = [1, START, *_filler(20), 7, 8, 7, 8, 7, 8]
    _, state = _loop_break_state(
        tokens, params=SamplingParams(thinking_loop_break=False)
    )

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[8], local_pos=[0])

    assert torch.all(out == 0)
    assert state.loop_break_fired[3].item() == -1


def test_v2_loop_break_opt_in_cannot_enable_an_unconfigured_server():
    tokens = [1, START, *_filler(20), 7, 8, 7, 8, 7, 8]
    _, state = _loop_break_state(
        tokens,
        config=MockReasoningConfig(),
        params=SamplingParams(thinking_loop_break=True),
    )

    assert not state.loop_break_enabled
    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[8], local_pos=[0])

    assert torch.all(out == 0)


def test_v2_loop_break_rejects_a_degenerate_min_count():
    """``min_count`` below 2 makes the tail comparison vacuously true, so it has
    to disable the feature rather than fire on every request."""

    class Degenerate(MockReasoningConfig):
        loop_break_max_pattern_size = 8
        loop_break_min_pattern_size = 2
        loop_break_min_count = 1

    tokens = [1, START, *_filler(20), 7, 8, 7, 8, 7, 8]
    _, state = _loop_break_state(tokens, config=Degenerate())

    assert not state.loop_break_enabled
    out = _apply(
        state,
        torch.zeros((1, VOCAB_SIZE), device=DEVICE),
        input_ids=[8],
        local_pos=[0],
    )
    assert torch.all(out == 0)


def test_v2_loop_break_survives_a_plain_sampling_request():
    """A loop-break-only request sets none of the usual logits-processing flags,
    so the sampler's fast path must still be opened for it."""
    tokens = [1, START, *_filler(20), 7, 8, 7, 8, 7, 8]
    req_states = _make_req_states(tokens, prompt_len=1)
    sampler = Sampler(
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
        reasoning_config=MockLoopBreakReasoningConfig(),
    )
    sampler.add_request(
        req_idx=3,
        prompt_len=1,
        sampling_params=SamplingParams(temperature=0.0),
    )
    sampler.apply_staged_writes()

    assert sampler.needs_logits_processing[3]

    idx_mapping = torch.tensor([3], dtype=torch.int32, device=DEVICE)
    out = sampler.apply_sampling_params(
        torch.zeros((1, VOCAB_SIZE), device=DEVICE),
        idx_mapping.clone(),
        idx_mapping,
        idx_mapping.cpu().numpy(),
        torch.tensor([len(tokens) - 1], dtype=torch.int32, device=DEVICE),
        torch.tensor([tokens[-1]], dtype=torch.int32, device=DEVICE),
        torch.tensor([0], dtype=torch.int32, device=DEVICE),
    )

    assert out[0, END].item() == pytest.approx(1.0e9)


def test_v2_loop_break_coexists_with_a_thinking_budget():
    """Loop breaking must not disturb the budget countdown it shares state
    with, and must still fire for a request that also carries a budget."""
    tokens = [1, START, *_filler(20), 7, 8, 7, 8, 7, 8]
    _, state = _loop_break_state(
        tokens, params=SamplingParams(thinking_token_budget=10_000)
    )

    assert state.use_thinking_budget[3]
    out = _apply(
        state,
        torch.zeros((1, VOCAB_SIZE), device=DEVICE),
        input_ids=[8],
        local_pos=[0],
    )
    assert out[0, END] == pytest.approx(1.0e9)


def test_v2_loop_break_does_not_leak_into_a_reused_slot():
    """Request slots are recycled; a fired flag left behind would force the end
    sequence for whoever lands in the slot next."""
    tokens = [1, START, *_filler(20), 7, 8, 7, 8, 7, 8]
    _, state = _loop_break_state(tokens)
    _apply(
        state,
        torch.zeros((1, VOCAB_SIZE), device=DEVICE),
        input_ids=[8],
        local_pos=[0],
    )
    assert state.loop_break_fired[3].item() == 1

    # The next occupant carries a budget, so the forcing kernel still runs for
    # it; only the cleared flag keeps it out of a forced end it never asked for.
    state.add_request(
        3,
        SamplingParams(thinking_token_budget=10_000, thinking_loop_break=False),
    )
    state.apply_staged_writes()
    out = _apply(
        state,
        torch.zeros((1, VOCAB_SIZE), device=DEVICE),
        input_ids=[8],
        local_pos=[0],
    )
    assert torch.all(out == 0)


class MockLoopBreakMultiTokenEndConfig(MockMultiTokenEndReasoningConfig):
    loop_break_max_pattern_size = 8
    loop_break_min_pattern_size = 2
    loop_break_min_count = 3
    loop_break_min_reasoning_tokens = 16
    loop_break_check_interval = 1


def test_v2_loop_break_continues_the_multi_token_end_marker():
    """A loop break reaches the same forced-end state the budget path uses, so
    it writes the whole end sequence across draft positions rather than
    restarting it at every step."""
    tokens = [1, START, *_filler(20), 7, 8, 7, 8, 7, 8]
    _, state = _loop_break_state(tokens, config=MockLoopBreakMultiTokenEndConfig())

    logits = torch.zeros((2, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[tokens[-1], END_A], local_pos=[0, 1])

    assert out[0, END_A] == pytest.approx(1.0e9)
    assert out[0, END_B] == 0
    assert out[1, END_B] == pytest.approx(1.0e9)


# --- A forced end sequence distinct from the parser's natural marker ---------


class MockLoopBreakDistinctEndConfig(MockDistinctEndReasoningConfig):
    loop_break_max_pattern_size = 8
    loop_break_min_pattern_size = 2
    loop_break_min_count = 3
    loop_break_min_reasoning_tokens = 16
    loop_break_check_interval = 1


def test_v2_thinking_budget_stops_after_a_committed_forced_end_sequence():
    """``reasoning_end_str`` may carry a transition phrase, so the sequence
    forcing writes can differ from the parser's natural marker. Committing it
    ends the section; otherwise forcing restarts and the answer never begins."""
    tokens = [1, START, 10, 11, 12]
    req_states = _make_req_states(tokens, prompt_len=1)
    state = ThinkingBudgetState(req_states, MockDistinctEndReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    out = _apply(state, torch.zeros((1, VOCAB_SIZE), device=DEVICE), [12], [0])
    assert out[0, END_A] == pytest.approx(1.0e9)

    # One token per step, as a plain decode commits them: the second half of the
    # marker only matches if the incremental scan overlaps the previous edge.
    _append_committed(req_states, len(tokens), [END_A])
    out = _apply(state, torch.zeros((1, VOCAB_SIZE), device=DEVICE), [END_A], [0])
    assert out[0, END_B] == pytest.approx(1.0e9)

    _append_committed(req_states, len(tokens) + 1, [END_B])
    out = _apply(state, torch.zeros((1, VOCAB_SIZE), device=DEVICE), [END_B], [0])
    assert torch.all(out == 0)


def test_v2_thinking_budget_stops_forcing_inside_the_draft_window():
    """The forced sequence can complete among speculative positions; the ones
    after it belong to the answer."""
    tokens = [1, START, 10, 11, 12]
    req_states = _make_req_states(tokens, prompt_len=1)
    state = ThinkingBudgetState(req_states, MockDistinctEndReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((3, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[12, END_A, END_B], local_pos=[0, 1, 2])

    assert out[0, END_A] == pytest.approx(1.0e9)
    assert out[1, END_B] == pytest.approx(1.0e9)
    assert torch.all(out[2] == 0)


def test_v2_loop_break_closes_on_a_committed_forced_end_sequence():
    """The loop-break flag is sticky until the section closes, so a forced end
    the committed scan cannot see would keep the request forcing forever."""
    tokens = [1, START, *_filler(20), 7, 8, 7, 8, 7, 8]
    req_states, state = _loop_break_state(
        tokens, config=MockLoopBreakDistinctEndConfig()
    )

    out = _apply(
        state,
        torch.zeros((1, VOCAB_SIZE), device=DEVICE),
        input_ids=[tokens[-1]],
        local_pos=[0],
    )
    assert out[0, END_A] == pytest.approx(1.0e9)
    assert state.loop_break_fired[3].item() == 1

    _append_committed(req_states, len(tokens), [END_A])
    out = _apply(
        state,
        torch.zeros((1, VOCAB_SIZE), device=DEVICE),
        input_ids=[END_A],
        local_pos=[0],
    )
    assert out[0, END_B] == pytest.approx(1.0e9)
    assert state.loop_break_fired[3].item() == 1

    _append_committed(req_states, len(tokens) + 1, [END_B])
    out = _apply(
        state,
        torch.zeros((1, VOCAB_SIZE), device=DEVICE),
        input_ids=[END_B],
        local_pos=[0],
    )
    assert torch.all(out == 0), "still forcing after the forced end was committed"
    assert state.loop_break_fired[3].item() == 0


class MockTransitionEndReasoningConfig:
    # The documented shape: a transition phrase before the parser's own marker,
    # so the natural sequence is a SUFFIX of the forced one.
    reasoning_start_token_ids = [START]
    reasoning_end_token_ids = [END_A, END]
    natural_reasoning_end_token_ids = [END]


def test_v2_thinking_budget_natural_close_survives_forced_end_tracking():
    """With a shared suffix both scans see the same close. Tracking the forced
    sequence must not shadow a section the model ended on its own."""
    req_states = _make_req_states([1, START, 10, 11, 12, END, 13], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockTransitionEndReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    out = _apply(state, torch.zeros((1, VOCAB_SIZE), device=DEVICE), [13], [0])
    assert torch.all(out == 0)


def test_v2_thinking_budget_transition_phrase_end_closes_the_section():
    """The same config closed by forcing: transition phrase, then the marker."""
    tokens = [1, START, 10, 11, 12]
    req_states = _make_req_states(tokens, prompt_len=1)
    state = ThinkingBudgetState(req_states, MockTransitionEndReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    out = _apply(state, torch.zeros((1, VOCAB_SIZE), device=DEVICE), [12], [0])
    assert out[0, END_A] == pytest.approx(1.0e9)

    _append_committed(req_states, len(tokens), [END_A])
    out = _apply(state, torch.zeros((1, VOCAB_SIZE), device=DEVICE), [END_A], [0])
    assert out[0, END] == pytest.approx(1.0e9)

    _append_committed(req_states, len(tokens) + 1, [END])
    out = _apply(state, torch.zeros((1, VOCAB_SIZE), device=DEVICE), [END], [0])
    assert torch.all(out == 0)
