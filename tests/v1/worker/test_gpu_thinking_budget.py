# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

pytest.importorskip("triton")
if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
    pytest.skip(
        "CUDA required for Model Runner V2 thinking budget tests",
        allow_module_level=True,
    )

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logits_processor import LogitsContext
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.thinking_budget import (
    ThinkingBudgetState,
    apply_thinking_budget,
)
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device(current_platform.device_type)
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
        LogitsContext(
            expanded_idx_mapping=expanded_idx_mapping,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_local_pos=torch.tensor(
                local_pos, dtype=torch.int32, device=DEVICE
            ),
            input_ids=torch.tensor(input_ids, dtype=torch.int32, device=DEVICE),
            pos=torch.zeros(len(input_ids), dtype=torch.int32, device=DEVICE),
            seq_lens_upper_bound_np=np.full(1, len(input_ids), dtype=np.int64),
        ),
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
        vllm_config=SimpleNamespace(reasoning_config=MockReasoningConfig()),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
    )
    sampler.add_request(
        req_idx=3,
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
        np.full(1, 4, dtype=np.int64),
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


TOOL_START = 94


def _implicit_budget_step(
    tokens,
    input_ids,
    budget=2,
    implicit_ids=(TOOL_START,),
    cache=None,
    natural_end=(END,),
    forced_end=(END_A, END_B),
    implicit_sequences=None,
):
    """Run real verification rows against committed history and a draft prefix."""
    if cache is None:
        cache = (
            torch.full((1,), -1, dtype=torch.int32, device=DEVICE),
            torch.full((1,), -1, dtype=torch.int32, device=DEVICE),
            torch.zeros(1, dtype=torch.int32, device=DEVICE),
        )

    def ids(values):
        return torch.tensor(values, dtype=torch.int32, device=DEVICE)

    logits = torch.zeros((len(input_ids), VOCAB_SIZE), device=DEVICE)
    if implicit_sequences is None:
        implicit_sequences = [(token_id,) for token_id in implicit_ids]
    apply_thinking_budget(
        logits,
        ids([0]),
        ids([0] * len(input_ids)),
        ids([budget]),
        ids([tokens]),
        ids([len(tokens)]),
        ids(input_ids),
        ids(list(range(len(input_ids)))),
        cache[0],
        cache[1],
        cache[2],
        ids([START]),
        ids(natural_end),
        ids(forced_end),
        ids([token_id for sequence in implicit_sequences for token_id in sequence]),
        tuple(map(len, implicit_sequences)),
    )
    return logits.cpu(), cache


@pytest.mark.parametrize("budget", [0, 2, 3])
@pytest.mark.parametrize("natural_end", [(END,), (END_A, END_B)])
def test_v2_implicit_end_stops_forcing_inside_committed_tool_arguments(
    budget, natural_end
):
    """Even at exhaustion, a committed tool start wins over forcing."""
    out, _ = _implicit_budget_step(
        [START, 10, TOOL_START, 11, 12],
        [12],
        budget=budget,
        natural_end=natural_end,
    )
    assert torch.count_nonzero(out) == 0


def test_v2_implicit_end_in_verification_prefix_protects_bonus_row():
    out, cache = _implicit_budget_step([START, 10], [10, TOOL_START, 11, 12])
    # Without the boundary, rows 1..3 would reach/exceed the budget.
    assert torch.count_nonzero(out) == 0
    assert cache[1].item() == -1, "draft boundaries must not enter committed cache"


def test_v2_rejected_implicit_end_does_not_disable_next_step_budget():
    _, cache = _implicit_budget_step([START, 10], [10, TOOL_START, 11])
    # The tool start was rejected; target instead committed ordinary token 12.
    out, cache = _implicit_budget_step([START, 10, 12], [12], cache=cache)
    assert out[0, END_A] == 1.0e9
    assert cache[1].item() == -1


def test_v2_committed_implicit_end_survives_incremental_rescan():
    _, cache = _implicit_budget_step([START, 10], [10])
    out, cache = _implicit_budget_step([START, 10, TOOL_START, 11], [11], cache=cache)
    assert torch.count_nonzero(out) == 0
    out, cache = _implicit_budget_step(
        [START, 10, TOOL_START, 11, 12], [12], cache=cache
    )
    assert torch.count_nonzero(out) == 0
    assert cache[1].item() == 2


@pytest.mark.parametrize("budget", [-1, 2])
def test_v2_implicit_end_disabled_preserves_legacy_budget(budget):
    out, _ = _implicit_budget_step(
        [START, 10, TOOL_START, 11], [11], budget=budget, implicit_ids=()
    )
    expected = torch.zeros_like(out)
    if budget >= 0:
        expected[0, END_A] = 1.0e9
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_v2_implicit_end_does_not_interrupt_explicit_multi_token_force():
    out, _ = _implicit_budget_step(
        [START, 10, 11], [11, END_A, END_B], natural_end=(END_A, END_B)
    )
    expected = torch.zeros_like(out)
    expected[0, END_A] = 1.0e9
    expected[1, END_B] = 1.0e9
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_v2_all_parser_implicit_ends_are_honored():
    for tool_id in (TOOL_START, TOOL_START + 1):
        out, _ = _implicit_budget_step(
            [START, tool_id, 10, 11],
            [11],
            implicit_ids=(TOOL_START, TOOL_START + 1),
        )
        assert torch.count_nonzero(out) == 0


def test_v2_implicit_boundary_isolation_with_reordered_batch():
    def ids(values):
        return torch.tensor(values, dtype=torch.int32, device=DEVICE)

    logits = torch.zeros((2, VOCAB_SIZE), device=DEVICE)
    last_start, last_end, scan_pos = ids([-1, -1]), ids([-1, -1]), ids([0, 0])
    apply_thinking_budget(
        logits,
        ids([1, 0]),
        ids([1, 0]),
        ids([2, 2]),
        ids([[START, 10, TOOL_START, 11], [START, 10, 11, 12]]),
        ids([4, 4]),
        ids([12, 11]),
        ids([0, 0]),
        last_start,
        last_end,
        scan_pos,
        ids([START]),
        ids([END]),
        ids([END_A, END_B]),
        ids([TOOL_START]),
    )
    expected = torch.zeros_like(logits)
    expected[0, END_A] = 1.0e9
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    assert last_end.cpu().tolist() == [2, -1]


@pytest.mark.parametrize("sequence", [(94, 95), (94, 95, 96)])
def test_v2_multi_token_implicit_end_in_committed_history(sequence):
    out, cache = _implicit_budget_step(
        [START, 10, *sequence, 11, 12],
        [12],
        implicit_sequences=[sequence],
    )
    assert torch.count_nonzero(out) == 0
    assert cache[1].item() == 2


def test_v2_multi_token_implicit_end_across_committed_and_draft_boundary():
    out, cache = _implicit_budget_step(
        [START, 94],
        [94, 95, 10, 11],
        implicit_sequences=[(94, 95)],
    )
    assert torch.count_nonzero(out) == 0
    assert cache[1].item() == -1


def test_v2_multi_token_implicit_end_completed_on_next_decode_step():
    _, cache = _implicit_budget_step(
        [START, 94],
        [94],
        implicit_sequences=[(94, 95)],
    )
    out, cache = _implicit_budget_step(
        [START, 94, 95, 10],
        [10],
        cache=cache,
        implicit_sequences=[(94, 95)],
    )
    assert torch.count_nonzero(out) == 0
    assert cache[1].item() == 1


def test_v2_partial_implicit_marker_gets_bounded_grace():
    out, cache = _implicit_budget_step(
        [START, 10, 94],
        [94],
        implicit_sequences=[(94, 95)],
    )
    assert torch.count_nonzero(out) == 0
    assert cache[1].item() == -1


def test_v2_rejected_multi_token_implicit_end_does_not_commit_boundary():
    _, cache = _implicit_budget_step(
        [START, 94],
        [94, 95, 10],
        implicit_sequences=[(94, 95)],
    )
    out, cache = _implicit_budget_step(
        [START, 94, 10],
        [10],
        cache=cache,
        implicit_sequences=[(94, 95)],
    )
    assert out[0, END_A] == 1.0e9
    assert cache[1].item() == -1


def test_v2_mixed_single_and_multi_token_implicit_markers():
    for sequence in [(94,), (95, 96)]:
        out, _ = _implicit_budget_step(
            [START, *sequence, 10, 11],
            [11],
            implicit_sequences=[(94,), (95, 96)],
        )
        assert torch.count_nonzero(out) == 0


@pytest.mark.parametrize("prefix", [(94,), (94, 95)])
def test_v2_budget_exhaustion_allows_implicit_marker_to_finish(prefix):
    out, _ = _implicit_budget_step(
        [START, 10, *prefix], [prefix[-1]], implicit_sequences=[(94, 95, 96)]
    )
    assert torch.count_nonzero(out) == 0


def test_v2_prefix_grace_is_not_renewed_by_repeated_prefixes():
    for count in range(1, 7):
        out, _ = _implicit_budget_step(
            [START, 10, *([94] * count)],
            [94],
            implicit_sequences=[(94, 94, 95)],
        )
        if count <= 2:
            assert torch.count_nonzero(out) == 0
        else:
            assert out[0, END_A] == 1.0e9


def test_v2_prefix_guard_does_not_start_after_budget_exhaustion():
    out, _ = _implicit_budget_step(
        [START, 10, 11, 94],
        [94],
        implicit_sequences=[(94, 95, 96)],
    )
    assert out[0, END_A] == 1.0e9


def test_v2_diverged_prefix_forces_end_without_using_remaining_grace():
    out, _ = _implicit_budget_step(
        [START, 10, 94, 12],
        [12],
        implicit_sequences=[(94, 95, 96)],
    )
    assert out[0, END_A] == 1.0e9


def test_v2_prefix_guard_in_speculative_block_is_bounded_per_row():
    out, cache = _implicit_budget_step(
        [START, 10],
        [10, 94, 94, 94, 94],
        implicit_sequences=[(94, 94, 95)],
    )
    expected = torch.zeros_like(out)
    expected[3:, END_A] = 1.0e9
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert cache[1].item() == -1


def test_v2_rejected_prefix_does_not_consume_committed_grace():
    _, cache = _implicit_budget_step(
        [START, 10],
        [10, 94, 94, 94],
        implicit_sequences=[(94, 94, 95)],
    )
    out, _ = _implicit_budget_step(
        [START, 10, 94],
        [94],
        cache=cache,
        implicit_sequences=[(94, 94, 95)],
    )
    assert torch.count_nonzero(out) == 0


@pytest.mark.parametrize("natural_end", [(END_A,), (END_A, END_B)])
def test_v2_started_force_finishes_after_natural_end_inside_sequence(natural_end):
    out, _ = _implicit_budget_step(
        [START, 10, 11],
        [11, END_A, END_B, 96, 12],
        natural_end=natural_end,
        forced_end=(END_A, END_B, 96),
    )
    expected = torch.zeros_like(out)
    expected[0, END_A] = 1.0e9
    expected[1, END_B] = 1.0e9
    expected[2, 96] = 1.0e9
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("prefix", [(94,), (94, 95)])
def test_v2_started_force_has_priority_over_implicit_prefix_and_end(prefix):
    out, _ = _implicit_budget_step(
        [START, 10, 11, *prefix],
        [prefix[-1]],
        forced_end=(94, 95, 96),
        implicit_sequences=[(94, 95)],
    )
    assert out[0, (94, 95, 96)[len(prefix)]] == 1.0e9


def test_v2_force_prefix_in_tool_arguments_does_not_restart_thinking():
    out, _ = _implicit_budget_step(
        [START, 94, 95, 10, 11, END_A],
        [END_A],
        natural_end=(END_A,),
        implicit_sequences=[(94, 95)],
    )
    assert torch.count_nonzero(out) == 0


def test_v2_force_continues_across_embedded_start_marker():
    out, _ = _implicit_budget_step(
        [START, 10, 11, END_A, START],
        [START],
        forced_end=(END_A, START, END_B),
    )
    assert out[0, END_B] == 1.0e9


def test_v2_completed_force_stays_ended_across_decode_steps():
    _, cache = _implicit_budget_step(
        [START, 10, 11, END_A],
        [END_A],
        forced_end=(END_A, START, END_B),
    )
    out, cache = _implicit_budget_step(
        [START, 10, 11, END_A, START, END_B, 12],
        [12],
        cache=cache,
        forced_end=(END_A, START, END_B),
    )
    assert torch.count_nonzero(out) == 0


def test_v2_rejected_force_prefix_does_not_lock_later_tool_call():
    _, cache = _implicit_budget_step(
        [START, 10],
        [10, END_A],
    )
    out, _ = _implicit_budget_step(
        [START, 10, TOOL_START, 12],
        [12],
        cache=cache,
    )
    assert torch.count_nonzero(out) == 0


@pytest.mark.parametrize("forced_end", [(94, 94), (94, 95, 94), (94, 95, 94, 95)])
@pytest.mark.parametrize("tail_len", [0, 1, 2])
def test_v2_self_overlapping_force_does_not_restart_after_completion(
    forced_end, tail_len
):
    tail = forced_end[:tail_len]
    tokens = [START, 10, 11, *forced_end, *tail]
    out, _ = _implicit_budget_step(
        tokens,
        [tokens[-1]],
        forced_end=forced_end,
        implicit_sequences=[(99, 100)],
    )
    assert torch.count_nonzero(out) == 0


def test_v2_force_protection_disabled_preserves_legacy_behavior():
    out, _ = _implicit_budget_step(
        [START, 10, 11, END_A],
        [END_A],
        forced_end=(END_A, END_B, 96),
        natural_end=(END_A,),
        implicit_ids=(),
    )
    assert torch.count_nonzero(out) == 0


@pytest.mark.parametrize("tail_len", [1022, 2046])
def test_v2_completed_force_crossing_cold_scan_block_stays_ended(tail_len):
    out, _ = _implicit_budget_step(
        [START, 10, 11, END_A, START, END_B, *([12] * tail_len)],
        [12],
        forced_end=(END_A, START, END_B),
    )
    assert torch.count_nonzero(out) == 0
