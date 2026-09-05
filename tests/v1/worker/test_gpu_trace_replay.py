# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Model Runner V2 inference trace-replay.

Trace-replay forces the sampler to emit a predetermined sequence of decode
token IDs. The replay step for a request is derived from GPU state as
``total_len - prompt_len`` (output tokens produced so far), and the trace token
overwrites the sampled token in place before logprobs are computed.
"""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for trace-replay tests", allow_module_level=True)

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.trace_replay import (
    TraceReplayState,
    apply_trace_tokens,
)
from vllm.v1.worker.gpu.states import RequestState

DEVICE = "cuda"
TEST_MAX_MODEL_LEN = 2048


def _i32(x) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.int32, device=DEVICE)


def _i64(x) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.int64, device=DEVICE)


def _trace_state(max_num_reqs: int) -> TraceReplayState:
    req_states = RequestState(
        max_num_reqs=max_num_reqs,
        max_model_len=TEST_MAX_MODEL_LEN,
        max_num_batched_tokens=TEST_MAX_MODEL_LEN,
        num_speculative_steps=0,
        vocab_size=1024,
        device=torch.device(DEVICE),
    )
    return TraceReplayState(req_states)


def _admit(
    state: TraceReplayState,
    req_id: str,
    prompt: list[int],
    params: SamplingParams,
    prefill: list[int] | None = None,
    num_computed_tokens: int = 0,
) -> int:
    prefill = prompt if prefill is None else prefill
    req_states = state.req_states
    suffix = state.get_token_suffix(len(prompt), len(prefill), params)
    req_states.add_request(
        req_id=req_id,
        prompt_len=len(prompt),
        all_token_ids=prefill,
        num_computed_tokens=num_computed_tokens,
        max_tokens=params.max_tokens,
        future_token_ids=suffix,
    )
    req_idx = req_states.req_id_to_index[req_id]
    state.add_request(req_idx, params)
    return req_idx


def _apply_admissions(state: TraceReplayState) -> None:
    state.req_states.apply_staged_writes()
    state.apply_staged_writes()


def _set_lens(state: TraceReplayState, total_len, prompt_len) -> None:
    """Position each request at a replay step via total_len - prompt_len."""
    state.req_states.total_len.gpu[: len(total_len)] = _i32(total_len)
    state.req_states.prompt_len.gpu[: len(prompt_len)] = _i32(prompt_len)


# ------------------------------ Kernel ------------------------------------


def test_replay_overwrites_sampled_at_each_step():
    """The trace token for the current step replaces the sampled token."""
    trace = [[100, 101, 102], [200, 201, 202]]
    trace_len = _i32([3, 3])
    all_token_ids = torch.zeros(2, TEST_MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE)
    for i, t in enumerate(trace):
        start = (5, 8)[i]
        all_token_ids[i, start : start + len(t)] = _i32(t)
    prompt_len = _i32([5, 8])
    idx_mapping = _i32([0, 1])

    for step in range(3):
        sampled = _i64([-7, -7])  # sentinel that must be overwritten
        total_len = _i32([5 + step, 8 + step])
        apply_trace_tokens(
            sampled, idx_mapping, all_token_ids, trace_len, total_len, prompt_len
        )
        assert sampled.tolist() == [trace[0][step], trace[1][step]]


def test_past_end_of_trace_leaves_sampled_untouched():
    """Once step >= trace_len, the sampler's own token is kept."""
    all_token_ids = torch.zeros(1, TEST_MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE)
    all_token_ids[0, 4:6] = _i32([100, 101])
    trace_len = _i32([2])
    prompt_len = _i32([4])
    idx_mapping = _i32([0])

    sampled = _i64([999])
    total_len = _i32([4 + 2])  # step == 2 == trace_len -> out of range
    apply_trace_tokens(
        sampled, idx_mapping, all_token_ids, trace_len, total_len, prompt_len
    )
    assert sampled.tolist() == [999]


def test_non_trace_request_untouched():
    """trace_len == 0 means the request never uses replay."""
    all_token_ids = torch.zeros(1, TEST_MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE)
    trace_len = _i32([0])
    prompt_len = _i32([3])
    idx_mapping = _i32([0])

    sampled = _i64([42])
    total_len = _i32([3])
    apply_trace_tokens(
        sampled, idx_mapping, all_token_ids, trace_len, total_len, prompt_len
    )
    assert sampled.tolist() == [42]


def test_idx_mapping_indirection_and_negative_skip():
    """batch_idx -> req_state_idx indirection, and negative entries are skipped."""
    # req_state 0: no trace; req_state 1: trace [500, 501].
    all_token_ids = torch.zeros(2, TEST_MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE)
    all_token_ids[1, 6:8] = _i32([500, 501])
    trace_len = _i32([0, 2])
    prompt_len = _i32([10, 6])

    # batch: [maps to state 1, masked (-1), maps to state 0].
    idx_mapping = _i32([1, -1, 0])
    sampled = _i64([1, 2, 3])
    # Indexed by req_state_idx: state 0 at total_len=10, state 1 at step 1.
    total_len = _i32([10, 6 + 1])
    apply_trace_tokens(
        sampled, idx_mapping, all_token_ids, trace_len, total_len, prompt_len
    )
    # batch 0 -> state 1 step 1 -> 501; batch 1 masked; batch 2 -> state 0 no trace.
    assert sampled.tolist() == [501, 2, 3]


# --------------------------- TraceReplayState -----------------------------


def test_state_end_to_end():
    """add_request -> apply_staged_writes -> apply_trace overwrites correctly."""
    state = _trace_state(4)
    prompt = [1, 2, 3, 4, 5, 6, 7]
    trace_idx = _admit(
        state,
        "trace",
        prompt,
        SamplingParams(trace_decode_token_ids=[11, 22, 33]),
    )
    normal_idx = _admit(state, "normal", prompt, SamplingParams())
    _apply_admissions(state)

    assert not hasattr(state, "trace_token_ids")
    assert state.req_states.all_token_ids.gpu[trace_idx, 7:10].tolist() == [11, 22, 33]

    idx_mapping = _i32([trace_idx, normal_idx])
    sampled = _i64([-1, -1])
    state.req_states.total_len.gpu[trace_idx] = 7 + 1
    state.apply_trace(sampled, idx_mapping)
    assert sampled.tolist() == [22, -1]


def test_state_leaves_non_trace_batch_unchanged():
    """Requests with trace_len == 0 remain unchanged after a trace was seen."""
    state = _trace_state(4)
    prompt = [1, 2, 3, 4, 5, 6, 7]
    _admit(
        state,
        "trace",
        prompt,
        SamplingParams(trace_decode_token_ids=[11, 22]),
    )
    normal_idx = _admit(state, "normal", prompt, SamplingParams())
    _apply_admissions(state)

    idx_mapping = _i32([normal_idx])
    sampled = _i64([555])
    state.apply_trace(sampled, idx_mapping)
    assert sampled.tolist() == [555]


def test_slot_reuse_clears_trace():
    """Reusing a slot for a non-trace request must not replay stale tokens."""
    state = _trace_state(1)
    trace_idx = _admit(
        state,
        "trace",
        [1, 2, 3],
        SamplingParams(trace_decode_token_ids=[11, 22]),
    )
    _apply_admissions(state)
    assert state.req_states.remove_request("trace") == trace_idx
    normal_idx = _admit(state, "normal", [7, 8, 9], SamplingParams())
    _apply_admissions(state)
    assert normal_idx == trace_idx

    idx_mapping = _i32([normal_idx])
    sampled = _i64([888])
    state.apply_trace(sampled, idx_mapping)
    assert sampled.tolist() == [888]


def test_resume_stages_only_unconsumed_trace_suffix():
    """A resumed request derives its trace offset from its prefill history."""
    state = _trace_state(2)
    prompt = [1, 2, 3]
    trace = [11, 22, 33]
    req_idx = _admit(
        state,
        "resumed",
        prompt,
        SamplingParams(trace_decode_token_ids=trace),
        prefill=prompt + trace[:1],
    )
    _apply_admissions(state)

    assert state.req_states.all_token_ids.gpu[req_idx, :6].tolist() == prompt + trace
    sampled = _i64([-1])
    state.apply_trace(sampled, _i32([req_idx]))
    assert sampled.tolist() == [22]


def test_full_admission_batch_uses_one_staged_write_per_request():
    """Trace suffixes must not overflow staged-write metadata at full capacity."""
    state = _trace_state(2)
    params = SamplingParams(trace_decode_token_ids=[11, 22])
    first_idx = _admit(state, "first", [1, 2, 3], params)
    second_idx = _admit(state, "second", [4, 5, 6], params)

    assert len(state.req_states.all_token_ids._staged_write_indices) == 2
    _apply_admissions(state)
    assert state.req_states.all_token_ids.gpu[first_idx, 3:5].tolist() == [11, 22]
    assert state.req_states.all_token_ids.gpu[second_idx, 3:5].tolist() == [11, 22]


def test_partial_prefill_admission_preserves_trace_start():
    state = _trace_state(1)
    prompt = [1, 2, 3, 4]
    req_idx = _admit(
        state,
        "partial-prefill",
        prompt,
        SamplingParams(trace_decode_token_ids=[11, 22]),
        num_computed_tokens=2,
    )
    _apply_admissions(state)

    assert state.req_states.num_computed_tokens.gpu[req_idx].item() == 2
    assert state.req_states.all_token_ids.gpu[req_idx, :6].tolist() == [
        *prompt,
        11,
        22,
    ]


def test_exhausted_trace_adds_no_future_suffix():
    state = _trace_state(1)
    prompt = [1, 2, 3]
    trace = [11, 22]
    req_idx = _admit(
        state,
        "exhausted",
        prompt,
        SamplingParams(trace_decode_token_ids=trace),
        prefill=prompt + trace,
    )

    assert len(state.req_states.all_token_ids._staged_write_indices) == 1
    assert state.req_states.all_token_ids._staged_write_contents == prompt + trace
    _apply_admissions(state)

    sampled = _i64([999])
    state.apply_trace(sampled, _i32([req_idx]))
    assert sampled.tolist() == [999]


def test_empty_future_suffix_preserves_empty_write_behavior():
    req_states = _trace_state(1).req_states
    req_states.add_request(
        req_id="empty",
        prompt_len=0,
        all_token_ids=[],
        num_computed_tokens=0,
        max_tokens=1,
        future_token_ids=[],
    )

    assert req_states.all_token_ids._staged_write_indices == []
    assert req_states.all_token_ids._staged_write_contents == []
