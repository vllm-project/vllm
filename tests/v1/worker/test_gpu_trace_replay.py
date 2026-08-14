# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Model Runner V2 inference trace-replay.

Trace-replay forces the sampler to emit a predetermined sequence of decode
token IDs. The replay step for a request is derived from GPU state as
``total_len - prompt_len`` (output tokens produced so far), and the trace token
overwrites the sampled token in place before logprobs are computed.
"""

from types import SimpleNamespace
from typing import cast

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
    req_states = cast(
        RequestState,
        SimpleNamespace(
            max_num_reqs=max_num_reqs,
            max_model_len=TEST_MAX_MODEL_LEN,
            device=torch.device(DEVICE),
        ),
    )
    return TraceReplayState(req_states)


# ------------------------------ Kernel ------------------------------------


def test_replay_overwrites_sampled_at_each_step():
    """The trace token for the current step replaces the sampled token."""
    trace = [[100, 101, 102], [200, 201, 202]]
    trace_len = _i32([3, 3])
    trace_token_ids = torch.zeros(
        2, TEST_MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE
    )
    for i, t in enumerate(trace):
        trace_token_ids[i, : len(t)] = _i32(t)
    prompt_len = _i32([5, 8])
    idx_mapping = _i32([0, 1])

    for step in range(3):
        sampled = _i64([-7, -7])  # sentinel that must be overwritten
        total_len = _i32([5 + step, 8 + step])
        apply_trace_tokens(
            sampled, idx_mapping, trace_token_ids, trace_len, total_len, prompt_len
        )
        assert sampled.tolist() == [trace[0][step], trace[1][step]]


def test_past_end_of_trace_leaves_sampled_untouched():
    """Once step >= trace_len, the sampler's own token is kept."""
    trace_token_ids = torch.zeros(
        1, TEST_MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE
    )
    trace_token_ids[0, :2] = _i32([100, 101])
    trace_len = _i32([2])
    prompt_len = _i32([4])
    idx_mapping = _i32([0])

    sampled = _i64([999])
    total_len = _i32([4 + 2])  # step == 2 == trace_len -> out of range
    apply_trace_tokens(
        sampled, idx_mapping, trace_token_ids, trace_len, total_len, prompt_len
    )
    assert sampled.tolist() == [999]


def test_non_trace_request_untouched():
    """trace_len == 0 means the request never uses replay."""
    trace_token_ids = torch.zeros(
        1, TEST_MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE
    )
    trace_len = _i32([0])
    prompt_len = _i32([3])
    idx_mapping = _i32([0])

    sampled = _i64([42])
    total_len = _i32([3])
    apply_trace_tokens(
        sampled, idx_mapping, trace_token_ids, trace_len, total_len, prompt_len
    )
    assert sampled.tolist() == [42]


def test_idx_mapping_indirection_and_negative_skip():
    """batch_idx -> req_state_idx indirection, and negative entries are skipped."""
    # req_state 0: no trace; req_state 1: trace [500, 501].
    trace_token_ids = torch.zeros(
        2, TEST_MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE
    )
    trace_token_ids[1, :2] = _i32([500, 501])
    trace_len = _i32([0, 2])
    prompt_len = _i32([10, 6])

    # batch: [maps to state 1, masked (-1), maps to state 0].
    idx_mapping = _i32([1, -1, 0])
    sampled = _i64([1, 2, 3])
    # Indexed by req_state_idx: state 0 at total_len=10, state 1 at step 1.
    total_len = _i32([10, 6 + 1])
    apply_trace_tokens(
        sampled, idx_mapping, trace_token_ids, trace_len, total_len, prompt_len
    )
    # batch 0 -> state 1 step 1 -> 501; batch 1 masked; batch 2 -> state 0 no trace.
    assert sampled.tolist() == [501, 2, 3]


# --------------------------- TraceReplayState -----------------------------


def test_state_skips_work_before_any_trace(monkeypatch: pytest.MonkeyPatch):
    """State operations return before touching buffers without a trace."""
    state = _trace_state(1)

    def fail_if_called(*_args, **_kwargs):
        pytest.fail("trace replay work must be skipped before any trace request")

    monkeypatch.setattr(type(state.trace_len), "copy_to_uva", fail_if_called)
    monkeypatch.setattr(type(state.trace_token_ids), "apply_write", fail_if_called)
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.sample.trace_replay.apply_trace_tokens",
        fail_if_called,
    )

    state.apply_staged_writes()
    state.apply_trace(_i64([7]), _i32([0]), _i32([0]), _i32([0]))


def test_state_end_to_end():
    """add_request -> apply_staged_writes -> apply_trace overwrites correctly."""
    state = _trace_state(4)
    state.add_request(0, SamplingParams(trace_decode_token_ids=[11, 22, 33]))
    state.add_request(1, SamplingParams())  # no trace
    state.apply_staged_writes()

    idx_mapping = _i32([0, 1])
    prompt_len = _i32([7, 7, 0, 0])
    sampled = _i64([-1, -1])
    total_len = _i32([7 + 1, 7, 0, 0])  # req 0 at step 1
    state.apply_trace(sampled, idx_mapping, total_len, prompt_len)
    assert sampled.tolist() == [22, -1]


def test_state_leaves_non_trace_batch_unchanged():
    """Requests with trace_len == 0 remain unchanged after a trace was seen."""
    state = _trace_state(4)
    state.add_request(0, SamplingParams(trace_decode_token_ids=[11, 22]))
    state.add_request(1, SamplingParams())
    state.apply_staged_writes()

    # Only the non-trace request (state 1) is in this batch.
    idx_mapping = _i32([1])
    prompt_len = _i32([7, 7, 0, 0])
    sampled = _i64([555])
    total_len = _i32([0, 7, 0, 0])
    state.apply_trace(sampled, idx_mapping, total_len, prompt_len)
    assert sampled.tolist() == [555]


def test_any_trace_is_sticky_after_slot_reuse():
    """The O(1) guard remains enabled after a trace request leaves its slot."""
    state = _trace_state(1)
    assert not state.any_trace

    state.add_request(0, SamplingParams(trace_decode_token_ids=[11, 22]))
    assert state.any_trace

    state.add_request(0, SamplingParams())
    assert state.any_trace


def test_slot_reuse_clears_trace():
    """Reusing a slot for a non-trace request must not replay stale tokens."""
    state = _trace_state(2)
    state.add_request(0, SamplingParams(trace_decode_token_ids=[11, 22]))
    state.apply_staged_writes()
    # Slot 0 reused by a request without a trace.
    state.add_request(0, SamplingParams())
    state.apply_staged_writes()

    idx_mapping = _i32([0])
    prompt_len = _i32([3, 0])
    sampled = _i64([888])
    total_len = _i32([3, 0])
    state.apply_trace(sampled, idx_mapping, total_len, prompt_len)
    assert sampled.tolist() == [888]
