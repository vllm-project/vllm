# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Window settlement must follow delivered tokens, including speculative replay."""

from dataclasses import asdict

import numpy as np
import pytest
import torch

from vllm.distributed.aux_output_connector.connector import AuxRequestOutput
from vllm.distributed.aux_output_connector.hidden_states import (
    HiddenStateCaptureChunk,
    HiddenStateCapturePlan,
    HiddenStateScheduler,
    capture_scheduled_hidden_states,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus


def test_recomputed_history_needs_no_logits_after_delivery():
    plan = HiddenStateCapturePlan.from_window("r", 1, 0, 16, "test", logprob_topk=2)
    assert (
        capture_scheduled_hidden_states(
            {"r": plan},
            ["r"],
            {"r": 8},
            {"r": 0},
            torch.zeros(8, 2),
            accepted_ends={"r": 8},
        )
        == {}
    )


def test_rejected_rows_remain_eligible_after_recomputation():
    plan = HiddenStateCapturePlan.from_window("r", 1, 0, 16, "test")
    hidden = torch.arange(20).reshape(10, 2)
    chunk = capture_scheduled_hidden_states(
        {"r": plan},
        ["r"],
        {"r": 10},
        {"r": 0},
        hidden,
        accepted_ends={"r": 8},
    )["r"]
    np.testing.assert_array_equal(chunk.positions, [8, 9])
    torch.testing.assert_close(chunk.hidden_states, hidden[8:10])


def test_rejected_window_suffix_is_collected_on_replay():
    plan = HiddenStateCapturePlan.from_window("r", 3, 1, 4, "test", min_rows=3)
    request = Request(
        "r",
        [1, 2, 3],
        SamplingParams(max_tokens=8, extra_args={"hidden_state_capture": asdict(plan)}),
        None,
    )
    scheduler = HiddenStateScheduler()
    scheduler.build_metadata({"r": 3}, {"r": request})
    request.append_output_token_ids([4, 5])
    first = HiddenStateCaptureChunk(
        np.array([3, 4, 5]), torch.tensor([[3], [40], [50]])
    )
    result, reason = scheduler.take_output(
        request, AuxRequestOutput(hidden_chunk=first), 2, False
    )
    assert result is None and reason is None
    request.append_output_token_ids([6, 7])
    replay = HiddenStateCaptureChunk(np.array([4, 5, 6]), torch.tensor([[4], [5], [6]]))
    result, reason = scheduler.take_output(
        request, AuxRequestOutput(hidden_chunk=replay), 2, False
    )
    assert reason is None
    np.testing.assert_array_equal(result.hidden_positions, [3, 4, 5])
    torch.testing.assert_close(result.hidden_states, torch.tensor([[3], [4], [5]]))
    plans, finished = scheduler.build_metadata({"r": 1}, {"r": request})
    assert plans == {} and finished == ("r",)
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    scheduler.request_finished(request)
    assert not scheduler.buffers and not scheduler.done


def test_mixed_batch_snapshots_only_intersecting_rows_and_owns_storage():
    plan = HiddenStateCapturePlan.from_window("a", 3, 1, 3, "test")
    hidden = torch.arange(20).reshape(10, 2)
    chunks = capture_scheduled_hidden_states(
        {"a": plan},
        ["b", "a"],
        {"a": 6, "b": 4},
        {"a": 0, "b": 0},
        hidden,
        query_start_loc=np.array([0, 4, 10]),
    )
    expected = hidden[7:9].clone()
    hidden.fill_(-1)
    assert list(chunks) == ["a"]
    np.testing.assert_array_equal(chunks["a"].positions, [3, 4])
    torch.testing.assert_close(chunks["a"].hidden_states, expected)
    assert (
        capture_scheduled_hidden_states({"a": plan}, ["a"], {"a": 1}, {"a": 9}, hidden)
        == {}
    )


@pytest.mark.parametrize(
    ("count", "reason"), [(2, "ended_before_window"), (4, "insufficient_rows")]
)
def test_early_stop_reports_missing_window_and_releases_buffer(count, reason):
    plan = HiddenStateCapturePlan.from_window("r", 3, 3, 8, "test", min_rows=5)
    request = Request(
        "r",
        [1, 2, 3],
        SamplingParams(max_tokens=8, extra_args={"hidden_state_capture": asdict(plan)}),
        None,
    )
    scheduler = HiddenStateScheduler()
    scheduler.build_metadata({"r": 3}, {"r": request})
    request.append_output_token_ids([4] * count)
    result, skipped = scheduler.take_output(request, None, count, True)
    assert result is None and skipped == reason
    assert not scheduler.buffers


def test_prefill_capture_error_is_delivered_with_the_first_output_token():
    plan = HiddenStateCapturePlan.from_window("r", 3, 1, 4, "test")
    request = Request(
        "r",
        [1, 2, 3],
        SamplingParams(max_tokens=8, extra_args={"hidden_state_capture": asdict(plan)}),
        None,
    )
    scheduler = HiddenStateScheduler()
    scheduler.build_metadata({"r": 1}, {"r": request})
    output = AuxRequestOutput(hidden_error="aux_layers_unavailable")
    assert scheduler.take_output(request, output, 0, False) == (None, None)
    plans, finished = scheduler.build_metadata({"r": 1}, {"r": request})
    assert not plans and finished == ("r",)
    request.append_output_token_ids([4])
    assert scheduler.take_output(request, None, 1, False) == (
        None,
        "aux_layers_unavailable",
    )
    assert not scheduler.buffers
