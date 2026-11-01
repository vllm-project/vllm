# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Window settlement must follow delivered tokens, including speculative replay."""

from dataclasses import asdict

import numpy as np
import pytest
import torch

from vllm.distributed.aux_output_connector.connector import AuxRequestOutput
from vllm.distributed.aux_output_connector.hidden_states import (
    CaptureSegment,
    HiddenCaptureCarry,
    HiddenStateCaptureBuffer,
    HiddenStateCaptureChunk,
    HiddenStateCapturePlan,
    HiddenStateScheduler,
    capture_plan,
    capture_scheduled_hidden_states,
)
from vllm.distributed.aux_output_connector.outputs import AuxOutputs
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test


def _chunk(positions, values) -> HiddenStateCaptureChunk:
    return HiddenStateCaptureChunk(
        np.array(positions), {"hidden_states": torch.tensor(values)}
    )


def _request(plan_spec, prompt=(1, 2, 3), max_tokens=8) -> Request:
    return Request(
        "r",
        list(prompt),
        SamplingParams(
            max_tokens=max_tokens, extra_args={"hidden_state_capture": plan_spec}
        ),
        None,
    )


def test_capture_window_binds_to_actual_prompt_length():
    legacy = HiddenStateCapturePlan.from_window("client", 7, 2, 6, "test")
    for spec in (asdict(legacy), {"start": 2, "end": 6, "collection_id": "test"}):
        plan = capture_plan(
            SamplingParams(extra_args={"hidden_state_capture": spec}), "engine", 12
        )
        assert (plan.window_start_abs, plan.window_end_abs) == (13, 17)
        assert plan.request_id == "engine"


@pytest.mark.parametrize(
    "spec", [[], {"unknown": 1}, {"start": 0.5, "end": 2}, {"start": 4, "end": 2}]
)
def test_invalid_capture_spec_is_a_request_validation_error(spec):
    with pytest.raises(VLLMValidationError):
        capture_plan(SamplingParams(extra_args={"hidden_state_capture": spec}), "r", 3)


def test_absolute_ranges_split_into_prompt_and_output_segments():
    plan = HiddenStateCapturePlan.from_window(
        "r", 8, 2, 10, "test", coordinate="absolute"
    )
    assert plan.segments == (
        CaptureSegment("prompt", 2, 7),
        CaptureSegment("output", 7, 10),
    )
    response = HiddenStateCapturePlan.from_window("r", 8, 0, 3, "test")
    assert response.segments == (CaptureSegment("output", 7, 10),)


def test_prompt_rows_cannot_request_capture_logprobs():
    """Prefill computes logits only for the last row; use prompt logprobs."""
    with pytest.raises(VLLMValidationError, match="prompt_logprob_token_ids"):
        capture_plan(
            SamplingParams(
                extra_args={
                    "hidden_state_capture": {
                        "start": 0,
                        "end": 2,
                        "coordinate": "absolute",
                        "collection_id": "test",
                        "logprob_topk": 1,
                    }
                }
            ),
            "r",
            4,
        )


def test_prompt_capture_bounds_prefix_cache_hits():
    """Cached prompt rows are not recomputed, so they cannot be captured."""
    spec = {"start": 2, "end": 7, "coordinate": "absolute", "collection_id": "t"}
    assert _request(spec, prompt=range(1, 9)).prefix_cache_hit_limit == 2
    output_only = {"start": 0, "end": 2, "collection_id": "t"}
    assert _request(output_only, prompt=range(1, 9)).prefix_cache_hit_limit is None


def test_missing_logits_rows_skip_only_affected_capture():
    plans = {
        name: HiddenStateCapturePlan.from_window(
            name, 1, 0, 2, "test", logprob_topk=int(name == "a")
        )
        for name in ("a", "b")
    }
    errors: dict[str, str] = {}
    chunks = capture_scheduled_hidden_states(
        plans,
        ["a", "b"],
        {"a": 2, "b": 2},
        {"a": 0, "b": 0},
        torch.zeros(4, 2),
        logits=torch.ones(3, 4),
        cu_num_logits=np.array([0, 1, 3]),
        errors=errors,
    )
    assert errors == {"a": "logits_rows_unavailable"}
    assert list(chunks) == ["b"]


def test_missing_logprobs_settle_as_skip_instead_of_engine_error():
    buffer = HiddenStateCaptureBuffer(
        HiddenStateCapturePlan.from_window("r", 1, 0, 1, "test", logprob_topk=1)
    )
    result, reason = buffer.update(_chunk([0], [[0.0, 0.0]]), 1, stopped=True)
    assert result is None and reason == "logprob_rows_unavailable"


def test_recomputed_history_needs_no_logits_after_delivery():
    plan = HiddenStateCapturePlan.from_window("r", 1, 0, 16, "test", logprob_topk=2)
    assert (
        capture_scheduled_hidden_states(
            {"r": plan},
            ["r"],
            {"r": 8},
            {"r": 0},
            torch.zeros(8, 2),
            settled_ends={"r": 8},
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
        settled_ends={"r": 8},
    )["r"]
    np.testing.assert_array_equal(chunk.positions, [8, 9])
    torch.testing.assert_close(chunk.hidden_states, hidden[8:10])


def test_rejected_window_suffix_is_collected_on_replay():
    plan = HiddenStateCapturePlan.from_window("r", 3, 1, 4, "test", min_rows=3)
    request = _request(asdict(plan))
    scheduler = HiddenStateScheduler()
    scheduler.build_metadata({"r": 3}, {"r": request})
    request.append_output_token_ids([4, 5])
    first = _chunk([3, 4, 5], [[3], [40], [50]])
    result, reason = scheduler.take_output(
        request, AuxRequestOutput(hidden_chunk=first), False
    )
    assert result is None and reason is None
    _, settled_ends, _ = scheduler.build_metadata({"r": 2}, {"r": request})
    assert settled_ends == {"r": 4}
    request.append_output_token_ids([6, 7])
    replay = _chunk([4, 5, 6], [[4], [5], [6]])
    result, reason = scheduler.take_output(
        request, AuxRequestOutput(hidden_chunk=replay), False
    )
    assert reason is None
    np.testing.assert_array_equal(result.hidden_positions, [3, 4, 5])
    torch.testing.assert_close(result.hidden_states, torch.tensor([[3], [4], [5]]))
    plans, _, finished = scheduler.build_metadata({"r": 1}, {"r": request})
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


def test_chunked_prefill_prompt_rows_settle_with_the_sampling_chunk():
    """Teacher scoring: rows from a non-sampling prefill chunk are not lost."""
    prompt = list(range(1, 9))
    spec = {"start": 2, "end": 7, "coordinate": "absolute", "collection_id": "t"}
    request = _request(spec, prompt=prompt, max_tokens=1)
    scheduler = HiddenStateScheduler()
    plans, settled_ends, _ = scheduler.build_metadata({"r": 4}, {"r": request})
    assert settled_ends == {"r": 2}
    hidden = torch.arange(8).reshape(8, 1)
    carry = HiddenCaptureCarry()
    released = {}
    for computed, emitted in ((0, []), (4, ["r"])):
        chunks = capture_scheduled_hidden_states(
            plans,
            ["r"],
            {"r": 4},
            {"r": computed},
            hidden[computed : computed + 4],
            settled_ends=settled_ends,
        )
        released = carry.release(chunks, {}, emitted, {"vocab_size": 8})
        if not emitted:
            assert released == {}
    chunk, error = released["r"]
    assert error is None and carry.empty()
    request.append_output_token_ids([9])
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    result, reason = scheduler.take_output(
        request, AuxRequestOutput(hidden_chunk=chunk), True
    )
    assert reason is None
    np.testing.assert_array_equal(result.hidden_positions, [2, 3, 4, 5, 6])
    torch.testing.assert_close(result.hidden_states, hidden[2:7])
    assert result.segments == (CaptureSegment("prompt", 2, 7),)
    assert result.logits_contract == {"vocab_size": 8}


@pytest.mark.parametrize(
    ("count", "reason"), [(2, "ended_before_window"), (4, "insufficient_rows")]
)
def test_early_stop_reports_missing_window_and_releases_buffer(count, reason):
    plan = HiddenStateCapturePlan.from_window("r", 3, 3, 8, "test", min_rows=5)
    request = _request(asdict(plan))
    scheduler = HiddenStateScheduler()
    scheduler.build_metadata({"r": 3}, {"r": request})
    request.append_output_token_ids([4] * count)
    result, skipped = scheduler.take_output(request, None, True)
    assert result is None and skipped == reason
    assert not scheduler.buffers


def test_prefill_capture_error_is_delivered_with_the_first_output_token():
    carry = HiddenCaptureCarry()
    assert carry.release({}, {"r": "aux_layers_unavailable"}, []) == {}
    released = carry.release({}, {}, ["r"])
    assert released == {"r": (None, "aux_layers_unavailable")}
    plan = HiddenStateCapturePlan.from_window("r", 3, 1, 4, "test")
    request = _request(asdict(plan))
    scheduler = HiddenStateScheduler()
    scheduler.build_metadata({"r": 1}, {"r": request})
    request.append_output_token_ids([4])
    output = AuxRequestOutput(hidden_error=released["r"][1])
    assert scheduler.take_output(request, output, False) == (
        None,
        "aux_layers_unavailable",
    )
    assert not scheduler.buffers


def test_aux_outputs_keep_settled_fields_across_merges():
    first = AuxOutputs(hidden_capture_skip_reason="ended_before_window")
    assert AuxOutputs().combine(first) == first
    assert AuxOutputs(hidden_capture_skip_reason="x").combine(first) == AuxOutputs(
        hidden_capture_skip_reason="x"
    )
