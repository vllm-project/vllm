# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import copy

from vllm import SamplingParams
from vllm.outputs import CompletionOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.metrics.stats import RequestStateStats


def test_parent_request_to_output_stream() -> None:
    parent_request = ParentRequest(make_request(SamplingParams(n=2)))
    parent_request.child_requests = {"child_id_0", "child_id_1"}
    output_0 = CompletionOutput(
        index=0, text="child 0", token_ids=[], cumulative_logprob=None, logprobs=None
    )
    output_1 = CompletionOutput(
        index=1, text="child 1", token_ids=[], cumulative_logprob=None, logprobs=None
    )
    # Request not finished
    assert ([output_0], False) == parent_request.get_outputs("child_id_0", output_0)
    assert ([output_1], False) == parent_request.get_outputs("child_id_1", output_1)
    assert ([output_0], False) == parent_request.get_outputs("child_id_0", output_0)
    assert ([output_1], False) == parent_request.get_outputs("child_id_1", output_1)

    # output_1 finished
    output_1.finish_reason = "ended"
    assert ([output_0], False) == parent_request.get_outputs("child_id_0", output_0)
    assert ([output_1], False) == parent_request.get_outputs("child_id_1", output_1)
    # Finished output_1 had already returned, DO NOT returned again
    assert ([output_0], False) == parent_request.get_outputs("child_id_0", output_0)
    assert parent_request.get_outputs("child_id_1", output_1) == ([], False)

    # output_0 finished
    output_0.finish_reason = "ended"
    assert ([output_0], True) == parent_request.get_outputs("child_id_0", output_0)
    assert parent_request.get_outputs("child_id_1", output_1) == ([], True)
    # Finished output_0 had already returned, DO NOT returned again
    assert parent_request.get_outputs("child_id_0", output_0) == ([], True)
    assert parent_request.get_outputs("child_id_1", output_1) == ([], True)


def test_parent_request_to_output_final_only() -> None:
    parent_request = ParentRequest(
        make_request(SamplingParams(n=2, output_kind=RequestOutputKind.FINAL_ONLY))
    )
    parent_request.child_requests = {"child_id_0", "child_id_1"}
    output_0 = CompletionOutput(
        index=0, text="child 0", token_ids=[], cumulative_logprob=None, logprobs=None
    )
    output_1 = CompletionOutput(
        index=1, text="child 1", token_ids=[], cumulative_logprob=None, logprobs=None
    )
    # Request not finished, return nothing
    assert parent_request.get_outputs("child_id_0", output_0) == ([], False)
    assert parent_request.get_outputs("child_id_1", output_1) == ([], False)
    # output_1 finished, but outputs won't be returned until all child requests finished
    output_1.finish_reason = "ended"
    assert parent_request.get_outputs("child_id_0", output_0) == ([], False)
    assert parent_request.get_outputs("child_id_1", output_1) == ([], False)
    # output_0 finished, as all child requests finished, the output would be returned
    output_0.finish_reason = "ended"
    assert ([output_0, output_1], True) == parent_request.get_outputs(
        "child_id_0", output_0
    )
    assert ([output_0, output_1], True) == parent_request.get_outputs(
        "child_id_1", output_1
    )


def test_parallel_sampling_child_requests_preserve_session_id() -> None:
    request = make_request(SamplingParams(n=2))
    request.session_id = "session-1"
    parent_request = ParentRequest(request)

    for idx in range(parent_request.n):
        request_id, child_params = parent_request.get_child_info(idx)
        child_request = request if idx == parent_request.n - 1 else copy(request)
        child_request.request_id = request_id
        child_request.sampling_params = child_params

        assert child_request.session_id == "session-1"


def test_parent_request_aggregates_child_stats() -> None:
    parent_request = ParentRequest(make_request(SamplingParams(n=2)))
    parent_request.register_child_stats(
        "child_id_0",
        RequestStateStats(
            num_generation_tokens=2,
            num_preemptions=1,
            arrival_time=1.0,
            queued_ts=2.0,
            scheduled_ts=4.0,
            first_token_ts=6.0,
            last_token_ts=8.0,
            first_token_latency=5.0,
            is_corrupted=False,
        ),
    )
    parent_request.register_child_stats(
        "child_id_1",
        RequestStateStats(
            num_generation_tokens=3,
            num_preemptions=2,
            arrival_time=0.5,
            queued_ts=1.5,
            scheduled_ts=3.0,
            first_token_ts=5.0,
            last_token_ts=10.0,
            first_token_latency=4.5,
            is_corrupted=True,
        ),
    )

    stats = parent_request.aggregate_stats()

    assert stats is not None
    assert stats.num_generation_tokens == 5
    assert stats.num_preemptions == 3
    assert stats.arrival_time == 0.5
    assert stats.queued_ts == 1.5
    assert stats.scheduled_ts == 3.0
    assert stats.first_token_ts == 5.0
    assert stats.last_token_ts == 10.0
    assert stats.first_token_latency == 4.5
    assert stats.is_corrupted


def make_request(sampling_params: SamplingParams) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="parent_id",
        external_req_id="ext_parent_id",
        prompt_token_ids=None,
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
