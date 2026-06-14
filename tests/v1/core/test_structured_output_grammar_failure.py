# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future

from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.engine import FinishReason
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output.request import StructuredOutputRequest

from .utils import create_requests, create_scheduler


def test_structured_output_request_records_grammar_future_error():
    structured_output_req = StructuredOutputRequest(
        params=StructuredOutputsParams(regex="[a-z]+")
    )
    future: Future = Future()
    exc = ValueError("bad grammar")
    future.set_exception(exc)
    structured_output_req.grammar = future

    assert structured_output_req.grammar is None
    assert structured_output_req.grammar_error is exc
    assert structured_output_req.is_grammar_ready is True

    structured_output_req.grammar = object()
    assert structured_output_req.grammar_error is None


def test_scheduler_skips_failed_structured_output_grammar_during_schedule():
    scheduler = create_scheduler()
    request = create_requests(num_requests=1)[0]
    request.sampling_params = SamplingParams(
        max_tokens=1,
        structured_outputs=StructuredOutputsParams(regex="[a-z]+"),
    )
    request.structured_output_request = StructuredOutputRequest.from_sampling_params(
        request.sampling_params
    )
    assert request.structured_output_request is not None
    request.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    future: Future = Future()
    future.set_exception(RuntimeError("xgrammar compile failed"))
    request.structured_output_request.grammar = future
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.scheduled_new_reqs == []
    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.request_id not in scheduler.requests
    assert output.finished_req_ids == {request.request_id}
    assert list(scheduler.waiting) == []
    assert list(scheduler.skipped_waiting) == []

    engine_core_outputs = scheduler.update_from_output(
        output, EMPTY_MODEL_RUNNER_OUTPUT
    )
    client_outputs = engine_core_outputs[0].outputs
    assert len(client_outputs) == 1
    assert client_outputs[0].request_id == request.request_id
    assert client_outputs[0].finish_reason == FinishReason.ERROR
