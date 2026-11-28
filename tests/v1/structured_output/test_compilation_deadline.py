# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from concurrent.futures import Future

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.outputs import RequestError
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.engine import FinishReason
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output.backend_types import StructuredOutputGrammar
from vllm.v1.structured_output.request import StructuredOutputRequest


def test_pending_compilation_is_cancelled_at_deadline():
    future: Future[StructuredOutputGrammar] = Future()
    request = StructuredOutputRequest(
        params=StructuredOutputsParams(regex="[a-z]+"),
        compilation_submit_time=10,
    )
    request.grammar = future

    assert request.grammar is None
    assert request.cancel_grammar_if_expired(1, now=12)
    assert future.cancelled()


def test_disabled_deadline_keeps_pending_compilation():
    future: Future[StructuredOutputGrammar] = Future()
    request = StructuredOutputRequest(
        params=StructuredOutputsParams(regex="[a-z]+"),
        compilation_submit_time=10,
    )
    request.grammar = future

    assert not request.cancel_grammar_if_expired(0, now=12)
    assert not future.done()


def test_completed_failure_is_preserved_after_deadline():
    failure = ValueError("compilation failed")
    future: Future[StructuredOutputGrammar] = Future()
    future.set_exception(failure)
    request = StructuredOutputRequest(
        params=StructuredOutputsParams(regex="[a-z]+"),
        compilation_submit_time=10,
    )
    request.grammar = future

    assert not request.cancel_grammar_if_expired(1, now=12)
    assert request.grammar is failure


def _expired_request(timeout: int) -> tuple[Request, Future[StructuredOutputGrammar]]:
    request = Request(
        request_id="grammar-timeout",
        prompt_token_ids=[0, 1],
        sampling_params=SamplingParams(
            structured_outputs=StructuredOutputsParams(regex="[a-z]+"),
        ),
        pooling_params=None,
    )
    structured_request = request.structured_output_request
    assert structured_request is not None
    structured_request.compilation_submit_time = time.monotonic() - timeout
    future: Future[StructuredOutputGrammar] = Future()
    structured_request.grammar = future
    return request, future


def test_scheduler_reports_timeout_even_if_it_schedules_again_before_output():
    scheduler = create_scheduler(skip_tokenizer_init=True)
    timeout = scheduler.vllm_config.structured_outputs_config.compilation_timeout
    request, future = _expired_request(timeout)
    scheduler.add_request(request)

    assert not scheduler.schedule().num_scheduled_tokens
    scheduler_output = scheduler.schedule()
    assert not scheduler_output.num_scheduled_tokens
    outputs = scheduler.update_from_output(
        scheduler_output, ModelRunnerOutput(req_ids=[])
    )

    assert future.cancelled()
    output = outputs[0].outputs[0]
    assert output.request_id == request.request_id
    assert output.finish_reason == FinishReason.ERROR
    assert output.error == RequestError(
        code="grammar_compilation_timeout",
        message=f"Grammar compilation timed out after {timeout}s.",
    )
    healthy_request = create_requests(num_requests=1)[0]
    scheduler.add_request(healthy_request)
    assert scheduler.schedule().num_scheduled_tokens.keys() == {
        healthy_request.request_id
    }


def test_scheduler_expires_grammar_while_all_slots_are_active():
    scheduler = create_scheduler(max_num_seqs=1, skip_tokenizer_init=True)
    healthy_request = create_requests(num_requests=1, ignore_eos=True)[0]
    scheduler.add_request(healthy_request)
    healthy_output = ModelRunnerOutput(
        req_ids=[healthy_request.request_id], sampled_token_ids=[[1]]
    )
    scheduler.update_from_output(scheduler.schedule(), healthy_output)

    timeout = scheduler.vllm_config.structured_outputs_config.compilation_timeout
    request, future = _expired_request(timeout)
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    assert scheduler_output.num_scheduled_tokens.keys() == {healthy_request.request_id}
    outputs = scheduler.update_from_output(scheduler_output, healthy_output)

    assert future.cancelled()
    output = next(o for o in outputs[0].outputs if o.request_id == request.request_id)
    assert output.finish_reason == FinishReason.ERROR
    assert output.error == RequestError(
        code="grammar_compilation_timeout",
        message=f"Grammar compilation timed out after {timeout}s.",
    )
