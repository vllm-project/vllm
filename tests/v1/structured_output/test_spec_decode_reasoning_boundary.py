# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.reasoning import ReasoningParser
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import validate_spec_tokens_with_reasoning_boundary


def make_structured_request(
    *,
    reasoning_ended: bool | None = False,
    valid_tokens: list[int] | None = None,
) -> SimpleNamespace:
    grammar = Mock()
    grammar.validate_tokens.return_value = valid_tokens or []
    grammar.accept_tokens.return_value = True
    return SimpleNamespace(reasoning_ended=reasoning_ended, grammar=grammar)


def make_request(structured_req: SimpleNamespace) -> Mock:
    request = Mock(spec=Request)
    request.request_id = "req-0"
    request.prompt_token_ids = [1, 2, 3]
    request.all_token_ids = [1, 2, 3, 4, 5]
    request.use_structured_output = True
    request.structured_output_request = structured_req
    return request


@pytest.mark.parametrize(
    (
        "boundary_end",
        "token_ids",
        "valid_suffix",
        "expected",
        "validated",
        "accepted",
    ),
    [
        (None, [9, 10], [], [9, 10], None, None),
        (1, [9, 99], [], [9, 99], None, None),
        (1, [9, 99, 11, 12], [11, 12], [9, 99, 11, 12], [11, 12], [11, 12]),
        (1, [9, 99, 11, 13], [11], [9, 99, 11], [11, 13], [11]),
        (1, [9, 99, 13], [], [9, 99], [13], None),
    ],
)
def test_validate_spec_tokens_splits_reasoning_boundary_suffix(
    boundary_end: int | None,
    token_ids: list[int],
    valid_suffix: list[int],
    expected: list[int],
    validated: list[int] | None,
    accepted: list[int] | None,
):
    structured_req = make_structured_request(valid_tokens=valid_suffix)
    request = make_request(structured_req)
    reasoner = Mock(spec=ReasoningParser)
    reasoner.find_reasoning_end_index.return_value = boundary_end

    result = validate_spec_tokens_with_reasoning_boundary(
        request,
        token_ids=token_ids,
        reasoner=reasoner,
    )

    assert result == expected
    assert structured_req.reasoning_ended is (boundary_end is not None)
    if validated is None:
        structured_req.grammar.validate_tokens.assert_not_called()
    else:
        structured_req.grammar.validate_tokens.assert_called_once_with(validated)
    if accepted is None:
        structured_req.grammar.accept_tokens.assert_not_called()
    else:
        structured_req.grammar.accept_tokens.assert_called_once_with("req-0", accepted)


def make_scheduler_output(request_id: str) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request_id: 4},
        total_num_scheduled_tokens=4,
        scheduled_spec_decode_tokens={request_id: [99, 11, 13]},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def make_model_runner_output(
    request_id: str, token_ids: list[int]
) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[request_id],
        req_id_to_index={request_id: 0},
        sampled_token_ids=[token_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def prepare_running_request(
    reasoning_ended: bool | None,
) -> tuple[Scheduler, Request]:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.enable_spec_reasoning_boundary_validation = True
    scheduler.log_stats = False
    scheduler.perf_metrics = None
    scheduler.max_model_len = 128
    scheduler.requests = {}
    scheduler.running = []
    scheduler.finished_req_ids_dict = {}
    scheduler.connector = None
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.make_stats = Mock(return_value=None)
    scheduler.structured_output_manager = SimpleNamespace(
        reasoner=None,
        enable_in_reasoning=False,
        should_advance=Mock(
            side_effect=lambda req: (
                req.structured_output_request.reasoning_ended is True
            )
        ),
    )

    request = Request(
        request_id="req-0",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10, ignore_eos=True),
        pooling_params=None,
    )
    request.num_computed_tokens = request.num_tokens + 4
    request.num_output_placeholders = 4
    request.status = RequestStatus.RUNNING
    request.structured_output_request = make_structured_request(
        reasoning_ended=reasoning_ended,
        valid_tokens=[11],
    )
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)
    return scheduler, request


def test_scheduler_validates_and_truncates_post_boundary_spec_tokens():
    scheduler, request = prepare_running_request(reasoning_ended=False)
    reasoner = Mock(spec=ReasoningParser)
    reasoner.is_reasoning_end.return_value = False
    reasoner.may_have_reasoning_end_in_delta.return_value = True
    reasoner.find_reasoning_end_index.return_value = 1
    scheduler.structured_output_manager.reasoner = reasoner

    scheduler.update_from_output(
        make_scheduler_output(request.request_id),
        make_model_runner_output(request.request_id, [90, 99, 11, 13]),
    )

    grammar = request.structured_output_request.grammar
    assert list(request.output_token_ids) == [90, 99, 11]
    assert request.num_computed_tokens == len(request.all_token_ids)
    assert request.num_output_placeholders == 3
    grammar.validate_tokens.assert_called_once_with([11, 13])
    grammar.accept_tokens.assert_called_once_with(request.request_id, [11])


def test_scheduler_initializes_prompt_reasoning_state_before_boundary_path():
    scheduler, request = prepare_running_request(reasoning_ended=None)
    reasoner = Mock(spec=ReasoningParser)
    reasoner.is_reasoning_end.return_value = True
    scheduler.structured_output_manager.reasoner = reasoner

    scheduler.update_from_output(
        make_scheduler_output(request.request_id),
        make_model_runner_output(request.request_id, [10, 11, 12]),
    )

    grammar = request.structured_output_request.grammar
    assert request.structured_output_request.reasoning_ended is True
    assert list(request.output_token_ids) == [10, 11, 12]
    reasoner.is_reasoning_end.assert_called_once_with(request.prompt_token_ids)
    reasoner.may_have_reasoning_end_in_delta.assert_not_called()
    grammar.validate_tokens.assert_not_called()
    grammar.accept_tokens.assert_called_once_with(request.request_id, [10, 11, 12])
