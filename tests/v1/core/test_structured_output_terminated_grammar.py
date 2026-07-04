# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for issue #42619: scheduler must not call accept_tokens() or
validate_tokens() on a grammar that has already reached a terminal state.

Once the FSM accepts its final token (e.g. EOS) it returns
is_terminated() == True.  On the *next* scheduler step the grammar is
still attached to the request; calling accept_tokens() again returns
False, which was misinterpreted as a decode error and caused spurious
FINISHED_ERROR outcomes.
"""

from unittest.mock import Mock

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .utils import EOS_TOKEN_ID

pytestmark = pytest.mark.cpu_test


def _make_scheduler_stub(request: Request) -> Scheduler:
    """Return a minimally-wired Scheduler stub for update_from_output tests."""
    scheduler = object.__new__(Scheduler)
    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.waiting = Mock()
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.kv_event_publisher = Mock()
    scheduler.vllm_config = Mock()
    scheduler.vllm_config.model_config.enable_return_routed_experts = False
    scheduler.enable_return_routed_experts = False
    scheduler.recompute_kv_load_failures = False
    scheduler.make_stats = Mock(return_value=None)
    scheduler.max_model_len = 128

    def free_request(req: Request, delay_free_blocks: bool = False):
        scheduler.finished_req_ids.add(req.request_id)
        scheduler.requests.pop(req.request_id, None)
        return None

    scheduler._free_request = Mock(side_effect=free_request)
    return scheduler


def _make_running_request() -> Request:
    sampling_params = SamplingParams(ignore_eos=True, max_tokens=4)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    request = Request(
        request_id="0",
        prompt_token_ids=[0, 1],
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
    )
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = request.num_tokens
    return request


def _make_scheduler_output(req_id: str, num_tokens: int = 1) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_id: num_tokens},
        total_num_scheduled_tokens=num_tokens,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _make_model_output(req_id: str, tokens: list[int]) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[tokens],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_accept_tokens_not_called_when_grammar_terminated():
    """
    When is_terminated() returns True, accept_tokens() must be skipped.

    Before the fix, accept_tokens() returned False on a terminated grammar and
    the scheduler set FINISHED_ERROR.  After the fix, the terminated check short-
    circuits and accept_tokens() is never reached.
    """
    request = _make_running_request()

    grammar = Mock()
    grammar.is_terminated.return_value = True
    # Simulate what the backend returns when called on a terminated matcher.
    grammar.accept_tokens.return_value = False

    request.structured_output_request = Mock()
    request.structured_output_request.grammar = grammar

    scheduler = _make_scheduler_stub(request)
    scheduler.structured_output_manager = Mock()
    scheduler.structured_output_manager.should_advance.return_value = True

    sched_output = _make_scheduler_output(request.request_id)
    model_output = _make_model_output(request.request_id, [42])

    scheduler.update_from_output(sched_output, model_output)

    # The termination guard must have fired; accept_tokens() must not run.
    grammar.accept_tokens.assert_not_called()
    # Request must NOT be marked as an error.
    assert request.status != RequestStatus.FINISHED_ERROR


def test_accept_tokens_called_when_grammar_not_terminated():
    """
    When is_terminated() returns False and accept_tokens() succeeds (True),
    the request proceeds normally without error — regression guard.
    """
    request = _make_running_request()

    grammar = Mock()
    grammar.is_terminated.return_value = False
    grammar.accept_tokens.return_value = True

    request.structured_output_request = Mock()
    request.structured_output_request.grammar = grammar

    scheduler = _make_scheduler_stub(request)
    scheduler.structured_output_manager = Mock()
    scheduler.structured_output_manager.should_advance.return_value = True

    sched_output = _make_scheduler_output(request.request_id)
    model_output = _make_model_output(request.request_id, [99])

    scheduler.update_from_output(sched_output, model_output)

    grammar.accept_tokens.assert_called_once_with(request.request_id, [99])
    assert request.status != RequestStatus.FINISHED_ERROR


def test_accept_tokens_failure_still_errors_when_not_terminated():
    """
    When is_terminated() is False and accept_tokens() returns False (genuine
    grammar violation), the scheduler must still mark the request FINISHED_ERROR.
    This ensures the guard does not suppress real errors.
    """
    request = _make_running_request()

    grammar = Mock()
    grammar.is_terminated.return_value = False
    grammar.accept_tokens.return_value = False

    request.structured_output_request = Mock()
    request.structured_output_request.grammar = grammar

    scheduler = _make_scheduler_stub(request)
    scheduler.structured_output_manager = Mock()
    scheduler.structured_output_manager.should_advance.return_value = True

    sched_output = _make_scheduler_output(request.request_id)
    model_output = _make_model_output(request.request_id, [123])

    scheduler.update_from_output(sched_output, model_output)

    grammar.accept_tokens.assert_called_once_with(request.request_id, [123])
    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.resumable is False


def test_validate_tokens_not_called_when_grammar_terminated_update_draft():
    """
    update_draft_token_ids: validate_tokens() must be skipped when the grammar
    is terminated (issue #42619). All spec tokens should be kept as-is since
    the structured-output constraint is already satisfied.
    """
    sampling_params = SamplingParams(ignore_eos=True, max_tokens=16)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    request = Request(
        request_id="req0",
        prompt_token_ids=[0, 1, 2],
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
    )
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = request.num_tokens
    request.is_prefill_chunk = False

    grammar = Mock()
    grammar.is_terminated.return_value = True
    grammar.validate_tokens.return_value = []  # would reject everything if called

    request.structured_output_request = Mock()
    request.structured_output_request.grammar = grammar

    scheduler = object.__new__(Scheduler)
    scheduler.requests = {request.request_id: request}
    scheduler.structured_output_manager = Mock()
    scheduler.structured_output_manager.should_advance.return_value = True

    spec_tokens = [10, 20, 30]
    draft = DraftTokenIds(
        req_ids=[request.request_id],
        draft_token_ids=[list(spec_tokens)],
    )
    scheduler.update_draft_token_ids(draft)

    grammar.validate_tokens.assert_not_called()
    # All spec tokens should be preserved unchanged.
    assert request.spec_token_ids == spec_tokens


def test_validate_tokens_called_when_grammar_not_terminated_update_draft():
    """
    update_draft_token_ids: validate_tokens() should be called when grammar is
    active (not terminated) — regression guard.
    """
    sampling_params = SamplingParams(ignore_eos=True, max_tokens=16)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    request = Request(
        request_id="req1",
        prompt_token_ids=[0, 1, 2],
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
    )
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = request.num_tokens
    request.is_prefill_chunk = False

    grammar = Mock()
    grammar.is_terminated.return_value = False
    grammar.validate_tokens.return_value = [10, 20]  # first two accepted

    request.structured_output_request = Mock()
    request.structured_output_request.grammar = grammar

    scheduler = object.__new__(Scheduler)
    scheduler.requests = {request.request_id: request}
    scheduler.structured_output_manager = Mock()
    scheduler.structured_output_manager.should_advance.return_value = True

    spec_tokens = [10, 20, 30]
    draft = DraftTokenIds(
        req_ids=[request.request_id],
        draft_token_ids=[list(spec_tokens)],
    )
    scheduler.update_draft_token_ids(draft)

    grammar.validate_tokens.assert_called_once_with(spec_tokens)
    assert request.spec_token_ids == [10, 20]
