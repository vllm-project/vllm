# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for GHSA-55m4-88pw-2875 (grammar-rejected spec token padding)."""

from unittest.mock import Mock

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .utils import EOS_TOKEN_ID

pytestmark = pytest.mark.cpu_test


def _make_scheduler_with_structured_output_request() -> tuple[Scheduler, Request]:
    """Minimal scheduler + running request for update_from_output tests."""
    scheduler = object.__new__(Scheduler)
    sampling_params = SamplingParams(ignore_eos=True, max_tokens=32)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)

    request = Request(
        request_id="req-0",
        prompt_token_ids=[0, 1],
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
    )
    request.structured_output_request = Mock()
    request.structured_output_request.grammar = Mock()
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = request.num_tokens + 5
    request.num_output_placeholders = 0

    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.structured_output_manager = Mock()
    scheduler.structured_output_manager.should_advance.return_value = True
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.waiting = Mock()
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.kv_event_publisher = Mock()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.vllm_config = Mock()
    scheduler.vllm_config.model_config.enable_return_routed_experts = False
    scheduler.enable_return_routed_experts = False
    scheduler.recompute_kv_load_failures = False
    scheduler.make_stats = Mock(return_value=None)
    scheduler.max_model_len = 128
    scheduler.log_stats = False
    scheduler._free_request = Mock()

    return scheduler, request


def _scheduler_output_with_spec(
    request: Request,
    scheduled_spec: list[int],
    num_invalid: int | None,
) -> SchedulerOutput:
    num_draft_tokens = len(scheduled_spec)
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 1 + num_draft_tokens},
        total_num_scheduled_tokens=1 + num_draft_tokens,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={request.request_id: scheduled_spec},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        num_invalid_spec_tokens=(
            {request.request_id: num_invalid} if num_invalid is not None else None
        ),
    )


@pytest.mark.parametrize(
    ("num_invalid", "sampled_token_ids", "expected_grammar_tokens"),
    [
        # All four draft slots grammar-invalid (-1 padding); only bonus kept.
        (4, [100, 101, 102, 103, 104], [100]),
        # Two valid drafts, two -1 pads; reject tokens beyond valid prefix.
        (2, [10, 20, 30, 40, 99], [10, 20, 99]),
    ],
)
def test_update_from_output_caps_tokens_when_grammar_invalidates_spec_drafts(
    num_invalid: int,
    sampled_token_ids: list[int],
    expected_grammar_tokens: list[int],
):
    """Rejection at unconstrained -1 slots must not reach grammar advancement."""
    scheduler, request = _make_scheduler_with_structured_output_request()
    num_draft_tokens = 4
    scheduled_spec = [-1] * num_draft_tokens

    def accept_tokens(_req_id: str, tokens: list[int]) -> bool:
        # Simulate grammar that only accepts the capped prefix.
        return tokens == expected_grammar_tokens

    request.structured_output_request.grammar.accept_tokens.side_effect = accept_tokens

    scheduler_output = _scheduler_output_with_spec(request, scheduled_spec, num_invalid)

    num_computed_before = request.num_computed_tokens
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[sampled_token_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)

    request.structured_output_request.grammar.accept_tokens.assert_called_once_with(
        request.request_id, expected_grammar_tokens
    )
    assert request.status == RequestStatus.RUNNING
    assert list(request.output_token_ids) == expected_grammar_tokens
    num_valid_drafts = num_draft_tokens - num_invalid
    num_rejected = num_draft_tokens - num_valid_drafts
    assert request.num_computed_tokens == num_computed_before - num_rejected
