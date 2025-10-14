# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import FinishReason, RequestStatus

from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)


def _make_get_num_new_matched_tokens(
    req_num_new_matched_tokens: dict[str, int],
    async_load: bool,
):
    def get_num_new_matched_tokens(request, _):
        value = req_num_new_matched_tokens.get(request.request_id, 0)
        return value, async_load

    return get_num_new_matched_tokens


@pytest.fixture
def abort_scheduler():
    """scheduler with kv_load_retry_policy='abort'"""
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config.kv_load_retry_policy = "abort"
    return create_scheduler(vllm_config)


def test_502_error_propagation_sync_load(abort_scheduler: Scheduler):
    """test invalid_block_ids with abort policy -> FINISHED_ERROR"""
    num_prompt_blocks = 100
    num_external_computed_blocks = 99
    invalid_block_idx = 50

    num_prompt_tokens = num_prompt_blocks * abort_scheduler.block_size
    num_external_computed_tokens = (
        num_external_computed_blocks * abort_scheduler.block_size
    )

    request = create_request(num_tokens=num_prompt_tokens)
    abort_scheduler.add_request(request=request)

    req_num_new_matched_tokens = {
        request.request_id: num_external_computed_tokens,
    }

    abort_scheduler.connector = Mock()
    abort_scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, async_load=False)
    )
    abort_scheduler.connector.request_finished.return_value = (False, None)
    abort_scheduler.connector.take_events.return_value = ()

    scheduler_output = abort_scheduler.schedule()

    assert len(abort_scheduler.running) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 1
    assert abort_scheduler.connector.get_num_new_matched_tokens.call_count == 1

    req_block_ids = scheduler_output.scheduled_new_reqs[0].block_ids[0]
    invalid_block_ids = {req_block_ids[invalid_block_idx]}
    model_runner_output = create_model_runner_output(
        [request],
        invalid_block_ids=invalid_block_ids,
        use_eos=True,
    )

    outputs = abort_scheduler.update_from_output(scheduler_output, model_runner_output)

    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.get_finished_reason() == FinishReason.ERROR

    assert len(outputs) == 1
    engine_outputs = next(iter(outputs.values()))
    assert len(engine_outputs.outputs) == 1
    output = engine_outputs.outputs[0]
    assert output.request_id == request.request_id
    assert output.finish_reason == FinishReason.ERROR

    assert len(abort_scheduler.running) == 0


def test_502_error_propagation_async_load(abort_scheduler: Scheduler):
    """test invalid_block_ids with abort + async load -> FINISHED_ERROR"""
    num_prompt_blocks = 100
    num_external_computed_blocks = 99
    invalid_block_idx = 50

    num_prompt_tokens = num_prompt_blocks * abort_scheduler.block_size
    num_external_computed_tokens = (
        num_external_computed_blocks * abort_scheduler.block_size
    )

    request = create_request(num_tokens=num_prompt_tokens)
    abort_scheduler.add_request(request=request)

    req_num_new_matched_tokens = {
        request.request_id: num_external_computed_tokens,
    }

    abort_scheduler.connector = Mock()
    abort_scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, async_load=True)
    )
    abort_scheduler.connector.request_finished.return_value = (False, None)
    abort_scheduler.connector.take_events.return_value = ()

    scheduler_output = abort_scheduler.schedule()

    assert len(abort_scheduler.waiting) == 1
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.num_computed_tokens == 0

    (req_block_ids,) = abort_scheduler.kv_cache_manager.get_block_ids(
        request.request_id
    )
    invalid_block_ids = {req_block_ids[invalid_block_idx]}
    model_runner_output = create_model_runner_output(
        reqs=[],
        finished_recving=set(),
        invalid_block_ids=invalid_block_ids,
        use_eos=True,
    )

    outputs = abort_scheduler.update_from_output(scheduler_output, model_runner_output)

    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.get_finished_reason() == FinishReason.ERROR

    assert len(outputs) == 1
    engine_outputs = next(iter(outputs.values()))
    assert len(engine_outputs.outputs) == 1
    output = engine_outputs.outputs[0]
    assert output.request_id == request.request_id
    assert output.finish_reason == FinishReason.ERROR

    assert len(abort_scheduler.waiting) == 0
