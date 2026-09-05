# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from vllm.config import SchedulerConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.core.sched.request_queue import ResidualSJFRequestQueue
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler, mock_kv

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _queue_requests() -> tuple[list, dict[str, int]]:
    requests = create_requests(
        num_requests=4,
        req_ids=["recovery", "aged", "short", "same_cost"],
    )
    costs = {"recovery": 100, "aged": 100, "short": 1, "same_cost": 1}
    return requests, costs


def test_residual_sjf_queue_orders_recovery_aging_cost_and_ties():
    requests, costs = _queue_requests()
    recovery, aged, short, same_cost = requests
    recovery.arrival_time = 95.0
    recovery.num_computed_tokens = 1
    aged.arrival_time = 90.0
    short.arrival_time = 99.0
    same_cost.arrival_time = 99.0

    queue = ResidualSJFRequestQueue(
        lambda request: costs[request.request_id], max_wait_ms=5_000, clock=lambda: 100
    )
    for request in requests:
        queue.add_request(request)

    assert queue.peek_request() is recovery
    assert queue.pop_request() is recovery
    assert queue.peek_request() is aged
    assert queue.pop_request() is aged
    assert queue.peek_request() is same_cost
    assert queue.pop_request() is same_cost
    assert queue.pop_request() is short


def test_residual_sjf_queue_recomputes_cost_but_pins_peeked_request():
    requests = create_requests(num_requests=2, req_ids=["a", "b"])
    for request in requests:
        request.arrival_time = 100.0
    costs = {"a": 10, "b": 20}
    queue = ResidualSJFRequestQueue(
        lambda request: costs[request.request_id], max_wait_ms=60_000, clock=lambda: 100
    )
    for request in requests:
        queue.add_request(request)

    assert queue.peek_request() is requests[0]
    costs["b"] = 1
    assert queue.pop_request() is requests[0]
    assert queue.peek_request() is requests[1]


def test_residual_sjf_queue_supports_requeue_and_removal():
    requests = create_requests(num_requests=3, req_ids=["a", "b", "c"])
    costs = {"a": 3, "b": 2, "c": 1}
    queue = ResidualSJFRequestQueue(
        lambda request: costs[request.request_id], max_wait_ms=60_000, clock=lambda: 100
    )
    for request in requests:
        queue.add_request(request)

    assert list(queue) == [requests[2], requests[1], requests[0]]
    queue.pin_request(requests[2])
    queue.remove_request(requests[2])
    assert queue.pop_request() is requests[1]
    queue.prepend_request(requests[1])
    queue.remove_requests([requests[0]])
    assert queue.pop_request() is requests[1]


def test_residual_sjf_prefers_warm_long_prompt_over_cold_short_prompt():
    scheduler = create_scheduler(
        scheduling_policy="residual_sjf",
        enable_prefix_caching=True,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        block_size=16,
    )
    warm, cached = create_requests(
        num_requests=2,
        num_tokens=33,
        same_prompt=True,
        max_tokens=1,
        req_ids=["warm", "cached"],
    )
    _, cold = create_requests(num_requests=2, num_tokens=18, req_ids=["unused", "cold"])

    scheduler.add_request(warm)
    output = scheduler.schedule()
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=[warm.request_id],
            req_id_to_index={warm.request_id: 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    assert scheduler.kv_cache_manager.get_num_local_computed_tokens(cached) == 32
    assert scheduler.kv_cache_manager.get_num_local_computed_tokens(cold) == 0
    assert scheduler._get_residual_sjf_cost(cached) == 1
    scheduler.add_request(cold)
    scheduler.add_request(cached)

    output = scheduler.schedule()
    assert [request.req_id for request in output.scheduled_new_reqs] == ["cached"]


@pytest.mark.parametrize("enable_prefix_caching", [False, True])
def test_residual_sjf_zero_local_hit_uses_prompt_length(enable_prefix_caching: bool):
    scheduler = create_scheduler(
        scheduling_policy="residual_sjf", enable_prefix_caching=enable_prefix_caching
    )
    long_request, _ = create_requests(
        num_requests=2, num_tokens=20, req_ids=["long", "unused"]
    )
    _, short_request = create_requests(
        num_requests=2, num_tokens=10, req_ids=["unused", "short"]
    )

    assert scheduler._get_residual_sjf_cost(long_request) == 2
    assert scheduler._get_residual_sjf_cost(short_request) == 1
    short_request.skip_reading_prefix_cache = True
    assert scheduler._get_residual_sjf_cost(short_request) == 1


def test_residual_sjf_compares_waiting_and_skipped_waiting_globally():
    scheduler = create_scheduler(scheduling_policy="residual_sjf")
    waiting, skipped = create_requests(
        num_requests=2, num_tokens=20, req_ids=["waiting", "skipped"]
    )
    skipped.prompt_token_ids = [1] * 10
    skipped.all_token_ids = skipped.prompt_token_ids.copy()
    skipped.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    waiting.arrival_time = 1.0
    skipped.arrival_time = 0.0
    scheduler.add_request(waiting)
    scheduler.skipped_waiting.add_request(skipped)

    selected_queue = scheduler._select_waiting_queue_for_scheduling()
    assert selected_queue is scheduler.skipped_waiting
    assert selected_queue.peek_request() is skipped


def test_residual_sjf_does_not_probe_connector_while_ranking():
    scheduler = create_scheduler(
        scheduling_policy="residual_sjf",
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    scheduler.connector.get_num_new_matched_tokens = Mock()
    for request in create_requests(num_requests=2):
        scheduler.add_request(request)

    scheduler._select_waiting_queue_for_scheduling()
    scheduler.connector.get_num_new_matched_tokens.assert_not_called()
    stats = scheduler.kv_cache_manager.prefix_cache_stats
    assert stats is not None
    assert (stats.requests, stats.queries, stats.hits) == (0, 0, 0)


def test_residual_sjf_config_and_cli_validation():
    config = SchedulerConfig(
        max_model_len=32,
        max_num_batched_tokens=32,
        max_num_seqs=1,
        is_encoder_decoder=False,
        policy="residual_sjf",
    )
    assert config.residual_sjf_max_wait_ms == 10_000

    with pytest.raises(ValueError, match="residual_sjf"):
        SchedulerConfig(
            max_model_len=32,
            max_num_batched_tokens=32,
            max_num_seqs=1,
            is_encoder_decoder=False,
            is_multimodal_model=True,
            policy="residual_sjf",
        )
    with pytest.raises(ValueError, match="residual_sjf"):
        SchedulerConfig(
            max_model_len=32,
            max_num_batched_tokens=32,
            max_num_seqs=1,
            is_encoder_decoder=True,
            policy="residual_sjf",
        )
    with pytest.raises(ValueError, match="residual_sjf"):
        SchedulerConfig(
            max_model_len=32,
            max_num_batched_tokens=32,
            max_num_seqs=1,
            is_encoder_decoder=False,
            runner_type="pooling",
            policy="residual_sjf",
        )
    with pytest.raises(ValueError):
        SchedulerConfig(
            max_model_len=32,
            max_num_batched_tokens=32,
            max_num_seqs=1,
            is_encoder_decoder=False,
            residual_sjf_max_wait_ms=0,
        )

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--scheduling-policy",
            "residual_sjf",
            "--residual-sjf-max-wait-ms",
            "1234",
        ]
    )
    assert args.scheduling_policy == "residual_sjf"
    assert args.residual_sjf_max_wait_ms == 1234
