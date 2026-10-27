# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from unittest.mock import Mock

import pytest

from vllm.config import SchedulerConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import ResidualSJFRequestQueue
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
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
    assert queue.peek_request() is requests[2]
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


def test_residual_sjf_does_not_probe_connector_while_ranking():
    scheduler = create_scheduler(
        scheduling_policy="residual_sjf",
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    scheduler.connector.get_num_new_matched_tokens = Mock()
    for request in create_requests(num_requests=2):
        scheduler.add_request(request)

    scheduler.waiting.peek_request()
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
            "--residual-sjf-preempt-prefills",
        ]
    )
    assert args.scheduling_policy == "residual_sjf"
    assert args.residual_sjf_max_wait_ms == 1234
    assert args.residual_sjf_preempt_prefills
    with pytest.raises(ValueError, match="requires residual_sjf"):
        SchedulerConfig(
            max_model_len=32,
            max_num_batched_tokens=32,
            max_num_seqs=1,
            is_encoder_decoder=False,
            residual_sjf_preempt_prefills=True,
        )


def test_connector_estimate_changes_order_but_does_not_replace_admission():
    scheduler = create_scheduler(
        scheduling_policy="residual_sjf",
        max_num_seqs=1,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    long_req = create_requests(1, num_tokens=97, req_ids=["long"])[0]
    short_req = create_requests(1, num_tokens=33, req_ids=["short"])[0]
    scheduler.add_request(short_req)
    scheduler.add_request(long_req)
    assert scheduler.waiting.peek_request() is short_req
    long_req.kv_transfer_params = {"cached_tokens": 96}
    assert scheduler.waiting.peek_request() is long_req
    output = scheduler.schedule()
    assert [r.req_id for r in output.scheduled_new_reqs] == ["long"]
    # The actual connector misses, so the estimate must not skip computation.
    assert output.num_scheduled_tokens[long_req.request_id] == 97


def test_local_estimate_refreshes_after_cache_reset():
    scheduler = create_scheduler(
        scheduling_policy="residual_sjf",
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    warm, waiting = create_requests(2, num_tokens=33, same_prompt=True)
    manager = scheduler.kv_cache_manager
    assert manager.allocate_slots(warm, 32) is not None
    manager.cache_blocks(warm, 32)
    manager.free(warm)
    scheduler.add_request(waiting)
    assert scheduler._get_residual_sjf_cost(waiting) == 1
    assert scheduler._get_residual_sjf_cost(waiting) == 1
    assert scheduler.reset_prefix_cache()
    assert scheduler._get_residual_sjf_cost(waiting) == 3


def _finish_step(scheduler, output, finished_recving=None):
    req_ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=[
                [] if scheduler.requests[rid].is_prefill_chunk else [0]
                for rid in req_ids
            ],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            kv_connector_output=KVConnectorOutput(finished_recving=finished_recving),
        ),
    )


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
def test_ineligible_async_decodes_leave_prefill_budget_available(
    pipeline_parallel_size,
):
    scheduler = create_scheduler(
        scheduling_policy="residual_sjf",
        residual_sjf_preempt_prefills=True,
        async_scheduling=True,
        pipeline_parallel_size=pipeline_parallel_size,
        use_v2_model_runner=pipeline_parallel_size > 1,
        max_num_batched_tokens=32,
        max_num_seqs=32,
    )
    scheduler.add_request(create_requests(1, num_tokens=129, req_ids=["long"])[0])
    _finish_step(scheduler, scheduler.schedule())
    shorts = create_requests(
        31, num_tokens=1, max_tokens=1 if pipeline_parallel_size == 1 else 32
    )
    for request in shorts:
        scheduler.add_request(request)
    scheduler.schedule()  # Leave terminal or PP-cadence-gated outputs in flight.
    assert all(request.num_output_placeholders == 1 for request in shorts)
    assert scheduler.schedule().num_scheduled_tokens == {"long": 32}


def test_decode_preemption_does_not_reserve_budget_for_blocked_admission():
    scheduler = create_scheduler(
        scheduling_policy="residual_sjf",
        residual_sjf_preempt_prefills=True,
        scheduler_reserve_full_isl=False,
        max_num_batched_tokens=34,
        num_blocks=9,
        max_num_seqs=3,
        max_model_len=128,
    )
    for name, prompt, output in [
        ("decode", 30, 64),
        ("resident", 114, 8),
        ("victim", 65, 8),
    ]:
        scheduler.add_request(
            create_requests(1, num_tokens=prompt, max_tokens=output, req_ids=[name])[0]
        )
        _finish_step(scheduler, scheduler.schedule())
    assert scheduler.schedule().num_scheduled_tokens == {"decode": 1, "resident": 33}
    assert scheduler.requests["victim"].num_preemptions == 1
    assert scheduler.requests["resident"].num_preemptions == 0


def test_kv_holder_reservation_precedes_ordinary_preempted_request():
    scheduler = create_scheduler(
        scheduling_policy="residual_sjf",
        residual_sjf_preempt_prefills=True,
        max_num_batched_tokens=32,
        max_num_seqs=2,
    )
    scheduler.add_request(create_requests(1, num_tokens=129, req_ids=["resident"])[0])
    _finish_step(scheduler, scheduler.schedule())
    holder = create_requests(1, num_tokens=64, req_ids=["holder"])[0]
    assert scheduler.kv_cache_manager.allocate_slots(holder, 32) is not None
    holder.num_computed_tokens = 32
    scheduler.add_request(holder)
    ordinary = create_requests(1, num_tokens=1, req_ids=["ordinary"])[0]
    ordinary.status = RequestStatus.PREEMPTED
    ordinary.num_preemptions = 1
    ordinary.arrival_time = holder.arrival_time - 0.01
    scheduler.add_request(ordinary)
    assert scheduler.schedule().num_scheduled_tokens == {"holder": 32}


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("async_scheduling", [False, True])
def test_residual_sjf_chunk_yield_keeps_kv_and_resumes(enabled, async_scheduling):
    scheduler = create_scheduler(
        scheduling_policy="residual_sjf",
        residual_sjf_preempt_prefills=enabled,
        async_scheduling=async_scheduling,
        enable_chunked_prefill=True,
        max_num_batched_tokens=32,
        max_num_seqs=2,
    )
    long_req = create_requests(1, num_tokens=129, req_ids=["long"])[0]
    scheduler.add_request(long_req)
    _finish_step(scheduler, scheduler.schedule())
    blocks = scheduler.kv_cache_manager.get_block_ids("long")
    short_req = create_requests(1, num_tokens=17, max_tokens=1, req_ids=["short"])[0]
    scheduler.add_request(short_req)

    output = scheduler.schedule()
    if enabled:
        assert output.num_scheduled_tokens == {"short": 17}
        assert long_req.num_computed_tokens == 32
        assert scheduler.kv_cache_manager.get_block_ids("long") == blocks
        _finish_step(scheduler, output)
        assert scheduler.schedule().num_scheduled_tokens == {"long": 32}
    else:
        assert output.num_scheduled_tokens == {"long": 32}
    assert long_req.num_preemptions == 0


@pytest.mark.parametrize("guard", ["full_slots", "paused_new", "aged"])
def test_residual_sjf_chunk_yield_respects_admission_and_aging(guard):
    scheduler = create_scheduler(
        scheduling_policy="residual_sjf",
        residual_sjf_preempt_prefills=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=32,
        max_num_seqs=1 if guard == "full_slots" else 2,
    )
    long_req = create_requests(1, num_tokens=129, req_ids=["long"])[0]
    if guard == "aged":
        long_req.arrival_time = time.time() - 20
    scheduler.add_request(long_req)
    _finish_step(scheduler, scheduler.schedule())
    scheduler.add_request(create_requests(1, num_tokens=17, req_ids=["short"])[0])
    if guard == "paused_new":
        scheduler.set_pause_state(PauseState.PAUSED_NEW)
    assert scheduler.schedule().num_scheduled_tokens == {"long": 32}


def test_residual_sjf_failed_reservation_does_not_stall_resident_prefill():
    scheduler = create_scheduler(
        scheduling_policy="residual_sjf",
        residual_sjf_preempt_prefills=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=24,
        max_num_seqs=3,
        num_blocks=8,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    long_req = create_requests(1, num_tokens=97, req_ids=["long"])[0]
    scheduler.add_request(long_req)
    _finish_step(scheduler, scheduler.schedule())
    scheduler.add_request(create_requests(1, num_tokens=1, req_ids=["decode"])[0])
    _finish_step(scheduler, scheduler.schedule())
    waiting = create_requests(1, num_tokens=257, req_ids=["waiting"])[0]
    waiting.kv_transfer_params = {"cached_tokens": 225}
    scheduler.add_request(waiting)
    output = scheduler.schedule()
    # The estimate promises a short suffix, but the actual miss cannot fit KV.
    assert output.num_scheduled_tokens == {"decode": 1}
    _finish_step(scheduler, output)
    assert scheduler.schedule().num_scheduled_tokens == {"long": 23, "decode": 1}
    assert long_req.num_preemptions == 0


def test_residual_sjf_yield_preserves_decode_budget():
    scheduler = create_scheduler(
        scheduling_policy="residual_sjf",
        residual_sjf_preempt_prefills=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=32,
        max_num_seqs=3,
    )
    scheduler.add_request(create_requests(1, num_tokens=129, req_ids=["long"])[0])
    _finish_step(scheduler, scheduler.schedule())
    scheduler.add_request(create_requests(1, num_tokens=1, req_ids=["decode"])[0])
    _finish_step(scheduler, scheduler.schedule())
    scheduler.add_request(create_requests(1, num_tokens=17, req_ids=["short"])[0])
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"decode": 1, "short": 17}


def test_completed_connector_estimate_can_preempt_a_running_chunk():
    scheduler = create_scheduler(
        scheduling_policy="residual_sjf",
        residual_sjf_preempt_prefills=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=32,
        max_num_seqs=2,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    scheduler.add_request(create_requests(1, num_tokens=129, req_ids=["running"])[0])
    _finish_step(scheduler, scheduler.schedule())
    waiting = create_requests(1, num_tokens=257, req_ids=["waiting"])[0]
    scheduler.add_request(waiting)
    _finish_step(scheduler, scheduler.schedule())
    waiting.kv_transfer_params = {"cached_tokens": 256}
    output = scheduler.schedule()
    assert "waiting" in output.num_scheduled_tokens
    # The real connector misses: estimated hits only affect order/budget.
    assert output.num_scheduled_tokens["waiting"] == 16
    assert waiting.num_computed_tokens == 16


def test_residual_sjf_reserves_remote_compute_only_after_load_completion():
    scheduler = create_scheduler(
        scheduling_policy="residual_sjf",
        residual_sjf_preempt_prefills=True,
        max_num_batched_tokens=32,
        max_num_seqs=2,
        use_kv_connector=mock_kv(matched_tokens=32, is_async=True),
    )
    scheduler.add_request(create_requests(1, num_tokens=257, req_ids=["long"])[0])
    _finish_step(scheduler, scheduler.schedule(), finished_recving={"long"})
    _finish_step(scheduler, scheduler.schedule())
    scheduler.add_request(create_requests(1, num_tokens=49, req_ids=["loading"])[0])
    _finish_step(scheduler, scheduler.schedule())  # start the asynchronous load
    _finish_step(scheduler, scheduler.schedule())  # retry resident work

    pending = scheduler.schedule()
    assert pending.num_scheduled_tokens == {"long": 32}
    _finish_step(scheduler, pending, finished_recving={"loading"})
    assert scheduler.schedule().num_scheduled_tokens == {"loading": 17}
