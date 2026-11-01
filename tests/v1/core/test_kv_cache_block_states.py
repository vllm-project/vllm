# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the `vllm:kv_cache_blocks_by_state` block counts."""

from unittest.mock import PropertyMock, patch

import prometheus_client
import pytest

from tests.v1.kv_connector.unit.utils import (
    create_model_runner_output,
    create_request,
    create_vllm_config,
)
from tests.v1.kv_connector.unit.utils import (
    create_scheduler as create_connector_scheduler,
)
from vllm.config import ModelConfig, VllmConfig
from vllm.config.observability import ObservabilityConfig
from vllm.v1.core.kv_cache_metrics import count_kv_cache_blocks_by_state
from vllm.v1.core.single_type_kv_cache_manager import HiSparseResidentManager
from vllm.v1.kv_cache_interface import HiSparseResidentSpec
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.prometheus import unregister_vllm_metrics
from vllm.v1.metrics.stats import KV_CACHE_BLOCK_STATES, SchedulerStats
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler, mock_kv

pytestmark = pytest.mark.cpu_test

NUM_BLOCKS = 100
BLOCK_SIZE = 16
# 2 full blocks and 1 partial block.
NUM_PROMPT_TOKENS = 2 * BLOCK_SIZE + 8


def _expected(**counts: int) -> dict[str, int]:
    """All states, with `empty_free` holding the remaining blocks."""
    expected = dict.fromkeys(KV_CACHE_BLOCK_STATES, 0)
    expected.update(counts)
    expected["empty_free"] = NUM_BLOCKS - 1 - sum(counts.values())
    return expected


def _count(scheduler) -> dict[str, int]:
    counts = count_kv_cache_blocks_by_state(
        scheduler.kv_cache_manager.block_pool,
        scheduler.kv_cache_manager.coordinator.single_type_managers,
        scheduler.requests,
        scheduler.finished_recving_kv_req_ids,
        scheduler.deferred_frees,
    )
    assert sum(counts.values()) == NUM_BLOCKS - 1
    return counts


def _stats_counts(engine_core_outputs) -> dict[str, int] | None:
    stats = next(iter(engine_core_outputs.values())).scheduler_stats
    counts = stats.kv_cache_blocks_by_state
    if counts is not None:
        assert sum(counts.values()) == NUM_BLOCKS - 1
    return counts


def _step(scheduler, requests, token_id: int = 0):
    scheduler_output = scheduler.schedule()
    return scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(reqs=requests, token_id=token_id),
    )


def test_running_requests_split_shared_and_private():
    scheduler = create_scheduler(enable_prefix_caching=True, num_blocks=NUM_BLOCKS)
    first, second = create_requests(
        num_requests=2, num_tokens=NUM_PROMPT_TOKENS, same_prompt=True
    )
    scheduler.add_request(first)
    _step(scheduler, [first])
    scheduler.add_request(second)
    scheduler.schedule()

    # The 2 full prompt blocks are shared; each partial block is private.
    assert _count(scheduler) == _expected(active_shared=2, active_private=2)


def _run_to_idle(scheduler, request):
    """Run a 2-token request to completion and return the stats of the step
    that leaves the engine idle (the step on which the request finishes).
    """
    scheduler.add_request(request)
    _step(scheduler, [request])
    engine_core_outputs = _step(scheduler, [request])
    assert request.is_finished()
    assert not scheduler.requests
    assert _stats_counts(_step(scheduler, [])) is None
    assert not scheduler.has_requests()
    return _stats_counts(engine_core_outputs)


@patch("time.monotonic", return_value=100.0)
def test_idle_after_busy_sample_is_emitted_once(_):
    """Going idle after a busy sample publishes even inside the busy interval;
    the same idle state with no busy sample in between is not recounted
    unless the prefix cache is reset.
    """
    scheduler = create_scheduler(
        enable_prefix_caching=True, num_blocks=NUM_BLOCKS, kv_cache_metrics=True
    )
    first, second = create_requests(
        num_requests=2, num_tokens=NUM_PROMPT_TOKENS, max_tokens=2, same_prompt=True
    )
    assert _run_to_idle(scheduler, first) == _expected(cached_evictable=2)
    assert _run_to_idle(scheduler, second) is None

    assert scheduler.reset_prefix_cache()
    assert _stats_counts(_step(scheduler, [])) == _expected(cached_evictable=0)
    assert _stats_counts(_step(scheduler, [])) is None


def test_busy_sample_interval_ignores_idle_samples():
    """Idle samples do not delay the next busy sample, and a busy sample makes
    the next idle state publish even if it repeats an earlier idle state.
    """
    scheduler = create_scheduler(num_blocks=NUM_BLOCKS, kv_cache_metrics=True)
    (one_step,) = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, max_tokens=1, req_ids=["0"]
    )
    (first,) = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, max_tokens=2, req_ids=["1"]
    )
    (second,) = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, max_tokens=3, req_ids=["2"]
    )
    # A 1-step request finishes in update_from_output with self.requests empty;
    # it emits the idle sample without starting the 10s busy cooldown.
    scheduler.add_request(one_step)
    with patch("time.monotonic", return_value=95.0):
        assert _stats_counts(_step(scheduler, [one_step])) == _expected()
        assert one_step.is_finished()
        assert _stats_counts(_step(scheduler, [])) is None

    scheduler.add_request(first)
    with patch("time.monotonic", return_value=100.0):
        assert _stats_counts(_step(scheduler, [first])) == _expected(active_private=3)
    with patch("time.monotonic", return_value=105.0):
        assert _stats_counts(_step(scheduler, [first])) == _expected()
        assert first.is_finished()
        assert _stats_counts(_step(scheduler, [])) is None

    scheduler.add_request(second)
    with patch("time.monotonic", return_value=108.0):
        assert _stats_counts(_step(scheduler, [second])) is None
    with patch("time.monotonic", return_value=110.0):
        assert _stats_counts(_step(scheduler, [second])) == _expected(active_private=3)
    with patch("time.monotonic", return_value=111.0):
        assert _stats_counts(_step(scheduler, [second])) == _expected()
        assert second.is_finished()
        assert _stats_counts(_step(scheduler, [])) is None


def test_busy_samples_are_rate_limited():
    scheduler = create_scheduler(num_blocks=NUM_BLOCKS, kv_cache_metrics=True)
    (request,) = create_requests(num_requests=1, num_tokens=NUM_PROMPT_TOKENS)
    scheduler.add_request(request)

    sampled = []
    for now in (100.0, 105.0, 110.0):
        with patch("time.monotonic", return_value=now):
            counts = _stats_counts(_step(scheduler, [request]))
        sampled.append(counts is not None)
        if counts is not None:
            assert counts == _expected(active_private=3)
    assert sampled == [True, False, True]


@patch("time.monotonic", return_value=100.0)
def test_deferred_frees_count_as_pending_free(_):
    # Deferred freeing needs a KV consumer with more than one batch in flight.
    with patch.object(
        VllmConfig,
        "max_concurrent_batches",
        new_callable=PropertyMock,
        return_value=2,
    ):
        scheduler = create_scheduler(
            num_blocks=NUM_BLOCKS,
            async_scheduling=True,
            use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
            kv_cache_metrics=True,
        )
    assert scheduler.defer_block_free
    (request,) = create_requests(num_requests=1, num_tokens=NUM_PROMPT_TOKENS)
    scheduler.add_request(request)
    # Prefill plus one over-scheduled decode step, both in flight.
    out0 = scheduler.schedule()
    out1 = scheduler.schedule()

    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert request.request_id not in scheduler.requests
    assert _count(scheduler) == _expected(pending_free=3)

    # Idle, but step 2 is still in flight: the blocks stay pending.
    engine_core_outputs = scheduler.update_from_output(
        out0, create_model_runner_output(reqs=[request])
    )
    assert not scheduler.requests
    assert _stats_counts(engine_core_outputs) == _expected(pending_free=3)

    # Step 2 drains the deferred free; the changed idle state is published.
    engine_core_outputs = scheduler.update_from_output(
        out1, create_model_runner_output(reqs=[request])
    )
    assert not scheduler.deferred_frees
    assert _stats_counts(engine_core_outputs) == _expected()


def _create_connector_scheduler():
    return create_connector_scheduler(create_vllm_config(), num_blocks=NUM_BLOCKS)


def test_remote_prefill_load_then_waiting_hold():
    scheduler = _create_connector_scheduler()
    request = create_request(num_tokens=NUM_PROMPT_TOKENS, do_remote_prefill=True)
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert _count(scheduler) == _expected(loading_kv=3)

    scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(
        reqs=[], finished_recving={request.request_id}
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert _count(scheduler) == _expected(waiting_hold=3)

    scheduler.schedule()
    assert request.status == RequestStatus.RUNNING
    assert _count(scheduler) == _expected(active_private=3)


def test_remote_decode_blocks_held_for_send():
    scheduler = _create_connector_scheduler()
    request = create_request(num_tokens=NUM_PROMPT_TOKENS, do_remote_decode=True)
    scheduler.add_request(request)

    _step(scheduler, [request])
    assert request.is_finished()
    assert _count(scheduler) == _expected(held_for_send=3)

    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(
        reqs=[], finished_sending={request.request_id}
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert request.request_id not in scheduler.requests
    assert _count(scheduler) == _expected(cached_evictable=2)


def test_aborted_load_counts_as_held_for_send():
    """An aborted async load keeps its blocks until the load reports finished;
    the request is finished by then, so the blocks count as held_for_send.
    """
    scheduler = _create_connector_scheduler()
    request = create_request(num_tokens=NUM_PROMPT_TOKENS, do_remote_prefill=True)
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)

    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert _count(scheduler) == _expected(held_for_send=3)

    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(
        reqs=[], finished_recving={request.request_id}
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert request.request_id not in scheduler.requests
    assert _count(scheduler) == _expected()


def test_external_pin_is_unattributed():
    """A block pinned outside any request (as connectors and HiSparse do) is
    in use but has no owner.
    """
    scheduler = create_scheduler(enable_prefix_caching=True, num_blocks=NUM_BLOCKS)
    (request,) = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, max_tokens=1
    )
    scheduler.add_request(request)
    _step(scheduler, [request])
    assert request.is_finished()
    block_pool = scheduler.kv_cache_manager.block_pool
    cached_block = next(b for b in block_pool.blocks if b.block_hash is not None)

    block_pool.touch([cached_block])

    assert _count(scheduler) == _expected(unattributed=1, cached_evictable=1)


def test_hisparse_unpinned_blocks_and_residency_changes():
    """Unpinned (`ref_cnt == 0`) blocks on `free_block_queue` count as free,
    and `HiSparseResidentManager.residency_changes` overrides `req_to_blocks`.
    """
    scheduler = create_scheduler(
        num_blocks=NUM_BLOCKS,
        kv_cache_spec=HiSparseResidentSpec(block_size=BLOCK_SIZE, page_size=BLOCK_SIZE),
    )
    (resident_manager,) = scheduler.kv_cache_manager.coordinator.single_type_managers
    assert isinstance(resident_manager, HiSparseResidentManager)
    block_pool = scheduler.kv_cache_manager.block_pool

    (request,) = create_requests(num_requests=1, num_tokens=NUM_PROMPT_TOKENS)
    request.status = RequestStatus.RUNNING
    scheduler.requests[request.request_id] = request
    resident_manager.allocate_external_computed_blocks(
        request.request_id,
        num_local_computed_tokens=BLOCK_SIZE,
        num_external_computed_tokens=BLOCK_SIZE,
    )
    tail_block = resident_manager.req_to_blocks[request.request_id][1]
    assert _count(scheduler) == _expected(active_private=1)

    (adopted_block,) = block_pool.get_new_blocks(1)
    assert resident_manager.adopt_resident_page(request.request_id, 0, adopted_block)
    assert _count(scheduler) == _expected(active_private=2)

    block_pool.unpin_blocks([tail_block], lambda _: None)
    assert tail_block.ref_cnt == 0
    assert _count(scheduler) == _expected(active_private=1)

    assert resident_manager.drop_resident_page(request.request_id, 0) is adopted_block
    assert _count(scheduler) == _expected(unattributed=1)
    block_pool.free_blocks([adopted_block])
    assert _count(scheduler) == _expected()


def _block_state_value(engine: str, state: str) -> float | None:
    return prometheus_client.REGISTRY.get_sample_value(
        "vllm:kv_cache_blocks_by_state",
        {"model_name": "distilbert/distilgpt2", "engine": engine, "state": state},
    )


@pytest.mark.parametrize("kv_cache_metrics", [True, False])
def test_prometheus_gauge_per_engine(kv_cache_metrics: bool):
    vllm_config = VllmConfig(
        model_config=ModelConfig(model="distilbert/distilgpt2"),
        observability_config=ObservabilityConfig(kv_cache_metrics=kv_cache_metrics),
    )
    counts = dict.fromkeys(KV_CACHE_BLOCK_STATES, 0)
    counts.update(active_private=5, empty_free=94)
    try:
        stat_logger = PrometheusStatLogger(vllm_config, engine_indexes=[0, 1])
        stat_logger.record(
            SchedulerStats(kv_cache_blocks_by_state=counts), None, engine_idx=1
        )
        if not kv_cache_metrics:
            assert _block_state_value("1", "active_private") is None
            return
        assert _block_state_value("1", "active_private") == 5
        assert _block_state_value("1", "empty_free") == 94
        assert _block_state_value("0", "active_private") == 0

        # Stats without a new sample keep the last values.
        stat_logger.record(SchedulerStats(), None, engine_idx=1)
        assert _block_state_value("1", "active_private") == 5
    finally:
        unregister_vllm_metrics()
