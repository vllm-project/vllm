# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for EngineCoreSentinel fault handling."""

from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.distributed.kv_transfer.kv_connector.utils import KVConnectorOutput
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.fault_tolerance.engine_core_sentinel import EngineCoreSentinel
from vllm.v1.outputs import ModelRunnerOutput


def _model_output(kv_connector_output: KVConnectorOutput | None) -> ModelRunnerOutput:
    output = ModelRunnerOutput(req_ids=[])
    output.kv_connector_output = kv_connector_output
    return output


def _done_future(result=None, exception=None) -> Future:
    future: Future = Future()
    if exception is not None:
        future.set_exception(exception)
    else:
        future.set_result(result)
    return future


def _make_sentinel(batch_queue: deque) -> tuple[EngineCoreSentinel, MagicMock]:
    scheduler = MagicMock()
    sentinel = EngineCoreSentinel.__new__(EngineCoreSentinel)
    sentinel.engine = SimpleNamespace(batch_queue=batch_queue, scheduler=scheduler)
    return sentinel, scheduler


def test_drain_batch_queue_surfaces_kv_connector_output():
    """Pre-fault KV output must reach the scheduler, not be lost on clear."""
    kv_output = KVConnectorOutput(finished_sending={"req1"})
    batch_queue = deque([(_done_future(_model_output(kv_output)), None, None)])
    sentinel, scheduler = _make_sentinel(batch_queue)

    sentinel._drain_batch_queue()

    assert not batch_queue
    scheduler.update_from_output.assert_called_once()
    sched_output, model_output = scheduler.update_from_output.call_args[0]
    assert isinstance(sched_output, SchedulerOutput)
    assert sched_output.num_scheduled_tokens == {}
    assert model_output.kv_connector_output is kv_output


def test_drain_batch_queue_drops_failures_and_empty_outputs():
    """Failed futures and empty KV outputs are dropped without scheduler calls."""
    batch_queue = deque(
        [
            (_done_future(exception=RuntimeError("boom")), None, None),
            (_done_future(_model_output(None)), None, None),
            (_done_future(_model_output(KVConnectorOutput())), None, None),
        ]
    )
    sentinel, scheduler = _make_sentinel(batch_queue)

    sentinel._drain_batch_queue()

    assert not batch_queue
    scheduler.update_from_output.assert_not_called()


def test_clear_contaminated_blocks_evicts_from_each_managers_pool():
    """Block IDs are per block pool, so each group's written blocks are evicted
    from its own pool (the drafter's KV pool reuses the shared pool's IDs)."""

    def blocks(*ids):
        return [SimpleNamespace(block_id=i, is_null=i == 0) for i in ids]

    shared_pool, draft_pool = MagicMock(), MagicMock()
    managers = [
        SimpleNamespace(
            block_size=4,
            block_pool=shared_pool,
            req_to_blocks={"r": blocks(5, 6, 7)},
        ),
        SimpleNamespace(
            block_size=4,
            block_pool=draft_pool,
            req_to_blocks={"r": blocks(0, 2, 3)},
        ),
    ]
    sentinel, scheduler = _make_sentinel(deque())
    scheduler.running = [
        SimpleNamespace(request_id="r", num_in_flight_tokens=6, num_computed_tokens=12)
    ]
    scheduler.kv_cache_manager.coordinator.single_type_managers = managers

    sentinel._clear_contaminated_blocks()

    shared_pool.evict_blocks.assert_called_once_with({6, 7})
    draft_pool.evict_blocks.assert_called_once_with({2, 3})
    scheduler.kv_cache_manager.evict_blocks.assert_not_called()
