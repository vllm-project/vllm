# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for EngineCoreSentinel fault handling."""

import queue
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import KVConnectorOutput
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineStatusType
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.fault_tolerance.engine_core_sentinel import (
    EngineCoreSentinel,
    FaultTolerancePause,
)
from vllm.v1.outputs import ModelRunnerOutput


def _model_output(kv_connector_output: KVConnectorOutput | None) -> ModelRunnerOutput:
    output = ModelRunnerOutput(req_ids=[], req_id_to_index={})
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


def test_pause_makes_healthy_engine_recoverable():
    """A healthy engine must accept recovery after ``pause``, without first
    faulting in a collective, so a controller can act on external signals."""
    engine = SimpleNamespace(
        engine_index=0,
        input_queue=queue.Queue(),
        output_queue=queue.Queue(),
        scheduler=MagicMock(),
        model_executor=SimpleNamespace(is_failed=False),
        _send_abort_outputs=MagicMock(),
    )
    ft_config = SimpleNamespace(engine_recovery_timeout_sec=60)
    sentinel = EngineCoreSentinel(
        engine, SimpleNamespace(fault_tolerance_config=ft_config)
    )
    engine.ft_sentinel = sentinel
    pause = {"instruction": "pause", "params": {}}

    sentinel.handle_command(0, 1, pause)
    sentinel.handle_command(0, 2, pause)
    assert engine.input_queue.qsize() == 1

    # The busy loop picks the marker up at its next step boundary.
    marker = engine.input_queue.get_nowait()
    with pytest.raises(FaultTolerancePause) as exc_info:
        EngineCoreProc._handle_client_request(engine, *marker)
    sentinel.on_fault(exc_info.value)
    assert sentinel.status_type == EngineStatusType.UNHEALTHY
    engine.scheduler.finish_requests.assert_called_once()

    # Pausing an unhealthy engine is a no-op, and a stale marker is ignored.
    sentinel.handle_command(0, 3, pause)
    assert engine.input_queue.empty()
    EngineCoreProc._handle_client_request(engine, *marker)
