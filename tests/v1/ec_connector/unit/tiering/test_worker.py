# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Timing must preserve nonblocking transfer completion and metric ownership."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.distributed.ec_transfer.ec_connector.cpu.worker import ECCPUWorker
from vllm.distributed.ec_transfer.ec_connector.tiering import worker as module
from vllm.distributed.ec_transfer.ec_connector.tiering.metrics import ECTieringStats
from vllm.distributed.ec_transfer.ec_connector.tiering.worker import ECTieringWorker


@pytest.mark.parametrize(
    "direction,metric",
    [
        ("load", "h2d_device_seconds"),
        ("save", "d2h_device_seconds"),
    ],
)
def test_device_timing_reports_only_completed_batches_without_synchronizing(
    direction, metric
):
    worker = object.__new__(ECTieringWorker)
    worker._timing_stats = ECTieringStats()
    worker._buf_pool = Mock()
    worker._stream_pool = []
    worker._event_pool = []
    start = Mock()
    start.elapsed_time.return_value = 3.0
    end = Mock()
    end.query.return_value = True
    pending_end = Mock()
    pending_end.query.return_value = False
    transfer = SimpleNamespace(
        start_event=start,
        end_event=end,
        completions=[1],
        bufs=object(),
        stream=object(),
        num_bytes=8,
    )
    pending = SimpleNamespace(end_event=pending_end)
    inflight = deque([transfer, pending])
    assert worker._collect_finished(inflight, direction) == [1]
    assert list(inflight) == [pending]
    assert worker.get_stats().data["histograms"][metric] == [0.003]
    assert worker.get_stats() is None
    end.synchronize.assert_not_called()
    pending_end.synchronize.assert_not_called()
    assert worker._collect_finished(inflight, direction) == []
    assert worker.get_stats() is None


def test_host_submit_timing_wraps_original_load_without_changing_arguments(monkeypatch):
    worker = object.__new__(ECTieringWorker)
    worker._timing_stats = ECTieringStats()
    original = Mock()
    monkeypatch.setattr(ECCPUWorker, "start_load_caches", original)
    clock = iter((1.0, 1.01))
    monkeypatch.setattr(module, "monotonic", lambda: next(clock))
    cache = {}
    meta = SimpleNamespace(loads={"x": object()})
    worker.start_load_caches(cache, meta)
    original.assert_called_once_with(cache, meta)
    assert worker.get_stats().data["histograms"][
        "h2d_host_submit_seconds"
    ] == pytest.approx([0.01])
    worker.start_load_caches(cache, SimpleNamespace(loads={}))
    assert worker.get_stats() is None


def test_worker_timing_is_exported_through_tiering_connector():
    from vllm.distributed.ec_transfer.ec_connector.tiering.connector import (
        ECTieringConnector,
    )

    worker = object.__new__(ECTieringWorker)
    worker._timing_stats = ECTieringStats()
    worker._timing_stats.observe("h2d_device_seconds", 0.01)
    connector = object.__new__(ECTieringConnector)
    connector.connector_scheduler = None
    connector.connector_worker = worker
    assert connector.get_ec_connector_stats().data["histograms"][
        "h2d_device_seconds"
    ] == [0.01]
    assert connector.get_ec_connector_stats() is None
