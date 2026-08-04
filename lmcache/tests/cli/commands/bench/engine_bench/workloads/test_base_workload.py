# SPDX-License-Identifier: Apache-2.0
"""Tests for the BaseWorkload abstract class."""

# Standard
from unittest.mock import AsyncMock, MagicMock
import queue

# First Party
from lmcache.cli.commands.bench.engine_bench.stats import RequestResult
from lmcache.cli.commands.bench.engine_bench.workloads.base import BaseWorkload

# ---------------------------------------------------------------------------
# Stub workload for testing
# ---------------------------------------------------------------------------


class StubWorkload(BaseWorkload):
    """Minimal concrete workload for testing the base class."""

    def __init__(self, request_sender, stats_collector, progress_monitor):
        super().__init__(request_sender, stats_collector, progress_monitor)
        self.warmup_called = False
        self.step_calls: list[float] = []
        self.finished_calls: list[tuple[str, str]] = []
        self._step_returns: list[float] = []

    def log_config(self) -> None:
        pass

    async def warmup(self) -> None:
        self.warmup_called = True

    async def step(self, time_offset: float) -> float:
        self.step_calls.append(time_offset)
        if self._step_returns:
            return self._step_returns.pop(0)
        return -1.0

    def on_request_finished(self, request_id: str, output: str) -> None:
        self.finished_calls.append((request_id, output))


def _make_mock_result(request_id: str = "req_0") -> RequestResult:
    return MagicMock(spec=RequestResult, request_id=request_id)


def _make_stub(**kwargs) -> StubWorkload:
    sender = MagicMock()
    sender.close = AsyncMock()
    collector = MagicMock()
    monitor = MagicMock()
    return StubWorkload(sender, collector, monitor, **kwargs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestBaseWorkloadConstruction:
    def test_stores_references(self) -> None:
        sender = MagicMock()
        collector = MagicMock()
        monitor = MagicMock()
        w = StubWorkload(sender, collector, monitor)
        assert w._request_sender is sender
        assert w._stats_collector is collector
        assert w._progress_monitor is monitor
        assert isinstance(w._finished_queue, queue.Queue)


# ---------------------------------------------------------------------------
# request_finished / drain
# ---------------------------------------------------------------------------


class TestBaseWorkloadRequestFinished:
    def test_enqueues_to_queue(self) -> None:
        w = _make_stub()
        result = _make_mock_result("r1")
        w.request_finished(result, "hello")
        item = w._finished_queue.get_nowait()
        assert item == ("r1", "hello")

    def test_multiple_enqueues(self) -> None:
        w = _make_stub()
        for i in range(3):
            w.request_finished(_make_mock_result(f"r{i}"), f"text_{i}")
        assert w._finished_queue.qsize() == 3


class TestBaseWorkloadDrainQueue:
    def test_drain_empty_queue(self) -> None:
        w = _make_stub()
        w._drain_finished_queue()
        assert w.finished_calls == []

    def test_drain_calls_on_request_finished(self) -> None:
        w = _make_stub()
        w.request_finished(_make_mock_result("r0"), "text_0")
        w.request_finished(_make_mock_result("r1"), "text_1")
        w._drain_finished_queue()
        assert w.finished_calls == [("r0", "text_0"), ("r1", "text_1")]

    def test_drain_empties_queue(self) -> None:
        w = _make_stub()
        w.request_finished(_make_mock_result("r0"), "text")
        w._drain_finished_queue()
        assert w._finished_queue.empty()


# ---------------------------------------------------------------------------
# run loop
# ---------------------------------------------------------------------------


class TestBaseWorkloadRunLoop:
    def test_run_calls_warmup(self) -> None:
        w = _make_stub()
        w.run()
        assert w.warmup_called is True

    def test_run_calls_stats_reset(self) -> None:
        w = _make_stub()
        w.run()
        w._stats_collector.reset.assert_called_once()  # type: ignore[attr-defined]

    def test_run_calls_step(self) -> None:
        w = _make_stub()
        w._step_returns = [0.0, -1.0]
        w.run()
        assert len(w.step_calls) == 2

    def test_run_drains_queue_after_warmup(self) -> None:
        class WarmupEnqueuer(StubWorkload):
            async def warmup(self) -> None:
                self.warmup_called = True
                self.request_finished(_make_mock_result("warmup_0"), "warmup_text")

        sender = MagicMock()
        sender.close = AsyncMock()
        collector = MagicMock()
        monitor = MagicMock()
        w = WarmupEnqueuer(sender, collector, monitor)
        w.run()
        # Warmup completion was drained (on_request_finished called)
        assert ("warmup_0", "warmup_text") in w.finished_calls

    def test_negative_step_stops_loop(self) -> None:
        w = _make_stub()
        # Default: step returns -1.0
        w.run()
        assert len(w.step_calls) == 1
