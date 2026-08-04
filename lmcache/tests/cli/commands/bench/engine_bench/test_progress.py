# SPDX-License-Identifier: Apache-2.0
"""Tests for bench engine progress monitor."""

# Standard
import io
import sys
import threading
import time

# First Party
from lmcache.cli.commands.bench.engine_bench.progress import (
    ProgressMonitor,
)
from lmcache.cli.commands.bench.engine_bench.stats import (
    RequestResult,
    StatsCollector,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_collector() -> StatsCollector:
    return StatsCollector()


def _make_result(request_id: str = "req_0") -> RequestResult:
    now = time.time()
    return RequestResult(
        request_id=request_id,
        successful=True,
        ttft=0.3,
        request_latency=2.0,
        num_input_tokens=10000,
        num_output_tokens=128,
        decode_speed=48.0,
        submit_time=now,
        first_token_time=now + 0.3,
        finish_time=now + 2.0,
        error="",
    )


# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------


class TestProgressMonitorState:
    def test_initial_state(self) -> None:
        m = ProgressMonitor(_make_collector(), quiet=True)
        assert m._in_flight == 0
        assert m._log_messages == []

    def test_in_flight_tracking(self) -> None:
        m = ProgressMonitor(_make_collector(), quiet=True)
        m.on_request_sent("r0")
        m.on_request_sent("r1")
        m.on_request_sent("r2")
        assert m._in_flight == 3
        m.on_request_finished("r0", True)
        m.on_request_finished("r1", False)
        assert m._in_flight == 1

    def test_on_request_finished_decrements_to_zero(self) -> None:
        m = ProgressMonitor(_make_collector(), quiet=True)
        m.on_request_sent("r0")
        m.on_request_finished("r0", True)
        assert m._in_flight == 0

    def test_on_request_finished_does_not_go_negative(self) -> None:
        m = ProgressMonitor(_make_collector(), quiet=True)
        m.on_request_finished("r0", True)
        assert m._in_flight == 0

    def test_multiple_log_messages(self) -> None:
        m = ProgressMonitor(_make_collector(), quiet=True)
        m.log_message("first")
        m.log_message("second")
        m.log_message("third")
        assert m._log_messages == ["first", "second", "third"]

    def test_log_message_max_lines(self) -> None:
        m = ProgressMonitor(_make_collector(), quiet=True)
        for i in range(10):
            m.log_message(f"msg_{i}")
        assert len(m._log_messages) == ProgressMonitor.LOG_LINES
        assert m._log_messages[0] == "msg_5"
        assert m._log_messages[-1] == "msg_9"


# ---------------------------------------------------------------------------
# Quiet mode
# ---------------------------------------------------------------------------


class TestProgressMonitorQuiet:
    def test_quiet_no_thread(self) -> None:
        m = ProgressMonitor(_make_collector(), quiet=True)
        m.start()
        assert m._thread is None

    def test_quiet_stop_safe(self) -> None:
        m = ProgressMonitor(_make_collector(), quiet=True)
        m.start()
        m.stop()  # should not raise


# ---------------------------------------------------------------------------
# Thread lifecycle
# ---------------------------------------------------------------------------


class TestProgressMonitorThread:
    def test_start_stop_lifecycle(self) -> None:
        collector = _make_collector()
        # Redirect stdout to suppress ANSI output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            m = ProgressMonitor(collector)
            m.start()
            assert m._thread is not None
            assert m._thread.is_alive()
            time.sleep(0.1)
            m.stop()
            assert not m._thread.is_alive()
        finally:
            sys.stdout = old_stdout

    def test_draw_calls_stats_collector(self) -> None:
        collector = _make_collector()
        call_count = 0
        original_get = collector.get_current_stats

        def counting_get():
            nonlocal call_count
            call_count += 1
            return original_get()

        collector.get_current_stats = counting_get  # type: ignore[assignment]

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            m = ProgressMonitor(collector)
            m.start()
            time.sleep(1.5)  # allow at least one tick
            m.stop()
            assert call_count >= 1
        finally:
            sys.stdout = old_stdout

    def test_stop_without_start(self) -> None:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            m = ProgressMonitor(_make_collector())
            m.stop()  # should not raise
        finally:
            sys.stdout = old_stdout


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestProgressMonitorThreadSafety:
    def test_concurrent_send_finish_log(self) -> None:
        collector = _make_collector()

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            m = ProgressMonitor(collector)
            m.start()

            num_threads = 10
            ops_per_thread = 50
            barrier = threading.Barrier(num_threads)

            def worker(thread_id: int) -> None:
                barrier.wait()
                for i in range(ops_per_thread):
                    rid = f"t{thread_id}_r{i}"
                    m.on_request_sent(rid)
                    m.log_message(f"sent {rid}")
                    m.on_request_finished(rid, True)

            threads = [
                threading.Thread(target=worker, args=(t,)) for t in range(num_threads)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            m.stop()
            # All sends matched by finishes → in-flight == 0
            assert m._in_flight == 0
        finally:
            sys.stdout = old_stdout
