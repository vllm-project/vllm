# SPDX-License-Identifier: Apache-2.0
"""Tests for bench engine stats module."""

# Standard
import csv
import json
import threading
import time

# First Party
from lmcache.cli.commands.bench.engine_bench.config import EngineBenchConfig
from lmcache.cli.commands.bench.engine_bench.stats import (
    AggregatedStats,
    FinalStats,
    RequestResult,
    StatsCollector,
    _percentile,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    request_id: str = "req_0",
    successful: bool = True,
    ttft: float = 0.3,
    request_latency: float = 2.0,
    num_input_tokens: int = 10000,
    num_output_tokens: int = 128,
    decode_speed: float = 48.0,
    error: str = "",
) -> RequestResult:
    now = time.time()
    return RequestResult(
        request_id=request_id,
        successful=successful,
        ttft=ttft,
        request_latency=request_latency,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        decode_speed=decode_speed,
        submit_time=now,
        first_token_time=now + ttft,
        finish_time=now + request_latency,
        error=error,
    )


def _make_config() -> EngineBenchConfig:
    return EngineBenchConfig(
        engine_url="http://localhost:8000",
        model="test-model",
        workload="long-doc-qa",
        kv_cache_volume_gb=100.0,
        tokens_per_gb_kvcache=50000,
        seed=42,
        output_dir=".",
        export_csv=True,
        export_json=True,
        quiet=False,
    )


# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_empty_list(self) -> None:
        assert _percentile([], 50) == 0.0

    def test_single_element(self) -> None:
        assert _percentile([5.0], 50) == 5.0

    def test_p50_even_count(self) -> None:
        # k = (4-1) * 50/100 = 1.5 → lerp(2, 3, 0.5) = 2.5
        assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5

    def test_p50_odd_count(self) -> None:
        # k = (5-1) * 50/100 = 2.0 → exact index 2 = 3.0
        assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0

    def test_p90(self) -> None:
        data = list(range(1, 101))  # 1..100
        # k = 99 * 90/100 = 89.1 → lerp(90, 91, 0.1) = 90.1
        result = _percentile([float(x) for x in data], 90)
        assert abs(result - 90.1) < 1e-9

    def test_p0_and_p100(self) -> None:
        data = [1.0, 2.0, 3.0]
        assert _percentile(data, 0) == 1.0
        assert _percentile(data, 100) == 3.0


# ---------------------------------------------------------------------------
# RequestResult
# ---------------------------------------------------------------------------


class TestRequestResult:
    def test_construction(self) -> None:
        r = _make_result(request_id="r1", ttft=0.5)
        assert r.request_id == "r1"
        assert r.ttft == 0.5
        assert r.successful is True
        assert r.error == ""

    def test_failed_result(self) -> None:
        r = _make_result(successful=False, error="timeout")
        assert r.successful is False
        assert r.error == "timeout"


# ---------------------------------------------------------------------------
# StatsCollector — basic operations
# ---------------------------------------------------------------------------


class TestStatsCollector:
    def test_empty_stats(self) -> None:
        c = StatsCollector()
        s = c.get_current_stats()
        assert s.total_requests == 0
        assert s.successful_requests == 0
        assert s.failed_requests == 0
        assert s.mean_ttft_ms == 0.0
        assert s.mean_decode_speed == 0.0

    def test_single_success(self) -> None:
        c = StatsCollector()
        c.on_request_finished(
            _make_result(ttft=0.3, decode_speed=48.0, request_latency=2.0)
        )
        s = c.get_current_stats()
        assert s.total_requests == 1
        assert s.successful_requests == 1
        assert abs(s.mean_ttft_ms - 300.0) < 1e-9
        assert abs(s.mean_decode_speed - 48.0) < 1e-9
        assert abs(s.mean_request_latency_ms - 2000.0) < 1e-9

    def test_single_failure(self) -> None:
        c = StatsCollector()
        c.on_request_finished(_make_result(successful=False, error="fail"))
        s = c.get_current_stats()
        assert s.total_requests == 1
        assert s.failed_requests == 1
        # Means stay 0 because no successful requests
        assert s.mean_ttft_ms == 0.0

    def test_multiple_results(self) -> None:
        c = StatsCollector()
        c.on_request_finished(_make_result(ttft=0.2, decode_speed=40.0))
        c.on_request_finished(_make_result(ttft=0.4, decode_speed=60.0))
        s = c.get_current_stats()
        assert s.total_requests == 2
        assert abs(s.mean_ttft_ms - 300.0) < 1e-9  # (200+400)/2
        assert abs(s.mean_decode_speed - 50.0) < 1e-9  # (40+60)/2

    def test_mixed_success_failure(self) -> None:
        c = StatsCollector()
        c.on_request_finished(_make_result(ttft=0.2, decode_speed=40.0))
        c.on_request_finished(_make_result(ttft=0.4, decode_speed=60.0))
        c.on_request_finished(_make_result(successful=False, error="err"))
        s = c.get_current_stats()
        assert s.total_requests == 3
        assert s.successful_requests == 2
        assert s.failed_requests == 1
        # Means over successful only
        assert abs(s.mean_ttft_ms - 300.0) < 1e-9

    def test_get_all_results_returns_copy(self) -> None:
        c = StatsCollector()
        c.on_request_finished(_make_result(request_id="r1"))
        results = c.get_all_results()
        results.clear()
        assert len(c.get_all_results()) == 1


# ---------------------------------------------------------------------------
# StatsCollector — final stats
# ---------------------------------------------------------------------------


class TestStatsCollectorFinalStats:
    def test_percentiles_single_result(self) -> None:
        c = StatsCollector()
        c.on_request_finished(_make_result(ttft=0.3))
        f = c.get_final_stats()
        assert isinstance(f, FinalStats)
        assert isinstance(f, AggregatedStats)
        assert abs(f.p50_ttft_ms - 300.0) < 1e-9
        assert abs(f.p90_ttft_ms - 300.0) < 1e-9
        assert abs(f.p99_ttft_ms - 300.0) < 1e-9

    def test_percentiles_multiple(self) -> None:
        c = StatsCollector()
        # 10 results with TTFT 0.1, 0.2, ..., 1.0
        for i in range(1, 11):
            c.on_request_finished(
                _make_result(
                    request_id=f"r{i}",
                    ttft=i * 0.1,
                    decode_speed=float(i * 10),
                    request_latency=float(i),
                )
            )
        f = c.get_final_stats()
        # P50 TTFT: k = 9 * 50/100 = 4.5 → lerp(500, 600, 0.5) = 550
        assert abs(f.p50_ttft_ms - 550.0) < 1e-6

    def test_final_stats_no_successful(self) -> None:
        c = StatsCollector()
        c.on_request_finished(_make_result(successful=False, error="err"))
        f = c.get_final_stats()
        assert f.total_requests == 1
        assert f.failed_requests == 1
        assert f.p50_ttft_ms == 0.0
        assert f.p90_ttft_ms == 0.0
        assert f.p99_ttft_ms == 0.0


# ---------------------------------------------------------------------------
# StatsCollector — thread safety
# ---------------------------------------------------------------------------


class TestStatsCollectorThreadSafety:
    def test_concurrent_writes(self) -> None:
        c = StatsCollector()
        num_threads = 10
        results_per_thread = 100
        barrier = threading.Barrier(num_threads)

        def writer(thread_id: int) -> None:
            barrier.wait()
            for i in range(results_per_thread):
                c.on_request_finished(_make_result(request_id=f"t{thread_id}_r{i}"))

        threads = [
            threading.Thread(target=writer, args=(t,)) for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        s = c.get_current_stats()
        assert s.total_requests == num_threads * results_per_thread

    def test_concurrent_read_write(self) -> None:
        c = StatsCollector()
        num_writes = 200
        read_count = 0
        stop_event = threading.Event()
        reader_started = threading.Event()

        def writer() -> None:
            reader_started.wait()
            for i in range(num_writes):
                c.on_request_finished(_make_result(request_id=f"r{i}"))
            stop_event.set()

        def reader() -> None:
            nonlocal read_count
            reader_started.set()
            while not stop_event.is_set():
                c.get_current_stats()
                read_count += 1

        w = threading.Thread(target=writer)
        r = threading.Thread(target=reader)
        r.start()
        w.start()
        w.join()
        r.join()

        s = c.get_current_stats()
        assert s.total_requests == num_writes
        assert read_count > 0


# ---------------------------------------------------------------------------
# StatsCollector — export
# ---------------------------------------------------------------------------


class TestStatsCollectorExport:
    def test_export_csv(self, tmp_path) -> None:
        c = StatsCollector()
        for i in range(3):
            c.on_request_finished(_make_result(request_id=f"r{i}"))
        path = str(tmp_path / "results.csv")
        c.export_csv(path)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["request_id"] == "r0"
        assert "ttft" in rows[0]
        assert "decode_speed" in rows[0]

    def test_export_csv_empty(self, tmp_path) -> None:
        c = StatsCollector()
        path = str(tmp_path / "empty.csv")
        c.export_csv(path)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 0

    def test_export_json(self, tmp_path) -> None:
        c = StatsCollector()
        for i in range(2):
            c.on_request_finished(_make_result(request_id=f"r{i}"))
        path = str(tmp_path / "summary.json")
        c.export_json(path, _make_config())

        with open(path) as f:
            data = json.load(f)

        assert "config" in data
        assert "results" in data
        assert data["config"]["engine_url"] == "http://localhost:8000"
        assert "p50_ttft_ms" in data["results"]
        assert "p90_ttft_ms" in data["results"]
        assert data["results"]["total_requests"] == 2


# ---------------------------------------------------------------------------
# StatsCollector — reset
# ---------------------------------------------------------------------------


class TestStatsCollectorReset:
    def test_reset_clears_results(self) -> None:
        c = StatsCollector()
        for i in range(3):
            c.on_request_finished(_make_result(request_id=f"r{i}"))
        assert c.get_current_stats().total_requests == 3
        c.reset()
        assert c.get_current_stats().total_requests == 0
        assert len(c.get_all_results()) == 0

    def test_reset_resets_accumulators(self) -> None:
        c = StatsCollector()
        c.on_request_finished(_make_result(ttft=0.3, decode_speed=48.0))
        c.reset()
        s = c.get_current_stats()
        assert s.mean_ttft_ms == 0.0
        assert s.mean_decode_speed == 0.0
        assert s.total_input_tokens == 0
        assert s.total_output_tokens == 0

    def test_reset_restarts_timer(self) -> None:
        c = StatsCollector()
        time.sleep(0.05)
        c.reset()
        s = c.get_current_stats()
        assert s.elapsed_time < 0.1

    def test_accumulate_after_reset(self) -> None:
        c = StatsCollector()
        c.on_request_finished(_make_result(request_id="r0"))
        c.on_request_finished(_make_result(request_id="r1"))
        c.reset()
        c.on_request_finished(
            _make_result(request_id="r2", ttft=0.5, decode_speed=30.0)
        )
        s = c.get_current_stats()
        assert s.total_requests == 1
        assert abs(s.mean_ttft_ms - 500.0) < 1e-9
        assert abs(s.mean_decode_speed - 30.0) < 1e-9

    def test_final_stats_after_reset(self) -> None:
        c = StatsCollector()
        c.on_request_finished(_make_result(request_id="r0", ttft=0.1))
        c.on_request_finished(_make_result(request_id="r1", ttft=0.9))
        c.reset()
        c.on_request_finished(_make_result(request_id="r2", ttft=0.5))
        f = c.get_final_stats()
        assert f.total_requests == 1
        assert abs(f.p50_ttft_ms - 500.0) < 1e-9

    def test_reset_thread_safe(self) -> None:
        c = StatsCollector()
        stop_event = threading.Event()
        writer_started = threading.Event()

        def writer() -> None:
            writer_started.set()
            while not stop_event.is_set():
                c.on_request_finished(_make_result())

        t = threading.Thread(target=writer)
        t.start()
        writer_started.wait()
        # Reset while writer is running
        for _ in range(10):
            c.reset()
        stop_event.set()
        t.join()
        # No crash — that's the assertion
