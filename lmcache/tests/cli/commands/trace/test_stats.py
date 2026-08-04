# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`ReplayStatsCollector`."""

# Standard
import json
import threading

# Third Party
import pytest

# First Party
from lmcache.cli.commands.trace._stats import (
    OpStats,
    ReplayStatsCollector,
    _percentile,
)


class TestPercentile:
    def test_empty_returns_zero(self):
        assert _percentile([], 50) == 0.0

    def test_p0_returns_min(self):
        assert _percentile([1.0, 2.0, 3.0], 0) == 1.0

    def test_p100_returns_max(self):
        assert _percentile([1.0, 2.0, 3.0], 100) == 3.0

    def test_p50_on_100_values(self):
        vals = [float(i) for i in range(1, 101)]
        # nearest-rank with floor((50/100)*100)=50 → index 50 → value 51
        assert _percentile(vals, 50) == 51.0


class TestReplayStatsCollector:
    def test_record_and_summary(self):
        s = ReplayStatsCollector()
        for latency_ms in (1.0, 2.0, 3.0, 4.0, 100.0):
            s.record("op.foo", latency_ms / 1000.0)
        summary = s.summary()
        assert "op.foo" in summary
        stats = summary["op.foo"]
        assert isinstance(stats, OpStats)
        assert stats.count == 5
        assert stats.error_count == 0
        assert stats.min_ms == pytest.approx(1.0)
        assert stats.max_ms == pytest.approx(100.0)
        # 22 = mean of {1,2,3,4,100}
        assert stats.mean_ms == pytest.approx(22.0)

    def test_records_failed_separately(self):
        s = ReplayStatsCollector()
        s.record("op.foo", 0.001, failed=False)
        s.record("op.foo", 0.001, failed=True)
        summary = s.summary()
        assert summary["op.foo"].count == 1
        assert summary["op.foo"].error_count == 1

    def test_error_only_bucket(self):
        """A qualname that only ever failed still appears with zero
        latency stats."""
        s = ReplayStatsCollector()
        s.record("op.failing", 0.0, failed=True)
        summary = s.summary()
        assert summary["op.failing"].count == 0
        assert summary["op.failing"].error_count == 1
        assert summary["op.failing"].mean_ms == 0.0

    def test_duration(self):
        s = ReplayStatsCollector()
        assert s.total_duration_s() == 0.0
        s.mark_start(100.0)
        s.mark_end(105.5)
        assert s.total_duration_s() == pytest.approx(5.5)

    def test_export_csv(self, tmp_path):
        s = ReplayStatsCollector()
        s.record("op.a", 0.001)
        s.record("op.a", 0.002)
        s.record("op.b", 0.003)
        path = str(tmp_path / "out.csv")
        s.export_csv(path)
        with open(path) as f:
            lines = f.read().splitlines()
        assert lines[0].split(",") == [
            "qualname",
            "count",
            "errors",
            "mean_ms",
            "p50_ms",
            "p90_ms",
            "p99_ms",
            "min_ms",
            "max_ms",
        ]
        # two op rows sorted alphabetically
        assert lines[1].startswith("op.a,")
        assert lines[2].startswith("op.b,")

    def test_export_json(self, tmp_path):
        s = ReplayStatsCollector()
        s.mark_start(0.0)
        s.record("op.a", 0.001)
        s.mark_end(1.0)
        path = str(tmp_path / "out.json")
        s.export_json(path)
        with open(path) as f:
            data = json.load(f)
        assert data["duration_s"] == pytest.approx(1.0)
        assert "op.a" in data["ops"]
        assert data["ops"]["op.a"]["count"] == 1

    def test_thread_safety(self):
        s = ReplayStatsCollector()

        def worker():
            for _ in range(100):
                s.record("op.x", 0.0001)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert s.summary()["op.x"].count == 400
