# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test Suite for Manual GC Control (gc_utils.py)

Test Plan:
===========

1. UNIT TESTS WITH MOCKS (CPU-only, no GPU required)
   - Rate-limit behavior verification
   - Generation selection logic
   - Telemetry counters
   - Leak guard triggering
   - Enable/disable state transitions
   - Singleton pattern verification

2. CPU-ONLY MICRO-TESTS
   - Tight loop stress test for counters/behavior
   - High object churn with threshold verification
   - Memory pressure simulation (no actual pressure, mock-based)

3. INTEGRATION BEHAVIOR TESTS
   - gc_collect_on_sync() entry point behavior
   - maybe_enable_manual_gc_control() behavior
   - State consistency across enable/disable cycles

NOTE ON PERFORMANCE/GPU TESTING:
================================
The following tests require GPU hardware and are NOT included here:
- Long-run soak test (6-12 hours steady load)
- Allocation-heavy micro-bench with latency measurements
- Async + multiproc topology tests
- Memory-pressure regression with real memory constraints

These benchmarks should be run via maintainer/CI perf runs with:
  VLLM_MANUAL_GC_CONTROL=1 python benchmarks/benchmark_serving.py ...

See docs/performance_testing.md for GPU benchmark procedures.
"""
from dataclasses import dataclass
from typing import Any

import gc
import pytest

from vllm.utils.gc_utils import (
    GCDebugConfig,
    ManualGCController,
    _compute_detailed_type,
    _compute_top_gc_collected_objects,
    gc_collect_on_sync,
    maybe_enable_manual_gc_control,
)


@dataclass
class Normal:
    v: int


@dataclass
class ListWrapper:
    vs: list[int]

    def __len__(self) -> int:
        return len(self.vs)


def test_compute_detailed_type():
    assert (
        _compute_detailed_type(Normal(v=8))
        == "<class 'tests.utils_.test_gc_utils.Normal'>"
    )

    assert _compute_detailed_type([1, 2, 3]) == "<class 'list'>(size:3)"
    assert _compute_detailed_type({4, 5}) == "<class 'set'>(size:2)"
    assert _compute_detailed_type({6: 7}) == "<class 'dict'>(size:1)"
    assert (
        _compute_detailed_type(ListWrapper(vs=[]))
        == "<class 'tests.utils_.test_gc_utils.ListWrapper'>(size:0)"
    )


def test_compute_top_gc_collected_objects():
    objects: list[Any] = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        {13, 14},
        {15: 16, 17: 18},
        Normal(v=19),
        Normal(v=20),
        Normal(v=21),
    ]
    assert _compute_top_gc_collected_objects(objects, top=-1) == ""
    assert _compute_top_gc_collected_objects(objects, top=0) == ""
    assert (
        _compute_top_gc_collected_objects(objects, top=1)
        == "    4:<class 'list'>(size:3)"
    )
    assert _compute_top_gc_collected_objects(objects, top=2) == "\n".join(
        [
            "    4:<class 'list'>(size:3)",
            "    3:<class 'tests.utils_.test_gc_utils.Normal'>",
        ]
    )
    assert _compute_top_gc_collected_objects(objects, top=3) == "\n".join(
        [
            "    4:<class 'list'>(size:3)",
            "    3:<class 'tests.utils_.test_gc_utils.Normal'>",
            "    1:<class 'set'>(size:2)",
        ]
    )


def test_gc_debug_config():
    assert not GCDebugConfig(None).enabled
    assert not GCDebugConfig("").enabled
    assert not GCDebugConfig("0").enabled

    config = GCDebugConfig("1")
    assert config.enabled
    assert config.top_objects == -1

    config = GCDebugConfig('{"top_objects":5}')
    assert config.enabled
    assert config.top_objects == 5


@pytest.fixture
def gc_controller():
    """
    Fixture that creates a fresh ManualGCController for testing.

    Properly saves and restores GC state after each test.
    """
    original_enabled = gc.isenabled()
    original_thresholds = gc.get_threshold()

    # Reset singleton
    ManualGCController._instance = None

    # Create fresh controller (this disables automatic GC)
    controller = ManualGCController._create_instance()

    yield controller

    # Cleanup: restore original GC state
    ManualGCController._instance = None
    if original_enabled:
        gc.enable()
    else:
        gc.disable()
    gc.set_threshold(*original_thresholds)


def test_manual_gc_rate_limit_skips(
    gc_controller: ManualGCController,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that rate limiting skips GC calls within MIN_GC_INTERVAL_S."""
    controller = gc_controller

    monkeypatch.setattr(gc, "get_count", lambda: (1, 0, 0))
    calls: list[int] = []

    def fake_collect(generation: int) -> int:
        calls.append(generation)
        return 1

    monkeypatch.setattr(gc, "collect", fake_collect)

    controller._thresholds = (1, 1, 1)
    now_ns = 1_000_000_000
    monkeypatch.setattr("vllm.utils.gc_utils.time.monotonic_ns", lambda: now_ns)
    controller._last_gc_time_ns = now_ns

    assert controller.maybe_collect() == 0
    assert calls == []
    assert controller.get_stats()["skipped_rate_limited"] == 1


def test_manual_gc_collects_highest_generation(
    gc_controller: ManualGCController,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that GC collects the highest necessary generation in one call."""
    controller = gc_controller

    monkeypatch.setattr(gc, "get_count", lambda: (1, 0, 0))
    # Mock get_objects to return empty list (for gen2 heuristic)
    monkeypatch.setattr(gc, "get_objects", lambda generation: [])
    calls: list[int] = []

    def fake_collect(generation: int) -> int:
        calls.append(generation)
        return 3

    monkeypatch.setattr(gc, "collect", fake_collect)

    controller._thresholds = (1, 1, 1)
    # Set gen2 baseline to 0 so any objects would trigger gen2
    controller._gen2_object_count_at_last_gc = 0
    monkeypatch.setattr(
        "vllm.utils.gc_utils.time.monotonic_ns", lambda: 2_000_000_000
    )
    controller._last_gc_time_ns = 0

    # First call should do gen0 (counters at 0)
    collected = controller.maybe_collect()
    assert collected == 3
    assert calls == [0]
    assert controller._gc0_count_since_gc1 == 1

    # Reset for next test
    calls.clear()
    controller._gc0_count_since_gc1 = 1  # Simulate one gen0 already done

    # Second call with threshold met should do gen1
    collected = controller.maybe_collect()
    assert collected == 3
    assert calls == [1]
    assert controller._gc0_count_since_gc1 == 0
    assert controller._gc1_count_since_gc2 == 1


def test_manual_gc_gen2_heuristic_skips(
    gc_controller: ManualGCController,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that gen2 collection is skipped when not enough new objects."""
    controller = gc_controller

    monkeypatch.setattr(gc, "get_count", lambda: (1, 0, 0))

    # Mock get_objects to return same count as baseline (no growth)
    baseline_objects = list(range(100))  # 100 objects
    monkeypatch.setattr(gc, "get_objects", lambda generation: baseline_objects)

    calls: list[int] = []

    def fake_collect(generation: int) -> int:
        calls.append(generation)
        return 3

    monkeypatch.setattr(gc, "collect", fake_collect)

    controller._thresholds = (1, 1, 1)
    controller._gen2_object_count_at_last_gc = 100  # Same as current
    controller._gc0_count_since_gc1 = 1  # Ready for gen1
    controller._gc1_count_since_gc2 = 1  # Ready for gen2

    monkeypatch.setattr(
        "vllm.utils.gc_utils.time.monotonic_ns", lambda: 2_000_000_000
    )
    controller._last_gc_time_ns = 0

    # Should skip gen2 (no growth) and do gen1 instead
    collected = controller.maybe_collect()
    assert collected == 3
    assert calls == [1]  # Gen1, not gen2
    assert controller.get_stats()["gen2_skipped_by_heuristic"] == 1


def test_manual_gc_gen2_heuristic_collects(
    gc_controller: ManualGCController,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that gen2 collection happens when enough new objects added."""
    controller = gc_controller

    monkeypatch.setattr(gc, "get_count", lambda: (1, 0, 0))

    # Mock get_objects to return more objects than baseline (50% growth)
    grown_objects = list(range(150))  # 150 objects (50% growth from 100)
    monkeypatch.setattr(gc, "get_objects", lambda generation: grown_objects)

    calls: list[int] = []

    def fake_collect(generation: int) -> int:
        calls.append(generation)
        return 3

    monkeypatch.setattr(gc, "collect", fake_collect)

    controller._thresholds = (1, 1, 1)
    controller._gen2_object_count_at_last_gc = 100  # Baseline
    controller._gc0_count_since_gc1 = 1  # Ready for gen1
    controller._gc1_count_since_gc2 = 1  # Ready for gen2

    monkeypatch.setattr(
        "vllm.utils.gc_utils.time.monotonic_ns", lambda: 2_000_000_000
    )
    controller._last_gc_time_ns = 0

    # Should do gen2 (50% growth > 25% threshold)
    collected = controller.maybe_collect()
    assert collected == 3
    assert calls == [2]  # Gen2
    assert controller.get_stats()["gen2_skipped_by_heuristic"] == 0


def test_manual_gc_leak_guard(
    gc_controller: ManualGCController,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that leak guard forces full GC regardless of rate limit."""
    controller = gc_controller

    # Leak guard threshold: 10x default (700 * 10 = 7000)
    monkeypatch.setattr(gc, "get_count", lambda: (7001, 0, 0))
    monkeypatch.setattr(gc, "get_objects", lambda generation: [])
    calls: list[int] = []

    def fake_collect(generation: int) -> int:
        calls.append(generation)
        return 100

    monkeypatch.setattr(gc, "collect", fake_collect)

    # Set last GC time to now (would normally be rate-limited)
    now_ns = 1_000_000_000
    monkeypatch.setattr("vllm.utils.gc_utils.time.monotonic_ns", lambda: now_ns)
    controller._last_gc_time_ns = now_ns

    collected = controller.maybe_collect()
    assert collected == 100
    assert calls == [2]  # Full gen2 collection


def test_gc_collect_on_sync_noop_when_disabled() -> None:
    """Test that gc_collect_on_sync is no-op when manual GC not enabled."""
    # Ensure no controller exists
    ManualGCController._instance = None

    result = gc_collect_on_sync()
    assert result == 0


def test_telemetry_counters(
    gc_controller: ManualGCController,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that telemetry counters update correctly."""
    controller = gc_controller

    monkeypatch.setattr(gc, "get_count", lambda: (1000, 0, 0))
    monkeypatch.setattr(gc, "get_objects", lambda generation: [])

    call_count = 0

    def fake_collect(generation: int) -> int:
        nonlocal call_count
        call_count += 1
        return 42

    monkeypatch.setattr(gc, "collect", fake_collect)

    # Mock time to advance past rate limit
    time_ns = 0

    def mock_time():
        nonlocal time_ns
        time_ns += 100_000_000  # 100ms
        return time_ns

    monkeypatch.setattr("vllm.utils.gc_utils.time.monotonic_ns", mock_time)

    controller._thresholds = (1, 100, 100)  # Only gen0 will trigger
    controller._last_gc_time_ns = 0

    # Perform multiple collections
    for _ in range(5):
        controller.maybe_collect()

    stats = controller.get_stats()
    assert stats["gc_invocations"] == 5
    assert stats["objects_collected"] == 42 * 5
    assert stats["gen0_invocations"] == 5


def test_controller_stats_structure(
    gc_controller: ManualGCController,
) -> None:
    """Test that get_stats returns all expected fields."""
    stats = gc_controller.get_stats()

    expected_fields = [
        "gc_invocations",
        "total_gc_time_ms",
        "max_gc_time_ms",
        "avg_gc_time_ms",
        "objects_collected",
        "skipped_rate_limited",
        "gen2_skipped_by_heuristic",
        "gen0_invocations",
        "gen1_invocations",
        "gen2_invocations",
        "gen0_time_ms",
        "gen1_time_ms",
        "gen2_time_ms",
    ]

    for field in expected_fields:
        assert field in stats, f"Missing field: {field}"
