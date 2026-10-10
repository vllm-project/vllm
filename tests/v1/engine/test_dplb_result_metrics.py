# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior tests for the DP LB result-metric scoring.

Executes the real ``DPLBAsyncMPClient`` class source (extracted via ast from
vllm/v1/engine/core_client.py) with stubbed external dependencies, so these
tests need no GPU, distributed runtime, or heavy imports.

Run with pytest, or directly:
    python tests/v1/engine/test_dplb_result_metrics.py
"""

import ast
import sys
import textwrap
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

VLLM_ROOT = Path(__file__).resolve().parents[3]
CORE_CLIENT_PATH = VLLM_ROOT / "vllm" / "v1" / "engine" / "core_client.py"
STATS_PATH = VLLM_ROOT / "vllm" / "v1" / "metrics" / "stats.py"
SCHEDULER_PATH = VLLM_ROOT / "vllm" / "v1" / "core" / "sched" / "scheduler.py"

# Fake monotonic clock shared by all tests (injected into the exec'd class
# globals as the ``time`` module).
_CLOCK = {"now": 1000.0}


def _extract_class_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            # Include class-level decorators (e.g. @dataclass).
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            lines = source.splitlines(keepends=True)
            return "".join(lines[start - 1 : node.end_lineno])
    raise AssertionError(f"class {name} not found in {path}")


def _extract_method_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            lines = source.splitlines(keepends=True)
            return textwrap.dedent("".join(lines[node.lineno - 1 : node.end_lineno]))
    raise AssertionError(f"method {name} not found in {path}")


def _build_dplb_class() -> type:
    class DPAsyncMPClient:
        def get_core_engine_for_request(self, request):
            return self.core_engine

        def _apply_snapshot_metrics(self) -> None:
            pass

    class _Stub:
        pass

    class _EnvStub:
        VLLM_DP_LB_RESULT_METRICS = True

    class _ReconfigureRankType:
        KEEP_CURRENT_RANK = 0
        SHUTDOWN_CURRENT_RANK = 1

    namespace: dict[str, Any] = {
        "DPAsyncMPClient": DPAsyncMPClient,
        "EngineCoreRequest": _Stub,
        "EngineCoreOutputs": _Stub,
        "EngineIdentity": int,
        "VllmConfig": _Stub,
        "Executor": _Stub,
        "BaseRenderer": _Stub,
        "Any": Any,
        "Counter": Counter,
        "defaultdict": __import__("collections").defaultdict,
        "asyncio": __import__("asyncio"),
        "envs": _EnvStub,
        "get_late_interaction_engine_index": lambda *args, **kwargs: None,
        "sys": sys,
        "time": SimpleNamespace(monotonic=lambda: _CLOCK["now"]),
        # Names below are only referenced at call time, but cheap to stub.
        "ReconfigureRankType": _ReconfigureRankType,
        "ReconfigureDistributedRequest": _Stub,
        "ElasticScalingCache": _Stub,
        "CoreEngineActorManager": _Stub,
        "EEPNotificationType": _Stub,
        "EEP_NOTIFICATION_CALL_ID": "eep",
        "VLLM_ENGINE_READY_TIMEOUT_S": 1.0,
        "logger": SimpleNamespace(
            info=lambda *a, **k: None, debug=lambda *a, **k: None
        ),
    }
    exec(
        compile(
            _extract_class_source(CORE_CLIENT_PATH, "DPLBAsyncMPClient"),
            str(CORE_CLIENT_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace["DPLBAsyncMPClient"]


def _build_scheduler_stats_class() -> type:
    class _Stub:
        def __init__(self, *args, **kwargs) -> None:
            pass

    namespace: dict[str, Any] = {
        "dataclass": __import__("dataclasses").dataclass,
        "field": __import__("dataclasses").field,
        "Any": Any,
        "PrefixCacheStats": _Stub,
        "SchedulerIterationDetails": _Stub,
        "KVCacheEvictionEvent": _Stub,
    }
    exec(
        compile(
            _extract_class_source(STATS_PATH, "SchedulerStats"),
            str(STATS_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace["SchedulerStats"]


DPLBAsyncMPClient = _build_dplb_class()


def _make_client(snapshots, client_count=1, result_metrics=True):
    client = DPLBAsyncMPClient.__new__(DPLBAsyncMPClient)
    num_engines = len(snapshots)
    client.core_engines = list(range(num_engines))
    client.client_count = client_count
    client.engine_inflight = Counter()
    client.reqs_in_flight = {}
    client.lb_engines = [list(snapshot) for snapshot in snapshots]
    client._dp_lb_result_metrics = result_metrics
    client._static_scores = [0.0] * num_engines
    client._smoothed_queue_time = [None] * num_engines
    client._last_preempted_total = [0] * num_engines
    client._preempt_delta_acc = [0] * num_engines
    client._preempt_window_elapsed = 0.0
    client._last_preempt_rates = [0.0] * num_engines
    client._last_snapshot_time = 0.0
    client._smoothed_baseline = None
    client.eng_start_index = 0
    return client


def _request(i):
    return SimpleNamespace(
        data_parallel_rank=None, pooling_params=None, request_id=f"req-{i}"
    )


def _route(client, n):
    return [client.get_core_engine_for_request(_request(i)) for i in range(n)]


def _reset_clock():
    _CLOCK["now"] = 1000.0


def test_flag_off_burst_round_robin():
    """Flag off: burst on empty snapshots spreads like main (round-robin)."""
    client = _make_client([[0, 0, 0.0, 0.0, 0]] * 3, result_metrics=False)
    picks = _route(client, 30)
    assert Counter(picks) == {0: 10, 1: 10, 2: 10}


def test_flag_on_burst_round_robin_preserved():
    """Flag on: small bursts (in-flight <= 10) keep exact round-robin."""
    client = _make_client([[0, 0, 0.0, 0.0, 0]] * 3, result_metrics=True)
    picks = _route(client, 30)
    assert Counter(picks) == {0: 10, 1: 10, 2: 10}


def _production_snapshot_client():
    """Client fed the production snapshot (waiting 0/8, running 3/5, KV
    8%/92%, wait-proxy 0s/17s, preempts 0 -> 12 in one 100ms window)."""
    _reset_clock()
    client = _make_client(
        [
            [0, 3, 0.08, 0.0, 0],
            [8, 5, 0.92, 17.0, 0],
        ],
        result_metrics=True,
    )
    client._apply_snapshot_metrics()  # seeds EMA and preempt counters
    _CLOCK["now"] += 0.1
    client.lb_engines[1][4] = 12
    client._apply_snapshot_metrics()  # preempt rate = 12 / 0.1s = 120/s
    return client


def test_production_snapshot_penalty_capped():
    """Result-metric penalties hit the cap base_est * 0.5 + 500 = 506.5 and
    the uncapped KV-pressure term (8 * 6 * 0.42 = 20.16) adds on top; DP0
    stays 0."""
    client = _production_snapshot_client()
    assert client._static_scores[0] == 0.0
    assert abs(client._static_scores[1] - 526.66) < 1e-6


def test_kv_pressure_term_not_capped():
    """The KV-pressure term is main's own signal and stays uncapped; only
    the result-metric penalties are capped. waiting=500 at kv=1.0 gives
    500 * 6 * 0.5 = 1500, above the base_est * 0.5 + 500 = 750 cap."""
    _reset_clock()
    client = _make_client(
        [
            [0, 0, 0.0, 0.0, 0],
            [500, 0, 1.0, 0.0, 0],
        ],
        result_metrics=True,
    )
    client._apply_snapshot_metrics()
    assert client._static_scores[0] == 0.0
    assert client._static_scores[1] == 1500.0


def test_queue_wait_baseline_hysteresis():
    """The cross-engine baseline is slow-EMA smoothed (beta=0.1): a sudden
    jump in the healthiest engine's queue wait moves the baseline only 10%
    per snapshot instead of instantly, so other engines' ratio penalties
    do not collapse or spike within one snapshot."""
    _reset_clock()
    client = _make_client(
        [
            [0, 0, 0.0, 0.0, 0],
            [0, 0, 0.0, 10.0, 0],
        ],
        result_metrics=True,
    )
    client._apply_snapshot_metrics()  # baseline seeds at the 1.0 floor
    assert client._smoothed_baseline == 1.0
    # The healthiest engine's queue wait jumps 0 -> 10s; its EMA lands at
    # 3.0, the raw min baseline at 3.0, and the smoothed baseline moves
    # 10% of the way: 0.1 * 3.0 + 0.9 * 1.0 = 1.2.
    client.lb_engines[0][3] = 10.0
    _CLOCK["now"] += 0.1
    client._apply_snapshot_metrics()
    assert abs(client._smoothed_baseline - 1.2) < 1e-9


def test_preempt_rate_survives_rapid_snapshots():
    """Storm snapshots (<50ms apart) accumulate instead of dropping deltas.

    Two preemptions land across five 30ms intervals; under the per-interval
    gate every delta was dropped (each elapsed < 50ms) and the penalty
    stayed 0. The accumulated window publishes rate 1/0.12 = 8.33/s on the
    fourth snapshot and the rate persists, so the penalty reaches
    30 * (8.33 - 5) = 100.
    """
    _reset_clock()
    client = _make_client(
        [
            [0, 3, 0.5, 0.0, 0],
            [0, 3, 0.5, 0.0, 0],
        ],
        result_metrics=True,
    )
    client._apply_snapshot_metrics()  # seed baseline
    for i in range(5):
        _CLOCK["now"] += 0.03
        if i >= 3:
            client.lb_engines[1][4] += 1
        client._apply_snapshot_metrics()
    assert client._static_scores[0] == 0.0
    assert abs(client._static_scores[1] - 100.0) < 1e-6


def test_production_snapshot_burst_no_inversion():
    """30 in-flight on the healthy rank never flips to the overloaded rank."""
    client = _production_snapshot_client()
    picks = _route(client, 30)
    assert picks == [0] * 30


def test_global_degradation_penalty_zero():
    """Uniform degradation: queue-wait ratios collapse, penalties stay 0."""
    _reset_clock()
    client = _make_client(
        [
            [0, 3, 0.5, 20.0, 0],
            [0, 3, 0.5, 22.0, 0],
        ],
        result_metrics=True,
    )
    client._apply_snapshot_metrics()
    _CLOCK["now"] += 0.1
    client._apply_snapshot_metrics()
    assert client._static_scores == [0.0, 0.0]


def test_flag_off_snapshot_refresh_noop():
    """Flag off: snapshot refresh never populates static penalties."""
    _reset_clock()
    client = _make_client(
        [
            [0, 3, 0.9, 30.0, 50],
            [0, 3, 0.9, 0.0, 0],
        ],
        result_metrics=False,
    )
    client._apply_snapshot_metrics()
    _CLOCK["now"] += 0.1
    client.lb_engines[1][4] = 60
    client._apply_snapshot_metrics()
    assert client._static_scores == [0.0, 0.0]
    assert Counter(_route(client, 6)) == {0: 3, 1: 3}


def test_flag_off_superlinear_inflight_always_active():
    """Flag off: the superlinear in-flight term stays active.

    It is an inversion bugfix, not part of the result-metric feature.
    With 20 in-flight on the empty engine its score (10 + 10^1.5 = 41.6)
    exceeds the queued peer's base (30) and routing sheds; main (linear,
    20 < 30) would keep picking the empty engine.
    """
    client = _make_client(
        [
            [0, 0, 0.0, 0.0, 0],
            [30, 0, 0.0, 0.0, 0],
        ],
        result_metrics=False,
    )
    client.engine_inflight[0] = 20
    assert _route(client, 1) == [1]


def test_flag_off_linear_inflight_up_to_ten():
    """Flag off: at in-flight <= 10 scoring matches main exactly."""
    client = _make_client(
        [
            [0, 0, 0.0, 0.0, 0],
            [20, 0, 0.0, 0.0, 0],
        ],
        result_metrics=False,
    )
    client.engine_inflight[0] = 10
    assert _route(client, 1) == [0]


def test_superlinear_inflight_drains_queued_engine_early():
    """With a queued peer (base 40) the burst flips at in-flight 20, not 41.

    Linear in-flight (main) would keep all 25 requests on the empty engine;
    the superlinear term starts shedding to the queued engine at 20.
    """
    _reset_clock()
    client = _make_client(
        [
            [0, 0, 0.0, 0.0, 0],
            [40, 0, 0.0, 0.0, 0],
        ],
        result_metrics=True,
    )
    client._apply_snapshot_metrics()
    assert client._static_scores == [0.0, 0.0]
    picks = _route(client, 25)
    assert picks[:20] == [0] * 20
    assert picks[20] == 1
    assert Counter(picks)[1] >= 3


def test_scheduler_stats_backward_compatible():
    """New SchedulerStats fields default to 0; old constructors still work."""
    scheduler_stats_cls = _build_scheduler_stats_class()
    stats = scheduler_stats_cls()
    assert stats.mean_queue_time == 0.0
    assert stats.preempted_total == 0
    stats = scheduler_stats_cls(num_running_reqs=1, num_waiting_reqs=2)
    assert stats.num_running_reqs == 1
    assert stats.mean_queue_time == 0.0
    assert stats.preempted_total == 0


def test_scheduler_stats_getters():
    """get_mean_queue_time averages the queue age of waiting + skipped
    requests (0.0 when empty) and get_preempted_count returns the
    cumulative counter — the signals make_stats() now publishes."""
    src = "\n".join(
        _extract_method_source(SCHEDULER_PATH, name)
        for name in ("get_mean_queue_time", "get_preempted_count")
    )
    namespace: dict[str, Any] = {
        "time": SimpleNamespace(time=lambda: 100.0),
    }
    exec(compile(src, str(SCHEDULER_PATH), "exec"), namespace)
    stub = SimpleNamespace(
        waiting=[
            SimpleNamespace(arrival_time=90.0),
            SimpleNamespace(arrival_time=80.0),
        ],
        skipped_waiting=[SimpleNamespace(arrival_time=70.0)],
        total_preempted_reqs=7,
    )
    assert namespace["get_mean_queue_time"](stub) == 20.0
    assert namespace["get_preempted_count"](stub) == 7
    empty = SimpleNamespace(waiting=[], skipped_waiting=[], total_preempted_reqs=0)
    assert namespace["get_mean_queue_time"](empty) == 0.0
    assert namespace["get_preempted_count"](empty) == 0


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
