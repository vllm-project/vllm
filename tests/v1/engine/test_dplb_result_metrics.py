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
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

VLLM_ROOT = Path(__file__).resolve().parents[3]
CORE_CLIENT_PATH = VLLM_ROOT / "vllm" / "v1" / "engine" / "core_client.py"
STATS_PATH = VLLM_ROOT / "vllm" / "v1" / "metrics" / "stats.py"

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
    client._last_snapshot_time = 0.0
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
    client = _make_client(
        [[0, 0, 0.0, 0.0, 0]] * 3, result_metrics=False
    )
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
    """DP1 penalty hits the cap base_est * 0.5 + 500 = 506.5; DP0 stays 0."""
    client = _production_snapshot_client()
    assert client._static_scores[0] == 0.0
    assert client._static_scores[1] == 506.5


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
