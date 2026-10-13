# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior tests for the DP LB superlinear in-flight scoring.

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


DPLBAsyncMPClient = _build_dplb_class()


def _make_client(snapshots):
    client = DPLBAsyncMPClient.__new__(DPLBAsyncMPClient)
    num_engines = len(snapshots)
    client.core_engines = list(range(num_engines))
    client.client_count = 1
    client.engine_inflight = Counter()
    client.reqs_in_flight = {}
    client.lb_engines = [list(snapshot) for snapshot in snapshots]
    client.eng_start_index = 0
    return client


def _request(i):
    return SimpleNamespace(
        data_parallel_rank=None, pooling_params=None, request_id=f"req-{i}"
    )


def _route(client, n):
    return [client.get_core_engine_for_request(_request(i)) for i in range(n)]


def test_burst_no_inversion():
    """20 in-flight on the empty engine: the superlinear score (41.6)
    exceeds the queued peer's base (30) and routing sheds; linear main
    (20 < 30) would keep picking the empty engine."""
    client = _make_client(
        [
            [0, 0, 0.0],
            [30, 0, 0.0],
        ]
    )
    client.engine_inflight[0] = 20
    assert _route(client, 1) == [1]


def test_small_burst_round_robin():
    """In-flight <= 10 keeps exact round-robin (identical to main)."""
    client = _make_client([[0, 0, 0.0]] * 3)
    picks = _route(client, 30)
    assert Counter(picks) == {0: 10, 1: 10, 2: 10}


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
