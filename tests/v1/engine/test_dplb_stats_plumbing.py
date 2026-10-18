# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DP LB snapshot plumbing (SchedulerStats extension).

Executes real source (extracted via ast) with stubbed external dependencies,
so these tests need no GPU, distributed runtime, or heavy imports.

Run with pytest, or directly:
    python tests/v1/engine/test_dplb_stats_plumbing.py
"""

import ast
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

VLLM_ROOT = Path(__file__).resolve().parents[3]
STATS_PATH = VLLM_ROOT / "vllm" / "v1" / "metrics" / "stats.py"
SCHEDULER_PATH = VLLM_ROOT / "vllm" / "v1" / "core" / "sched" / "scheduler.py"


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
    cumulative counter — the signals make_stats() publishes."""
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
