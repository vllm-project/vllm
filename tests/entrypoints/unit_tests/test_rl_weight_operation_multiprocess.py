# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-process aggregation of the weight-operation metrics.

These use real child processes plus a real ``PROMETHEUS_MULTIPROC_DIR``, because a
single-process ``CollectorRegistry`` cannot show whether
``multiprocess_mode="livesum"`` and the mmap-backed values behave correctly.

They rely on POSIX semantics for ``mark_process_dead`` (on Windows the mmap files
are locked), so they are skipped there.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.cpu_test,
    pytest.mark.skip_global_cleanup,
    pytest.mark.skipif(
        sys.platform == "win32", reason="multiprocess registry needs POSIX semantics"
    ),
]

PREFIX = "vllm:rl_weight_update_"
REPO_ROOT = Path(__file__).resolve().parents[3]

# Record one successful operation, then hold the gauge so the parent can scrape a
# non-zero in-flight value while the child is still alive.
CHILD_RECORD = """
import sys
from vllm.entrypoints.serve.dev.rlhf.metrics import weight_operation_metrics

operation = sys.argv[1]
metrics = weight_operation_metrics()
with metrics.record(operation):
    metrics.in_flight.labels(operation=operation).inc()
    print("ready", flush=True)
    sys.stdin.readline()
"""


def _child_env(multiproc_dir: str) -> dict[str, str]:
    pythonpath = os.pathsep.join(
        [str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return {
        **os.environ,
        "PROMETHEUS_MULTIPROC_DIR": multiproc_dir,
        "PYTHONPATH": pythonpath,
        "PYTHONUNBUFFERED": "1",
    }


def _spawn_recorder(multiproc_dir: str, operation: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-c", CHILD_RECORD, operation],
        env=_child_env(multiproc_dir),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ready = proc.stdout.readline() if proc.stdout else ""
    if ready.strip() != "ready":
        stderr = proc.stderr.read() if proc.stderr else ""
        proc.kill()
        raise AssertionError(f"recorder child failed to start: {ready!r}\n{stderr}")
    return proc


def _stop_recorder(proc: subprocess.Popen) -> int:
    pid = proc.pid
    if proc.stdin is not None:
        try:
            proc.stdin.write("\n")
            proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass
    try:
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
        if proc.stdin is not None:
            proc.stdin.close()
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
    return pid


def _scrape(monkeypatch, multiproc_dir: str) -> str:
    """Scrape the multiprocess registry the way ``/metrics`` does."""
    from prometheus_client import CollectorRegistry, generate_latest, multiprocess

    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", multiproc_dir)
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return generate_latest(registry).decode()


def _sample(text: str, name: str, labels: str) -> float | None:
    target = f"{name}{{{labels}}}"
    for line in text.splitlines():
        if line.startswith(target + " "):
            return float(line.rsplit(" ", 1)[1])
    return None


def test_multiprocess_aggregates_operations_duration_and_in_flight(
    tmp_path, monkeypatch
):
    multiproc_dir = str(tmp_path)
    children = [
        _spawn_recorder(multiproc_dir, "update"),
        _spawn_recorder(multiproc_dir, "update"),
    ]
    try:
        text = _scrape(monkeypatch, multiproc_dir)
    finally:
        for child in children:
            _stop_recorder(child)

    # Two processes each recorded one operation: counters and histogram counts
    # sum, and livesum makes the held gauges add up.
    assert (
        _sample(
            text, PREFIX + "operations_total", 'operation="update",status="success"'
        )
        == 2
    )
    assert (
        _sample(text, PREFIX + "operation_duration_seconds_count", 'operation="update"')
        == 2
    )
    assert _sample(text, PREFIX + "operations_in_flight", 'operation="update"') == 2


def test_different_operations_stay_separate_across_processes(tmp_path, monkeypatch):
    multiproc_dir = str(tmp_path)
    update = _spawn_recorder(multiproc_dir, "update")
    finish = _spawn_recorder(multiproc_dir, "finish")
    try:
        text = _scrape(monkeypatch, multiproc_dir)
    finally:
        _stop_recorder(update)
        _stop_recorder(finish)

    assert (
        _sample(
            text, PREFIX + "operations_total", 'operation="update",status="success"'
        )
        == 1
    )
    assert (
        _sample(
            text, PREFIX + "operations_total", 'operation="finish",status="success"'
        )
        == 1
    )
    assert _sample(text, PREFIX + "operations_in_flight", 'operation="update"') == 1
    assert _sample(text, PREFIX + "operations_in_flight", 'operation="finish"') == 1
    # The operation label is what separates them; summing is not meaningful.
    assert _sample(text, PREFIX + "operations_total", 'status="success"') is None


def test_mark_process_dead_clears_stale_in_flight(tmp_path, monkeypatch):
    from prometheus_client import multiprocess

    multiproc_dir = str(tmp_path)
    alive = _spawn_recorder(multiproc_dir, "update")
    dead = _spawn_recorder(multiproc_dir, "finish")
    dead_pid = _stop_recorder(dead)
    try:
        before = _scrape(monkeypatch, multiproc_dir)
        assert (
            _sample(before, PREFIX + "operations_in_flight", 'operation="finish"') == 1
        )

        multiprocess.mark_process_dead(dead_pid, multiproc_dir)

        after = _scrape(monkeypatch, multiproc_dir)
        # The dead process's held gauge is gone instead of lingering forever.
        assert (
            _sample(after, PREFIX + "operations_in_flight", 'operation="finish"')
            is None
        )
        # The live process is unaffected.
        assert (
            _sample(after, PREFIX + "operations_in_flight", 'operation="update"') == 1
        )
    finally:
        _stop_recorder(alive)


def test_metrics_are_created_lazily_after_multiprocess_setup(tmp_path):
    """Importing the module must not create collectors; first use must."""
    multiproc_dir = str(tmp_path)
    code = (
        "from vllm.entrypoints.serve.dev.rlhf import metrics;"
        "print('before', metrics._metrics is not None);"
        "metrics.weight_operation_metrics();"
        "print('after', metrics._metrics is not None)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=_child_env(multiproc_dir),
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "before False" in out.stdout
    assert "after True" in out.stdout

    files = os.listdir(multiproc_dir)
    assert any(name.startswith("counter_") for name in files), files
    assert any(name.startswith("gauge_livesum_") for name in files), files
    assert any(name.startswith("histogram_") for name in files), files
