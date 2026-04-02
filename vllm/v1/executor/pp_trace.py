# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight per-process trace recorder for PP async scheduling debugging.

Produces Chrome Trace Event Format JSON files viewable in chrome://tracing
or https://ui.perfetto.dev. Each process (driver + each PP worker) writes its
own file; load them all together for a unified cross-rank timeline.

Enable:
    export VLLM_PP_TRACE_DIR=/tmp/vllm_pp_trace

Then run inference. Trace files appear in that directory:
    /tmp/vllm_pp_trace/vllm_trace_<pid>.json    ← one per process

How to view:
    Open chrome://tracing → Load all JSON files at once.
    OR: go to ui.perfetto.dev → Open trace file(s).

Each process row is labeled by its role (driver, pp_rank_0, pp_rank_1, …).
Spans of interest:
    driver/schedule          — scheduler.schedule() CPU overhead
    driver/dag_submit        — DAG submission latency (should be ~0)
    driver/dag_wait          — time blocked in future.result() per step
    driver/update_from_output — scheduler update after output
    worker/pp_recv           — time non-last ranks block on dist.recv (our fix)
    worker/execute_model     — actual model forward pass per PP stage
    worker/sample_tokens     — pp3 sampling time
    worker/execute_model_ray — total time in execute_model_ray per call
"""

import atexit
import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Generator

# ---------------------------------------------------------------------------
# Configuration (resolved once at module import, so workers inherit it)
# ---------------------------------------------------------------------------
_TRACE_DIR: str = os.environ.get("VLLM_PP_TRACE_DIR", "")
_ENABLED: bool = bool(_TRACE_DIR)
# Flush trace every this many steps (for both driver and workers).
_FLUSH_STEPS: int = int(os.environ.get("VLLM_PP_TRACE_FLUSH_STEPS", "100"))

# ---------------------------------------------------------------------------
# Module-level state (per process)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_events: list[dict[str, Any]] = []
_pid: int = os.getpid()
_process_name: str = f"pid_{_pid}"  # overridden by set_process_name()


def is_enabled() -> bool:
    return _ENABLED


def set_process_name(name: str) -> None:
    """Call once per process to label it in the trace (e.g. 'pp_rank_2')."""
    global _process_name
    _process_name = name
    if not _ENABLED:
        return
    # Chrome trace metadata event that renames the PID row.
    with _lock:
        _events.append({
            "name": "process_name",
            "ph": "M",
            "pid": _pid,
            "tid": 0,
            "args": {"name": name},
        })


@contextmanager
def span(name: str, **kwargs: Any) -> Generator[None, None, None]:
    """Context manager that records a complete (X-phase) trace event.

    Usage::

        with pp_trace.span("my_op", step=42, tokens=128):
            do_work()

    When VLLM_PP_TRACE_DIR is not set this is a no-op with zero overhead
    (no branch taken inside the context body).
    """
    if not _ENABLED:
        yield
        return

    t0 = time.perf_counter() * 1_000_000.0  # µs, relative to process start
    try:
        yield
    finally:
        dur = time.perf_counter() * 1_000_000.0 - t0
        ev: dict[str, Any] = {
            "name": name,
            "ph": "X",
            "ts": t0,
            "dur": dur,
            "pid": _pid,
            "tid": threading.get_ident(),
        }
        if kwargs:
            ev["args"] = {
                k: (v if isinstance(v, (int, float, str, bool, type(None)))
                    else str(v))
                for k, v in kwargs.items()
            }
        with _lock:
            _events.append(ev)


def instant(name: str, **kwargs: Any) -> None:
    """Record an instant (zero-duration) marker event."""
    if not _ENABLED:
        return
    ev: dict[str, Any] = {
        "name": name,
        "ph": "i",
        "s": "p",  # process scope
        "ts": time.perf_counter() * 1_000_000.0,
        "pid": _pid,
        "tid": threading.get_ident(),
    }
    if kwargs:
        ev["args"] = {
            k: (v if isinstance(v, (int, float, str, bool, type(None)))
                else str(v))
            for k, v in kwargs.items()
        }
    with _lock:
        _events.append(ev)


def flush(tag: str = "") -> str | None:
    """Write all buffered events to a JSON file (non-destructive: events are
    kept in memory so subsequent flushes include earlier events too).

    The file is always overwritten with the FULL accumulated event list, so
    every flush produces a valid, loadable Chrome Trace file.  Events are
    never cleared — the final atexit flush therefore always captures the
    complete run.

    Returns the path written, or None if tracing is disabled.
    """
    if not _ENABLED:
        return None

    os.makedirs(_TRACE_DIR, exist_ok=True)
    tag_part = f"_{tag}" if tag else ""
    path = os.path.join(_TRACE_DIR, f"vllm_trace_{_pid}{tag_part}.json")

    with _lock:
        snapshot = list(_events)  # snapshot without clearing

    payload = {
        "traceEvents": snapshot,
        # Perfetto and chrome://tracing both honour this.
        "displayTimeUnit": "ms",
        "meta": {
            "pid": _pid,
            "process_name": _process_name,
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    return path


# Register a final flush at process exit so no events are ever lost,
# even if the process is killed or exits before a periodic flush runs.
if _ENABLED:
    atexit.register(lambda: flush(tag="final") if _events else None)


def record_complete(name: str, t0_us: float, **kwargs: Any) -> None:
    """Record a complete event using a pre-captured start timestamp (in µs).

    Typical usage (avoids the overhead of a context manager for functions with
    multiple return points)::

        _t0 = time.perf_counter() * 1e6
        try:
            ...
            return result
        finally:
            pp_trace.record_complete("my_fn", _t0, step=42)
    """
    if not _ENABLED:
        return
    dur = time.perf_counter() * 1_000_000.0 - t0_us
    ev: dict[str, Any] = {
        "name": name,
        "ph": "X",
        "ts": t0_us,
        "dur": dur,
        "pid": _pid,
        "tid": threading.get_ident(),
    }
    if kwargs:
        ev["args"] = {
            k: (v if isinstance(v, (int, float, str, bool, type(None)))
                else str(v))
            for k, v in kwargs.items()
        }
    with _lock:
        _events.append(ev)


def maybe_flush(step_count: int) -> None:
    """Flush trace every _FLUSH_STEPS steps.

    Called from both workers (execute_model_ray) and the driver
    (step_with_batch_queue).  Each flush writes the *full* accumulated event
    list to disk without clearing, so the file is always a valid snapshot of
    everything recorded so far.
    """
    if not _ENABLED or _FLUSH_STEPS <= 0:
        return
    if step_count % _FLUSH_STEPS == 0:
        flush()


# Keep the old name as an alias so existing call-sites still compile.
maybe_flush_worker = maybe_flush
