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
    worker/sample_tokens     — last pp sampling time
    worker/execute_model_ray — total time in execute_model_ray per call
"""

import atexit
import gzip
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

# CUDA / torch-profiler tracing window.
#
# Three supported values:
#
#   VLLM_PP_CUDA_TRACE_STEPS=all          capture every step; export at exit
#   VLLM_PP_CUDA_TRACE_STEPS=10:20        capture steps 10-20 (closed interval)
#   (unset / empty)                        GPU tracing disabled
#
# One persistent torch.profiler.profile is kept open per worker process for
# the duration of the window.  A single Chrome Trace JSON file is exported at
# the end of the window (or at process exit for "all") and merged into the PP
# trace so CUDA kernels appear in the same Perfetto timeline as the CPU spans.
_INF = 10 ** 9  # sentinel for "no upper bound"


def _parse_cuda_trace_steps(env_val: str) -> tuple[int, int] | None:
    """Return (first, last) step range, or None if disabled.

    ``last == _INF`` means "run until process exit".
    """
    val = env_val.strip()
    if not val:
        return None
    if val.lower() == "all":
        return (0, _INF)
    try:
        parts = val.split(":")
        if len(parts) != 2:
            raise ValueError
        first, last = int(parts[0]), int(parts[1])
        if first > last:
            raise ValueError
        return (first, last)
    except ValueError:
        raise ValueError(
            f"VLLM_PP_CUDA_TRACE_STEPS must be 'all' or 'first:last' "
            f"(e.g. '10:20'), got: {env_val!r}"
        )


_CUDA_TRACE_STEPS: tuple[int, int] | None = _parse_cuda_trace_steps(
    os.environ.get("VLLM_PP_CUDA_TRACE_STEPS", "")
)

# Per-process persistent CUDA profiler state.
# Guarded by _cuda_prof_lock; each worker is a separate OS process so
# there is no cross-rank sharing — this is truly per-rank state.
_cuda_prof_lock = threading.Lock()
_cuda_prof: Any = None          # torch.profiler.profile instance while active
_cuda_prof_rank: int = -1       # pp_rank that started the profiler
_cuda_prof_started: bool = False  # True after __enter__ has been called

# ---------------------------------------------------------------------------
# Module-level state (per process)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_events: list[dict[str, Any]] = []
_external_trace_files: set[str] = set()
_pid: int = os.getpid()
_process_name: str = f"pid_{_pid}"  # overridden by set_process_name()
_CLOCK_SYNC_NAME = "vllm_pp_trace_clock_sync"
_clock_sync_emitted: bool = False
_CLOCK_OFFSET_FILE = "pp_trace_clock_offset_us.json"


def is_enabled() -> bool:
    return _ENABLED


def _now_us() -> float:
    return time.monotonic_ns() / 1_000.0


def _emit_clock_sync_if_needed() -> None:
    """Emit a one-time marker into both PP trace and torch profiler trace.

    The marker lets us rebase PP trace timestamps onto the torch profiler clock
    domain when we merge the files for Perfetto/Chrome.
    """
    global _clock_sync_emitted
    if not _ENABLED or _clock_sync_emitted:
        return

    sync_ts_us = _now_us()
    with _lock:
        _events.append({
            "name": _CLOCK_SYNC_NAME,
            "ph": "i",
            "s": "p",
            "ts": sync_ts_us,
            "pid": _pid,
            "tid": threading.get_ident(),
        })
    _clock_sync_emitted = True

    try:
        torch = __import__("torch")
        with torch.profiler.record_function(_CLOCK_SYNC_NAME):
            pass
    except Exception:
        # Profiling sync is best-effort only.
        return


def emit_profiler_clock_sync() -> None:
    """Emit a sync marker while the torch profiler is actively recording.

    Torch profiler traces only capture `record_function()` ranges after
    profiling has started, so we need a second explicit sync point at profiler
    start time to align PP spans with the exported torch trace clock domain.
    """
    if not _ENABLED:
        return
    with _lock:
        _events.append({
            "name": _CLOCK_SYNC_NAME,
            "ph": "i",
            "s": "p",
            "ts": _now_us(),
            "pid": _pid,
            "tid": threading.get_ident(),
            "args": {"source": "torch_profiler"},
        })


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

    _emit_clock_sync_if_needed()
    t0 = _now_us()
    try:
        yield
    finally:
        dur = _now_us() - t0
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
    _emit_clock_sync_if_needed()
    ev: dict[str, Any] = {
        "name": name,
        "ph": "i",
        "s": "p",  # process scope
        "ts": _now_us(),
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


def register_external_trace(path: str) -> None:
    """Register another Chrome-trace file to merge into this PP trace.

    This is used by the torch profiler so a worker's CUDA/CPU timeline can be
    viewed in the same Perfetto/Chrome trace JSON as the PP async spans.
    """
    if not _ENABLED or not path:
        return
    with _lock:
        _external_trace_files.add(os.path.abspath(path))


def _load_external_trace_events(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        payload = json.load(f)

    trace_events = payload.get("traceEvents", [])
    # Keep PP process naming authoritative for this pid.
    return [
        ev
        for ev in trace_events
        if not (ev.get("ph") == "M" and ev.get("name") == "process_name")
    ]


def _find_sync_ts(
    events: list[dict[str, Any]],
    *,
    choose_last: bool = False,
) -> float | None:
    iterable = reversed(events) if choose_last else events
    for ev in iterable:
        if ev.get("name") == _CLOCK_SYNC_NAME and "ts" in ev:
            return float(ev["ts"])
    return None


def _shift_events(events: list[dict[str, Any]], delta_us: float) -> list[dict[str, Any]]:
    shifted: list[dict[str, Any]] = []
    for ev in events:
        if "ts" not in ev:
            shifted.append(ev)
            continue
        shifted_ev = dict(ev)
        shifted_ev["ts"] = float(shifted_ev["ts"]) + delta_us
        shifted.append(shifted_ev)
    return shifted


def _clock_offset_path() -> str:
    return os.path.join(_TRACE_DIR, _CLOCK_OFFSET_FILE)


def _load_global_clock_offset_us() -> float | None:
    if not _ENABLED:
        return None
    path = _clock_offset_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
        return float(payload["offset_us"])
    except Exception:
        return None


def _store_global_clock_offset_us(offset_us: float) -> None:
    if not _ENABLED:
        return
    os.makedirs(_TRACE_DIR, exist_ok=True)
    path = _clock_offset_path()
    tmp_path = f"{path}.tmp.{_pid}"
    payload = {"offset_us": offset_us}
    try:
        with open(tmp_path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


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
        external_trace_files = sorted(_external_trace_files)

    applied_global_offset = False
    for external_trace_path in external_trace_files:
        try:
            external_events = _load_external_trace_events(external_trace_path)
        except Exception:
            continue

        local_sync_ts = _find_sync_ts(snapshot, choose_last=True)
        external_sync_ts = _find_sync_ts(external_events)
        if local_sync_ts is not None and external_sync_ts is not None:
            offset_us = external_sync_ts - local_sync_ts
            _store_global_clock_offset_us(offset_us)
            snapshot = _shift_events(snapshot, offset_us)
            applied_global_offset = True
            break

    if not applied_global_offset:
        global_offset_us = _load_global_clock_offset_us()
        if global_offset_us is not None:
            snapshot = _shift_events(snapshot, global_offset_us)

    for external_trace_path in external_trace_files:
        try:
            snapshot.extend(_load_external_trace_events(external_trace_path))
        except Exception:
            # Tracing is best-effort only; keep PP spans even if an external
            # trace is malformed or still being finalized by another writer.
            continue

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

        _t0 = time.monotonic() * 1e6
        try:
            ...
            return result
        finally:
            pp_trace.record_complete("my_fn", _t0, step=42)
    """
    if not _ENABLED:
        return
    _emit_clock_sync_if_needed()
    dur = _now_us() - t0_us
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


# ---------------------------------------------------------------------------
# CUDA / GPU kernel tracing via torch.profiler
# ---------------------------------------------------------------------------

def _cuda_export(pp_rank: int, step_id: int | None) -> None:
    """Stop the active CUDA profiler and write its Chrome Trace JSON.

    Must be called with ``_cuda_prof_lock`` already held, or from atexit
    (single-threaded by that point).  Safe to call even if no profiler is
    active (no-op in that case).
    """
    global _cuda_prof, _cuda_prof_started
    if _cuda_prof is None or not _cuda_prof_started:
        return
    prof = _cuda_prof
    _cuda_prof = None
    _cuda_prof_started = False
    try:
        prof.__exit__(None, None, None)
        os.makedirs(_TRACE_DIR, exist_ok=True)
        step_tag = "final" if step_id is None else str(step_id)
        cuda_trace_path = os.path.join(
            _TRACE_DIR,
            f"cuda_trace_rank{pp_rank}_step{step_tag}_{_pid}.json",
        )
        prof.export_chrome_trace(cuda_trace_path)
        register_external_trace(cuda_trace_path)
        flush(tag=f"after_cuda_step{step_tag}")
    except Exception:
        pass  # best-effort; never crash the serving process


@contextmanager
def cuda_trace_step(
    step_id: int | None,
    pp_rank: int,
) -> Generator[None, None, None]:
    """Wrap one DAG step inside a persistent torch.profiler session.

    Lifecycle
    ---------
    * **First step in the window** — starts ``torch.profiler.profile`` once,
      emits clock-sync markers so CPU and GPU clocks can be aligned.
    * **Middle steps** — the profiler stays open; just ``yield`` through.
    * **Last step in the window** — stops the profiler, exports a single
      Chrome Trace JSON for the entire window, and registers it for merging.
    * **"all" mode** (``VLLM_PP_CUDA_TRACE_STEPS=all``) — profiler stays open
      until process exit; an ``atexit`` handler exports the final trace.

    Zero overhead when ``VLLM_PP_CUDA_TRACE_STEPS`` is unset.
    """
    global _cuda_prof, _cuda_prof_rank, _cuda_prof_started

    if (
        not _ENABLED
        or _CUDA_TRACE_STEPS is None
        or step_id is None
        or step_id < _CUDA_TRACE_STEPS[0]
        or step_id > _CUDA_TRACE_STEPS[1]
    ):
        yield
        return

    try:
        import torch
    except ImportError:
        yield
        return

    first_step, last_step = _CUDA_TRACE_STEPS
    is_first = (step_id == first_step)
    is_last = (last_step < _INF and step_id == last_step)

    with _cuda_prof_lock:
        if is_first and not _cuda_prof_started:
            # Emit clock-sync BEFORE the profiler starts so the PP trace has a
            # reference point on its own (monotonic) clock.
            emit_profiler_clock_sync()

            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                with_stack=False,
                with_flops=False,
            )
            prof.__enter__()

            # Second sync emitted INSIDE the profiler so the CUDA-aligned
            # profiler clock gets the same reference marker.
            emit_profiler_clock_sync()

            _cuda_prof = prof
            _cuda_prof_rank = pp_rank
            _cuda_prof_started = True

            # For "all" mode: register an atexit so the trace is exported even
            # if we never see the last step (e.g. server is killed).
            if last_step >= _INF:
                atexit.register(
                    lambda: _cuda_export(pp_rank, step_id=None)
                )

    try:
        yield
    finally:
        if is_last:
            with _cuda_prof_lock:
                _cuda_export(pp_rank, step_id)
