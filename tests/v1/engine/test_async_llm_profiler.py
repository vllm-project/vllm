# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The AsyncLLM frontend profiler has to start and stop on one thread.

Kineto binds its client to the thread that starts the profiler. Starting it from
a worker thread silently records nothing, and stopping it from a different
thread than it was started on crashes the process, so the lifecycle must stay on
the event loop thread. Exporting the trace has no such constraint and must not
stay there: it serializes every recorded event.
"""

import gzip
import json
import threading
from functools import partial
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from vllm.v1.engine.async_llm import AsyncLLM

NUM_SPANS = 64


def make_engine(profiler, trace_handler=None):
    engine = MagicMock(spec=AsyncLLM)
    engine.engine_core = MagicMock()
    engine.engine_core.profile_async = AsyncMock()
    engine.profiler = profiler
    engine.frontend_trace_handler = trace_handler
    # The mock would otherwise swallow the helpers these tests are about.
    engine._start_frontend_profiler = partial(AsyncLLM._start_frontend_profiler, engine)
    engine._stop_frontend_profiler = partial(AsyncLLM._stop_frontend_profiler, engine)
    return engine


def read_spans(trace_dir):
    traces = list(trace_dir.glob("*.pt.trace.json*"))
    assert len(traces) == 1, f"expected one trace, got {traces}"
    opener = gzip.open if traces[0].suffix == ".gz" else open
    with opener(traces[0], "rt") as f:
        events = json.load(f)["traceEvents"]
    return [e for e in events if e.get("name") == "frontend_span"]


@pytest.mark.asyncio
async def test_frontend_profiler_records_what_happened(tmp_path):
    """The exported trace must actually contain the frontend's spans."""
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
    )
    handler = torch.profiler.tensorboard_trace_handler(
        str(tmp_path), worker_name="async_llm", use_gzip=True
    )
    engine = make_engine(profiler, handler)

    await AsyncLLM.start_profile(engine, "trace")
    for _ in range(NUM_SPANS):
        with torch.profiler.record_function("frontend_span"):
            pass
    await AsyncLLM.stop_profile(engine)

    assert len(read_spans(tmp_path)) == NUM_SPANS
    engine.engine_core.profile_async.assert_any_await(True, "trace")
    engine.engine_core.profile_async.assert_any_await(False)


@pytest.mark.asyncio
async def test_profiler_lifecycle_stays_on_the_calling_thread():
    """Both calls must run on the caller's thread, not a worker thread."""
    profiler = MagicMock()
    threads: list[int] = []
    profiler.start.side_effect = lambda: threads.append(threading.get_ident())
    profiler.stop.side_effect = lambda: threads.append(threading.get_ident())
    engine = make_engine(profiler, MagicMock())

    await AsyncLLM.start_profile(engine)
    await AsyncLLM.stop_profile(engine)

    assert threads == [threading.get_ident(), threading.get_ident()]


@pytest.mark.asyncio
async def test_trace_export_does_not_run_on_the_event_loop():
    """Exporting blocks for as long as it takes to serialize the trace.

    On the event loop that stalls every other request for its whole duration,
    so it has to happen off-thread even though stop() cannot.
    """
    profiler = MagicMock()
    ran_on: dict[str, object] = {}
    profiler.stop.side_effect = lambda: ran_on.update(stop=threading.get_ident())

    def handler(prof):
        ran_on.update(export=threading.get_ident(), exported=prof)

    engine = make_engine(profiler, handler)

    await AsyncLLM.stop_profile(engine)

    assert ran_on["stop"] == threading.get_ident()
    assert ran_on["export"] != threading.get_ident()
    assert ran_on["exported"] is profiler


@pytest.mark.asyncio
async def test_engine_core_is_told_even_when_the_trace_export_fails():
    """A frontend trace-export failure must not leave EngineCore profiling."""
    profiler = MagicMock()

    def handler(prof):
        raise OSError("no space left on device")

    engine = make_engine(profiler, handler)

    with pytest.raises(OSError):
        await AsyncLLM.stop_profile(engine)

    engine.engine_core.profile_async.assert_any_await(False)


@pytest.mark.asyncio
async def test_profile_endpoints_still_work_without_a_frontend_profiler():
    engine = make_engine(None)

    await AsyncLLM.start_profile(engine)
    await AsyncLLM.stop_profile(engine)

    engine.engine_core.profile_async.assert_any_await(True, None)
    engine.engine_core.profile_async.assert_any_await(False)
