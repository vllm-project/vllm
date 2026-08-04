# SPDX-License-Identifier: Apache-2.0

"""Trace recording subsystem.

This package implements the **capture** half of the ``lmcache trace``
feature.  Public API:

* :func:`~lmcache.v1.mp_observability.trace.decorator.enable_tracing` —
  decorator that publishes a unified ``TRACE_CALL`` event on each call
  to a decorated function (entry-only, inputs only).
* :class:`~lmcache.v1.mp_observability.trace.recorder.StorageTraceRecorder`
  — :class:`EventSubscriber` that writes ``TRACE_CALL`` events to a
  binary trace file.
* :class:`~lmcache.v1.mp_observability.trace.reader.TraceReader` —
  streaming reader for inspection (replay lives in a separate package).

When tracing is disabled (the default), the decorator is a thin
wrapper around the original function; only a single boolean attribute
load is added per call.
"""

# First Party
from lmcache.v1.mp_observability.trace.decorator import (
    enable_tracing,
    is_tracing_enabled,
    publish_call_event,
    set_tracing_enabled,
)
from lmcache.v1.mp_observability.trace.format import Header, Record
from lmcache.v1.mp_observability.trace.lifecycle import (
    maybe_initialize_trace_recorder,
)
from lmcache.v1.mp_observability.trace.reader import TraceReader
from lmcache.v1.mp_observability.trace.recorder import (
    StorageTraceRecorder,
    TraceRecorder,
)

__all__ = [
    "Header",
    "Record",
    "StorageTraceRecorder",
    "TraceReader",
    "TraceRecorder",
    "enable_tracing",
    "is_tracing_enabled",
    "maybe_initialize_trace_recorder",
    "publish_call_event",
    "set_tracing_enabled",
]
