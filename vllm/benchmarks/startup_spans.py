# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect the OTEL spans emitted during a startup benchmark."""

import threading
import time
from collections.abc import Iterator
from concurrent import futures
from contextlib import contextmanager
from typing import Any

import grpc
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceResponse,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import (
    TraceServiceServicer,
    add_TraceServiceServicer_to_server,
)


class InMemorySpanSink(TraceServiceServicer):
    """A local OTLP/gRPC receiver that keeps every span it is sent."""

    def __init__(self, address: str = "localhost:0"):
        self._requested_address = address
        self._lock = threading.Lock()
        self._spans: list[dict[str, Any]] = []
        self._server: Any = None
        self._bound_address: str | None = None

    @property
    def address(self) -> str:
        if self._bound_address is None:
            raise RuntimeError("span sink has not been started")
        return self._bound_address

    def start(self) -> str:
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        add_TraceServiceServicer_to_server(self, self._server)
        port = self._server.add_insecure_port(self._requested_address)
        if not port:
            raise RuntimeError(
                f"cannot bind OTLP receiver to {self._requested_address}"
            )
        host = self._requested_address.rsplit(":", 1)[0]
        self._bound_address = f"{host}:{port}"
        self._server.start()
        return self._bound_address

    def stop(self, grace: float = 1.0) -> None:
        if self._server is not None:
            self._server.stop(grace=grace)
            self._server = None

    def Export(self, request, context):  # noqa: N802
        with self._lock:
            # Keep each span's identity and trace: the same span name can be
            # emitted by several processes (engine core, one worker per rank),
            # so the ids are what tell those emissions apart.
            for resource_spans in request.resource_spans:
                for scope_spans in resource_spans.scope_spans:
                    for span in scope_spans.spans:
                        self._spans.append(
                            {
                                "name": span.name,
                                "trace_id": span.trace_id.hex(),
                                "span_id": span.span_id.hex(),
                                "parent_span_id": span.parent_span_id.hex() or None,
                                "start_time_unix_nano": span.start_time_unix_nano,
                                "end_time_unix_nano": span.end_time_unix_nano,
                            }
                        )
        return ExportTraceServiceResponse()

    def get_all_spans(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()

    def collect(
        self, *, timeout: float = 2.0, quiet: float = 0.2
    ) -> list[dict[str, Any]]:
        """Return the spans received so far once exports have settled.

        vLLM's ``BatchSpanProcessor`` flushes on process exit, so a benchmark
        iteration has every span on the wire before its subprocess returns; the
        loop only covers the exporter's in-flight batch. Wait until no new span
        has arrived for ``quiet`` seconds, bounded by ``timeout``, so a runner
        that does not instrument a phase cannot stall the benchmark.
        """
        deadline = time.monotonic() + timeout
        last_count = -1
        quiet_deadline = time.monotonic() + quiet
        while time.monotonic() < deadline:
            count = len(self.get_all_spans())
            if count != last_count:
                last_count = count
                quiet_deadline = time.monotonic() + quiet
            elif time.monotonic() >= quiet_deadline:
                break
            time.sleep(0.02)
        return self.get_all_spans()


@contextmanager
def span_sink(address: str = "localhost:0") -> Iterator[InMemorySpanSink]:
    """Run an in-memory OTLP receiver for the duration of the block."""
    sink = InMemorySpanSink(address)
    sink.start()
    try:
        yield sink
    finally:
        sink.stop()


def collect_startup_spans(sink: InMemorySpanSink) -> list[dict[str, Any]]:
    """Snapshot every received span as a flat record, one per span."""
    return [
        {
            "name": span["name"],
            "duration_s": max(
                0.0,
                (int(span["end_time_unix_nano"]) - int(span["start_time_unix_nano"]))
                / 1e9,
            ),
            "trace_id": span["trace_id"],
            "span_id": span["span_id"],
            "parent_span_id": span["parent_span_id"],
        }
        for span in sink.collect()
    ]


def phase_durations(spans) -> dict[str, float]:
    """Duration per span name in seconds.

    A name can be emitted by several processes (engine core, one worker per
    rank), so those emissions are summed. A span nested inside another span of
    the same name is skipped instead: its cost is already inside the parent's.
    """
    ids_by_name: dict[str, set[str]] = {}
    for span in spans:
        ids_by_name.setdefault(span["name"], set()).add(span["span_id"])
    durations: dict[str, float] = {}
    for span in spans:
        if span["parent_span_id"] in ids_by_name[span["name"]]:
            continue
        durations[span["name"]] = durations.get(span["name"], 0.0) + span["duration_s"]
    return durations
