# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-phase cold-start OTEL span consumer for vllm bench startup."""

import logging
import threading
import time
from concurrent import futures
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

COLD_START_PHASES: tuple[str, ...] = (
    "Worker init", "Init device", "Loading (GPU)", "Capture model",
    "Allocate KV cache", "Warmup (GPU)",
)
PHASE_LIST_VERSION: int = 1


@dataclass(frozen=True)
class StartupPhaseSpan:
    name: str
    start_ns: int
    end_ns: int
    duration_s: float
    missing: bool = False


try:
    import grpc
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceResponse,
    )
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import (
        add_TraceServiceServicer_to_server,
    )
    _OTEL_PROTO_AVAILABLE = True
except ImportError:
    _OTEL_PROTO_AVAILABLE = False


class InMemorySpanSink:
    def __init__(self, address: str = "localhost:0"):
        if not _OTEL_PROTO_AVAILABLE:
            raise RuntimeError(
                "opentelemetry protos not installed; per-phase consumer "
                "requires the opentelemetry packages."
            )
        self._address = address
        self._lock = threading.Lock()
        self._spans: list[dict[str, Any]] = []
        self._server: Any = None
        self._bound: str | None = None

    def start(self) -> str:
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        add_TraceServiceServicer_to_server(self, self._server)
        port = self._server.add_insecure_port(self._address)
        host = self._address.rsplit(":", 1)[0]
        self._bound = f"{host}:{port}"
        self._server.start()
        return self._bound

    def stop(self, grace: float = 1.0) -> None:
        if self._server is not None:
            self._server.stop(grace=grace)
            self._server = None

    def Export(self, request, context):  # noqa: N802
        with self._lock:
            it = (s for rs in request.resource_spans
                  for ss in rs.scope_spans for s in ss.spans)
            for span in it:
                self._spans.append({
                    "name": span.name,
                    "start_time_unix_nano": span.start_time_unix_nano,
                    "end_time_unix_nano": span.end_time_unix_nano,
                })
        return ExportTraceServiceResponse()

    def get_all_spans(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()

    def wait_for_spans(self, names, timeout: float = 30.0, poll: float = 0.1) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                present = {s["name"] for s in self._spans}
            if set(names).issubset(present):
                return True
            time.sleep(poll)
        return False


def _phase_span_from_dict(name, span: dict[str, Any] | None) -> StartupPhaseSpan:
    if span is None:
        return StartupPhaseSpan(
            name=name, start_ns=0, end_ns=0, duration_s=0.0, missing=True)
    start_ns = int(span["start_time_unix_nano"])
    end_ns = int(span["end_time_unix_nano"])
    return StartupPhaseSpan(
        name=name, start_ns=start_ns, end_ns=end_ns,
        duration_s=max(0.0, (end_ns - start_ns) / 1e9))


def select_phase_spans(
    all_spans, phase_names=COLD_START_PHASES,
) -> list[StartupPhaseSpan]:
    out: list[StartupPhaseSpan] = []
    for name in phase_names:
        matches = [s for s in all_spans if s["name"] == name]
        if not matches:
            out.append(_phase_span_from_dict(name, None))
            continue
        chosen = min(matches, key=lambda s: int(s["start_time_unix_nano"]))
        out.append(_phase_span_from_dict(name, chosen))
    known = set(phase_names)
    extras = {s["name"] for s in all_spans if s["name"] not in known}
    if extras:
        logger.warning("spans not in COLD_START_PHASES (v=%d): %s",
                       PHASE_LIST_VERSION, ", ".join(sorted(extras)))
    return out


def collect_startup_spans(sink, phase_names=COLD_START_PHASES, *, timeout=30.0):
    sink.wait_for_spans(phase_names, timeout=timeout)
    return select_phase_spans(sink.get_all_spans(), phase_names)


def build_phase_report(spans, *, wall_clock_startup_s=None) -> dict[str, Any]:
    return {
        "phase_list_version": PHASE_LIST_VERSION,
        "phases": [
            {"name": s.name, "duration_s": s.duration_s,
             "start_ns": s.start_ns, "end_ns": s.end_ns, "missing": s.missing}
            for s in spans],
        "total_phase_time_s": sum(s.duration_s for s in spans),
        "wall_clock_startup_s": wall_clock_startup_s,
    }
