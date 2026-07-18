# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation tests for the per-phase cold-start startup span consumer.

This test file is RED on master because ``vllm.benchmarks.startup_spans``
does not exist there (the imports below raise ``ImportError`` at collection
time). It turns GREEN on the branch once the consumer ships.

The integration test ``test_startup_per_phase_spans`` drives the real
instrumented worker path (no mocking of the function under fix): it boots
``LLM(model="facebook/opt-125m")`` against the in-tree ``FakeTraceService``
OTLP harness — the same proven pattern as
``tests/v1/tracing/test_tracing.py::test_traces`` — waits for the 6
canonical cold-start phase spans, and asserts the consumer's report shape.

The CLI smoke test ``test_bench_startup_per_phase_flag`` runs
``vllm bench startup --per-phase`` end-to-end and asserts it exits 0 and
prints the per-phase table header, guarding the wiring in
``vllm/benchmarks/startup.py``.
"""

import json
import subprocess
import time

import pytest
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_TRACES_INSECURE

from tests.tracing.conftest import (  # noqa: F401, F811
    FAKE_TRACE_SERVER_ADDRESS,
    FakeTraceService,
    trace_service,
)
from vllm import LLM
from vllm.benchmarks.startup_spans import (
    COLD_START_PHASES,
    PHASE_LIST_VERSION,
    InMemorySpanSink,
    build_phase_report,
    compare_to_baseline,
    format_phase_report,
    select_phase_spans,
)
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform


def test_startup_per_phase_spans(
    monkeypatch: pytest.MonkeyPatch,
    trace_service: FakeTraceService,  # noqa: F811 - pytest fixture injection
):
    """Boot LLM(opt-125m) against the fake OTLP server and assert the
    per-phase consumer produces a well-formed report from the real
    cold-start phase spans.

    RED on master: ``vllm.benchmarks.startup_spans`` does not exist, so this
    module fails to import at collection time. GREEN on the branch: the
    consumer ships, the 6 phase spans emit via ``@instrument``, and the
    report has the expected shape.
    """
    with monkeypatch.context() as m:
        m.setenv(OTEL_EXPORTER_OTLP_TRACES_INSECURE, "true")
        # gRPC's C-core is not fork-safe; mirror test_traces' spawn setting.
        m.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        llm = None
        try:
            llm = LLM(
                model="facebook/opt-125m",
                otlp_traces_endpoint=FAKE_TRACE_SERVER_ADDRESS,
                gpu_memory_utilization=0.3,
            )

            # The BatchSpanProcessor batches spans and exports them
            # periodically; poll for the 6 canonical phase spans with the
            # same 15s budget test_traces uses (x2 for headroom).
            deadline = time.time() + 30.0
            all_spans: list[dict] = []
            while time.time() < deadline:
                all_spans = trace_service.get_all_spans()
                present = {s["name"] for s in all_spans}
                if set(COLD_START_PHASES).issubset(present):
                    break
                time.sleep(0.5)

            # 1. Every canonical phase name appears at least once.
            present = {s["name"] for s in all_spans}
            missing_names = set(COLD_START_PHASES) - present
            assert not missing_names, (
                f"Expected all {len(COLD_START_PHASES)} cold-start phase "
                f"spans; missing: {sorted(missing_names)}. "
                f"Got: {sorted(present)}"
            )

            # 2. select_phase_spans + build_phase_report produce the
            #    expected report shape.
            phase_spans = select_phase_spans(all_spans, COLD_START_PHASES)
            assert len(phase_spans) == len(COLD_START_PHASES)
            report = build_phase_report(phase_spans)
            assert report["phase_list_version"] == PHASE_LIST_VERSION == 1
            assert len(report["phases"]) == len(COLD_START_PHASES)

            # 3. Every reported duration is non-negative (guards the #40698
            #    timing-fix edge cases where end_ns could precede start_ns).
            for row in report["phases"]:
                assert row["duration_s"] >= 0.0, (
                    f"phase '{row['name']}' has negative duration "
                    f"{row['duration_s']}"
                )
                # The phase must be present (not a missing placeholder),
                # since we asserted all names are present above.
                assert not row["missing"], (
                    f"phase '{row['name']}' marked missing despite span present"
                )
        finally:
            if llm is not None:
                shutdown_timeout = 60.0 if current_platform.is_rocm() else 5.0
                llm.llm_engine.engine_core.shutdown(timeout=shutdown_timeout)
            cleanup_dist_env_and_memory()


@pytest.mark.benchmark
def test_bench_startup_per_phase_flag():
    """CLI smoke: ``vllm bench startup --per-phase`` exits 0 and prints the
    per-phase cold-start table header. Guards the CLI wiring in
    ``vllm/benchmarks/startup.py`` (``--per-phase`` / ``--phase-baseline``
    flags + sink lifecycle + main-loop report emission)."""
    command = [
        "vllm",
        "bench",
        "startup",
        "--model",
        "facebook/opt-125m",
        "--num-iters-cold",
        "1",
        "--num-iters-warmup",
        "0",
        "--num-iters-warm",
        "0",
        "--per-phase",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, (
        f"Benchmark failed (rc={result.returncode}): {result.stderr}"
    )
    assert "PER-PHASE COLD-START REPORT" in result.stdout, (
        "per-phase report banner not found in stdout; got:\n" + result.stdout
    )
    # Every canonical phase name should appear in the printed table.
    for phase_name in COLD_START_PHASES:
        assert phase_name in result.stdout, (
            f"phase '{phase_name}' not in stdout table"
        )


# ---------------------------------------------------------------------------
# Lightweight, non-GPU unit tests for the pure consumer logic. These guard
# regressions in select_phase_spans / build_phase_report / InMemorySpanSink /
# compare_to_baseline / format_phase_report on CI-skip lanes that have no GPU
# and no ``vllm`` console script. They never boot an LLM.
# ---------------------------------------------------------------------------


def test_select_phase_spans_picks_earliest():
    """select_phase_spans picks the earliest span per phase, marks absent
    phases missing, and clamps negative durations to 0.0 (#40698). No GPU.
    """
    spans = [
        # Duplicate "Worker init": the later-starting one must NOT win.
        {"name": "Worker init", "start_time_unix_nano": 200_000_000,
         "end_time_unix_nano": 300_000_000},
        {"name": "Worker init", "start_time_unix_nano": 100_000_000,
         "end_time_unix_nano": 150_000_000},
        # end < start (#40698 timing-fix edge case) -> duration clamped to 0.0.
        {"name": "Loading (GPU)", "start_time_unix_nano": 400_000_000,
         "end_time_unix_nano": 100_000_000},
        {"name": "Init device", "start_time_unix_nano": 200_000_000,
         "end_time_unix_nano": 250_000_000},
        {"name": "Allocate KV cache", "start_time_unix_nano": 500_000_000,
         "end_time_unix_nano": 550_000_000},
        {"name": "Warmup (GPU)", "start_time_unix_nano": 600_000_000,
         "end_time_unix_nano": 700_000_000},
        # "Capture model" is intentionally absent -> missing placeholder.
        # An unknown span is a logged warning, never a failure.
        {"name": "Mystery phase", "start_time_unix_nano": 0,
         "end_time_unix_nano": 0},
    ]
    result = select_phase_spans(spans, COLD_START_PHASES)

    # (a) exactly 6 rows, one per canonical phase, in COLD_START_PHASES order.
    assert len(result) == len(COLD_START_PHASES) == 6
    assert [s.name for s in result] == list(COLD_START_PHASES)
    # (b) duplicate "Worker init" collapses to the earliest start_ns.
    worker = result[0]
    assert worker.name == "Worker init"
    assert worker.start_ns == 100_000_000
    assert worker.duration_s == 0.05  # 50 ms
    assert not worker.missing
    # (c) absent "Capture model" -> missing=True, zero-duration placeholder.
    capture = result[3]
    assert capture.name == "Capture model"
    assert capture.missing is True
    assert capture.duration_s == 0.0
    # (d) end < start -> duration clamped to 0.0 (#40698 edge case).
    loading = result[2]
    assert loading.name == "Loading (GPU)"
    assert loading.duration_s == 0.0
    # All durations are non-negative.
    assert all(s.duration_s >= 0.0 for s in result)


def test_build_phase_report_shape():
    """build_phase_report returns the frozen version, 6 phase rows, the summed
    total, and echoes wall_clock_startup_s. No GPU.
    """
    spans = [
        {"name": "Worker init", "start_time_unix_nano": 0,
         "end_time_unix_nano": 100_000_000},   # 0.10 s
        {"name": "Init device", "start_time_unix_nano": 0,
         "end_time_unix_nano": 50_000_000},    # 0.05 s
        {"name": "Loading (GPU)", "start_time_unix_nano": 0,
         "end_time_unix_nano": 200_000_000},   # 0.20 s
        {"name": "Capture model", "start_time_unix_nano": 0,
         "end_time_unix_nano": 80_000_000},    # 0.08 s
        {"name": "Allocate KV cache", "start_time_unix_nano": 0,
         "end_time_unix_nano": 70_000_000},    # 0.07 s
        {"name": "Warmup (GPU)", "start_time_unix_nano": 0,
         "end_time_unix_nano": 150_000_000},   # 0.15 s
    ]
    phase_spans = select_phase_spans(spans, COLD_START_PHASES)
    report = build_phase_report(phase_spans, wall_clock_startup_s=1.0)

    assert report["phase_list_version"] == PHASE_LIST_VERSION == 1
    assert len(report["phases"]) == 6
    durations = [p["duration_s"] for p in report["phases"]]
    assert report["total_phase_time_s"] == sum(durations)
    assert report["wall_clock_startup_s"] == 1.0
    # Every row carries the frozen-shape fields.
    for p in report["phases"]:
        assert {"name", "duration_s", "start_ns", "end_ns", "missing"} <= set(p)


def test_in_memory_span_sink_export_and_poll():
    """InMemorySpanSink receives a span over gRPC and exposes it via
    get_all_spans / wait_for_spans / clear. No GPU. Confirms the sink
    functions as a TraceServiceServicer (inherited when opentelemetry is
    installed, duck-typed via the gRPC Export contract).
    """
    import grpc
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import (
        TraceServiceStub,
    )
    from opentelemetry.proto.trace.v1.trace_pb2 import Span

    sink = InMemorySpanSink(address="localhost:0")
    address = sink.start()
    try:
        channel = grpc.insecure_channel(address)
        # Block until the gRPC server is accepting connections.
        grpc.channel_ready_future(channel).result(timeout=5.0)
        stub = TraceServiceStub(channel)

        span = Span(name="Worker init",
                    start_time_unix_nano=100,
                    end_time_unix_nano=200)
        request = ExportTraceServiceRequest()
        rs = request.resource_spans.add()
        ss = rs.scope_spans.add()
        ss.spans.add().CopyFrom(span)

        stub.Export(request)

        assert sink.wait_for_spans(["Worker init"], timeout=5.0)
        all_spans = sink.get_all_spans()
        assert len(all_spans) >= 1
        got = next(s for s in all_spans if s["name"] == "Worker init")
        assert got["start_time_unix_nano"] == 100
        assert got["end_time_unix_nano"] == 200

        # clear() empties the in-memory buffer.
        sink.clear()
        assert sink.get_all_spans() == []
    finally:
        sink.stop()


def test_compare_to_baseline_and_format(tmp_path):
    """compare_to_baseline flags regressions via both the absolute and the
    relative threshold branch, skips on a missing file / phase_list_version
    mismatch (never raises); format_phase_report emits aligned rows. No GPU.
    Guards the --phase-baseline regression-gate path.
    """
    current = {
        "phase_list_version": PHASE_LIST_VERSION,
        "phases": [
            {"name": "Worker init", "duration_s": 0.15, "start_ns": 0,
             "end_ns": 0, "missing": False},      # +0.05 < 0.5 -> ok (abs)
            {"name": "Init device", "duration_s": 0.04, "start_ns": 0,
             "end_ns": 0, "missing": False},      # 0 delta -> ok
            {"name": "Loading (GPU)", "duration_s": 11.5, "start_ns": 0,
             "end_ns": 0, "missing": False},      # +1.5 > max(0.5,1.0) -> reg
            {"name": "Capture model", "duration_s": 0.0, "start_ns": 0,
             "end_ns": 0, "missing": True},       # missing -> skipped
            {"name": "Allocate KV cache", "duration_s": 0.07, "start_ns": 0,
             "end_ns": 0, "missing": False},      # 0 delta -> ok
            {"name": "Warmup (GPU)", "duration_s": 1.20, "start_ns": 0,
             "end_ns": 0, "missing": False},      # +0.70 > 0.5 -> reg (abs)
        ],
        "total_phase_time_s": 12.96,
        "wall_clock_startup_s": 13.5,
    }
    baseline = {
        "phase_list_version": PHASE_LIST_VERSION,
        "phases": [
            {"name": "Worker init", "duration_s": 0.10},
            {"name": "Init device", "duration_s": 0.04},
            {"name": "Loading (GPU)", "duration_s": 10.0},
            {"name": "Capture model", "duration_s": 0.08},
            {"name": "Allocate KV cache", "duration_s": 0.07},
            {"name": "Warmup (GPU)", "duration_s": 0.50},
        ],
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline))

    cmp = compare_to_baseline(current, str(baseline_path))
    assert cmp["skipped"] is None
    assert cmp["regressed"] is True
    by_name = {d["name"]: d for d in cmp["phases"]}
    # Loading (GPU): rel branch — threshold = max(0.5, 10.0*0.10=1.0) = 1.0;
    # delta +1.5 > 1.0 -> regressed.
    assert by_name["Loading (GPU)"]["regressed"] is True
    assert by_name["Loading (GPU)"]["delta_s"] == 1.5
    # Warmup (GPU): abs branch — threshold = max(0.5, 0.05) = 0.5;
    # delta +0.70 > 0.5 -> regressed.
    assert by_name["Warmup (GPU)"]["regressed"] is True
    assert by_name["Warmup (GPU)"]["delta_s"] == 0.70
    # Worker init: small delta under the 0.5 s floor -> not regressed.
    assert by_name["Worker init"]["regressed"] is False
    # Capture model is missing in the current report -> skipped (not in deltas).
    assert "Capture model" not in by_name

    # Missing baseline file -> warning + skipped, never raises.
    missing = compare_to_baseline(current, str(tmp_path / "absent.json"))
    assert missing["skipped"] == "baseline_missing"
    assert missing["regressed"] is False

    # phase_list_version mismatch -> warning + skipped, never raises.
    stale = {"phase_list_version": 999, "phases": []}
    stale_path = tmp_path / "stale.json"
    stale_path.write_text(json.dumps(stale))
    mismatch = compare_to_baseline(current, str(stale_path))
    assert mismatch["skipped"] == "version_mismatch"
    assert mismatch["regressed"] is False

    # format_phase_report emits one aligned row per phase (seconds to 3 dp).
    formatted = format_phase_report(current)
    assert "Worker init" in formatted
    assert "MISSING" in formatted  # Capture model is missing
    lines = [ln for ln in formatted.split("\n") if ln.strip()]
    assert len(lines) == 6  # one row per canonical phase
