# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the per-phase startup span collection in `vllm bench startup`.

``test_startup_per_phase_spans`` boots ``LLM(model="facebook/opt-125m")``
against the in-memory OTLP receiver (no mocking of the function under test),
so the spans come from the real ``@instrument`` call sites. The CLI smoke test
runs ``vllm bench startup --per-phase`` end-to-end. The remaining tests cover
the pure span decoding/aggregation without a GPU.
"""

import json
import subprocess
import time

import pytest
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_TRACES_INSECURE

from vllm import LLM
from vllm.benchmarks.startup_spans import (
    collect_startup_spans,
    phase_durations,
    span_sink,
)
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform


def test_startup_per_phase_spans(monkeypatch: pytest.MonkeyPatch):
    """Every startup span that arrives is reported with its identity.

    The consumer does not carry a phase allowlist: whatever ``@instrument``
    emits during the boot is reported, keyed by span name, with the
    ``span_id``/``parent_span_id`` that identify the emission.
    """
    with monkeypatch.context() as m:
        m.setenv(OTEL_EXPORTER_OTLP_TRACES_INSECURE, "true")
        # gRPC's C-core is not fork-safe; mirror test_traces' spawn setting.
        m.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        llm = None
        with span_sink() as sink:
            try:
                llm = LLM(
                    model="facebook/opt-125m",
                    otlp_traces_endpoint=sink.address,
                    gpu_memory_utilization=0.3,
                )

                # BatchSpanProcessor exports on a timer; poll for the loader
                # span with the same budget tests/tracing/test_tracing.py uses.
                # "Load model" is instrumented on every loader; "Init device"
                # only on the GPU worker, so it is not required on CPU/XPU.
                expected = {"Load model"}
                if current_platform.is_cuda() or current_platform.is_rocm():
                    expected.add("Init device")
                deadline = time.time() + 30.0
                while time.time() < deadline:
                    if expected <= {s["name"] for s in sink.get_all_spans()}:
                        break
                    time.sleep(0.5)

                spans = collect_startup_spans(sink)
            finally:
                if llm is not None:
                    shutdown_timeout = 60.0 if current_platform.is_rocm() else 5.0
                    llm.llm_engine.engine_core.shutdown(timeout=shutdown_timeout)
                cleanup_dist_env_and_memory()

        names = {span["name"] for span in spans}
        assert expected <= names, f"missing startup spans; got {sorted(names)}"

        for span in spans:
            # No span is dropped or collapsed: each keeps its own identity.
            assert span["trace_id"], span
            assert span["span_id"], span
            assert span["duration_s"] >= 0.0, (
                f"span '{span['name']}' has negative duration"
            )

        durations = phase_durations(spans)
        assert set(durations) == names
        assert all(value >= 0.0 for value in durations.values())


@pytest.mark.benchmark
def test_bench_startup_per_phase_flag(tmp_path):
    """CLI smoke: ``vllm bench startup --per-phase`` exits 0, prints the
    per-phase metrics and writes them as regular metric keys in --output-json.
    Guards the lazy import, the sink lifecycle and the MetricDesc wiring in
    ``vllm/benchmarks/startup.py``."""
    output = tmp_path / "startup.json"
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
        "1",
        "--per-phase",
        "--output-json",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, (
        f"Benchmark failed (rc={result.returncode}): {result.stderr}"
    )
    assert "load model phase" in result.stdout.lower(), (
        "per-phase metric not found in stdout; got:\n" + result.stdout
    )
    metrics = json.loads(output.read_text())
    assert "avg_cold_load_model_time" in metrics
    assert len(metrics["cold_load_model_times"]) == 1


def test_per_phase_rejects_user_otlp_endpoint(monkeypatch):
    """``--per-phase`` owns the OTLP endpoint; a user-provided one is an error
    instead of being silently overwritten."""
    from types import SimpleNamespace

    from vllm.benchmarks import startup

    # The endpoint check runs before anything starts; stub the process-global
    # start method so this test cannot change it for the rest of the shard.
    monkeypatch.setattr(
        startup,
        "multiprocessing",
        SimpleNamespace(set_start_method=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        startup.EngineArgs,
        "from_cli_args",
        classmethod(lambda cls, args: SimpleNamespace(otlp_traces_endpoint=args.otlp)),
    )
    with pytest.raises(ValueError, match="--otlp-traces-endpoint"):
        startup.main(
            SimpleNamespace(
                serve_cmd=None,
                per_phase=True,
                otlp="localhost:4317",
            )
        )


# ---------------------------------------------------------------------------
# Non-GPU unit tests for the pure span decoding and aggregation.
# ---------------------------------------------------------------------------


def test_in_memory_span_sink_keeps_span_identity():
    """The sink decodes trace/span identity over gRPC, so a span stays
    attributable after it leaves the exporter."""
    import grpc
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import (
        TraceServiceStub,
    )
    from opentelemetry.proto.trace.v1.trace_pb2 import Span

    with span_sink() as sink:
        channel = grpc.insecure_channel(sink.address)
        grpc.channel_ready_future(channel).result(timeout=5.0)
        stub = TraceServiceStub(channel)

        span = Span(
            name="Load model",
            trace_id=b"\x21\x22\x23\x24\x25\x26\x27\x28\x29\x2a\x2b\x2c\x2d\x2e\x2f\x30",
            span_id=b"\x01\x02\x03\x04\x05\x06\x07\x08",
            parent_span_id=b"\x11\x12\x13\x14\x15\x16\x17\x18",
            start_time_unix_nano=100,
            end_time_unix_nano=200,
        )
        request = ExportTraceServiceRequest()
        request.resource_spans.add().scope_spans.add().spans.add().CopyFrom(span)

        stub.Export(request)
        channel.close()

        [collected] = collect_startup_spans(sink)

    assert collected["name"] == "Load model"
    assert collected["duration_s"] == pytest.approx(100e-9)
    assert collected["trace_id"] == "2122232425262728292a2b2c2d2e2f30"
    assert collected["span_id"] == "0102030405060708"
    assert collected["parent_span_id"] == "1112131415161718"


def test_collect_startup_spans_clamps_negative_duration():
    """A span whose end precedes its start (#40698) reports 0.0, never a
    negative duration."""
    spans = [
        {
            "name": "Load model",
            "start_time_unix_nano": 400,
            "end_time_unix_nano": 100,
            "trace_id": "trace",
            "span_id": "abcd",
            "parent_span_id": None,
        }
    ]

    class _Sink:
        def collect(self, **_kwargs):
            return spans

    [collected] = collect_startup_spans(_Sink())
    assert collected["duration_s"] == 0.0


def test_phase_durations_sums_disjoint_names_and_skips_nested():
    """A name emitted by several processes is summed; a span nested inside
    another span of the same name is skipped so its time is not counted twice."""
    spans = [
        {
            "name": "Worker init",
            "duration_s": 0.2,
            "trace_id": "t1",
            "span_id": "a",
            "parent_span_id": None,
        },
        {
            "name": "Worker init",
            "duration_s": 0.3,
            "trace_id": "t2",
            "span_id": "b",
            "parent_span_id": None,
        },
        {
            # Outer loader span...
            "name": "Load model",
            "duration_s": 1.5,
            "trace_id": "t1",
            "span_id": "c",
            "parent_span_id": "a",
        },
        {
            # ...wraps an inner loader span with the same name.
            "name": "Load model",
            "duration_s": 1.2,
            "trace_id": "t1",
            "span_id": "d",
            "parent_span_id": "c",
        },
    ]

    assert phase_durations(spans) == pytest.approx(
        {"Worker init": 0.5, "Load model": 1.5}
    )


def test_per_phase_metrics_use_the_metric_pipeline():
    """Per-span-name durations flow through MetricDesc as one metric each; an
    iteration that did not emit a name is skipped, not counted as zero."""
    from vllm.benchmarks.startup import _per_phase_metrics

    iterations = [
        {"per_phase": {"Load model": 1.0, "Warmup (GPU)": 0.5}},
        {"per_phase": {"Load model": 3.0}},
    ]
    metrics = {m.key: m for m in _per_phase_metrics("cold", iterations)}

    assert set(metrics) == {"cold_load_model", "cold_warmup_gpu"}
    assert metrics["cold_load_model"].values == [1.0, 3.0]
    assert metrics["cold_load_model"].avg == 2.0
    assert metrics["cold_warmup_gpu"].values == [0.5]
