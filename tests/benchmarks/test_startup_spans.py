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
    build_phase_report,
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
