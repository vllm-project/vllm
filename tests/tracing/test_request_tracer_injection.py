# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-level spans must follow the tracer the engine configured.

If another OpenTelemetry integration (e.g. an auto-instrumentation
distribution) installs the global tracer provider before vLLM starts,
spans resolved from the global provider are misrouted to that
integration's exporter instead of the endpoint given via
--otlp-traces-endpoint. The engines therefore keep the tracer returned
by init_tracer() and hand it down to OutputProcessor, which creates the
request spans. See https://github.com/vllm-project/vllm/issues/69661.
"""

import time
from types import SimpleNamespace

import opentelemetry.trace as otel_trace
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from vllm.tracing import instrument_manual, is_otel_available

pytestmark = pytest.mark.skipif(
    not is_otel_available(), reason="OpenTelemetry is not installed"
)


@pytest.fixture()
def foreign_global_provider():
    """Install a foreign provider as the global one, and restore after.

    Stands in for an auto-instrumentation distribution that installs
    itself at interpreter start, before any vLLM code runs. The SDK's
    set_tracer_provider() refuses overrides, so swap the module-level
    global directly.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    original = otel_trace._TRACER_PROVIDER
    otel_trace._TRACER_PROVIDER = provider
    try:
        yield exporter
    finally:
        otel_trace._TRACER_PROVIDER = original
        provider.shutdown()


def make_tracer(exporter: InMemorySpanExporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("vllm.llm_engine")


def test_global_provider_used_when_no_tracer_passed(foreign_global_provider):
    """Without an explicit tracer, spans still resolve the global provider."""
    instrument_manual(span_name="llm_request", start_time=time.time_ns())

    names = [s.name for s in foreign_global_provider.get_finished_spans()]
    assert names == ["llm_request"]


def test_explicit_tracer_bypasses_global_provider(foreign_global_provider):
    """A passed tracer wins over whichever provider is installed globally."""
    vllm_exporter = InMemorySpanExporter()
    vllm_tracer = make_tracer(vllm_exporter)

    instrument_manual(
        span_name="llm_request",
        start_time=time.time_ns(),
        attributes={"gen_ai.request.model": "test-model"},
        tracer=vllm_tracer,
    )

    spans = vllm_exporter.get_finished_spans()
    assert [s.name for s in spans] == ["llm_request"]
    assert spans[0].attributes["gen_ai.request.model"] == "test-model"
    # Nothing was misrouted to the foreign global provider.
    assert len(foreign_global_provider.get_finished_spans()) == 0


def test_output_processor_passes_tracer_to_instrument_manual(monkeypatch):
    """OutputProcessor forwards the tracer it was constructed with."""
    import vllm.v1.engine.output_processor as op_mod
    from vllm.v1.engine.output_processor import OutputProcessor

    sentinel_tracer = object()
    captured: dict = {}

    def record(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(op_mod, "instrument_manual", record)

    processor = OutputProcessor(
        None,
        collect_stats=False,
        tracing_enabled=True,
        tracer=sentinel_tracer,
    )

    req_state = SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        prompt_embeds=None,
        external_req_id="req-1",
        top_p=None,
        max_tokens_param=None,
        temperature=None,
        n=None,
        stats=SimpleNamespace(
            arrival_time=1.0,
            scheduled_ts=2.0,
            queued_ts=1.5,
            first_token_ts=3.0,
            last_token_ts=4.0,
            first_token_latency=1.0,
            num_generation_tokens=5,
        ),
    )
    iteration_stats = SimpleNamespace(iteration_timestamp=5.0)

    processor.do_tracing(
        SimpleNamespace(trace_headers=None), req_state, iteration_stats
    )

    assert captured["span_name"] == "llm_request"
    assert captured["tracer"] is sentinel_tracer


def test_output_processor_default_tracer_is_none(monkeypatch):
    """Callers that don't pass a tracer keep the legacy global behavior."""
    import vllm.v1.engine.output_processor as op_mod
    from vllm.v1.engine.output_processor import OutputProcessor

    captured: dict = {}

    def record(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(op_mod, "instrument_manual", record)

    processor = OutputProcessor(None, collect_stats=False, tracing_enabled=True)
    assert processor.tracer is None

    req_state = SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        prompt_embeds=None,
        external_req_id="req-2",
        top_p=None,
        max_tokens_param=None,
        temperature=None,
        n=None,
        stats=SimpleNamespace(
            arrival_time=1.0,
            scheduled_ts=2.0,
            queued_ts=1.5,
            first_token_ts=3.0,
            last_token_ts=4.0,
            first_token_latency=1.0,
            num_generation_tokens=5,
        ),
    )
    processor.do_tracing(
        SimpleNamespace(trace_headers=None),
        req_state,
        SimpleNamespace(iteration_timestamp=5.0),
    )

    assert captured["tracer"] is None
