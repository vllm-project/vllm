# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only test of the attributes OutputProcessor.do_tracing puts on the
llm_request span, without starting an engine."""

from types import SimpleNamespace

import pytest

from vllm.tracing import SpanAttributes
from vllm.v1.engine import FinishReason
from vllm.v1.engine import output_processor as op


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [(FinishReason.STOP, ["stop"]), (FinishReason.LENGTH, ["length"]), (None, [])],
)
def test_do_tracing_genai_attributes(monkeypatch, finish_reason, expected):
    captured: dict = {}

    def fake_instrument_manual(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(op, "instrument_manual", fake_instrument_manual)

    stats = SimpleNamespace(
        arrival_time=100.0,
        queued_ts=100.1,
        scheduled_ts=100.2,
        first_token_ts=100.5,
        last_token_ts=101.0,
        first_token_latency=0.5,
        num_generation_tokens=7,
    )
    req_state = SimpleNamespace(
        stats=stats,
        prompt_token_ids=[1, 2, 3, 4, 5],
        prompt_embeds=None,
        num_cached_tokens=3,
        external_req_id="req-1",
        top_p=None,
        max_tokens_param=None,
        temperature=None,
        n=None,
    )
    engine_core_output = SimpleNamespace(trace_headers=None)
    iteration_stats = SimpleNamespace(iteration_timestamp=101.0)

    op.OutputProcessor.do_tracing(
        None, engine_core_output, req_state, iteration_stats, finish_reason
    )

    attrs = captured["attributes"]
    assert attrs[SpanAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 5
    assert attrs[SpanAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 7
    assert attrs[SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 3
    assert attrs[SpanAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == expected
    # The deprecated names stay for existing consumers.
    assert attrs[SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS] == 5
    assert attrs[SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS] == 7
    assert SpanAttributes.GEN_AI_USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"
    assert SpanAttributes.GEN_AI_USAGE_OUTPUT_TOKENS == "gen_ai.usage.output_tokens"
    assert (
        SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS
        == "gen_ai.usage.cache_read.input_tokens"
    )
    assert (
        SpanAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        == "gen_ai.response.finish_reasons"
    )
