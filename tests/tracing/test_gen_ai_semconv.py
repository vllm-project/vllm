# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the current OTel GenAI semconv opt-in helpers.

These are pure-function tests (no engine / GPU required)."""

import pytest

from vllm.tracing import (
    SpanAttributes,
    is_gen_ai_latest_semconv_enabled,
    latest_gen_ai_semconv_attributes,
)

OPT_IN = "OTEL_SEMCONV_STABILITY_OPT_IN"


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, False),  # unset -> legacy only
        ("", False),
        ("gen_ai_latest_experimental", True),  # exact token
        ("http/dup,gen_ai_latest_experimental", True),  # one of several tokens
        (" gen_ai_latest_experimental , http ", True),  # whitespace tolerated
        ("gen_ai_latest", False),  # partial token must not match
        ("database/dup", False),  # unrelated opt-in
    ],
)
def test_is_gen_ai_latest_semconv_enabled(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool
):
    monkeypatch.delenv(OPT_IN, raising=False)
    if value is not None:
        monkeypatch.setenv(OPT_IN, value)
    assert is_gen_ai_latest_semconv_enabled() is expected


def test_latest_semconv_attributes_both_known():
    attrs, span_name = latest_gen_ai_semconv_attributes("chat", "facebook/opt-125m")
    assert attrs == {
        SpanAttributes.GEN_AI_OPERATION_NAME: "chat",
        SpanAttributes.GEN_AI_REQUEST_MODEL: "facebook/opt-125m",
    }
    # Spec span name: "{operation} {model}".
    assert span_name == "chat facebook/opt-125m"


def test_latest_semconv_attributes_missing_operation():
    # Offline / non-wired endpoints: operation unknown -> omitted, keep default
    # span name (span_name is None).
    attrs, span_name = latest_gen_ai_semconv_attributes(None, "m")
    assert attrs == {SpanAttributes.GEN_AI_REQUEST_MODEL: "m"}
    assert span_name is None


def test_latest_semconv_attributes_missing_model():
    attrs, span_name = latest_gen_ai_semconv_attributes("chat", None)
    assert attrs == {SpanAttributes.GEN_AI_OPERATION_NAME: "chat"}
    assert span_name is None


def test_latest_semconv_attributes_both_unknown():
    assert latest_gen_ai_semconv_attributes(None, None) == ({}, None)
