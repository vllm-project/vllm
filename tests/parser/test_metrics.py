# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for parser Prometheus metrics.

These cover ``vllm.parser.metrics``, which records the
``vllm:tool_call_parser_invocations_total`` counter. The module keeps
process-global state (a cached model label and a lazily created counter), so
each test uses a unique ``model_name`` label and asserts on before/after deltas
read through the registry to stay isolated from other tests.
"""

from itertools import product

import pytest
from prometheus_client import REGISTRY

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser import metrics

# These tests are pure-CPU and never touch an accelerator, so skip the global
# GPU cleanup fixture (consistent with the reasoning parser test suites).
pytestmark = pytest.mark.skip_global_cleanup

_COUNTER = "vllm:tool_call_parser_invocations_total"
_MODES = ("streaming", "non_streaming")
_OUTCOMES = ("tool_call", "no_tool_call")
_REQUEST_TYPES = ("chat_completions", "responses", "other")


def _sample(model_name, mode, outcome, request_type):
    """Read the current counter value for one label combination."""
    return REGISTRY.get_sample_value(
        _COUNTER,
        {
            "model_name": model_name,
            "mode": mode,
            "outcome": outcome,
            "request_type": request_type,
        },
    )


def _chat_completion_request():
    return ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])


def _responses_request():
    return ResponsesRequest(input="hi")


def test_record_is_noop_before_init(monkeypatch):
    """Recording before init_parser_metrics must not raise or register."""
    # The counter is process-global; force the uninitialized state so this
    # test is independent of whether other tests already ran init.
    monkeypatch.setattr(metrics, "_tool_call_parser_invocations", None)

    # Must return silently rather than raising when the counter is absent.
    metrics.record_tool_parser_invocation(
        is_tool_called=True, is_streaming=False, request=object()
    )


def test_init_preinitializes_all_label_combinations():
    """init_parser_metrics seeds every (mode, outcome, request_type) at zero."""
    model_name = "test-init-all-labels"
    metrics.init_parser_metrics(model_name=model_name)

    for mode, outcome, request_type in product(_MODES, _OUTCOMES, _REQUEST_TYPES):
        value = _sample(model_name, mode, outcome, request_type)
        assert value == 0.0, (mode, outcome, request_type, value)


def test_init_is_idempotent():
    """Re-initializing reuses the existing collector instead of raising."""
    metrics.init_parser_metrics(model_name="test-idempotent-1")
    # A second call would hit prometheus' duplicate-registration ValueError if
    # the reuse branch regressed; it must return cleanly.
    metrics.init_parser_metrics(model_name="test-idempotent-2")


def test_record_true_increments_tool_call():
    model_name = "test-true-outcome"
    metrics.init_parser_metrics(model_name=model_name)
    before = _sample(model_name, "non_streaming", "tool_call", "other")

    metrics.record_tool_parser_invocation(
        is_tool_called=True, is_streaming=False, request=object()
    )

    after = _sample(model_name, "non_streaming", "tool_call", "other")
    assert after == before + 1.0


def test_record_false_increments_no_tool_call():
    model_name = "test-false-outcome"
    metrics.init_parser_metrics(model_name=model_name)
    before = _sample(model_name, "non_streaming", "no_tool_call", "other")

    metrics.record_tool_parser_invocation(
        is_tool_called=False, is_streaming=False, request=object()
    )

    after = _sample(model_name, "non_streaming", "no_tool_call", "other")
    assert after == before + 1.0


def test_record_exception_counts_as_no_tool_call():
    """A parser exception is currently bucketed as no_tool_call (documented)."""
    model_name = "test-exception-outcome"
    metrics.init_parser_metrics(model_name=model_name)
    before = _sample(model_name, "non_streaming", "no_tool_call", "other")

    metrics.record_tool_parser_invocation(
        is_tool_called=ValueError("parse failed"),
        is_streaming=False,
        request=object(),
    )

    after = _sample(model_name, "non_streaming", "no_tool_call", "other")
    assert after == before + 1.0


def test_record_streaming_uses_streaming_mode_label():
    model_name = "test-streaming-mode"
    metrics.init_parser_metrics(model_name=model_name)
    streaming_before = _sample(model_name, "streaming", "tool_call", "other")
    non_streaming_before = _sample(model_name, "non_streaming", "tool_call", "other")

    metrics.record_tool_parser_invocation(
        is_tool_called=True, is_streaming=True, request=object()
    )

    assert _sample(model_name, "streaming", "tool_call", "other") == (
        streaming_before + 1.0
    )
    # The non-streaming bucket must be untouched by a streaming invocation.
    assert (
        _sample(model_name, "non_streaming", "tool_call", "other")
        == non_streaming_before
    )


def test_record_maps_chat_completion_request_type():
    model_name = "test-chat-request-type"
    metrics.init_parser_metrics(model_name=model_name)
    before = _sample(model_name, "non_streaming", "tool_call", "chat_completions")

    metrics.record_tool_parser_invocation(
        is_tool_called=True,
        is_streaming=False,
        request=_chat_completion_request(),
    )

    after = _sample(model_name, "non_streaming", "tool_call", "chat_completions")
    assert after == before + 1.0


def test_record_maps_responses_request_type():
    model_name = "test-responses-request-type"
    metrics.init_parser_metrics(model_name=model_name)
    before = _sample(model_name, "non_streaming", "tool_call", "responses")

    metrics.record_tool_parser_invocation(
        is_tool_called=True,
        is_streaming=False,
        request=_responses_request(),
    )

    after = _sample(model_name, "non_streaming", "tool_call", "responses")
    assert after == before + 1.0


def test_record_maps_unknown_request_type_to_other():
    model_name = "test-other-request-type"
    metrics.init_parser_metrics(model_name=model_name)
    before = _sample(model_name, "non_streaming", "tool_call", "other")

    metrics.record_tool_parser_invocation(
        is_tool_called=True, is_streaming=False, request=object()
    )

    after = _sample(model_name, "non_streaming", "tool_call", "other")
    assert after == before + 1.0


@pytest.mark.parametrize(
    "outcome_enum, expected_label",
    [
        (metrics.ToolCallOutcome.TOOL_CALL, "tool_call"),
        (metrics.ToolCallOutcome.NO_TOOL_CALL, "no_tool_call"),
    ],
)
def test_tool_call_outcome_enum_values(outcome_enum, expected_label):
    assert outcome_enum.value == expected_label


@pytest.mark.parametrize(
    "request_type_enum, expected_label",
    [
        (metrics.RequestType.CHAT_COMPLETIONS, "chat_completions"),
        (metrics.RequestType.RESPONSES, "responses"),
        (metrics.RequestType.OTHER, "other"),
    ],
)
def test_request_type_enum_values(request_type_enum, expected_label):
    assert request_type_enum.value == expected_label
