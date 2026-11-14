# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the parser latency histograms."""

import os
import subprocess
import sys
from collections.abc import Callable

import pytest
from prometheus_client import REGISTRY, Histogram

from tests.parser.engine.replay_harness import MockTokenizer
from tests.parser.metrics_utils import reset_parser_metrics
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.stream_processor import (
    ChatStreamProcessor,
    observe_stream_processors,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser import Parser, ParserManager, metrics
from vllm.parser.metrics import (
    _PARSER_DURATION_SECONDS,
    _REQUEST_PARSER_DURATION_SECONDS,
    Endpoint,
    ParserMetrics,
    ParserMode,
    RequestType,
    init_parser_metrics,
    record_tool_parser_invocation,
)

pytestmark = pytest.mark.cpu_test

MODEL = "model"
PARSER = "hermes"
INVOCATION = _PARSER_DURATION_SECONDS
REQUEST = _REQUEST_PARSER_DURATION_SECONDS


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    """Isolate the global registry and the module's pre-bound children."""
    monkeypatch.setattr(metrics.envs, "VLLM_PARSER_SLOW_LOG", False)
    reset_parser_metrics()
    yield
    reset_parser_metrics()


def _value(name: str, suffix: str, *, parser: str = PARSER, **labels) -> float | None:
    return REGISTRY.get_sample_value(
        f"{name}{suffix}",
        {"model_name": MODEL, "parser": parser, **labels},
    )


def _counter_value(*, mode: str, outcome: str, request_type: str) -> float | None:
    return REGISTRY.get_sample_value(
        "vllm:tool_call_parser_invocations_total",
        {
            "model_name": MODEL,
            "mode": mode,
            "outcome": outcome,
            "request_type": request_type,
        },
    )


def _tokenizer() -> MockTokenizer:
    return MockTokenizer({}, [])


def _chat_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "hi"}],
    )


def _responses_request() -> ResponsesRequest:
    return ResponsesRequest(model=MODEL, input="hi")


def _parser_cls(tokenizer: MockTokenizer) -> type[Parser]:
    parser_cls = ParserManager.get_parser(
        tool_parser_name=PARSER,
        enable_auto_tools=True,
        model_name=MODEL,
        tokenizer=tokenizer,
    )
    assert parser_cls is not None
    return parser_cls


def _processor(
    tokenizer: MockTokenizer,
    request: ChatCompletionRequest,
    mode: ParserMode = ParserMode.STREAMING,
    endpoint: Endpoint = Endpoint.CHAT_COMPLETIONS,
    request_id: str | None = None,
) -> ChatStreamProcessor:
    parser_cls = _parser_cls(tokenizer)
    parser = parser_cls(tokenizer, request.tools, chat_template_kwargs={})
    return ChatStreamProcessor(
        request,
        parser,
        enable_auto_tools=True,
        parser_cls=parser_cls,
        mode=mode,
        endpoint=endpoint,
        request_id=request_id,
    )


def _init_hermes(*, log_request_ids: bool = False) -> None:
    init_parser_metrics(
        model_name=MODEL,
        tool_parser=PARSER,
        enable_auto_tool_choice=True,
        log_request_ids=log_request_ids,
    )


def _enable_slow_log(
    monkeypatch,
    *,
    call_s: float = 0.25,
    request_s: float = 5.0,
    log_request_ids: bool = False,
) -> None:
    monkeypatch.setattr(metrics.envs, "VLLM_PARSER_SLOW_LOG", True)
    monkeypatch.setattr(metrics.envs, "VLLM_PARSER_SLOW_CALL_SECONDS", call_s)
    monkeypatch.setattr(metrics.envs, "VLLM_PARSER_SLOW_REQUEST_SECONDS", request_s)
    _init_hermes(log_request_ids=log_request_ids)


def _fake_clock(monkeypatch, *times: float) -> None:
    values = iter(times)
    monkeypatch.setattr(metrics, "_clock", lambda: next(values))


def _run_three_steps(processor: ChatStreamProcessor) -> None:
    processor.step("one ", [1], None, None)
    processor.step("two ", [2], None, None)
    processor.step("three", [3], "stop", None)


@pytest.mark.parametrize(
    ("reasoning_parser", "tool_parser", "expected"),
    [
        ("deepseek_r1", None, "deepseek_r1"),
        (None, PARSER, PARSER),
        ("deepseek_r1", PARSER, f"deepseek_r1+{PARSER}"),
    ],
)
def test_parser_label_reflects_configured_parsers(
    reasoning_parser: str | None,
    tool_parser: str | None,
    expected: str,
) -> None:
    init_parser_metrics(
        model_name=MODEL,
        reasoning_parser=reasoning_parser,
        tool_parser=tool_parser,
        enable_auto_tool_choice=True,
    )

    parser_metrics = ParserMetrics(ParserMode.NON_STREAMING, Endpoint.OTHER)
    with parser_metrics.time_call():
        pass
    parser_metrics.observe()

    for name in (INVOCATION, REQUEST):
        assert (
            _value(
                name,
                "_count",
                parser=expected,
                mode=ParserMode.NON_STREAMING.value,
                endpoint=Endpoint.OTHER.value,
            )
            == 1.0
        )


def test_init_requires_an_enabled_parser() -> None:
    init_parser_metrics(
        model_name=MODEL,
        tool_parser=PARSER,
        enable_auto_tool_choice=False,
    )

    assert (
        _value(
            INVOCATION,
            "_count",
            mode=ParserMode.STREAMING.value,
            endpoint=Endpoint.CHAT_COMPLETIONS.value,
        )
        is None
    )


def test_children_bind_on_first_use() -> None:
    """Init registers the families but exports no label children."""
    _init_hermes()

    labels = {
        "mode": ParserMode.NON_STREAMING.value,
        "endpoint": Endpoint.CHAT_COMPLETIONS.value,
    }
    assert _value(INVOCATION, "_count", **labels) is None
    assert _value(REQUEST, "_count", **labels) is None

    parser_metrics = ParserMetrics(ParserMode.NON_STREAMING, Endpoint.CHAT_COMPLETIONS)
    with parser_metrics.time_call():
        pass
    parser_metrics.observe()

    assert _value(INVOCATION, "_count", **labels) == 1.0
    assert _value(REQUEST, "_count", **labels) == 1.0

    # Only the label set that was used is exported.
    assert (
        _value(
            INVOCATION,
            "_count",
            mode=ParserMode.NON_STREAMING.value,
            endpoint=Endpoint.RESPONSES.value,
        )
        is None
    )


def test_reinit_reuses_registered_collectors() -> None:
    _init_hermes()
    _init_hermes()

    labels = {
        "mode": ParserMode.NON_STREAMING.value,
        "endpoint": Endpoint.CHAT_COMPLETIONS.value,
    }
    parser_metrics = ParserMetrics(ParserMode.NON_STREAMING, Endpoint.CHAT_COMPLETIONS)
    with parser_metrics.time_call():
        pass
    assert _value(INVOCATION, "_count", **labels) == 1.0


def test_conflicting_collector_shape_fails_loudly() -> None:
    Histogram(
        "vllm:parser_duration_seconds",
        "conflict",
        ["model_name"],
        buckets=(1.0,),
    )

    with pytest.raises(ValueError, match="Duplicated timeseries"):
        _init_hermes()


def test_observe_invocation_records_elapsed(monkeypatch) -> None:
    _init_hermes()
    times = iter([1.0, 1.25])
    monkeypatch.setattr(metrics, "_clock", lambda: next(times))

    parser_metrics = ParserMetrics(ParserMode.NON_STREAMING, Endpoint.CHAT_COMPLETIONS)
    with parser_metrics.time_call():
        pass

    assert parser_metrics.seconds == pytest.approx(0.25)
    labels = {
        "mode": ParserMode.NON_STREAMING.value,
        "endpoint": Endpoint.CHAT_COMPLETIONS.value,
    }
    assert _value(INVOCATION, "_count", **labels) == 1.0
    assert _value(INVOCATION, "_sum", **labels) == pytest.approx(0.25)


def test_request_histogram_skips_replay_and_zero_calls() -> None:
    _init_hermes()

    replay = ParserMetrics(ParserMode.STREAMING_REPLAY, Endpoint.CHAT_COMPLETIONS)
    replay.seconds = 0.5
    replay.calls = 1
    replay.observe()

    unused = ParserMetrics(ParserMode.NON_STREAMING, Endpoint.OTHER)
    unused.seconds = 0.5
    unused.observe()

    recorded = ParserMetrics(ParserMode.NON_STREAMING, Endpoint.RESPONSES)
    recorded.seconds = 0.5
    recorded.calls = 2
    recorded.observe()

    assert (
        _value(
            REQUEST,
            "_count",
            mode=ParserMode.STREAMING_REPLAY.value,
            endpoint=Endpoint.CHAT_COMPLETIONS.value,
        )
        is None
    )
    # Zero calls does not bind the label set.
    assert (
        _value(
            REQUEST,
            "_count",
            mode=ParserMode.NON_STREAMING.value,
            endpoint=Endpoint.OTHER.value,
        )
        is None
    )
    assert (
        _value(
            REQUEST,
            "_count",
            mode=ParserMode.NON_STREAMING.value,
            endpoint=Endpoint.RESPONSES.value,
        )
        == 1.0
    )


def test_processor_binds_histograms_on_first_parse() -> None:
    """A request that never parses must not export zero histogram series."""
    _init_hermes()
    processor = _processor(_tokenizer(), _chat_request())

    labels = {
        "mode": ParserMode.STREAMING.value,
        "endpoint": Endpoint.CHAT_COMPLETIONS.value,
    }
    assert _value(INVOCATION, "_count", **labels) is None
    assert _value(REQUEST, "_count", **labels) is None

    processor.step("hello", [1], "stop", None)

    assert _value(INVOCATION, "_count", **labels) == 1.0


def test_streaming_records_one_sample_per_delta() -> None:
    _init_hermes()
    processor = _processor(_tokenizer(), _chat_request(), ParserMode.STREAMING)

    processor.step("hello ", [1], None, None)
    processor.step("world", [2], "stop", None)

    labels = {
        "mode": ParserMode.STREAMING.value,
        "endpoint": Endpoint.CHAT_COMPLETIONS.value,
    }
    assert processor.metrics.calls == 2
    assert processor.metrics.seconds > 0
    assert _value(INVOCATION, "_count", **labels) == 2.0

    processor.metrics.observe()
    assert _value(REQUEST, "_count", **labels) == 1.0
    assert _value(REQUEST, "_sum", **labels) == pytest.approx(processor.metrics.seconds)


def test_replay_has_no_per_request_sample() -> None:
    _init_hermes()
    processor = _processor(_tokenizer(), _chat_request(), ParserMode.STREAMING_REPLAY)

    processor.step("hello ", [1], None, None)
    processor.step("world", [2], "stop", None)

    labels = {
        "mode": ParserMode.STREAMING_REPLAY.value,
        "endpoint": Endpoint.CHAT_COMPLETIONS.value,
    }
    assert processor.metrics.calls == 2
    assert _value(INVOCATION, "_count", **labels) == 2.0

    processor.metrics.observe()
    assert _value(REQUEST, "_count", **labels) is None


@pytest.mark.parametrize(
    "endpoint",
    [Endpoint.CHAT_COMPLETIONS, Endpoint.OTHER],
)
def test_non_streaming_parse_records_one_invocation_and_request(
    endpoint: Endpoint,
) -> None:
    """A per-request timer records both histograms with the caller's labels."""
    _init_hermes()
    tokenizer = _tokenizer()
    request = _chat_request()
    parser_cls = _parser_cls(tokenizer)
    parser = parser_cls(tokenizer, request.tools, chat_template_kwargs={})

    parser_metrics = ParserMetrics(ParserMode.NON_STREAMING, endpoint)
    with parser_metrics.time_call():
        parser.parse("hello", request, enable_auto_tools=True)
    parser_metrics.observe()

    labels = {
        "mode": ParserMode.NON_STREAMING.value,
        "endpoint": endpoint.value,
    }
    assert _value(INVOCATION, "_count", **labels) == 1.0
    assert _value(REQUEST, "_count", **labels) == 1.0


def test_request_histogram_records_one_sample_per_accumulator(monkeypatch) -> None:
    """Several calls sum into one request sample, not one sample per call."""
    _init_hermes()
    _fake_clock(monkeypatch, 0.0, 0.4, 1.0, 1.4)
    parser_metrics = ParserMetrics(ParserMode.NON_STREAMING, Endpoint.CHAT_COMPLETIONS)

    with parser_metrics.time_call():
        pass
    with parser_metrics.time_call():
        pass
    parser_metrics.observe()

    labels = {
        "mode": ParserMode.NON_STREAMING.value,
        "endpoint": Endpoint.CHAT_COMPLETIONS.value,
    }
    assert _value(INVOCATION, "_count", **labels) == 2.0
    assert _value(REQUEST, "_count", **labels) == 1.0
    assert _value(REQUEST, "_sum", **labels) == pytest.approx(0.8)


def test_repeated_observe_records_one_request_sample(monkeypatch) -> None:
    """A late observe() after a failure must not double-count the request."""
    _init_hermes()
    _fake_clock(monkeypatch, 0.0, 0.4)
    parser_metrics = ParserMetrics(ParserMode.NON_STREAMING, Endpoint.CHAT_COMPLETIONS)

    with parser_metrics.time_call():
        pass
    parser_metrics.observe()
    parser_metrics.observe()

    labels = {
        "mode": ParserMode.NON_STREAMING.value,
        "endpoint": Endpoint.CHAT_COMPLETIONS.value,
    }
    assert _value(REQUEST, "_count", **labels) == 1.0
    assert _value(REQUEST, "_sum", **labels) == pytest.approx(0.4)


def test_non_streaming_request_sampled_when_parse_raises(monkeypatch) -> None:
    """The timer records the request when its caller's observe() is skipped."""
    _init_hermes()
    _fake_clock(monkeypatch, 0.0, 0.5)
    parser_metrics = ParserMetrics(ParserMode.NON_STREAMING, Endpoint.CHAT_COMPLETIONS)

    with (
        pytest.raises(RuntimeError, match="parse failed"),
        parser_metrics.time_call(),
    ):
        raise RuntimeError("parse failed")

    labels = {
        "mode": ParserMode.NON_STREAMING.value,
        "endpoint": Endpoint.CHAT_COMPLETIONS.value,
    }
    assert _value(INVOCATION, "_count", **labels) == 1.0
    assert _value(REQUEST, "_count", **labels) == 1.0
    assert _value(REQUEST, "_sum", **labels) == pytest.approx(0.5)


def test_streaming_choice_not_sampled_when_parse_raises(monkeypatch) -> None:
    """Streaming choices are merged into one summary, so they never self-observe."""
    _init_hermes()
    _fake_clock(monkeypatch, 0.0, 0.5)
    parser_metrics = ParserMetrics(ParserMode.STREAMING, Endpoint.CHAT_COMPLETIONS)

    with (
        pytest.raises(RuntimeError, match="parse failed"),
        parser_metrics.time_call(),
    ):
        raise RuntimeError("parse failed")

    labels = {
        "mode": ParserMode.STREAMING.value,
        "endpoint": Endpoint.CHAT_COMPLETIONS.value,
    }
    assert _value(INVOCATION, "_count", **labels) == 1.0
    # The label set is bound, but no request sample was recorded.
    assert _value(REQUEST, "_count", **labels) == 0.0


def test_finish_reason_accumulates_across_choices() -> None:
    """A hoisted accumulator reports every choice's finish reason."""
    parser_metrics = ParserMetrics(ParserMode.NON_STREAMING, Endpoint.CHAT_COMPLETIONS)
    parser_metrics.finish_reason = "stop"
    parser_metrics.finish_reason = "length"
    assert parser_metrics.finish_reason == "length,stop"


def test_tool_counter_takes_mode_from_timer() -> None:
    """Replay reports streaming_replay; request_type still follows the request."""
    _init_hermes()

    with ParserMetrics(
        ParserMode.STREAMING_REPLAY,
        Endpoint.OTHER,
    ).time_call():
        record_tool_parser_invocation(
            is_tool_called=False,
            is_streaming=True,
            request=_chat_request(),
        )

    labels = {
        "mode": ParserMode.STREAMING_REPLAY.value,
        "outcome": "no_tool_call",
    }
    assert (
        _counter_value(request_type=RequestType.CHAT_COMPLETIONS.value, **labels) == 1.0
    )
    # Endpoint.OTHER does not move a chat request to `other`.
    assert _counter_value(request_type=RequestType.OTHER.value, **labels) is None


@pytest.mark.parametrize(
    ("endpoint", "request_factory", "request_type"),
    [
        (Endpoint.CHAT_COMPLETIONS, _chat_request, RequestType.CHAT_COMPLETIONS),
        (Endpoint.ANTHROPIC_MESSAGES, _chat_request, RequestType.CHAT_COMPLETIONS),
        (Endpoint.COHERE_CHAT, _chat_request, RequestType.CHAT_COMPLETIONS),
        (Endpoint.RESPONSES, _responses_request, RequestType.RESPONSES),
        # Batch and TITO derender parse chat requests under `other`.
        (Endpoint.OTHER, _chat_request, RequestType.CHAT_COMPLETIONS),
        (Endpoint.OTHER, _responses_request, RequestType.RESPONSES),
    ],
)
def test_tool_counter_keeps_legacy_request_type_label(
    endpoint: Endpoint,
    request_factory: Callable[[], object],
    request_type: RequestType,
) -> None:
    """The counter keeps `request_type`; the endpoint does not select it."""
    _init_hermes()

    with ParserMetrics(ParserMode.STREAMING, endpoint).time_call():
        record_tool_parser_invocation(
            is_tool_called=False,
            is_streaming=True,
            request=request_factory(),
        )

    assert (
        _counter_value(
            mode=ParserMode.STREAMING.value,
            outcome="no_tool_call",
            request_type=request_type.value,
        )
        == 1.0
    )


def test_tool_counter_keeps_legacy_zero_series_at_init() -> None:
    """Existing ratio and absent() queries rely on these series at startup."""
    _init_hermes()

    for mode in (ParserMode.STREAMING.value, ParserMode.NON_STREAMING.value):
        for outcome in ("tool_call", "no_tool_call"):
            for request_type in RequestType:
                assert (
                    _counter_value(
                        mode=mode,
                        outcome=outcome,
                        request_type=request_type.value,
                    )
                    == 0.0
                )


def test_tool_counter_falls_back_to_request_class() -> None:
    """Outside a timer the request class still selects the request type."""
    _init_hermes()

    record_tool_parser_invocation(
        is_tool_called=False,
        is_streaming=True,
        request=_chat_request(),
    )

    assert (
        _counter_value(
            mode=ParserMode.STREAMING.value,
            outcome="no_tool_call",
            request_type=RequestType.CHAT_COMPLETIONS.value,
        )
        == 1.0
    )


class _RaisingParser(Parser):
    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return False

    def parse(self, *args, **kwargs):
        raise RuntimeError("parse failed")

    def parse_delta(self, *args, **kwargs):
        raise RuntimeError("parse failed")


def test_failed_parse_still_counts_time(monkeypatch) -> None:
    """A parse that raises still pays its time toward the request totals."""
    _init_hermes()
    tokenizer = _tokenizer()
    processor = ChatStreamProcessor(
        _chat_request(),
        _RaisingParser(tokenizer),
        enable_auto_tools=True,
        parser_cls=_parser_cls(tokenizer),
    )
    times = iter([0.0, 0.5])
    monkeypatch.setattr(metrics, "_clock", lambda: next(times))

    with pytest.raises(RuntimeError, match="parse failed"):
        processor.step("hello", [1], None, None)

    assert processor.metrics.calls == 1
    assert processor.metrics.seconds == pytest.approx(0.5)
    assert _value(
        INVOCATION,
        "_sum",
        mode=ParserMode.STREAMING.value,
        endpoint=Endpoint.CHAT_COMPLETIONS.value,
    ) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "initialized", [False, True], ids=["uninitialized", "no_enabled_parser"]
)
def test_no_samples_without_an_enabled_parser(initialized) -> None:
    if initialized:
        init_parser_metrics(model_name=MODEL)

    parser_metrics = ParserMetrics(ParserMode.STREAMING, Endpoint.CHAT_COMPLETIONS)
    assert parser_metrics.time_call() is metrics._NOOP_TIMER
    with parser_metrics.time_call():
        pass

    unobserved = ParserMetrics(ParserMode.STREAMING, Endpoint.CHAT_COMPLETIONS)
    unobserved.seconds = 0.1
    unobserved.calls = 1
    unobserved.observe()

    labels = {
        "mode": ParserMode.STREAMING.value,
        "endpoint": Endpoint.CHAT_COMPLETIONS.value,
    }
    assert _value(INVOCATION, "_count", **labels) is None
    assert _value(REQUEST, "_count", **labels) is None


def test_serving_handlers_label_their_api_surface() -> None:
    """Anthropic and Cohere reuse chat serving but keep their own endpoint."""
    from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
    from vllm.entrypoints.cohere.serving import CohereServingChatV2
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    assert OpenAIServingChat.parser_endpoint is Endpoint.CHAT_COMPLETIONS
    assert AnthropicServingMessages.parser_endpoint is Endpoint.ANTHROPIC_MESSAGES
    assert CohereServingChatV2.parser_endpoint is Endpoint.COHERE_CHAT


def test_slow_call_warning_requires_opt_in(monkeypatch, caplog_vllm) -> None:
    monkeypatch.setattr(metrics.envs, "VLLM_PARSER_SLOW_LOG", False)
    _init_hermes()
    _fake_clock(monkeypatch, 0.0, 5.0)

    with ParserMetrics(ParserMode.STREAMING, Endpoint.CHAT_COMPLETIONS).time_call(
        tokens=1
    ):
        pass

    assert "parser_slow_call" not in caplog_vllm.text


def test_slow_call_warning_reports_fields(monkeypatch, caplog_vllm) -> None:
    _enable_slow_log(monkeypatch, call_s=0.25)
    _fake_clock(monkeypatch, 10.0, 10.3)
    parser_metrics = ParserMetrics(
        ParserMode.STREAMING,
        Endpoint.CHAT_COMPLETIONS,
        request_id="req-1",
    )

    with parser_metrics.time_call(tokens=7):
        pass

    message = caplog_vllm.text
    for fragment in (
        "event=parser_slow_call",
        "duration_s=0.300",
        "threshold_s=0.250",
        f"parser={PARSER}",
        "mode=streaming",
        "endpoint=chat_completions",
        "call_index=1",
        "tokens=7",
        "tokens_total=7",
    ):
        assert fragment in message
    # Request IDs stay out unless --enable-log-requests was set.
    assert "request_id=" not in message


def test_slow_call_warning_skips_fast_calls(monkeypatch, caplog_vllm) -> None:
    _enable_slow_log(monkeypatch, call_s=0.25)
    _fake_clock(monkeypatch, 0.0, 0.2)

    with ParserMetrics(ParserMode.STREAMING, Endpoint.CHAT_COMPLETIONS).time_call(
        tokens=1
    ):
        pass

    assert "parser_slow_call" not in caplog_vllm.text


def test_slow_call_warnings_capped_per_accumulator(monkeypatch, caplog_vllm) -> None:
    """A degraded stream logs a bounded number of warnings, not one per delta."""
    _enable_slow_log(monkeypatch, call_s=0.25, request_s=1.0)
    times = iter(float(i) for i in range(30))
    monkeypatch.setattr(metrics, "_clock", lambda: next(times))
    parser_metrics = ParserMetrics(ParserMode.STREAMING, Endpoint.CHAT_COMPLETIONS)

    for _ in range(15):
        with parser_metrics.time_call(tokens=1):
            pass
    parser_metrics.observe()

    assert caplog_vllm.text.count("event=parser_slow_call") == 10
    # The summary still reports every slow call, including the capped ones.
    assert "slow_calls=15" in caplog_vllm.text


def test_slow_log_master_switch_disables_both_events(monkeypatch, caplog_vllm) -> None:
    _enable_slow_log(monkeypatch, call_s=0.25, request_s=0.5)
    monkeypatch.setattr(metrics.envs, "VLLM_PARSER_SLOW_LOG", False)
    _init_hermes()
    _fake_clock(monkeypatch, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    parser_metrics = ParserMetrics(ParserMode.STREAMING, Endpoint.CHAT_COMPLETIONS)

    with parser_metrics.time_call(tokens=1):
        pass
    with parser_metrics.time_call(tokens=1):
        pass
    parser_metrics.observe()

    assert "parser_slow_call" not in caplog_vllm.text
    assert "parser_slow_request" not in caplog_vllm.text


def test_slow_request_warning_reports_aggregates(monkeypatch, caplog_vllm) -> None:
    _enable_slow_log(monkeypatch, call_s=10.0, request_s=1.0, log_request_ids=True)
    processor = _processor(_tokenizer(), _chat_request(), request_id="req-2")
    _fake_clock(monkeypatch, 0.0, 0.4, 1.0, 1.4, 2.0, 2.4)

    _run_three_steps(processor)
    observe_stream_processors([processor])

    message = caplog_vllm.text
    for fragment in (
        "event=parser_slow_request",
        "duration_s=1.200",
        "threshold_s=1.000",
        f"parser={PARSER}",
        "mode=streaming",
        "endpoint=chat_completions",
        "calls=3",
        "max_call_s=0.400",
        "slow_calls=0",
        "tokens_total=3",
        "finish_reason=stop",
        "request_id=req-2",
    ):
        assert fragment in message
    # The per-call threshold is high, so only the summary fired.
    assert "parser_slow_call" not in message


def test_slow_request_warning_suppressed_for_single_call(
    monkeypatch, caplog_vllm
) -> None:
    """The per-call event stands in for a summary when it fires."""
    _enable_slow_log(monkeypatch, call_s=0.25, request_s=1.0)
    processor = _processor(_tokenizer(), _chat_request())
    _fake_clock(monkeypatch, 0.0, 2.0)

    processor.step("one", [1], "stop", None)
    observe_stream_processors([processor])

    assert "event=parser_slow_call" in caplog_vllm.text
    assert "parser_slow_request" not in caplog_vllm.text


def test_single_call_summarized_when_per_call_event_does_not_fire(
    monkeypatch, caplog_vllm
) -> None:
    """A call between the two thresholds must not be swallowed by neither."""
    _enable_slow_log(monkeypatch, call_s=10.0, request_s=1.0)
    processor = _processor(_tokenizer(), _chat_request())
    _fake_clock(monkeypatch, 0.0, 2.0)

    processor.step("one", [1], "stop", None)
    observe_stream_processors([processor])

    assert "parser_slow_call" not in caplog_vllm.text
    assert "event=parser_slow_request" in caplog_vllm.text
    assert "calls=1" in caplog_vllm.text


def test_slow_request_summarizes_single_call_when_per_call_disabled(
    monkeypatch, caplog_vllm
) -> None:
    """The per-call event is the single-call summary's fallback, not a given."""
    _enable_slow_log(monkeypatch, call_s=0.0, request_s=1.0)
    processor = _processor(_tokenizer(), _chat_request())
    _fake_clock(monkeypatch, 0.0, 2.0)

    processor.step("one", [1], "stop", None)
    observe_stream_processors([processor])

    assert "parser_slow_call" not in caplog_vllm.text
    assert "event=parser_slow_request" in caplog_vllm.text
    assert "calls=1" in caplog_vllm.text


def test_slow_call_warning_suppressed_for_replay(monkeypatch, caplog_vllm) -> None:
    _enable_slow_log(monkeypatch, call_s=0.25, request_s=10.0)
    processor = _processor(
        _tokenizer(),
        _chat_request(),
        mode=ParserMode.STREAMING_REPLAY,
    )
    _fake_clock(monkeypatch, 0.0, 2.0)

    processor.step("one", [1], None, None)

    assert "parser_slow_call" not in caplog_vllm.text


def test_summary_reports_summed_calls(monkeypatch, caplog_vllm) -> None:
    """The one summary uses the request's final totals."""
    _enable_slow_log(monkeypatch, call_s=10.0, request_s=0.5)
    _fake_clock(monkeypatch, 0.0, 0.4, 1.0, 1.4)
    parser_metrics = ParserMetrics(ParserMode.NON_STREAMING, Endpoint.CHAT_COMPLETIONS)

    with parser_metrics.time_call(tokens=1):
        pass
    with parser_metrics.time_call(tokens=1):
        pass
    parser_metrics.observe()

    assert caplog_vllm.text.count("event=parser_slow_request") == 1
    assert "calls=2" in caplog_vllm.text


def test_slow_request_warning_suppressed_for_replay(monkeypatch, caplog_vllm) -> None:
    _enable_slow_log(monkeypatch, call_s=10.0, request_s=0.1)
    processor = _processor(
        _tokenizer(),
        _chat_request(),
        mode=ParserMode.STREAMING_REPLAY,
    )
    _fake_clock(monkeypatch, 0.0, 0.4, 1.0, 1.4, 2.0, 2.4)

    _run_three_steps(processor)
    observe_stream_processors([processor])

    assert "parser_slow_request" not in caplog_vllm.text


def test_slow_request_threshold_zero_disables_summary(monkeypatch, caplog_vllm) -> None:
    _enable_slow_log(monkeypatch, call_s=10.0, request_s=0.0)
    processor = _processor(_tokenizer(), _chat_request())
    _fake_clock(monkeypatch, 0.0, 0.4, 1.0, 1.4, 2.0, 2.4)

    _run_three_steps(processor)
    observe_stream_processors([processor])

    assert "parser_slow_request" not in caplog_vllm.text


def test_call_threshold_zero_leaves_summary_enabled(monkeypatch, caplog_vllm) -> None:
    _enable_slow_log(monkeypatch, call_s=0.0, request_s=1.0)
    processor = _processor(_tokenizer(), _chat_request())
    _fake_clock(monkeypatch, 0.0, 0.4, 1.0, 1.4, 2.0, 2.4)

    _run_three_steps(processor)
    observe_stream_processors([processor])

    assert "parser_slow_call" not in caplog_vllm.text
    assert "event=parser_slow_request" in caplog_vllm.text
    assert "slow_calls=0" in caplog_vllm.text


def test_slow_request_summary_emitted_once(monkeypatch, caplog_vllm) -> None:
    _enable_slow_log(monkeypatch, call_s=10.0, request_s=0.1)
    processor = _processor(_tokenizer(), _chat_request())
    _fake_clock(monkeypatch, 0.0, 0.4, 1.0, 1.4, 2.0, 2.4)

    _run_three_steps(processor)
    processor.metrics.observe()
    processor.metrics.observe()

    assert caplog_vllm.text.count("event=parser_slow_request") == 1


def test_slow_call_includes_request_id_with_log_requests(
    monkeypatch, caplog_vllm
) -> None:
    _enable_slow_log(monkeypatch, call_s=0.25, log_request_ids=True)
    _fake_clock(monkeypatch, 0.0, 5.0)
    parser_metrics = ParserMetrics(
        ParserMode.STREAMING,
        Endpoint.CHAT_COMPLETIONS,
        request_id="req-1",
    )

    with parser_metrics.time_call(tokens=1):
        pass

    assert "request_id=req-1" in caplog_vllm.text


def test_slow_call_omits_unknown_token_count(monkeypatch, caplog_vllm) -> None:
    _enable_slow_log(monkeypatch, call_s=0.25)
    _fake_clock(monkeypatch, 0.0, 5.0)

    with ParserMetrics(ParserMode.STREAMING, Endpoint.CHAT_COMPLETIONS).time_call():
        pass

    message = caplog_vllm.text
    assert "event=parser_slow_call" in message
    assert " tokens=" not in message
    assert "tokens_total=0" in message


def test_slow_warning_never_logs_request_content(monkeypatch, caplog_vllm) -> None:
    _enable_slow_log(monkeypatch, call_s=0.25)
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "SECRET_PROMPT"}],
    )
    processor = _processor(_tokenizer(), request, request_id="req-1")
    _fake_clock(monkeypatch, 0.0, 5.0)

    processor.step("SECRET_OUTPUT", [1], None, None)

    assert "event=parser_slow_call" in caplog_vllm.text
    assert "SECRET_PROMPT" not in caplog_vllm.text
    assert "SECRET_OUTPUT" not in caplog_vllm.text
    assert "SamplingParams" not in caplog_vllm.text


def test_collectors_visible_through_multiprocess_registry(tmp_path) -> None:
    script = """
from prometheus_client import generate_latest
from vllm.parser.metrics import (
    ParserMode,
    ParserMetrics,
    Endpoint,
    init_parser_metrics,
)
from vllm.v1.metrics.prometheus import get_prometheus_registry

init_parser_metrics(
    model_name="mp", tool_parser="hermes", enable_auto_tool_choice=True
)
with ParserMetrics(ParserMode.STREAMING, Endpoint.CHAT_COMPLETIONS).time_call():
    pass
exposition = generate_latest(get_prometheus_registry()).decode()
assert "vllm:parser_duration_seconds_count" in exposition, exposition
print("OK")
"""
    env = os.environ | {"PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
