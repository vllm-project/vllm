# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metrics for the parsers."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextvars import ContextVar, Token
from enum import Enum
from itertools import product
from types import TracebackType
from typing import NamedTuple, TypeVar, cast

from prometheus_client import Counter, Histogram

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_model_name: str | None = None
_parser_name: str = ""

# Slow-parse warnings are opt-in and require an enabled parser, like the
# metrics. The values are read once at init so the hot path only compares
# floats. A zero threshold disables that event.
_slow_log_enabled = False
_slow_call_seconds = 0.0
_slow_request_seconds = 0.0
_log_request_ids = False

# A degraded streaming parser can cross the slow-call threshold on every
# delta. Cap the per-accumulator warnings; the request summary's `slow_calls`
# still reports the true count.
_MAX_SLOW_CALL_WARNINGS = 10

_TOOL_CALL_PARSER_INVOCATIONS_TOTAL = "vllm:tool_call_parser_invocations_total"

_PARSER_DURATION_SECONDS = "vllm:parser_duration_seconds"
_REQUEST_PARSER_DURATION_SECONDS = "vllm:request_parser_duration_seconds"

_TOOL_CALL_INVOCATION_LABELS = ("model_name", "mode", "outcome", "request_type")
_PARSER_DURATION_LABELS = ("model_name", "parser", "mode", "endpoint")

# 5us streaming deltas through minute-scale degenerate parses. The tail is
# deliberately long: outlier parses are the thing worth measuring, and
# histogram_quantile cannot see past the last finite bound.
_PARSER_DURATION_BUCKETS = (
    5e-6,
    1e-5,
    2.5e-5,
    5e-5,
    1e-4,
    2.5e-4,
    5e-4,
    1e-3,
    2.5e-3,
    5e-3,
    1e-2,
    2.5e-2,
    5e-2,
    0.1,
    0.25,
    0.5,
    1,
    2.5,
    5,
    10,
    20,
    40,
    60,
    120,
    240,
    480,
    960,
)

# A request sums every parser call it makes, so a parser that degrades with
# history can accumulate more time than any single call.
_REQUEST_PARSER_DURATION_BUCKETS = (
    1e-3,
    2.5e-3,
    5e-3,
    1e-2,
    2.5e-2,
    5e-2,
    0.1,
    0.25,
    0.5,
    1,
    2.5,
    5,
    10,
    20,
    40,
    60,
    120,
    240,
    480,
    960,
    1920,
)

_MetricT = TypeVar("_MetricT")


class ToolCallOutcome(Enum):
    TOOL_CALL = "tool_call"
    NO_TOOL_CALL = "no_tool_call"


class Endpoint(Enum):
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    COHERE_CHAT = "cohere_chat"
    OTHER = "other"


class ParserMode(Enum):
    STREAMING = "streaming"
    NON_STREAMING = "non_streaming"
    # Stateless scale-out derender replays the whole token history through
    # parse_delta once per HTTP chunk; it is not one request.
    STREAMING_REPLAY = "streaming_replay"


class RequestType(Enum):
    """Values of the tool-call counter's pre-existing `request_type` label."""

    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"
    OTHER = "other"


class _MetricChildren(NamedTuple):
    invocation: Histogram
    request: Histogram | None


# Histogram children per (mode, endpoint), bound on first use. The model_name /
# parser labels are fixed for the process. Replay has no request child: it is
# not one request. Lazy binding keeps deployments from exporting zero series
# for API surfaces they do not serve.
_metrics: dict[tuple[ParserMode, Endpoint], _MetricChildren] = {}

# Tool-call counter children per (mode, outcome, request_type). The legacy
# combinations are pre-populated at init so existing alerts that rely on their
# zero series keep working; new combinations bind on first use.
_tool_call_children: dict[tuple[ParserMode, ToolCallOutcome, RequestType], Counter] = {}

# The mode of the parse call a `ParserMetrics` timer is timing. The tool-call
# counter reads it so derender replay reports `streaming_replay`, which
# `is_streaming` alone cannot express.
_metrics_context: ContextVar[ParserMode | None] = ContextVar(
    "vllm_parser_metrics_context", default=None
)

# Collectors this module created, keyed by metric name. Reused on a repeated
# `init_parser_metrics` call; a collector registered by someone else under the
# same name makes the factory raise instead.
_collectors: dict[str, Counter | Histogram] = {}

# Whether `init_parser_metrics` registered the collectors for this process.
_enabled = False


def _parser_label(
    reasoning_parser: str | None,
    tool_parser: str | None,
) -> str:
    """Bounded identifier for the configured parser pair."""
    names = [name for name in (reasoning_parser, tool_parser) if name]
    return "+".join(names)


def _get_or_create_collector(
    name: str,
    factory: Callable[[], _MetricT],
) -> _MetricT:
    """Return the collector for `name`, registering it if it is not present.

    `factory()` registers the collector and raises `ValueError` when one with
    that name is already registered. In that case a collector this module
    created earlier is reused; a foreign collector under the same name is a
    caller error and re-raised. A repeated `init_parser_metrics()` call
    re-creates collectors that `unregister_vllm_metrics()` detached; nothing
    reacts to that unregistration on its own, so the cached children stay
    detached until init runs again.
    """
    try:
        collector = factory()
    except ValueError:
        cached = _collectors.get(name)
        if cached is None:
            raise
        return cast(_MetricT, cached)
    _collectors[name] = collector
    return collector


def _bind_metric_children(scope: tuple[ParserMode, Endpoint]) -> _MetricChildren:
    """Bind the histogram children for one (mode, endpoint) label set."""
    mode, endpoint = scope
    labels = {
        "model_name": _model_name,
        "parser": _parser_name,
        "mode": mode.value,
        "endpoint": endpoint.value,
    }
    invocation = cast(Histogram, _collectors[_PARSER_DURATION_SECONDS])
    request = cast(Histogram, _collectors[_REQUEST_PARSER_DURATION_SECONDS])
    return _MetricChildren(
        invocation.labels(**labels),
        None if mode is ParserMode.STREAMING_REPLAY else request.labels(**labels),
    )


def init_parser_metrics(
    *,
    model_name: str,
    reasoning_parser: str | None = None,
    tool_parser: str | None = None,
    enable_auto_tool_choice: bool = False,
    log_request_ids: bool = False,
) -> None:
    """Lazily register parser metrics and cache the shared label values.

    Args:
        model_name: The served model name.
        reasoning_parser: The resolved reasoning parser, normally
            `structured_outputs_config.reasoning_parser`. That value already
            merges `--reasoning-parser`, `--structured-outputs-config` and
            model-specific defaults.
        tool_parser: The configured `--tool-call-parser`, if any. The tool
            parser is frontend-only and has no engine config to resolve it.
        enable_auto_tool_choice: Whether `--enable-auto-tool-choice` is set.
            Without it the tool parser is not used, so it neither enables
            metrics nor appears in the `parser` label.
        log_request_ids: Whether slow-parse warnings may include the request
            ID. Set from `--enable-log-requests`, which already logs request
            IDs, so the warnings stay within the same logging policy.

    Neither metrics nor slow-parse warnings are active when no parser is
    enabled. Warnings additionally require `VLLM_PARSER_SLOW_LOG`.

    """
    global _model_name, _parser_name, _enabled
    global _slow_log_enabled, _slow_call_seconds, _slow_request_seconds
    global _log_request_ids

    effective_tool_parser = (tool_parser or None) if enable_auto_tool_choice else None
    has_parser = effective_tool_parser is not None or bool(reasoning_parser)
    _model_name = model_name
    _parser_name = _parser_label(reasoning_parser, effective_tool_parser)
    _log_request_ids = log_request_ids
    _slow_call_seconds = max(0.0, envs.VLLM_PARSER_SLOW_CALL_SECONDS)
    _slow_request_seconds = max(0.0, envs.VLLM_PARSER_SLOW_REQUEST_SECONDS)
    _slow_log_enabled = envs.VLLM_PARSER_SLOW_LOG and has_parser

    if not has_parser:
        _enabled = False
        _metrics.clear()
        _tool_call_children.clear()
        return

    counter = _get_or_create_collector(
        _TOOL_CALL_PARSER_INVOCATIONS_TOTAL,
        lambda: Counter(
            name=_TOOL_CALL_PARSER_INVOCATIONS_TOTAL,
            documentation=(
                "Total number of ToolParser invocations. "
                "Non-streaming increments once per choice; "
                "streaming increments once per delta."
            ),
            labelnames=_TOOL_CALL_INVOCATION_LABELS,
        ),
    )
    _get_or_create_collector(
        _PARSER_DURATION_SECONDS,
        lambda: Histogram(
            name=_PARSER_DURATION_SECONDS,
            documentation=(
                "Time spent in one parser call: Parser.parse or "
                "Parser.parse_delta, or process_chunk and flush on the Harmony "
                "/v1/responses path. Streaming records one sample per delta or "
                "chunk; non-streaming records one sample per parse call. "
                "streaming_replay samples come from stateless scale-out "
                "derender."
            ),
            labelnames=_PARSER_DURATION_LABELS,
            buckets=_PARSER_DURATION_BUCKETS,
        ),
    )
    _get_or_create_collector(
        _REQUEST_PARSER_DURATION_SECONDS,
        lambda: Histogram(
            name=_REQUEST_PARSER_DURATION_SECONDS,
            documentation=(
                "Parser time accumulated across one request, including time "
                "spent before a parse failure. Streaming sums every call "
                "across the request's choices; non-streaming sums the "
                "request's parse calls across choices. A batch request records "
                "one sample per conversation. Requests served by stateless "
                "derender replay are excluded."
            ),
            labelnames=_PARSER_DURATION_LABELS,
            buckets=_REQUEST_PARSER_DURATION_BUCKETS,
        ),
    )

    _metrics.clear()
    _tool_call_children.clear()
    for mode, outcome, request_type in product(
        (ParserMode.STREAMING, ParserMode.NON_STREAMING),
        ToolCallOutcome,
        RequestType,
    ):
        _tool_call_children[(mode, outcome, request_type)] = counter.labels(
            model_name=_model_name,
            mode=mode.value,
            outcome=outcome.value,
            request_type=request_type.value,
        )
    _enabled = True


def _request_id_fragment(request_id: str | None) -> str | None:
    """Return the request ID log fragment, when logging IDs is enabled."""
    if not _log_request_ids or not request_id:
        return None
    return f"request_id={request_id}"


def _warn_slow_call(
    owner: ParserMetrics,
    duration: float,
    tokens: int | None,
) -> None:
    """Log one parser call that crossed the slow-call threshold."""
    if owner.mode is ParserMode.STREAMING_REPLAY:
        return
    fields = [
        "event=parser_slow_call",
        f"duration_s={duration:.3f}",
        f"threshold_s={_slow_call_seconds:.3f}",
        f"parser={_parser_name or 'none'}",
        f"mode={owner.mode.value}",
        f"endpoint={owner.endpoint.value}",
        f"call_index={owner.calls}",
    ]
    if tokens is not None:
        fields.append(f"tokens={tokens}")
    fields.append(f"tokens_total={owner.tokens}")
    request_id = _request_id_fragment(owner.request_id)
    if request_id is not None:
        fields.append(request_id)
    logger.warning("Slow parser call: %s", " ".join(fields))


def _warn_slow_request(
    seconds: float,
    *,
    mode: ParserMode,
    endpoint: Endpoint,
    calls: int,
    max_call_s: float,
    slow_calls: int,
    tokens: int,
    finish_reason: str | None,
    request_id: str | None,
) -> bool:
    """Log one request whose summed parser time crossed the summary threshold.

    Returns:
        Whether a warning was logged.

    """
    if not _slow_log_enabled or _slow_request_seconds <= 0.0:
        return False
    if calls <= 0 or mode is ParserMode.STREAMING_REPLAY:
        return False
    # A single call is covered by the per-call event when that event fires.
    if calls == 1 and 0.0 < _slow_call_seconds <= seconds:
        return False
    if seconds < _slow_request_seconds:
        return False

    fields = [
        "event=parser_slow_request",
        f"duration_s={seconds:.3f}",
        f"threshold_s={_slow_request_seconds:.3f}",
        f"parser={_parser_name or 'none'}",
        f"mode={mode.value}",
        f"endpoint={endpoint.value}",
        f"calls={calls}",
        f"max_call_s={max_call_s:.3f}",
        f"slow_calls={slow_calls}",
        f"tokens_total={tokens}",
        f"finish_reason={finish_reason or 'none'}",
    ]
    request_fragment = _request_id_fragment(request_id)
    if request_fragment is not None:
        fields.append(request_fragment)
    logger.warning("Slow parser request: %s", " ".join(fields))
    return True


class _NoopTimer:
    """Stand-in timer used when no parser is enabled."""

    __slots__ = ()

    def __enter__(self) -> _NoopTimer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass


_NOOP_TIMER = _NoopTimer()

# Indirection so tests can substitute the clock without patching `time` for
# the whole process.
_clock: Callable[[], float] = time.perf_counter


class _ParserCallTimer:
    """Times one parser call into its owning `ParserMetrics`.

    Owned and reused by one `ParserMetrics`; parser calls for a request are
    sequential, so a single timer object is enough and the streaming hot path
    allocates nothing per call.
    """

    __slots__ = ("_owner", "_started", "_token", "_tokens")

    def __init__(self, owner: ParserMetrics) -> None:
        self._owner = owner
        self._started = 0.0
        self._token: Token[ParserMode | None] | None = None
        self._tokens: int | None = None

    def __enter__(self) -> _ParserCallTimer:
        self._owner._ensure_bound()
        self._started = _clock()
        self._token = _metrics_context.set(self._owner.mode)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        token, self._token = self._token, None
        if token is not None:
            _metrics_context.reset(token)
        owner = self._owner
        elapsed = _clock() - self._started
        tokens = self._tokens
        self._tokens = None
        if owner._invocation is not None:
            owner._invocation.observe(elapsed)
        owner.seconds += elapsed
        owner.calls += 1
        if _slow_log_enabled:
            if elapsed > owner.max_call_s:
                owner.max_call_s = elapsed
            if tokens is not None:
                owner.tokens += tokens
            if _slow_call_seconds > 0.0 and elapsed >= _slow_call_seconds:
                owner.slow_calls += 1
                if owner._slow_call_warnings < _MAX_SLOW_CALL_WARNINGS:
                    owner._slow_call_warnings += 1
                    _warn_slow_call(owner, elapsed, tokens)
        if exc_type is not None and owner.mode is ParserMode.NON_STREAMING:
            # A non-streaming accumulator is the request's own, so a raise
            # skips the caller's observe(). Streaming accumulators are merged
            # into a request summary instead and must not record a per-choice
            # request sample here.
            owner.observe()


class ParserMetrics:
    """Accumulates parser time for one request or one streamed choice.

    On the batch endpoint one instance covers one conversation, matching the
    request ID used in the slow-parse logs.

    Construct one with the `mode` and API `endpoint` its parser calls use.
    Wrap each `parse` / `parse_delta` call (or Harmony `process_chunk` /
    `flush`) in `with metrics.time_call():`. The call is recorded as an
    invocation and its time added to `seconds` / `calls`, including when the
    call raises.

    Call `observe()` once when the request finishes to record the per-request
    histogram. If a timed call raises, a non-streaming accumulator records the
    sample immediately because the request is not going to finish. Streaming
    accumulators are merged into a request summary instead. Replay requests
    have no per-request child and are dropped there. When no parser is enabled,
    `time_call()` returns a shared no-op timer and `observe()` records nothing.

    With `VLLM_PARSER_SLOW_LOG` enabled, `time_call()` also emits a warning
    for calls above `VLLM_PARSER_SLOW_CALL_SECONDS`, up to a fixed
    per-accumulator cap, and `observe()` emits a summary for requests whose
    summed time crosses `VLLM_PARSER_SLOW_REQUEST_SECONDS`. The summary's
    `slow_calls` counts every slow call, including ones past the cap. Replay
    calls are excluded from both.
    """

    __slots__ = (
        "mode",
        "endpoint",
        "seconds",
        "calls",
        "max_call_s",
        "slow_calls",
        "tokens",
        "request_id",
        "_summary_emitted",
        "_slow_call_warnings",
        "_observed",
        "_reasons",
        "_scope",
        "_invocation",
        "_request",
        "_bound",
        "_timer",
    )

    def __init__(
        self,
        mode: ParserMode,
        endpoint: Endpoint,
        *,
        request_id: str | None = None,
    ) -> None:
        self.mode = mode
        self.endpoint = endpoint
        self.seconds = 0.0
        self.calls = 0
        self.max_call_s = 0.0
        self.slow_calls = 0
        self.tokens = 0
        self.request_id = request_id
        self._reasons: set[str] = set()
        self._summary_emitted = False
        self._slow_call_warnings = 0
        self._observed = False
        self._scope = (mode, endpoint)
        self._bound = False
        self._invocation: Histogram | None = None
        self._request: Histogram | None = None
        if _enabled:
            self._timer: _ParserCallTimer | _NoopTimer = _ParserCallTimer(self)
        else:
            self._timer = _NOOP_TIMER

    def _ensure_bound(self) -> None:
        """Bind this (mode, endpoint) histogram children on first use."""
        if self._bound:
            return
        self._bound = True
        if not _enabled:
            return
        entry = _metrics.get(self._scope)
        if entry is None:
            entry = _metrics[self._scope] = _bind_metric_children(self._scope)
        self._invocation = entry.invocation
        self._request = entry.request

    @property
    def finish_reason(self) -> str | None:
        """The finish reason(s) seen, sorted, for the request summary."""
        if not self._reasons:
            return None
        return ",".join(sorted(self._reasons))

    @finish_reason.setter
    def finish_reason(self, value: str | None) -> None:
        if value is not None:
            self._reasons.add(value)

    def time_call(self, *, tokens: int | None = None) -> _ParserCallTimer | _NoopTimer:
        """Return the context manager timing one parser call.

        Args:
            tokens: Tokens this call parses, for the slow-call warning. None
                omits the count. Parser calls for a request are sequential, so
                the value is stored on the reused timer.

        """
        timer = self._timer
        if isinstance(timer, _ParserCallTimer):
            timer._tokens = tokens
        return timer

    def merge(self, other: ParserMetrics) -> None:
        """Accumulate another choice's totals into this one.

        Used to summarize a request that streams several choices.
        """
        self.seconds += other.seconds
        self.calls += other.calls
        self.max_call_s = max(self.max_call_s, other.max_call_s)
        self.slow_calls += other.slow_calls
        self.tokens += other.tokens
        self._reasons |= other._reasons

    def observe(self) -> None:
        """Record the accumulated time as one finished request."""
        if self.calls <= 0:
            return
        self._ensure_bound()
        if not self._observed:
            self._observed = True
            if self._request is not None:
                self._request.observe(self.seconds)
        if self._summary_emitted:
            return
        self._summary_emitted = _warn_slow_request(
            self.seconds,
            mode=self.mode,
            endpoint=self.endpoint,
            calls=self.calls,
            max_call_s=self.max_call_s,
            slow_calls=self.slow_calls,
            tokens=self.tokens,
            finish_reason=self.finish_reason,
            request_id=self.request_id,
        )


_request_classes: tuple[type, type] | None = None


def _request_type(request: object) -> RequestType:
    """Map a request object to the counter's pre-existing `request_type`.

    The endpoint label does not select this: batch and token-in-token-out
    derender parse chat-completion requests under `Endpoint.OTHER` but must
    keep reporting `chat_completions`.
    """
    global _request_classes
    if _request_classes is None:
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )
        from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

        _request_classes = (ChatCompletionRequest, ResponsesRequest)
    chat_completions, responses = _request_classes
    if isinstance(request, chat_completions):
        return RequestType.CHAT_COMPLETIONS
    if isinstance(request, responses):
        return RequestType.RESPONSES
    return RequestType.OTHER


def record_tool_parser_invocation(
    *,
    is_tool_called: bool | Exception,
    is_streaming: bool,
    request: object,
) -> None:
    """Increment the tool-call parser invocation counter when registered.
    Currently parser failures are treated as no tool calls.

    TODO: To accurately track parser failures, add a new ToolCallOutcome and
    more importantly, ensure exceptions are propagated out of the ToolParsers
    instead of being caught internally. This would require going through
    ToolParser implementation on a case-by-case basis.
    """
    if not _enabled:
        return

    mode = _metrics_context.get()
    if mode is None:
        mode = ParserMode.STREAMING if is_streaming else ParserMode.NON_STREAMING
    request_type = _request_type(request)

    match is_tool_called:
        case bool():
            outcome = (
                ToolCallOutcome.TOOL_CALL
                if is_tool_called
                else ToolCallOutcome.NO_TOOL_CALL
            )
        case _:
            outcome = ToolCallOutcome.NO_TOOL_CALL

    key = (mode, outcome, request_type)
    child = _tool_call_children.get(key)
    if child is None:
        counter = cast(Counter, _collectors[_TOOL_CALL_PARSER_INVOCATIONS_TOTAL])
        child = _tool_call_children[key] = counter.labels(
            model_name=_model_name,
            mode=mode.value,
            outcome=outcome.value,
            request_type=request_type.value,
        )
    child.inc()
