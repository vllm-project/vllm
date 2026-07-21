# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import dataclass, field

import openai
import pytest

from tests.tool_use.utils import (
    MESSAGES_ASKING_FOR_PARALLEL_TOOLS,
    MESSAGES_ASKING_FOR_TOOLS,
    MESSAGES_WITHOUT_TOOLS,
    SEARCH_TOOL,
    SEED,
    WEATHER_TOOL,
)

from .utils import MistralServerConfig, ensure_system_prompt

AUTO = object()

_MESSAGES_ASKING_FOR_JSON: list[dict] = [
    {
        "role": "user",
        "content": (
            "Provide information about Paris, France. "
            "Return only a JSON object with 'city' (string) "
            "and 'population' (integer) fields."
        ),
    }
]

_CITY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "population": {"type": "integer"},
    },
    "required": ["city", "population"],
}

_MESSAGES_MAP: dict[str, list] = {
    "asking_for_tools": MESSAGES_ASKING_FOR_TOOLS,
    "without_tools": MESSAGES_WITHOUT_TOOLS,
}


@dataclass
class Scenario:
    r"""One row of the tool-call scenario matrix."""

    id: str
    messages_key: str
    tools: list | None
    tool_choice: object
    expectation: str
    forced_tool_name: str | None
    requires_grammar: bool = False
    # Named tool_choice forces a tool via constrained generation, so the
    # response finishes with "stop"/"length" rather than "tool_calls".
    finish_is_tool_calls: bool = True


@dataclass
class SOScenario:
    r"""One row of the structured-output scenario matrix."""

    id: str
    response_format: dict
    schema: dict | None


@dataclass
class StreamedToolCallResult:
    r"""Accumulated result from streaming a single tool call."""

    function_name: str | None = None
    function_args_str: str = ""
    tool_call_id: str | None = None
    role_name: str | None = None
    finish_reason_count: int = 0
    finish_reason: str | None = None
    reasoning: str = ""


@dataclass
class StreamedContentResult:
    r"""Accumulated result from streaming a content-only response."""

    chunks: list[str] = field(default_factory=list)
    finish_reason_count: int = 0
    finish_reason: str | None = None
    role_sent: bool = False
    reasoning: str = ""


@dataclass
class StreamedParallelToolCallResult:
    r"""Accumulated result from streaming parallel tool calls."""

    function_names: list[str] = field(default_factory=list)
    function_args_strs: list[str] = field(default_factory=list)
    tool_call_ids: list[str] = field(default_factory=list)
    role_name: str | None = None
    finish_reason_count: int = 0


TOOL_SCENARIOS: list[Scenario] = [
    Scenario(
        id="auto_weather",
        messages_key="asking_for_tools",
        tools=[WEATHER_TOOL, SEARCH_TOOL],
        tool_choice=AUTO,
        expectation="tool_call",
        forced_tool_name="get_current_weather",
    ),
    Scenario(
        id="auto_joke",
        messages_key="without_tools",
        tools=[WEATHER_TOOL, SEARCH_TOOL],
        tool_choice=AUTO,
        expectation="content_only",
        forced_tool_name=None,
    ),
    Scenario(
        id="required_weather",
        messages_key="asking_for_tools",
        tools=[WEATHER_TOOL],
        tool_choice="required",
        expectation="tool_call",
        forced_tool_name=None,
    ),
    Scenario(
        id="named_weather",
        messages_key="asking_for_tools",
        tools=[WEATHER_TOOL, SEARCH_TOOL],
        tool_choice={
            "type": "function",
            "function": {"name": "get_current_weather"},
        },
        expectation="tool_call",
        forced_tool_name="get_current_weather",
        finish_is_tool_calls=False,
    ),
    Scenario(
        id="named_search_mismatch",
        messages_key="asking_for_tools",
        tools=[WEATHER_TOOL, SEARCH_TOOL],
        tool_choice={"type": "function", "function": {"name": "web_search"}},
        expectation="tool_call",
        forced_tool_name="web_search",
        requires_grammar=True,
        finish_is_tool_calls=False,
    ),
    Scenario(
        id="none_weather",
        messages_key="asking_for_tools",
        tools=[WEATHER_TOOL],
        tool_choice="none",
        expectation="content_only",
        forced_tool_name=None,
    ),
]

SO_SCENARIOS: list[SOScenario] = [
    SOScenario(
        id="json_object",
        response_format={"type": "json_object"},
        schema=None,
    ),
    SOScenario(
        id="json_schema",
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "city_info", "schema": _CITY_SCHEMA},
        },
        schema=_CITY_SCHEMA,
    ),
]


def _requires_tool_parser(server_config: MistralServerConfig) -> None:
    """Skip if server was not started with --tool-call-parser."""
    if "--tool-call-parser" not in server_config.get("arguments", []):
        pytest.skip(
            f"Skipping: {server_config['model']} not configured with --tool-call-parser"
        )


def _requires_parallel(server_config: MistralServerConfig) -> None:
    """Skip if the model does not support parallel tool calls."""
    if not server_config.get("supports_parallel"):
        pytest.skip(
            f"Skipping: {server_config['model']} does not support parallel tool calls"
        )


def _reasoning_effort_values(config: MistralServerConfig) -> list[str | None]:
    """Valid reasoning_effort values for a model."""
    if config.get("reasoning_mode") == "effort":
        return [None, "none", "high"]
    return [None]


def _expect_reasoning(
    config: MistralServerConfig, reasoning_effort: str | None
) -> bool:
    """Resolve whether reasoning_content must be present in the response."""
    mode = config.get("reasoning_mode")
    if mode == "intrinsic":
        return True
    if mode == "effort":
        return reasoning_effort == "high"
    return False


def _build_request_kwargs(
    model: str,
    messages: list,
    *,
    tools: list | None = None,
    tool_choice: object = AUTO,
    response_format: dict | None = None,
    reasoning_effort: str | None = None,
    stream: bool = False,
    expect_reasoning: bool = False,
) -> dict:
    """Assemble keyword arguments for chat.completions.create."""
    # Reasoning models must think before acting, so the reasoning block plus the
    # tool call / final answer needs a far larger budget than a direct answer.

    max_tokens = 2048 if expect_reasoning else (100 if tools is not None else 150)
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "seed": SEED,
        "max_completion_tokens": max_tokens,
        "stream": stream,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not AUTO:
        kwargs["tool_choice"] = tool_choice
    if response_format is not None:
        kwargs["response_format"] = response_format
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    return kwargs


def _assert_tool_call(
    *,
    function_name: str | None,
    function_args_str: str,
    tool_call_id: str,
    expected_name: str | None,
    config: MistralServerConfig,
) -> dict:
    """Validate a single tool call and return parsed args."""
    assert function_name is not None
    if expected_name is not None:
        assert function_name == expected_name
    parsed = json.loads(function_args_str)
    if config.get("supports_grammar"):
        assert len(tool_call_id) == 9
    return parsed


def _assert_content_only(
    *, content: str, tool_calls_present: bool, config: MistralServerConfig
) -> None:
    """Validate a content-only response.

    Grammar-capable models must emit non-empty content, no tool calls, and no
    literal [TOOL_CALLS]. Pre-v11 models lack enforcement, so those checks are
    relaxed (matches the tolerance of the pre-existing pre-v11 tests).
    """
    if config.get("supports_grammar"):
        assert len(content) > 0
        assert not tool_calls_present
        assert "[TOOL_CALLS]" not in content


def _assert_finish(
    finish_reason: str | None, *, is_tool_calls: bool, allow_length: bool = False
) -> None:
    """Assert the finish reason for a tool-call scenario.

    Named tool_choice forces the tool via constrained generation and finishes
    with "stop"/"length"; auto/required finish with "tool_calls".

    allow_length tolerates a "length" finish. Non-streamed "required" only gets
    promoted to "tool_calls" when the model emits EOS (serving.py gates on
    output.finish_reason == "stop"), and the "required" grammar is an unbounded
    tool-call array (minItems=1, no maxItems). An old, weaker model can keep
    appending items and hit the token budget without EOS, so it finishes as
    "length". The tool call itself is validated separately.
    """
    if is_tool_calls:
        allowed = {"tool_calls", "length"} if allow_length else {"tool_calls"}
        assert finish_reason in allowed
    else:
        assert finish_reason in ("stop", "length")


def _assert_valid_json(*, content: str, schema: dict | None) -> dict:
    """Validate structured-output content and return the parsed dict."""
    parsed = json.loads(content)
    assert isinstance(parsed, dict)
    if schema is not None:
        for key in schema.get("required", []):
            assert key in parsed
        if "city" in parsed:
            assert isinstance(parsed["city"], str)
        if "population" in parsed:
            assert isinstance(parsed["population"], int)
    return parsed


def _assert_reasoning(
    *, reasoning_content: str | None, expected: bool, optional: bool = False
) -> None:
    """Assert reasoning_content presence matches expected.

    optional relaxes the presence check: a reasoning model may skip its think
    block for a trivial content-only reply (e.g. a conversational joke), so
    reasoning is neither required nor forbidden there. It stays strict for
    tool-call cells (where the model does reason) and for non-reasoning models
    (which must never emit reasoning).
    """
    if optional:
        return
    if expected:
        assert reasoning_content is not None and len(reasoning_content) > 0
    else:
        assert not reasoning_content


async def _collect_streamed_tool_call(
    stream: openai.AsyncStream,
) -> StreamedToolCallResult:
    r"""Consume a streaming response and collect a single tool call.

    The finish reason is recorded but not asserted here; the caller checks it
    (named tool_choice finishes with "stop"/"length", not "tool_calls").
    """
    result = StreamedToolCallResult()

    async for chunk in stream:
        delta = chunk.choices[0].delta

        if chunk.choices[0].finish_reason:
            result.finish_reason_count += 1
            result.finish_reason = chunk.choices[0].finish_reason

        if delta.role:
            assert not result.role_name or result.role_name == "assistant"
            result.role_name = "assistant"

        reasoning_delta = getattr(delta, "reasoning", None) or getattr(
            delta, "reasoning_content", None
        )
        if reasoning_delta:
            result.reasoning += reasoning_delta

        streamed_tool_calls = delta.tool_calls
        if streamed_tool_calls and len(streamed_tool_calls) > 0:
            assert len(streamed_tool_calls) == 1
            tool_call = streamed_tool_calls[0]

            if tool_call.id:
                assert not result.tool_call_id
                result.tool_call_id = tool_call.id

            if tool_call.function:
                if tool_call.function.name:
                    assert result.function_name is None
                    result.function_name = tool_call.function.name
                if tool_call.function.arguments:
                    result.function_args_str += tool_call.function.arguments

    return result


async def _collect_streamed_content(
    stream: openai.AsyncStream,
    *,
    expected_finish_reason: str | None = None,
    no_tool_calls: bool = True,
) -> StreamedContentResult:
    r"""Consume a streaming response and collect text content."""
    result = StreamedContentResult()

    async for chunk in stream:
        delta = chunk.choices[0].delta

        if delta.role:
            assert not result.role_sent
            assert delta.role == "assistant"
            result.role_sent = True

        if delta.content:
            result.chunks.append(delta.content)

        reasoning_delta = getattr(delta, "reasoning", None) or getattr(
            delta, "reasoning_content", None
        )
        if reasoning_delta:
            result.reasoning += reasoning_delta

        if chunk.choices[0].finish_reason is not None:
            result.finish_reason_count += 1
            result.finish_reason = chunk.choices[0].finish_reason
            if expected_finish_reason is not None:
                assert result.finish_reason == expected_finish_reason

        if no_tool_calls:
            assert not delta.tool_calls or len(delta.tool_calls) == 0

    return result


async def _collect_streamed_parallel_tool_calls(
    stream: openai.AsyncStream,
) -> StreamedParallelToolCallResult:
    r"""Consume a streaming response and collect parallel tool calls."""
    result = StreamedParallelToolCallResult()
    tool_call_idx: int = -1

    async for chunk in stream:
        if chunk.choices[0].finish_reason:
            result.finish_reason_count += 1
            assert chunk.choices[0].finish_reason == "tool_calls"

        if chunk.choices[0].delta.role:
            assert not result.role_name or result.role_name == "assistant"
            result.role_name = "assistant"

        streamed_tool_calls = chunk.choices[0].delta.tool_calls
        if streamed_tool_calls and len(streamed_tool_calls) > 0:
            assert len(streamed_tool_calls) == 1
            tool_call = streamed_tool_calls[0]

            if tool_call.index != tool_call_idx:
                tool_call_idx = tool_call.index
                result.function_args_strs.append("")
                result.tool_call_ids.append("")

            if tool_call.id:
                result.tool_call_ids[tool_call.index] = tool_call.id

            if tool_call.function:
                if tool_call.function.name:
                    result.function_names.append(tool_call.function.name)
                if tool_call.function.arguments:
                    result.function_args_strs[tool_call.index] += (
                        tool_call.function.arguments
                    )

    return result


async def _run_tool_cell(
    client: openai.AsyncOpenAI,
    server_config: MistralServerConfig,
    scenario: Scenario,
    stream: bool,
    reasoning_effort: str | None,
    model_name: str,
) -> None:
    """Exercise one tool-call cell for a given reasoning_effort value."""
    supports_grammar = bool(server_config.get("supports_grammar"))
    messages = ensure_system_prompt(_MESSAGES_MAP[scenario.messages_key], server_config)
    expect_reasoning = _expect_reasoning(
        config=server_config, reasoning_effort=reasoning_effort
    )
    # An intrinsic reasoning model may skip its think block for a trivial
    # content-only reply (e.g. a conversational joke), so reasoning is optional
    # there. It stays required for tool-call cells (where the model does reason)
    # and for effort models at reasoning_effort="high", where reasoning is
    # explicitly requested and must be produced.
    reasoning_optional = (
        expect_reasoning
        and scenario.expectation == "content_only"
        and server_config.get("reasoning_mode") == "intrinsic"
    )
    kwargs = _build_request_kwargs(
        model=model_name,
        messages=messages,
        tools=scenario.tools,
        tool_choice=scenario.tool_choice,
        reasoning_effort=reasoning_effort,
        stream=stream,
        expect_reasoning=expect_reasoning,
    )

    if stream:
        raw = await client.chat.completions.create(**kwargs)
        if scenario.expectation == "tool_call":
            tc_result = await _collect_streamed_tool_call(raw)
            _assert_tool_call(
                function_name=tc_result.function_name,
                function_args_str=tc_result.function_args_str,
                tool_call_id=tc_result.tool_call_id or "",
                expected_name=scenario.forced_tool_name,
                config=server_config,
            )
            _assert_finish(
                tc_result.finish_reason,
                is_tool_calls=scenario.finish_is_tool_calls,
            )
            _assert_reasoning(
                reasoning_content=tc_result.reasoning or None,
                expected=expect_reasoning,
            )
        else:
            ct_result = await _collect_streamed_content(
                raw, no_tool_calls=supports_grammar
            )
            _assert_content_only(
                content="".join(ct_result.chunks),
                tool_calls_present=False,
                config=server_config,
            )
            assert ct_result.finish_reason != "tool_calls"
            _assert_reasoning(
                reasoning_content=ct_result.reasoning or None,
                expected=expect_reasoning,
                optional=reasoning_optional,
            )
    else:
        chat_completion = await client.chat.completions.create(**kwargs)
        choice = chat_completion.choices[0]
        reasoning_content = getattr(choice.message, "reasoning", None) or getattr(
            choice.message, "reasoning_content", None
        )

        if scenario.expectation == "tool_call":
            tool_calls = choice.message.tool_calls
            assert tool_calls is not None and len(tool_calls) >= 1
            _assert_tool_call(
                function_name=tool_calls[0].function.name,
                function_args_str=tool_calls[0].function.arguments,
                tool_call_id=tool_calls[0].id,
                expected_name=scenario.forced_tool_name,
                config=server_config,
            )
            # Mistral-7B-Instruct-v0.3 is an old pre-v11 model (no grammar) and
            # is not reliable at cleanly terminating a "required" tool-call
            # array, so it can run to the token budget and finish as "length"
            # instead of "tool_calls". Tolerate that only for this model and
            # exit gracefully; the tool call itself is still validated above.
            old_model = not server_config.get("supports_grammar")
            _assert_finish(
                choice.finish_reason,
                is_tool_calls=scenario.finish_is_tool_calls,
                allow_length=scenario.tool_choice == "required" and old_model,
            )
        else:
            _assert_content_only(
                content=choice.message.content or "",
                tool_calls_present=bool(choice.message.tool_calls),
                config=server_config,
            )
            assert choice.finish_reason != "tool_calls"

        _assert_reasoning(
            reasoning_content=reasoning_content,
            expected=expect_reasoning,
            optional=reasoning_optional,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("scenario", TOOL_SCENARIOS, ids=lambda s: s.id)
async def test_tool_matrix(
    client: openai.AsyncOpenAI,
    server_config: MistralServerConfig,
    scenario: Scenario,
    stream: bool,
) -> None:
    _requires_tool_parser(server_config)
    if scenario.requires_grammar and not server_config.get("supports_grammar"):
        pytest.skip(
            f"Skipping {scenario.id}: {server_config['model']} does not enforce "
            "grammar (named tool_choice against the prompt is not guaranteed)."
        )
    models = await client.models.list()
    model_name: str = models.data[0].id
    # reasoning_effort only applies to v15+ models; other models run once with
    # None. Looping here (instead of a pytest axis) avoids generating skipped
    # effort cells for non-effort models.
    for reasoning_effort in _reasoning_effort_values(server_config):
        await _run_tool_cell(
            client, server_config, scenario, stream, reasoning_effort, model_name
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("so_scenario", SO_SCENARIOS, ids=lambda s: s.id)
async def test_structured_output_matrix(
    client: openai.AsyncOpenAI,
    server_config: MistralServerConfig,
    so_scenario: SOScenario,
    stream: bool,
) -> None:
    if not server_config.get("supports_grammar"):
        pytest.skip(f"Skipping: {server_config['model']} does not support grammar")

    models = await client.models.list()
    model_name: str = models.data[0].id
    messages = ensure_system_prompt(_MESSAGES_ASKING_FOR_JSON, server_config)
    for reasoning_effort in _reasoning_effort_values(server_config):
        kwargs = _build_request_kwargs(
            model=model_name,
            messages=messages,
            tools=None,
            tool_choice="none",
            response_format=so_scenario.response_format,
            reasoning_effort=reasoning_effort,
            stream=stream,
            expect_reasoning=_expect_reasoning(
                config=server_config, reasoning_effort=reasoning_effort
            ),
        )
        if stream:
            raw = await client.chat.completions.create(**kwargs)
            ct_result = await _collect_streamed_content(raw, no_tool_calls=True)
            content = "".join(ct_result.chunks)
        else:
            chat_completion = await client.chat.completions.create(**kwargs)
            content = chat_completion.choices[0].message.content or ""

        _assert_valid_json(content=content, schema=so_scenario.schema)


@pytest.mark.asyncio
async def test_tool_call_parallel(
    client: openai.AsyncOpenAI, server_config: MistralServerConfig
) -> None:
    _requires_tool_parser(server_config)
    _requires_parallel(server_config)

    models = await client.models.list()
    model_name: str = models.data[0].id

    # Reasoning models must think before emitting the parallel tool calls, so
    # the reasoning block plus two tool calls needs a far larger budget than a
    # direct answer or the response finishes as "length" mid-generation.
    expect_reasoning = _expect_reasoning(config=server_config, reasoning_effort=None)
    max_tokens = 2048 if expect_reasoning else 200

    # --- non-streaming ---
    chat_completion = await client.chat.completions.create(
        messages=ensure_system_prompt(
            MESSAGES_ASKING_FOR_PARALLEL_TOOLS, server_config
        ),
        temperature=0,
        max_completion_tokens=max_tokens,
        model=model_name,
        tools=[WEATHER_TOOL],
        logprobs=False,
        seed=SEED,
    )

    choice = chat_completion.choices[0]
    tool_calls = choice.message.tool_calls

    assert choice.finish_reason == "tool_calls"
    assert tool_calls is not None and len(tool_calls) >= 2
    for tc in tool_calls:
        assert tc.type == "function"
        assert tc.function.name == "get_current_weather"
        assert isinstance(tc.function.arguments, str)
        parsed = json.loads(tc.function.arguments)
        assert "city" in parsed
        assert len(tc.id) == 9

    non_streaming_tool_calls = tool_calls

    # --- streaming ---
    stream = await client.chat.completions.create(
        messages=ensure_system_prompt(
            MESSAGES_ASKING_FOR_PARALLEL_TOOLS, server_config
        ),
        temperature=0,
        max_completion_tokens=max_tokens,
        model=model_name,
        tools=[WEATHER_TOOL],
        logprobs=False,
        seed=SEED,
        stream=True,
    )

    result = await _collect_streamed_parallel_tool_calls(stream)

    assert result.finish_reason_count == 1
    assert result.role_name == "assistant"
    assert len(result.function_names) >= 2
    assert all(name == "get_current_weather" for name in result.function_names)
    assert len(result.tool_call_ids) >= 2
    assert all(isinstance(tid, str) and len(tid) == 9 for tid in result.tool_call_ids)

    for args_str in result.function_args_strs:
        streamed_args = json.loads(args_str)
        assert "city" in streamed_args

    assert len(result.function_names) == len(non_streaming_tool_calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
async def test_tool_call_multi_turn(
    client: openai.AsyncOpenAI,
    server_config: MistralServerConfig,
    stream: bool,
) -> None:
    """Exercises repeated reason+tool turns with growing tool-call/result history.

    Two consecutive non-streaming assistant turns each produce a tool call
    (Dallas then Paris), building up assistant-tool-call + tool-result history
    in the prompt.  Turn 2's prompt therefore contains the prior tool call and
    tool result that single-turn tests never reach.  A final summarization turn
    (streamed or not per `stream`) asserts that reasoning does not leak into
    content across the full multi-turn conversation — the failure mode that
    prompted this test.
    """
    _requires_tool_parser(server_config)
    if not server_config.get("supports_multi_turn", True):
        pytest.skip(
            f"Skipping: {server_config['model']} is not trained for multi-turn "
            "tool-call conversations."
        )
    models = await client.models.list()
    model_name: str = models.data[0].id
    expect_reasoning = _expect_reasoning(config=server_config, reasoning_effort=None)

    cities = ["Dallas", "Paris"]
    tool_results = {
        "Dallas": "The weather in Dallas is 98 degrees fahrenheit and sunny.",
        "Paris": "The weather in Paris is 60 degrees fahrenheit and cloudy.",
    }

    messages = ensure_system_prompt(MESSAGES_ASKING_FOR_TOOLS, server_config)

    for i, city in enumerate(cities):
        kwargs = _build_request_kwargs(
            model=model_name,
            messages=messages,
            tools=[WEATHER_TOOL],
            expect_reasoning=expect_reasoning,
        )
        chat = await client.chat.completions.create(**kwargs)
        choice = chat.choices[0]
        turn_tool_calls = choice.message.tool_calls

        if not server_config.get("supports_grammar") and (
            turn_tool_calls is None or len(turn_tool_calls) == 0
        ):
            pytest.skip(
                f"Non-grammar model did not produce a tool call on turn {i}"
                f" (city={city}); skipping multi-turn test."
            )

        assert turn_tool_calls is not None and len(turn_tool_calls) >= 1
        tc = turn_tool_calls[0]
        _assert_tool_call(
            function_name=tc.function.name,
            function_args_str=tc.function.arguments,
            tool_call_id=tc.id,
            expected_name="get_current_weather",
            config=server_config,
        )

        turn_reasoning = getattr(choice.message, "reasoning", None) or getattr(
            choice.message, "reasoning_content", None
        )
        turn_content = choice.message.content or ""
        _assert_reasoning(
            reasoning_content=turn_reasoning,
            expected=expect_reasoning,
            optional=True,
        )
        assert "[THINK]" not in turn_content
        assert "[/THINK]" not in turn_content
        assert "</think>" not in turn_content

        messages = list(messages) + [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_results[city],
            },
        ]
        if i < len(cities) - 1:
            messages = list(messages) + [
                {"role": "user", "content": f"And what about {cities[i + 1]}?"}
            ]

    # Final turn: ask for a summary, respecting the `stream` parameter.
    messages = list(messages) + [
        {
            "role": "user",
            "content": "Thanks, can you summarize both weather results?",
        }
    ]
    final_kwargs = _build_request_kwargs(
        model=model_name,
        messages=messages,
        tools=[WEATHER_TOOL],
        stream=stream,
        expect_reasoning=expect_reasoning,
    )

    if stream:
        raw = await client.chat.completions.create(**final_kwargs)
        ct_result = await _collect_streamed_content(raw, no_tool_calls=False)
        content = "".join(ct_result.chunks)
        finish_reason = ct_result.finish_reason
        reasoning_content: str | None = ct_result.reasoning or None
    else:
        final_chat = await client.chat.completions.create(**final_kwargs)
        final_choice = final_chat.choices[0]
        content = final_choice.message.content or ""
        finish_reason = final_choice.finish_reason
        reasoning_content = getattr(final_choice.message, "reasoning", None) or getattr(
            final_choice.message, "reasoning_content", None
        )

    assert finish_reason in {"stop", "length", "tool_calls"}
    assert "[THINK]" not in content
    assert "[/THINK]" not in content
    assert "</think>" not in content
    if finish_reason != "tool_calls":
        assert len(content) > 0
    _assert_reasoning(
        reasoning_content=reasoning_content,
        expected=expect_reasoning,
        optional=True,
    )
