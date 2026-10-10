# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from copy import deepcopy

import pytest
import regex as re
from openai.types.responses import (
    FunctionTool,
    ToolChoiceFunction,
    WebSearchTool,
)
from pydantic import TypeAdapter

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError
from vllm.tool_parsers.streaming import (
    RequiredToolCallScanner,
    extract_required_tool_call_streaming,
)
from vllm.tool_parsers.utils import (
    find_tool_properties,
    get_json_schema_from_tools,
)

pytestmark = pytest.mark.cpu_test

EXAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for"
                        ", e.g. 'San Francisco'",
                    },
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
        "strict": True,
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "Get the weather forecast for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to get the forecast for, e.g. "
                        "'New York'",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days to get the forecast for (1-7)",
                    },
                },
                "required": ["city", "days"],
                "additionalProperties": False,
            },
        },
        "strict": True,
    },
]


def _compile_and_check(
    tools: list[ChatCompletionToolsParam],
    sample_output,
    should_match: bool,
    parallel_tool_calls: bool | None = None,
):
    # self = MagicMock(tool_choice="required", tools=tools)
    # schema = ChatCompletionRequest._get_json_schema_from_tool(self)
    schema = get_json_schema_from_tools(
        tools=tools,
        tool_choice="required",
        parallel_tool_calls=parallel_tool_calls,
    )
    assert isinstance(schema, dict)

    # use build_regex_from_schema used in JSONLogitsProcessor to create Guide
    from outlines_core.json_schema import build_regex_from_schema

    regex = build_regex_from_schema(json.dumps(schema))
    compiled = re.compile(regex)
    matches = compiled.fullmatch(json.dumps(sample_output)) is not None

    assert matches == should_match


VALID_TOOL_OUTPUTS = [
    ([{"name": "get_current_weather", "parameters": {"city": "Vienna"}}], True),
    (
        [
            {"name": "get_current_weather", "parameters": {"city": "Vienna"}},
            {"name": "get_current_weather", "parameters": {"city": "Berlin"}},
        ],
        True,
    ),
    ([{"name": "get_forecast", "parameters": {"city": "Vienna", "days": 7}}], True),
    (
        [
            {"name": "get_forecast", "parameters": {"city": "Vienna", "days": 7}},
            {"name": "get_current_weather", "parameters": {"city": "Vienna"}},
        ],
        True,
    ),
    (
        [
            {"name": "get_forecast", "parameters": {"city": "Vienna", "days": 7}},
            {"name": "get_current_weather", "parameters": {"city": "Vienna"}},
            {"name": "get_forecast", "parameters": {"city": "Berlin", "days": 7}},
            {"name": "get_current_weather", "parameters": {"city": "Berlin"}},
        ],
        True,
    ),
]

VALID_TOOLS = [t[0] for t in VALID_TOOL_OUTPUTS]


@pytest.mark.parametrize(
    "sample_output, should_match",
    VALID_TOOL_OUTPUTS
    + [
        (None, False),
        ([], False),  # empty list cannot be generated
        ({}, False),  # empty object cannot be generated
        ([{}], False),  # list with empty object cannot be generated
        (
            [
                {  # function without required parameters cannot be generated
                    "name": "get_current_weather"
                }
            ],
            False,
        ),
        (
            [
                {  # function without required parameters cannot be generated
                    "name": "get_current_weather",
                    "parameters": {},
                }
            ],
            False,
        ),
        (
            [
                {  # function without required parameters cannot be generated
                    "name": "get_current_weather",
                    "parameters": None,
                }
            ],
            False,
        ),
        (
            {  # tool call without lists cannot be generated
                "name": "get_current_weather",
                "parameters": {"city": "Vienna"},
            },
            False,
        ),
        (
            [
                {  # tool call with extra parameters cannot be generated
                    "name": "get_current_weather",
                    "parameters": {"city": "Vienna", "extra": "value"},
                }
            ],
            False,
        ),
        (
            [
                {  # tool call where parameters are first cannot be generated
                    "parameters": {"city": "Vienna"},
                    "name": "get_current_weather",
                }
            ],
            False,
        ),
        (
            [
                {  # tool call without all required parameters cannot be generated
                    "name": "get_forecast",
                    "parameters": {"city": "Vienna"},
                }
            ],
            False,
        ),
        (  # tool call with incorrect name/parameters cannot be generated
            [{"name": "get_weather", "parameters": {"city": "Vienna", "days": 7}}],
            False,
        ),
        (  #  tool call with both valid and empty function cannot be generated
            [{"name": "get_current_weather", "parameters": {"city": "Vienna"}}, {}],
            False,
        ),
    ],
)
def test_structured_outputs_json(sample_output, should_match):
    _compile_and_check(
        tools=TypeAdapter(list[ChatCompletionToolsParam]).validate_python(
            EXAMPLE_TOOLS
        ),
        sample_output=sample_output,
        should_match=should_match,
    )


def update_parameters_none(tool: ChatCompletionToolsParam) -> ChatCompletionToolsParam:
    tool.function.parameters = None
    return tool


def update_parameters_empty_dict(
    tool: ChatCompletionToolsParam,
) -> ChatCompletionToolsParam:
    tool.function.parameters = {}
    return tool


@pytest.mark.parametrize(
    "sample_output, should_match",
    [
        (None, False),
        ([], False),  # empty list cannot be generated
        ({}, False),  # empty object cannot be generated
        ([{}], False),  # list with empty object cannot be generated
        (
            [
                {  # function without required parameters cannot be generated
                    "name": "get_current_weather"
                }
            ],
            False,
        ),
        (
            [
                {  # function without required parameters cannot be generated
                    "name": "get_current_weather",
                    "parameters": None,
                }
            ],
            False,
        ),
        (
            [
                {  # function with extra parameters cannot be generated
                    "name": "get_current_weather",
                    "parameters": {"extra": "value"},
                }
            ],
            False,
        ),
        (
            [
                {  # only function with empty parameters object is valid
                    "name": "get_current_weather",
                    "parameters": {},
                }
            ],
            True,
        ),
    ],
)
@pytest.mark.parametrize(
    "update_parameters", [update_parameters_none, update_parameters_empty_dict]
)
def test_structured_outputs_json_without_parameters(
    sample_output, should_match, update_parameters
):
    updated_tools = [deepcopy(EXAMPLE_TOOLS[0])]
    tools = TypeAdapter(list[ChatCompletionToolsParam]).validate_python(updated_tools)
    tools = list(map(update_parameters, tools))
    assert all(
        [
            tool.function.parameters is None or tool.function.parameters == {}
            for tool in tools
        ]
    )
    _compile_and_check(
        tools=tools, sample_output=sample_output, should_match=should_match
    )


def _stream_required_tool_calls(
    output_json: str,
    deltas: list[str],
    tool_call_id_type: str = "random",
    tool_call_idx: int | None = None,
) -> list[dict]:
    """Feed ``deltas`` (which concatenate to ``output_json``) through the
    required-tool streaming helper and rebuild the calls per array index,
    checking that every index gets exactly one id/name chunk first."""
    assert "".join(deltas) == output_json
    scanner = RequiredToolCallScanner()
    calls: dict[int, dict] = {}
    for delta_text in deltas:
        delta_message, _ = extract_required_tool_call_streaming(
            delta_text=delta_text,
            tool_call_idx=tool_call_idx,
            tool_call_id_type=tool_call_id_type,
            scanner=scanner,
        )
        if delta_message is None:
            continue
        assert delta_message.tool_calls
        for tc in delta_message.tool_calls:
            assert tc.function is not None
            if tc.id is not None:
                assert tc.index not in calls, "id/name sent twice for one index"
                assert tc.type == "function"
                assert tc.function.name
                calls[tc.index] = {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "",
                }
            else:
                assert tc.index in calls, "arguments before id/name"
                assert tc.function.name is None
                assert tc.function.arguments
                calls[tc.index]["arguments"] += tc.function.arguments
    assert list(calls) == list(range(len(calls)))
    return [calls[i] for i in range(len(calls))]


def _fixed_len_deltas(text: str, delta_len: int) -> list[str]:
    return [text[i : i + delta_len] for i in range(0, len(text), delta_len)]


@pytest.mark.parametrize("delta_len", [1, 13])
def test_streaming_parameters_before_name(delta_len):
    output = [
        {"parameters": call["parameters"], "name": call["name"]} for call in TWO_CALLS
    ]
    _assert_streams_to(output, _fixed_len_deltas(json.dumps(output), delta_len))


def _assert_streams_to(output: list[dict], deltas: list[str]) -> None:
    calls = _stream_required_tool_calls(json.dumps(output), deltas)
    assert [
        {"name": c["name"], "parameters": json.loads(c["arguments"])} for c in calls
    ] == output
    for call, expected in zip(calls, output):
        assert call["arguments"] == json.dumps(expected["parameters"])


@pytest.mark.parametrize("output", VALID_TOOLS)
@pytest.mark.parametrize("empty_params", [False, True])
@pytest.mark.parametrize("delta_len", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_streaming_output_valid(output, empty_params, delta_len):
    output = deepcopy(output)
    if empty_params:
        output = [{"name": o["name"], "parameters": {}} for o in output]
    _assert_streams_to(output, _fixed_len_deltas(json.dumps(output), delta_len))


@pytest.mark.parametrize(
    "city",
    [
        "a { b",
        "a } b",
        "a }} b",
        'a " } b',
        r"a \ } b",
        # Text that looks like the end of one call and the start of another.
        '"}}, {"name": "get_forecast", "parameters": {"city": "x',
    ],
)
@pytest.mark.parametrize("delta_len", [1, 2, 3, 8, 9999])
def test_streaming_output_valid_with_braces_in_string(city, delta_len):
    output = [
        {"name": "get_current_weather", "parameters": {"city": city}},
        {"name": "get_forecast", "parameters": {"city": city, "days": 1}},
    ]
    _assert_streams_to(output, _fixed_len_deltas(json.dumps(output), delta_len))


def test_streaming_output_valid_with_trailing_extra_data():
    output = [{"name": "get_current_weather", "parameters": {"city": "Vienna"}}]
    output_json = json.dumps(output) + "\nDONE"
    calls = _stream_required_tool_calls(output_json, _fixed_len_deltas(output_json, 3))
    assert [
        {"name": c["name"], "parameters": json.loads(c["arguments"])} for c in calls
    ] == output


TWO_CALLS = [
    {"name": "get_current_weather", "parameters": {"city": "Dallas"}},
    {"name": "get_forecast", "parameters": {"city": "Dallas", "days": 3}},
]


def _split_at(text: str, cuts: list[int]) -> list[str]:
    bounds = [0, *cuts, len(text)]
    return [text[a:b] for a, b in zip(bounds, bounds[1:])]


def test_streaming_multiple_calls_completed_in_one_delta():
    """Regression test for #60351: a delta that completes more than one
    call must stream every call, not only the last one."""
    text = json.dumps(TWO_CALLS)
    _assert_streams_to(TWO_CALLS, [text[:1], text[1:]])
    _assert_streams_to(TWO_CALLS, [text])


def test_streaming_delta_crossing_call_boundary():
    """A delta that closes one call and opens the next must attribute the
    closing argument text to the first call and still announce the next."""
    text = json.dumps(TWO_CALLS)
    boundary = text.index("}, {") + 1
    name_end = text.index('"get_forecast"') + 6
    for cuts in (
        [boundary - 1, name_end],  # '}, {"name": "get_f' in one delta
        [boundary - 2, boundary + 2],  # '"}}, {' then the rest
        [boundary + 2],  # opening brace of the next call ends a delta
    ):
        _assert_streams_to(TWO_CALLS, _split_at(text, cuts))


def test_streaming_invalid_name_string_is_never_announced():
    """A closed name that is not a valid JSON string must not be streamed
    (the non-streaming path rejects such an array as well)."""
    text = '[{"name": "bad\\x", "parameters": {"city": "x"}}]'
    assert _stream_required_tool_calls(text, _fixed_len_deltas(text, 4)) == []


def test_streaming_tool_call_idx_increments_per_started_call():
    """kimi_k2 ids number the calls started in one delta consecutively."""
    text = json.dumps(TWO_CALLS)
    calls = _stream_required_tool_calls(
        text, [text], tool_call_id_type="kimi_k2", tool_call_idx=5
    )
    assert [c["id"] for c in calls] == [
        "functions.get_current_weather:5",
        "functions.get_forecast:6",
    ]


FUNCTION_TOOL = FunctionTool(
    type="function",
    name="get_weather",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)
WEB_SEARCH_TOOL = WebSearchTool(type="web_search")


class TestNonFunctionToolsSkipped:
    """Non-function tools (web_search, etc.) must be silently skipped
    by the tool-schema utilities instead of raising TypeError."""

    def test_find_tool_properties_skips_web_search(self):
        tools = [WEB_SEARCH_TOOL, FUNCTION_TOOL]
        props = find_tool_properties(tools, "get_weather")
        assert props == {"city": {"type": "string"}}

    def test_find_tool_properties_only_non_function_tools(self):
        props = find_tool_properties([WEB_SEARCH_TOOL], "get_weather")
        assert props == {}

    def test_get_json_schema_with_mixed_tools(self):
        tools = [WEB_SEARCH_TOOL, FUNCTION_TOOL]
        schema = get_json_schema_from_tools(tools=tools, tool_choice="required")
        assert isinstance(schema, dict)
        any_of = schema["items"]["anyOf"]
        assert len(any_of) == 1
        assert any_of[0]["properties"]["name"]["enum"] == ["get_weather"]

    def test_get_json_schema_rejects_only_non_function_tools(self):
        # An empty anyOf would compile to a grammar no output can satisfy.
        with pytest.raises(VLLMValidationError, match="no function tool"):
            get_json_schema_from_tools(tools=[WEB_SEARCH_TOOL], tool_choice="required")


class TestMalformedToolSchemaDefs:
    """Malformed `$defs` in caller-supplied tool parameters is a 400, not a 500."""

    @staticmethod
    def _tool(params: dict) -> ChatCompletionToolsParam:
        return TypeAdapter(ChatCompletionToolsParam).validate_python(
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": params,
                },
            }
        )

    @pytest.mark.parametrize(
        "defs",
        [
            pytest.param(None, id="null"),
            pytest.param([], id="list"),
            pytest.param("nope", id="string"),
            pytest.param(1, id="int"),
        ],
    )
    def test_non_object_defs_is_a_client_error(self, defs):
        tools = [
            self._tool(
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "$defs": defs,
                }
            )
        ]

        with pytest.raises(VLLMValidationError) as exc_info:
            get_json_schema_from_tools(tools=tools, tool_choice="required")

        assert exc_info.value.parameter == "tools"
        assert "$defs" in str(exc_info.value)
        assert "get_weather" in str(exc_info.value)

    def test_duplicate_def_names_is_a_client_error(self):
        """Also caller-caused, and it used to raise a plain `ValueError`."""
        tools = [
            self._tool(
                {
                    "type": "object",
                    "properties": {"a": {"$ref": "#/$defs/D"}},
                    "$defs": {"D": {"type": "string"}},
                }
            ),
            self._tool(
                {
                    "type": "object",
                    "properties": {"b": {"$ref": "#/$defs/D"}},
                    "$defs": {"D": {"type": "integer"}},
                }
            ),
        ]

        with pytest.raises(VLLMValidationError) as exc_info:
            get_json_schema_from_tools(tools=tools, tool_choice="required")

        assert exc_info.value.parameter == "tools"
        assert "multiple schemas" in str(exc_info.value)

    def test_well_formed_defs_still_hoisted(self):
        """The type check must not reject the shape it is guarding."""
        tools = [
            self._tool(
                {
                    "type": "object",
                    "properties": {"a": {"$ref": "#/$defs/D"}},
                    "$defs": {"D": {"type": "string"}},
                }
            )
        ]

        schema = get_json_schema_from_tools(tools=tools, tool_choice="required")

        assert isinstance(schema, dict)
        assert schema["$defs"] == {"D": {"type": "string"}}

    def test_absent_defs_still_works(self):
        tools = [
            self._tool({"type": "object", "properties": {"a": {"type": "string"}}})
        ]

        schema = get_json_schema_from_tools(tools=tools, tool_choice="required")

        assert isinstance(schema, dict)


class TestParallelToolCallsConstraint:
    """`parallel_tool_calls=false` must be enforced by the decoding grammar.

    Without a `maxItems` bound the model is free to emit an unbounded run of
    tool calls that are only discarded afterwards, wasting the token budget and
    risking truncation of the one call the client actually receives."""

    TOOLS = TypeAdapter(list[ChatCompletionToolsParam]).validate_python(EXAMPLE_TOOLS)
    ONE_CALL = [{"name": "get_current_weather", "parameters": {"city": "Vienna"}}]
    TWO_CALLS = [
        {"name": "get_current_weather", "parameters": {"city": "Vienna"}},
        {"name": "get_current_weather", "parameters": {"city": "Berlin"}},
    ]

    def test_disabled_rejects_a_second_tool_call(self):
        _compile_and_check(self.TOOLS, self.ONE_CALL, True, parallel_tool_calls=False)
        _compile_and_check(self.TOOLS, self.TWO_CALLS, False, parallel_tool_calls=False)

    @pytest.mark.parametrize("parallel_tool_calls", [True, None])
    def test_enabled_or_unset_still_allows_multiple(self, parallel_tool_calls):
        _compile_and_check(
            self.TOOLS,
            self.TWO_CALLS,
            True,
            parallel_tool_calls=parallel_tool_calls,
        )

    @pytest.mark.parametrize(
        "parallel_tool_calls,expected",
        [(False, 1), (True, None), (None, None)],
    )
    def test_max_items_bound(self, parallel_tool_calls, expected):
        schema = get_json_schema_from_tools(
            tools=self.TOOLS,
            tool_choice="required",
            parallel_tool_calls=parallel_tool_calls,
        )
        assert isinstance(schema, dict)
        assert schema["minItems"] == 1
        assert schema.get("maxItems") == expected

    def test_forced_named_tool_is_unaffected(self):
        # Named tool choice yields a bare parameters object, never an array.
        schema = get_json_schema_from_tools(
            tools=self.TOOLS,
            tool_choice=ChatCompletionNamedToolChoiceParam(
                function=ChatCompletionNamedFunction(name="get_current_weather")
            ),
            parallel_tool_calls=False,
        )
        assert isinstance(schema, dict)
        assert "maxItems" not in schema


class TestForcedNamedToolChoiceEmptyParams:
    """A forced named tool_choice with missing/empty parameters must still
    constrain the generated arguments to a JSON object, like the
    `tool_choice="required"` path, instead of leaving them unconstrained."""

    @pytest.mark.parametrize("params", [None, {}])
    def test_chat_empty_params_constrains_object(self, params):
        tool = ChatCompletionToolsParam.model_validate(
            {"type": "function", "function": {"name": "ping", "parameters": params}}
        )
        choice = ChatCompletionNamedToolChoiceParam.model_validate(
            {"type": "function", "function": {"name": "ping"}}
        )
        schema = get_json_schema_from_tools(choice, [tool])
        assert schema == {"type": "object", "properties": {}}

    @pytest.mark.parametrize("params", [None, {}])
    def test_responses_empty_params_constrains_object(self, params):
        tool = FunctionTool(type="function", name="ping", parameters=params)
        choice = ToolChoiceFunction(type="function", name="ping")
        schema = get_json_schema_from_tools(choice, [tool])
        assert schema == {"type": "object", "properties": {}}


class TestForcedFunctionShortNameAlias:
    """Forced choices must use the selected function's parameter schema."""

    PARAMS = {"type": "object", "properties": {"x": {"type": "integer"}}}

    @pytest.mark.parametrize("plain_first", [True, False])
    @pytest.mark.parametrize(
        "namespace,function_name,choice_name",
        [
            ("math", "add", "add"),
            ("math", "add", "math__add"),
            ("math", "vector__add", "vector__add"),
            ("math__vector", "vector__add", "vector__add"),
        ],
    )
    def test_namespace_schema_is_not_shadowed(
        self, plain_first, namespace, function_name, choice_name
    ):
        tools = [
            {
                "type": "function",
                "name": "other__add",
                "parameters": {
                    "type": "object",
                    "properties": {"y": {"type": "string"}},
                },
            },
            {
                "type": "namespace",
                "name": namespace,
                "description": "math tools",
                "tools": [
                    {
                        "type": "function",
                        "name": function_name,
                        "parameters": self.PARAMS,
                    }
                ],
            },
        ]
        if not plain_first:
            tools.reverse()
        request = ResponsesRequest.model_validate(
            {
                "model": "test-model",
                "input": "Add the numbers.",
                "tools": tools,
                "tool_choice": {"type": "function", "name": choice_name},
            }
        )
        assert (
            get_json_schema_from_tools(request.tool_choice, request.tools)
            == self.PARAMS
        )

    def test_plain_function_double_underscore_name_has_no_phantom_alias(self):
        tool = FunctionTool(type="function", name="math__add", parameters=self.PARAMS)
        choice = ToolChoiceFunction(type="function", name="add")
        with pytest.raises(ValueError, match="has not been passed in `tools`"):
            get_json_schema_from_tools(choice, [tool])

    def test_plain_function_full_name_still_resolves(self):
        tool = FunctionTool(type="function", name="math__add", parameters=self.PARAMS)
        choice = ToolChoiceFunction(type="function", name="math__add")
        assert get_json_schema_from_tools(choice, [tool]) == self.PARAMS
