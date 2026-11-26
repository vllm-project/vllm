# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai_harmony import (
    Message,
)
from pydantic import ValidationError

from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    ResponsesResponse,
    serialize_message,
    serialize_messages,
)
from vllm.entrypoints.openai.responses.utils import (
    construct_input_messages,
    construct_tool_dicts,
)
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams

pytestmark = pytest.mark.skip_global_cleanup


def test_serialize_message() -> None:
    dict_value = {"a": 1, "b": "2"}
    assert serialize_message(dict_value) == dict_value

    msg_value = {
        "role": "assistant",
        "name": None,
        "content": [{"type": "text", "text": "Test 1"}],
        "channel": "analysis",
    }
    msg = Message.from_dict(msg_value)
    assert serialize_message(msg) == msg_value


def test_serialize_messages() -> None:
    assert serialize_messages(None) is None
    assert serialize_messages([]) is None

    dict_value = {"a": 3, "b": "4"}
    msg_value = {
        "role": "assistant",
        "name": None,
        "content": [{"type": "text", "text": "Test 2"}],
        "channel": "analysis",
    }
    msg = Message.from_dict(msg_value)
    assert serialize_messages([msg, dict_value]) == [msg_value, dict_value]


def test_normalize_openresponses_input_item_stamps_type() -> None:
    """A typeless assistant item must keep its role and content.

    `type` is optional in the OpenResponses schema, but filling in
    `id`/`status` without it makes the item match `ItemReference` in the
    input union, which collapses it to `{"id": ...}` and silently drops the
    assistant turn from the conversation. Regression test for that.
    """
    item = {
        "role": "assistant",
        "content": [{"type": "output_text", "text": "hi"}],
    }

    normalized = ResponsesRequest._normalize_openresponses_input_item(item)

    assert normalized["type"] == "message"
    assert normalized["role"] == "assistant"
    assert normalized["id"].startswith("msg_")
    assert normalized["status"] == "completed"
    # annotations are filled in for output_text content
    assert normalized["content"][0]["annotations"] == []
    # the caller's dict is not mutated
    assert "type" not in item


def test_normalize_openresponses_input_item_reasoning_type() -> None:
    """An explicit reasoning item keeps its type and gets a summary default."""
    normalized = ResponsesRequest._normalize_openresponses_input_item(
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "why"}]}
    )

    assert normalized["type"] == "reasoning"
    assert normalized["id"].startswith("rs_")
    assert normalized["status"] == "completed"
    assert normalized["summary"] == []


def test_input_item_parsing_typeless_assistant_message() -> None:
    """End to end: a typeless assistant item survives as a real message."""
    request = ResponsesRequest.model_validate(
        {
            "input": [
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}],
                }
            ]
        }
    )

    assert isinstance(request.input, list)
    (parsed,) = request.input
    assert isinstance(parsed, ResponseOutputMessage)
    assert parsed.role == "assistant"
    assert parsed.content[0].text == "hi"


def test_input_item_parsing_leaves_non_assistant_items_alone() -> None:
    """User messages are untouched — normalization only targets assistant items."""
    item = {"role": "user", "content": "hello"}
    request = ResponsesRequest.model_validate({"input": [item]})

    assert request.input == [item]


def test_input_list_validation_stops_at_first_invalid_item() -> None:
    valid_item = {"role": "user", "content": "hello"}
    invalid_item = {"type": "message", "role": "user"}
    with pytest.raises(ValidationError) as first:
        ResponsesRequest(input=[valid_item, invalid_item])
    with pytest.raises(ValidationError) as multiple:
        ResponsesRequest(input=[valid_item, invalid_item, invalid_item])

    assert multiple.value.error_count() == first.value.error_count()


def test_custom_tool_grammar_format_rejected() -> None:
    grammar = {"type": "grammar", "syntax": "lark", "definition": 'start: "pwd"'}
    with pytest.raises(VLLMValidationError) as exc_info:
        ResponsesRequest(
            input="hi", tools=[{"type": "custom", "name": "emit", "format": grammar}]
        )
    assert exc_info.value.parameter == "tools"

    request = ResponsesRequest(
        input="hi",
        tools=[{"type": "custom", "name": "emit", "format": {"type": "text"}}],
    )
    assert request.tools is not None and request.tools[0].type == "custom"


@pytest.mark.parametrize(
    "tools",
    [
        "abc",
        {"type": "custom"},
        [None],
        [{"type": "custom", "name": "x", "format": "text"}],
    ],
)
def test_malformed_tools_rejected_by_field_validation(tools) -> None:
    """Malformed tools bodies fail pydantic field validation (4xx), not an
    AttributeError inside the custom-format validator (500)."""
    with pytest.raises(ValidationError):
        ResponsesRequest(input="hi", tools=tools)


def _codex_additional_tools_item() -> dict:
    """Shape sent by Codex >= 0.169 as an input item instead of `tools`."""
    return {
        "type": "additional_tools",
        "id": "at_123",
        "role": "developer",
        "tools": [
            {
                "type": "namespace",
                "name": "clock",
                "description": "",
                "tools": [
                    {
                        "type": "function",
                        "name": "sleep",
                        "description": "Sleep for a while.",
                        "parameters": {"type": "object", "properties": {}},
                        "strict": False,
                    }
                ],
            }
        ],
    }


def test_additional_tools_item_hoisted_into_tools() -> None:
    top_level_tool = {
        "type": "function",
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {}},
        "strict": False,
    }
    user_message = {"role": "user", "content": "Reply with OK"}
    request = ResponsesRequest.model_validate(
        {
            "input": [_codex_additional_tools_item(), user_message],
            "tools": [top_level_tool],
        }
    )

    assert request.input == [user_message]
    assert [tool.type for tool in request.tools] == ["function", "namespace"]
    assert request.tools[0].name == "get_weather"
    assert request.tools[1].name == "clock"
    assert request.tools[1].tools[0].name == "sleep"


def test_additional_tools_items_merged_without_top_level_tools() -> None:
    first = _codex_additional_tools_item()
    second = _codex_additional_tools_item()
    second["tools"][0]["name"] = "timer"
    request = ResponsesRequest.model_validate(
        {"input": [first, {"role": "user", "content": "hi"}, second]}
    )

    assert len(request.input) == 1
    assert [tool.name for tool in request.tools] == ["clock", "timer"]
    # Tool-choice validation must see the hoisted tools, or "auto" becomes "none".
    assert request.tool_choice == "auto"


@pytest.mark.parametrize(
    "tool_choice,expected",
    [
        ("required", "required"),
        ({"type": "function", "name": "sleep"}, "clock__sleep"),
    ],
)
def test_additional_tools_satisfy_tool_choice(tool_choice, expected) -> None:
    request = ResponsesRequest.model_validate(
        {
            "input": [
                _codex_additional_tools_item(),
                {"role": "user", "content": "hi"},
            ],
            "tool_choice": tool_choice,
        }
    )

    resolved = request.tool_choice
    assert (resolved if isinstance(resolved, str) else resolved.name) == expected


def test_additional_tools_custom_grammar_rejected() -> None:
    grammar = {"type": "grammar", "syntax": "lark", "definition": 'start: "pwd"'}
    item = {
        "type": "additional_tools",
        "role": "developer",
        "tools": [{"type": "custom", "name": "emit", "format": grammar}],
    }
    with pytest.raises(VLLMValidationError) as exc_info:
        ResponsesRequest.model_validate({"input": [item]})
    assert exc_info.value.parameter == "tools"


def test_additional_tools_item_without_tools_list_rejected() -> None:
    item = {"type": "additional_tools", "role": "developer", "tools": "oops"}
    with pytest.raises(VLLMValidationError) as exc_info:
        ResponsesRequest.model_validate({"input": [item]})
    assert exc_info.value.parameter == "input"


def _function_tool(name: str, description: str = "") -> dict:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {}},
    }


def test_additional_tools_deduplicated() -> None:
    """Repeated definitions are offered once; a namespace split across items
    is merged, so a named choice of a repeated namespaced tool is unambiguous."""
    clock = _codex_additional_tools_item()
    timer = _codex_additional_tools_item()
    timer["tools"][0]["tools"][0]["name"] = "timer"
    weather = _function_tool("get_weather")
    request = ResponsesRequest.model_validate(
        {
            "input": [
                clock,
                {**_codex_additional_tools_item(), "tools": [weather]},
                {"role": "user", "content": "hi"},
                _codex_additional_tools_item(),
                timer,
            ],
            "tools": [weather],
            "tool_choice": {"type": "function", "name": "sleep"},
        }
    )

    assert [tool.name for tool in request.tools] == ["get_weather", "clock"]
    assert [tool.name for tool in request.tools[1].tools] == ["sleep", "timer"]
    assert request.tool_choice.name == "clock__sleep"


def test_additional_tools_redefinition_last_wins_in_place() -> None:
    def item(*tools: dict) -> dict:
        return {"type": "additional_tools", "role": "developer", "tools": list(tools)}

    def clock(*tools: dict) -> dict:
        return {
            "type": "namespace",
            "name": "clock",
            "description": "",
            "tools": list(tools),
        }

    request = ResponsesRequest.model_validate(
        {
            "input": [
                item(clock(_function_tool("sleep", "v1"), _function_tool("timer"))),
                item(_function_tool("get_weather", "v2")),
                {"role": "user", "content": "hi"},
                item(clock(_function_tool("sleep", "v2"))),
            ],
            "tools": [_function_tool("get_weather", "v1"), _function_tool("get_time")],
        }
    )

    assert [tool.name for tool in request.tools] == ["get_weather", "get_time", "clock"]
    assert request.tools[0].description == "v2"
    assert [(t.name, t.description) for t in request.tools[2].tools] == [
        ("sleep", "v2"),
        ("timer", ""),
    ]


@pytest.mark.parametrize("with_item", [True, False])
@pytest.mark.parametrize("tools", [5, "abc", {"type": "function"}])
def test_non_list_tools_rejected_by_field_validation(tools, with_item) -> None:
    items = [_codex_additional_tools_item()] if with_item else []
    with pytest.raises(ValidationError) as exc_info:
        ResponsesRequest.model_validate(
            {"input": [*items, {"role": "user", "content": "hi"}], "tools": tools}
        )
    (error,) = exc_info.value.errors()
    assert error["loc"] == ("tools",) and error["type"] == "list_type"


_TOOL_SEARCH = {"type": "tool_search", "execution": "client"}
_JIRA = {
    "type": "function",
    "name": "jira_search",
    "description": "Search Jira",
    "defer_loading": True,
    "parameters": {"type": "object", "properties": {}},
}


def _tool_search_turn(loaded: list[dict]) -> list[dict]:
    return [
        {"role": "user", "content": "Find my open bugs"},
        {
            "type": "tool_search_call",
            "id": "tsc_1",
            "call_id": "call_1",
            "execution": "client",
            "status": "completed",
            "arguments": {"query": "jira"},
        },
        {
            "type": "tool_search_output",
            "id": "tso_1",
            "call_id": "call_1",
            "execution": "client",
            "status": "completed",
            "tools": loaded,
        },
    ]


def test_tool_search_output_keeps_declared_tools() -> None:
    """A loaded tool that is already declared keeps its declaration, so the
    tool list (and the prompt prefix) does not change."""
    request = ResponsesRequest.model_validate(
        {
            "input": _tool_search_turn([{**_JIRA, "defer_loading": None}]),
            "tools": [_TOOL_SEARCH, _JIRA],
        }
    )
    assert [tool.type for tool in request.tools] == ["tool_search", "function"]
    assert request.tools[1].defer_loading is True
    assert len(request.input) == 3


def test_tool_search_output_adds_undeclared_tools() -> None:
    open_pr = {
        "type": "function",
        "name": "open_pr",
        "parameters": {"type": "object", "properties": {}},
    }
    close_pr = {**open_pr, "name": "close_pr"}
    github = {"type": "namespace", "name": "github", "description": "GitHub"}
    request = ResponsesRequest.model_validate(
        {
            "input": _tool_search_turn(
                [_JIRA, {**github, "description": "new", "tools": [open_pr, close_pr]}]
            ),
            "tools": [_TOOL_SEARCH, _JIRA, {**github, "tools": [open_pr]}],
        }
    )
    assert [tool.type for tool in request.tools] == [
        "tool_search",
        "function",
        "namespace",
    ]
    namespace = request.tools[2]
    assert namespace.description == "GitHub"
    assert [tool.name for tool in namespace.tools] == ["open_pr", "close_pr"]


@pytest.mark.parametrize(
    "tools",
    [
        [_TOOL_SEARCH, _TOOL_SEARCH],
        [_TOOL_SEARCH, {**_JIRA, "name": "tool_search"}],
    ],
)
def test_tool_search_tool_conflicts_rejected(tools) -> None:
    with pytest.raises((VLLMValidationError, ValidationError)):
        ResponsesRequest.model_validate({"input": "hi", "tools": tools})


def _template_names(request: ResponsesRequest) -> list[str]:
    tool_dicts = construct_tool_dicts(request.template_tools(), request.tool_choice)
    return [tool["function"]["name"] for tool in tool_dicts or []]


@pytest.mark.parametrize("strict", [None, True])
def test_minimal_tool_search_items_render(strict) -> None:
    """Items with only the required fields validate, pair up, and render the
    same every time."""
    loaded = {"type": "function", "name": "open_pr", "parameters": {"type": "object"}}
    if strict is not None:
        loaded["strict"] = strict
    request = ResponsesRequest.model_validate(
        {
            "input": [
                {"role": "user", "content": "Open a PR"},
                {"type": "tool_search_call", "arguments": {"query": "github"}},
                {"type": "tool_search_output", "tools": [loaded]},
            ],
            "tools": [_TOOL_SEARCH],
        }
    )
    first = construct_input_messages(
        request_instructions=None, request_input=request.input
    )
    second = construct_input_messages(
        request_instructions=None, request_input=request.input
    )
    assert first == second
    [tool_call] = first[1]["tool_calls"]
    assert tool_call["function"]["name"] == "tool_search"
    assert first[2]["role"] == "tool"
    assert first[2]["tool_call_id"] == tool_call["id"]
    assert '"name": "open_pr"' in first[2]["content"]


def test_tool_search_output_only_loads_functions() -> None:
    request = ResponsesRequest.model_validate(
        {
            "input": _tool_search_turn(
                [
                    {"type": "web_search"},
                    {"type": "mcp", "server_label": "x", "server_url": "http://x"},
                    _TOOL_SEARCH,
                    _JIRA,
                ]
            ),
            "tools": [_TOOL_SEARCH],
        }
    )
    assert [tool.type for tool in request.tools] == ["tool_search", "function"]
    [message] = [
        m
        for m in construct_input_messages(
            request_instructions=None, request_input=request.input
        )
        if m["role"] == "tool"
    ]
    [line] = message["content"].splitlines()
    assert '"name": "jira_search"' in line


def test_tools_loaded_by_tool_search_stay_out_of_the_template() -> None:
    """Undeclared loaded tools reach the parser (`tools`) but not the chat
    template, so the tool list in the prompt does not change."""
    open_pr = {"type": "function", "name": "open_pr", "parameters": {}}
    github = {"type": "namespace", "name": "github", "description": "GitHub"}
    declared = [_TOOL_SEARCH, {**github, "tools": [open_pr]}]
    before = ResponsesRequest.model_validate({"input": "hi", "tools": declared})
    request = ResponsesRequest.model_validate(
        {
            "input": _tool_search_turn(
                [_JIRA, {**github, "tools": [{**open_pr, "name": "close_pr"}]}]
            ),
            "tools": declared,
        }
    )
    assert _template_names(request) == _template_names(before)
    assert _template_names(request) == ["tool_search", "github__open_pr"]
    assert [tool.type for tool in request.tools] == [
        "tool_search",
        "namespace",
        "function",
    ]
    assert [tool.name for tool in request.tools[1].tools] == ["open_pr", "close_pr"]


def test_response_echoes_declared_tools_only() -> None:
    request = ResponsesRequest.model_validate(
        {"input": _tool_search_turn([_JIRA]), "tools": [_TOOL_SEARCH]}
    )
    assert [tool.type for tool in request.tools] == ["tool_search", "function"]
    response = ResponsesResponse.from_request(
        request, SamplingParams(), "model", 0, [], "completed"
    )
    assert [tool.type for tool in response.tools] == ["tool_search"]


def test_tools_from_additional_tools_items_stay_in_the_template() -> None:
    """A tool declared by an `additional_tools` item counts as declared, even
    when a search loads it again."""
    declare = {"type": "additional_tools", "role": "developer", "tools": [_JIRA]}
    user = {"role": "user", "content": "Find my open bugs"}
    before = ResponsesRequest.model_validate(
        {"input": [declare, user], "tools": [_TOOL_SEARCH]}
    )
    after = ResponsesRequest.model_validate(
        {"input": [declare, *_tool_search_turn([_JIRA])], "tools": [_TOOL_SEARCH]}
    )
    assert _template_names(before) == ["tool_search", "jira_search"]
    assert _template_names(after) == _template_names(before)
    assert [tool.type for tool in after.tools] == ["tool_search", "function"]


def test_tool_search_outputs_without_call_id_answer_calls_in_order() -> None:
    def call(query: str) -> dict:
        return {"type": "tool_search_call", "arguments": {"query": query}}

    def output(name: str) -> dict:
        return {"type": "tool_search_output", "tools": [{**_JIRA, "name": name}]}

    request = ResponsesRequest.model_validate(
        {
            "input": [
                {"role": "user", "content": "hi"},
                {**call("a"), "call_id": "call_a"},
                call("b"),
                call("c"),
                output("b"),
                {**output("a"), "call_id": "call_a"},
                output("c"),
            ],
            "tools": [_TOOL_SEARCH],
        }
    )
    messages = construct_input_messages(
        request_instructions=None, request_input=request.input
    )
    a, b, c = (tool_call["id"] for tool_call in messages[1]["tool_calls"])
    assert a == "call_a"
    assert [message["tool_call_id"] for message in messages[2:]] == [b, a, c]


@pytest.mark.parametrize(
    "tools",
    ["abc", [1, "a"], [{"type": "function", "parameters": {}}]],
)
def test_invalid_tool_search_output_rejected(tools) -> None:
    with pytest.raises(VLLMValidationError, match="tool_search_output"):
        ResponsesRequest.model_validate(
            {
                "input": [{"type": "tool_search_output", "tools": tools}],
                "tools": [_TOOL_SEARCH],
            }
        )


def test_loaded_tools_never_shadow_declared_names() -> None:
    """A loaded tool with the name of a declared tool of another type, or of
    the tool_search function, is not added."""
    request = ResponsesRequest.model_validate(
        {
            "input": _tool_search_turn(
                [
                    {"type": "custom", "name": "jira_search"},
                    {**_JIRA, "name": "tool_search"},
                    {**_JIRA, "name": "gh__open_pr"},
                    {
                        "type": "namespace",
                        "name": "gh",
                        "description": "GitHub",
                        "tools": [{**_JIRA, "name": "open_pr"}],
                    },
                ]
            ),
            "tools": [_TOOL_SEARCH, _JIRA],
        }
    )
    assert [(tool.type, getattr(tool, "name", None)) for tool in request.tools] == [
        ("tool_search", None),
        ("function", "jira_search"),
        ("function", "gh__open_pr"),
    ]
