# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
from openai.types.responses import FunctionTool, ToolChoiceFunction
from xgrammar import Grammar
from xgrammar.testing import _is_grammar_accept_string

from vllm.config.reasoning import ReasoningConfig
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    ResponsesResponse,
)
from vllm.exceptions import VLLMValidationError
from vllm.parser.beam import BeamParser, BeamReasoningParser, BeamToolParser
from vllm.parser.parser_manager import ParserManager
from vllm.parser.response_template import ResponseTemplateParser
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager

# The `response_template` of the Beam checkpoints.
BEAM_RESPONSE_TEMPLATE: dict[str, Any] = {
    "defaults": {"role": "assistant"},
    "start_anchor_pattern": r"<\|start\|>assistant",
    "fields": {
        "thinking": {
            "open": "<|channel|>analysis<|message|>",
            "close": "<|end|><|start|>assistant",
            "content": "text",
        },
        "content": {
            "open": "<|channel|>final<|message|>",
            "close": ["<|return|>", "<|end|>", "<|call|>"],
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": (
                r" to=(?P<name>[^\s<]+)"
                r"<\|channel\|>analysis <\|constrain\|>xml<\|message\|>"
            ),
            "close": ["<|end|><|start|>assistant", "<|call|>"],
            "repeats": True,
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": (
                    r'<rfl:parameter name="(?P<key>[^"]+)">'
                    r'\n(?P<value>.*?)\n(?:</rfl:parameter>|(?=<rfl:parameter name="))'
                ),
                "value_parser": {"name": "text", "args": {"strip": False}},
            },
            "transform": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
    },
}

VOCAB = {
    "<|start|>": 200,
    "<|channel|>": 201,
    "<|message|>": 202,
    "<|end|>": 203,
    "<|call|>": 204,
    "<|return|>": 205,
    "<|constrain|>": 206,
}
TEXT_TOKENS = {100: "assistant", 101: "analysis", 102: "final", 103: "plan"}
PROMPT_TOKEN_IDS = [200, 100]
THOUGHT = "<|channel|>analysis<|message|>plan<|end|><|start|>assistant"
FINAL = "<|channel|>final<|message|>"
NEXT_MESSAGE = "<|end|><|start|>assistant"


def _tool(name, properties, *, strict=None):
    return ChatCompletionToolsParam(
        type="function",
        function={
            "name": name,
            "parameters": {"type": "object", "properties": properties},
            "strict": strict,
        },
    )


TOOLS = [
    _tool("get_weather", {"city": {"type": "string"}}),
    _tool("set_count", {"count": {"type": "integer"}}),
]
STRICT_TOOLS = [
    _tool("get_weather", {"city": {"type": "string"}}),
    _tool("set_count", {"count": {"type": "integer"}}, strict=True),
]


class BeamTokenizer:
    def __init__(self, response_template=BEAM_RESPONSE_TEMPLATE):
        self.response_template = response_template
        self.init_kwargs = {}
        self.name_or_path = ""
        self.all_special_tokens = list(VOCAB)
        self.all_special_ids = list(VOCAB.values())

    def decode(self, token_ids, skip_special_tokens=False, **_kwargs):
        inverse_vocab = {token_id: token for token, token_id in VOCAB.items()}
        return "".join(
            TEXT_TOKENS.get(token_id, inverse_vocab.get(token_id, ""))
            for token_id in token_ids
        )

    def get_vocab(self):
        return VOCAB

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(text.encode())


def request(**overrides):
    values = dict(
        tool_choice="auto",
        include_reasoning=True,
        tools=TOOLS,
        parallel_tool_calls=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def parameter(name, value):
    return f'<rfl:parameter name="{name}">\n{value}\n</rfl:parameter>'


def call(name, body, *, terminator="<|call|>"):
    return (
        f" to={name}<|channel|>analysis <|constrain|>xml<|message|>{body}{terminator}"
    )


def make_parser(prompt_token_ids=PROMPT_TOKEN_IDS, tools=TOOLS, **kwargs):
    parser_cls = ParserManager.get_parser(
        tool_parser_name="beam",
        reasoning_parser_name="beam",
        enable_auto_tools=True,
        tokenizer=BeamTokenizer(),
    )
    assert parser_cls is not None
    parser = parser_cls(BeamTokenizer(), tools, **kwargs)
    parser.set_prompt_token_ids(prompt_token_ids)
    return parser


def parse(text, prompt_token_ids=PROMPT_TOKEN_IDS):
    reasoning, content, calls = make_parser(prompt_token_ids).parse(
        text, request(), enable_auto_tools=True
    )
    return reasoning, content, normalize_calls(calls)


def normalize_calls(calls):
    return [(call.name, json.loads(call.arguments)) for call in calls or []]


def test_beam_parsers_are_registered():
    assert ToolParserManager.get_tool_parser("beam") is BeamToolParser
    assert ReasoningParserManager.get_reasoning_parser("beam") is BeamReasoningParser


def test_parser_manager_selects_beam_parser():
    parser_cls = ParserManager.get_parser(
        tool_parser_name="beam",
        reasoning_parser_name="beam",
        enable_auto_tools=True,
        tokenizer=BeamTokenizer(),
    )

    assert parser_cls is not None
    assert issubclass(parser_cls, BeamParser)
    assert parser_cls.reasoning_parser_cls is BeamReasoningParser
    assert parser_cls.tool_parser_cls is BeamToolParser
    assert parser_cls._enable_auto_tools


@pytest.mark.parametrize("other", ["hermes", "hf"])
def test_beam_parser_rejects_other_parsers(other):
    with pytest.raises(TypeError, match="cannot be combined"):
        ParserManager.get_parser(
            tool_parser_name=other,
            reasoning_parser_name="beam",
            enable_auto_tools=True,
            tokenizer=BeamTokenizer(),
        )


def test_beam_parser_requires_checkpoint_metadata_at_startup():
    with pytest.raises(
        TypeError, match="--reasoning-parser beam requires `response_template`"
    ):
        ParserManager.get_parser(
            reasoning_parser_name="beam",
            tokenizer=BeamTokenizer(response_template=None),
        )

    template = deepcopy(BEAM_RESPONSE_TEMPLATE)
    del template["fields"]["tool_calls"]
    with pytest.raises(TypeError, match="--tool-call-parser beam requires"):
        ParserManager.get_parser(
            tool_parser_name="beam",
            enable_auto_tools=True,
            tokenizer=BeamTokenizer(template),
        )


def test_reasoning_and_final_answer():
    assert parse(THOUGHT + FINAL + "Sunny.<|return|>") == ("plan", "Sunny.", [])


def test_prefilled_reasoning_continues_from_prompt():
    prompt = PROMPT_TOKEN_IDS + [201, 101, 202, 103]
    text = " more.<|end|><|start|>assistant" + FINAL + "Done.<|return|>"

    assert parse(text, prompt) == (" more.", "Done.", [])


def test_parallel_calls_are_coerced_to_their_schemas():
    text = (
        THOUGHT
        + call(
            "get_weather",
            parameter("city", "Paris"),
            terminator="<|end|><|start|>assistant",
        )
        + call("set_count", parameter("count", "7"))
    )

    assert parse(text) == (
        "plan",
        None,
        [("get_weather", {"city": "Paris"}), ("set_count", {"count": 7})],
    )


def test_call_stop_token_removed_from_the_text_still_ends_the_call():
    text = THOUGHT + call("get_weather", parameter("city", "Paris"), terminator="")

    assert parse(text) == ("plan", None, [("get_weather", {"city": "Paris"})])


@pytest.mark.parametrize("end", ["<|end|>", "<|end|><|start|>"])
def test_output_cut_before_the_next_message_start_ends_its_region(end):
    """Output that stops after `<|end|>`, such as at `max_tokens`, has ended
    its reasoning or call and lacks only the next message's start."""
    weather = call("get_weather", parameter("city", "Paris"), terminator=end)

    assert parse("<|channel|>analysis<|message|>plan" + end) == ("plan", None, [])
    assert parse(THOUGHT + weather) == (
        "plan",
        None,
        [("get_weather", {"city": "Paris"})],
    )


THOUGHT_IDS = [201, 101, 202, 103, 203, 200, 100]
FINAL_IDS = [201, 102, 202]


@pytest.mark.parametrize(
    ("prompt_token_ids", "token_ids", "count"),
    [
        (PROMPT_TOKEN_IDS, [*THOUGHT_IDS, *FINAL_IDS, 205], 1),
        (PROMPT_TOKEN_IDS + [201, 101, 202], [103, 103, 203, 200, 100, *FINAL_IDS], 2),
        (PROMPT_TOKEN_IDS, [201, 101, 202, 103, 103, 103], 3),
    ],
    ids=["reasoning_then_answer", "prefilled_reasoning", "cut_off"],
)
def test_reasoning_tokens_are_counted(prompt_token_ids, token_ids, count):
    parser = make_parser(prompt_token_ids)

    assert parser.count_reasoning_tokens(token_ids) == count


def test_reasoning_tokens_are_counted_as_the_generation_grows():
    token_ids = [*THOUGHT_IDS, *FINAL_IDS, 205]
    parser = make_parser()

    assert [
        parser.count_reasoning_tokens(token_ids[:end])
        for end in range(1, len(token_ids) + 1)
    ] == [
        make_parser().count_reasoning_tokens(token_ids[:end])
        for end in range(1, len(token_ids) + 1)
    ]
    assert parser.count_reasoning_tokens(FINAL_IDS) == 0


def test_call_stop_token_ends_the_answer():
    assert parse(FINAL + "done<|call|>") == (None, "done", [])


@pytest.mark.parametrize("chunk_size", [1, 7, 10_000])
def test_streaming_matches_non_streaming(chunk_size):
    text = (
        THOUGHT
        + call(
            "get_weather",
            parameter("city", "Paris"),
            terminator="<|end|><|start|>assistant",
        )
        + call("set_count", parameter("count", "7"))
    )
    parser = make_parser()
    reasoning, calls = [], []
    for start in range(0, len(text), chunk_size):
        end = min(start + chunk_size, len(text))
        delta = parser.parse_delta(
            text[start:end], [], request(), finished=end == len(text)
        )
        if delta is not None:
            reasoning.append(delta.reasoning or "")
            calls.extend(delta.tool_calls)

    assert "".join(reasoning) == "plan"
    assert [
        (call.function.name, json.loads(call.function.arguments)) for call in calls
    ] == parse(text)[2]


WEATHER_CALL = call("get_weather", parameter("city", "Paris"))
WEATHER = [("get_weather", {"city": "Paris"})]


def chat_request(tools=STRICT_TOOLS, **body):
    return ChatCompletionRequest(
        model="beam",
        messages=[{"role": "user", "content": "Weather?"}],
        tools=tools and [tool.model_dump() for tool in tools],
        **body,
    )


def named(name):
    return {"type": "function", "function": {"name": name}}


def adjust(req, prompt_token_ids=PROMPT_TOKEN_IDS, **kwargs):
    make_parser(prompt_token_ids, req.tools, **kwargs).adjust_request(req)
    return req


def accepts(req, text):
    grammar = Grammar.from_structural_tag(
        req.extract_structured_outputs().structural_tag
    )
    return _is_grammar_accept_string(grammar, text)


@pytest.mark.parametrize(
    ("tool_choice", "text", "calls"),
    [
        ("required", THOUGHT + WEATHER_CALL, WEATHER),
        ("required", WEATHER_CALL, None),
        (
            "required",
            THOUGHT
            + call("get_weather", parameter("city", "Paris"), terminator=NEXT_MESSAGE)
            + call("set_count", parameter("count", "7")),
            [*WEATHER, ("set_count", {"count": 7})],
        ),
        ("required", THOUGHT + call("set_count", parameter("count", "x")), None),
        (
            "required",
            THOUGHT + call("get_weather", parameter("city", "a<|call|>b")),
            None,
        ),
        ("required", THOUGHT + FINAL + "Paris<|return|>", None),
        (named("set_count"), THOUGHT + WEATHER_CALL, None),
        (named("get_weather"), THOUGHT + WEATHER_CALL, WEATHER),
        (
            named("get_weather"),
            THOUGHT
            + call("get_weather", parameter("city", "Paris"), terminator=NEXT_MESSAGE)
            + WEATHER_CALL,
            None,
        ),
    ],
)
def test_forced_tool_calls_follow_the_template_grammar(tool_choice, text, calls):
    """From the message start, the grammar opens with the reasoning, and
    accepts only calls the parser reads back."""
    req = adjust(chat_request(tool_choice=tool_choice))

    assert req._grammar_from_parser
    assert accepts(req, text) is (calls is not None)
    if calls is not None:
        assert parse(text)[2] == calls


def test_reasoning_may_be_skipped_when_thinking_is_disabled():
    req = adjust(
        chat_request(tool_choice="required"),
        chat_template_kwargs={"enable_thinking": False},
    )

    assert accepts(req, WEATHER_CALL)
    assert accepts(req, THOUGHT + WEATHER_CALL)


def test_parallel_tool_calls_false_allows_a_single_call():
    req = adjust(chat_request(tool_choice="required", parallel_tool_calls=False))
    calls = call("get_weather", parameter("city", "Paris"), terminator=NEXT_MESSAGE)

    assert accepts(req, THOUGHT + WEATHER_CALL)
    assert not accepts(req, THOUGHT + calls + WEATHER_CALL)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    ("separator", "accepted"), [("", True), ("\n", True), ("\n\n", False), (" ", False)]
)
def test_parameters_are_separated_by_at_most_a_newline(strict, separator, accepted):
    """A model that skips a parameter's opener must not write whitespace until
    `max_tokens`."""
    write = _tool(
        "write",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        strict=strict,
    )
    req = adjust(chat_request([write], tool_choice="required"))
    body = parameter("path", "a.py") + separator + parameter("content", "x")

    assert accepts(req, THOUGHT + call("write", body)) is accepted
    assert accepts(req, THOUGHT + call("write", separator + body)) is accepted


def test_forced_calls_in_prefilled_reasoning_start_after_it():
    req = adjust(
        chat_request(tool_choice="required"), PROMPT_TOKEN_IDS + [201, 101, 202, 103]
    )

    assert not req._grammar_from_parser
    assert accepts(req, WEATHER_CALL)


def test_forced_calls_cannot_follow_a_prefilled_answer():
    with pytest.raises(VLLMValidationError, match="required tool choice"):
        adjust(chat_request(tool_choice="required"), PROMPT_TOKEN_IDS + [201, 102, 202])


def test_prefilled_answer_can_only_be_finished():
    req = adjust(chat_request(TOOLS), PROMPT_TOKEN_IDS + [201, 102, 202])

    assert accepts(req, "Sunny.<|return|>")
    assert not accepts(req, "Sunny." + NEXT_MESSAGE + WEATHER_CALL)


def test_forced_call_with_unencodable_tool_name_is_rejected():
    tools = [_tool("get weather", {"city": {"type": "string"}})]

    with pytest.raises(VLLMValidationError, match="tool name"):
        adjust(chat_request(tools, tool_choice="required"))


def test_string_formats_and_patterns_are_held_in_raw_text():
    """Like XGrammar, a strict string keeps its format or pattern, unless a
    match could end the value early."""
    contact = _tool(
        "set_contact",
        {
            "email": {"type": "string", "format": "email"},
            "code": {"type": "string", "pattern": "^[A-Z]{3}$"},
            "n": {"type": "integer"},
        },
        strict=True,
    )
    contact.function.parameters["required"] = ["email", "code", "n"]
    req = adjust(chat_request([contact], tool_choice="required"))

    def contact_call(email, code, *rest):
        body = parameter("email", email) + parameter("code", code) + "".join(rest)
        return THOUGHT + call("set_contact", body)

    text = contact_call("a.b@x.co", "ABC", parameter("n", "1"))
    calls = make_parser(tools=[contact]).parse(text, req, enable_auto_tools=True)[2]
    assert accepts(req, text)
    assert normalize_calls(calls) == [
        ("set_contact", {"email": "a.b@x.co", "code": "ABC", "n": 1})
    ]
    assert not accepts(req, contact_call("ab", "ABC", parameter("n", "1")))
    assert not accepts(req, contact_call("a.b@x.co", "ABCD", parameter("n", "1")))
    assert not accepts(req, contact_call("a.b@x.co", "ABC"))
    # Line breaks could end the value.
    dotted = _tool(
        "set_code", {"code": {"type": "string", "pattern": "^a.b$"}}, strict=True
    )
    with pytest.raises(VLLMValidationError, match="required tool choice"):
        adjust(chat_request([dotted], tool_choice="required"))


@pytest.mark.parametrize(
    ("text", "accepted"),
    [
        (THOUGHT + WEATHER_CALL, True),
        (THOUGHT + FINAL + "Sunny.<|return|>", True),
        (THOUGHT + FINAL + "Sunny." + NEXT_MESSAGE + WEATHER_CALL, False),
        (THOUGHT + call("set_count", parameter("count", "x")), False),
    ],
)
def test_strict_tools_constrain_automatic_tool_choice(text, accepted):
    assert accepts(adjust(chat_request()), text) is accepted


@pytest.mark.parametrize(
    ("text", "accepted"),
    [
        (THOUGHT + FINAL + "Sunny.<|return|>", True),
        (FINAL + "Sunny.<|return|>", False),
        (WEATHER_CALL, False),
        (THOUGHT + WEATHER_CALL, False),
        (FINAL + "Sunny." + NEXT_MESSAGE + WEATHER_CALL, False),
    ],
)
def test_tool_choice_none_allows_only_an_answer(text, accepted):
    req = adjust(chat_request(TOOLS, tool_choice="none"))

    assert req._grammar_from_parser
    assert accepts(req, text) is accepted


@pytest.mark.parametrize(("tools", "tool_choice"), [(TOOLS, "auto"), (None, None)])
def test_always_constrained_output_follows_the_template_framing(tools, tool_choice):
    req = adjust(chat_request(tools, tool_choice=tool_choice))
    count_call = THOUGHT + call("set_count", parameter("count", "3"))

    assert accepts(req, THOUGHT + FINAL + "Sunny.<|return|>")
    assert not accepts(req, FINAL + "Sunny.<|return|>")
    assert not accepts(req, THOUGHT.replace("plan", "") + FINAL + "Sunny.<|return|>")
    assert not accepts(req, THOUGHT + "Sunny.")
    assert accepts(req, count_call) is (tool_choice == "auto")


@pytest.mark.parametrize("response_format", [None, {"type": "json_object"}])
@pytest.mark.parametrize("strict", [None, True])
@pytest.mark.parametrize("tool_choice", ["auto", "required"])
def test_automatic_choice_lets_calls_name_undeclared_tools(
    strict, tool_choice, response_format
):
    """Unless a forced choice limits calls to the declared tools. The parser
    passes such calls on, so the harness can tell the model the tool does not
    exist."""
    bash = _tool("bash", {"command": {"type": "string"}}, strict=strict)
    req = adjust(
        chat_request([bash], tool_choice=tool_choice, response_format=response_format)
    )
    undeclared = THOUGHT + call("read_file", parameter("path", "a.py"))
    parser = make_parser(tools=[bash])

    assert accepts(req, THOUGHT + call("bash", parameter("command", "ls")))
    assert accepts(req, undeclared) is (tool_choice == "auto")
    calls = parser.parse(undeclared, req, enable_auto_tools=True)[2]
    assert normalize_calls(calls) == [("read_file", {"path": "a.py"})]


@pytest.mark.parametrize(
    ("name", "body", "accepted"),
    [
        ("set_count", parameter("count", "1"), True),
        ("set_count", parameter("count", "x"), False),
        ("get_weather", parameter("town", "Paris"), False),
        # The grammar cannot enforce `minProperties`.
        ("find", parameter("x", "1") + parameter("x", "y"), True),
        ("search_files", parameter("x", "1") + parameter("x", "y"), True),
        ("set_coun", parameter("x", "1") + parameter("x", "y"), True),
        ("set_counts", parameter("x", "1") + parameter("x", "y"), True),
    ],
)
def test_automatic_calls_enforce_declared_arguments_beside_any_other_name(
    name, body, accepted
):
    find = _tool("find", {"q": {"type": "string"}})
    find.function.parameters["minProperties"] = 1
    req = adjust(chat_request([*TOOLS, find]))

    assert accepts(req, THOUGHT + call(name, body)) is accepted


def test_value_ends_before_an_opener_on_the_next_line():
    write = _tool("write", {"path": {"type": "string"}, "content": {"type": "string"}})
    req = adjust(chat_request([write], tool_choice="required"))
    body = '<rfl:parameter name="content">\nprint(1)\n\n' + parameter("path", "a.py")
    text = THOUGHT + call("write", body)

    assert accepts(req, text)
    calls = make_parser(tools=[write]).parse(text, req, enable_auto_tools=True)[2]
    assert normalize_calls(calls) == [
        ("write", {"content": "print(1)\n", "path": "a.py"})
    ]


def test_always_constrained_output_can_be_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_BEAM_ALWAYS_CONSTRAIN_OUTPUT", "0")

    assert adjust(chat_request(TOOLS)).structured_outputs is None


def test_strict_tools_the_template_cannot_hold_leave_automatic_choice_unconstrained():
    dotted = _tool(
        "set_code", {"code": {"type": "string", "pattern": "^a.b$"}}, strict=True
    )

    assert adjust(chat_request([dotted])).structured_outputs is None


def test_generic_parser_does_not_derive_the_template_grammar():
    req = chat_request(TOOLS, tool_choice="required")
    parser = ResponseTemplateParser(BeamTokenizer(), req.tools, enable_auto_tools=True)

    with pytest.raises(VLLMValidationError, match="required tool choice"):
        parser.adjust_request(req)


@pytest.mark.parametrize("tools", [None, TOOLS])
def test_structured_output_is_framed_as_the_answer(tools):
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    req = adjust(
        chat_request(
            tools,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema},
            },
        )
    )

    assert req.response_format is None
    assert accepts(req, THOUGHT + FINAL + '{"a": 1}<|return|>')
    assert not accepts(req, THOUGHT + FINAL + "Sunny.<|return|>")
    assert accepts(req, THOUGHT + WEATHER_CALL) is (tools is not None)


def test_responses_named_tool_choice_follows_the_template_grammar():
    req = adjust(
        ResponsesRequest(
            input="Weather?",
            tools=[
                FunctionTool(
                    type="function",
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                    strict=True,
                )
            ],
            tool_choice=ToolChoiceFunction(type="function", name="get_weather"),
            text={"format": {"type": "json_object"}},
        )
    )

    assert req._grammar_from_parser
    assert req.text.format.type == "json_object"
    assert accepts(req, THOUGHT + WEATHER_CALL)
    assert not accepts(req, FINAL + "{}<|return|>")


@pytest.mark.parametrize("status", ["in_progress", "completed"])
@pytest.mark.parametrize("inside_reasoning", [False, True])
def test_responses_preserves_text_config_with_template_grammar(
    status, inside_reasoning
):
    text = {
        "format": {
            "type": "json_schema",
            "name": "answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "required": ["a"],
                "additionalProperties": False,
            },
        },
        "verbosity": "low",
    }
    req = ResponsesRequest(input="Answer", text=text)
    expected_text = req.text.model_dump(by_alias=True)
    prompt = PROMPT_TOKEN_IDS + ([201, 101, 202] if inside_reasoning else [])
    adjust(req, prompt_token_ids=prompt)
    params = req.to_sampling_params(default_max_tokens=128)
    grammar = Grammar.from_structural_tag(params.structured_outputs.structural_tag)
    prefix = "" if inside_reasoning else THOUGHT
    assert _is_grammar_accept_string(grammar, prefix + FINAL + '{"a": 1}<|return|>')
    assert not _is_grammar_accept_string(grammar, prefix + FINAL + "Sunny.<|return|>")
    response = ResponsesResponse.from_request(req, params, "beam", 0, [], status)
    assert response.model_dump(by_alias=True)["text"] == expected_text


@pytest.mark.parametrize("reverse_alternatives", [False, True])
def test_reasoning_budget_initializes_from_checkpoint_pattern(
    monkeypatch, reverse_alternatives
):
    template = deepcopy(BEAM_RESPONSE_TEMPLATE)
    alternatives = [
        r"^<\|channel\|>analysis<\|message\|>",
        r"<\|start\|>assistant<\|channel\|>analysis<\|message\|>",
    ]
    if reverse_alternatives:
        alternatives.reverse()
    template["fields"]["thinking"] = {
        "open_pattern": "(?:" + "|".join(alternatives) + ")",
        "close": "<|end|>",
        "content": "text",
    }
    tokenizer = BeamTokenizer(template)
    monkeypatch.setattr(
        "vllm.config.reasoning.cached_tokenizer_from_config", lambda **_: tokenizer
    )
    config = ReasoningConfig(reasoning_parser="beam")
    config.initialize_token_ids(SimpleNamespace())

    assert config.enabled
    # The marker must work both after the prompt's assistant header and in
    # subsequent complete assistant messages, regardless of regex branch order.
    assert config.reasoning_start_token_ids == tokenizer.encode(
        "<|channel|>analysis<|message|>"
    )
    assert config.reasoning_end_token_ids == tokenizer.encode("<|end|>")
    assert config.natural_reasoning_end_token_ids == tokenizer.encode("<|end|>")


@pytest.mark.parametrize("pattern", [r"analysis.*", r"(?:analysis|thinking)"])
def test_reasoning_budget_does_not_guess_a_marker_for_other_patterns(pattern):
    template = deepcopy(BEAM_RESPONSE_TEMPLATE)
    thinking = template["fields"]["thinking"]
    del thinking["open"]
    thinking["open_pattern"] = pattern

    assert BeamReasoningParser(BeamTokenizer(template)).reasoning_start_str is None
