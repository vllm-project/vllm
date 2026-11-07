# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A tool call that a bundled chat template writes into conversation history
must be readable by the tool parser that template ships to be used with.

Every multi-turn tool conversation replays past assistant tool calls through
the chat template. If the template renders them in a shape the paired parser
cannot read, the format vLLM writes into the prompt and the format vLLM
accepts from the model have drifted apart, and nothing reports it.

The conversation is built the way the serving path builds it: tool call
arguments arrive as an OpenAI JSON string and ``_postprocess_messages`` turns
them into a dict before templating. The canonical call carries a string, a
bool, an int and a null, because a template whose arguments are all strings
hides every type error.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import jinja2.ext
import jinja2.sandbox
import pytest

from tests.tool_parsers.utils import run_tool_extraction
from vllm.entrypoints.chat_utils import _postprocess_messages
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParserManager

EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "examples"

ARGUMENTS = {"city": "Paris", "metric": True, "limit": 3, "unit": None}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "metric": {"type": "boolean"},
                    "limit": {"type": "integer"},
                    "unit": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    }
]

# Turn-end markers to drop: a parser sees the model's output, and the marker
# that closes the turn is the only template scaffolding inside it.
TURN_END_MARKERS = (
    "<end_of_turn>",
    "<|eot_id|>",
    "<|im_end|>",
    "<|eot|>",
    "<|end|>",
    "</s>",
)

# (template, parser) pairs that ship together for pythonic tool calling.
PYTHONIC_PAIRS = [
    ("tool_chat_template_gemma3_pythonic.jinja", "pythonic"),
    pytest.param(
        "tool_chat_template_llama3.2_pythonic.jinja",
        "pythonic",
        marks=pytest.mark.xfail(
            reason="renders string arguments unquoted (city=Paris), which is "
            "not a Python literal, so the parser rejects the whole call",
            strict=True,
        ),
    ),
    pytest.param(
        "tool_chat_template_toolace.jinja",
        "pythonic",
        marks=pytest.mark.xfail(
            reason="renders string arguments unquoted (city=Paris), which is "
            "not a Python literal, so the parser rejects the whole call",
            strict=True,
        ),
    ),
    pytest.param(
        "tool_chat_template_llama4_pythonic.jinja",
        "llama4_pythonic",
        marks=pytest.mark.xfail(
            reason="renders every argument as a string, so bools and ints "
            "round-trip as 'True' and '3' with no error raised",
            strict=True,
        ),
    ),
]


def _jinja_env() -> jinja2.sandbox.ImmutableSandboxedEnvironment:
    """A Jinja environment matching the one chat templates are rendered in.

    Mirrors ``transformers`` so a template behaves here as it does in serving:
    ``tojson`` must not HTML-escape, and ``raise_exception`` must be callable.
    """
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[jinja2.ext.loopcontrols],
    )

    def raise_exception(message: str):
        raise jinja2.exceptions.TemplateError(message)

    env.filters["tojson"] = lambda obj, **kwargs: json.dumps(
        obj, ensure_ascii=False, **kwargs
    )
    env.globals["raise_exception"] = raise_exception
    return env


def _conversation() -> list[dict]:
    messages: list[dict] = [
        {"role": "user", "content": "What is the weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps(ARGUMENTS),
                    },
                }
            ],
        },
    ]
    _postprocess_messages(messages)
    return messages


def _model_output(template_name: str) -> str:
    """Render the assistant tool call the way the model would have emitted it.

    Rendering the full conversation and the same conversation with a generation
    prompt differ only in the assistant turn's content: the role header is on
    both sides and cancels out. What remains is what a parser receives, since a
    parser never sees the prompt.
    """
    template = _jinja_env().from_string((EXAMPLES_DIR / template_name).read_text())
    specials = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }
    messages = _conversation()
    with_call = template.render(
        messages=messages,
        tools=TOOLS,
        add_generation_prompt=False,
        **specials,
    )
    with_prompt = template.render(
        messages=messages[:-1],
        tools=TOOLS,
        add_generation_prompt=True,
        **specials,
    )

    index = 0
    limit = min(len(with_call), len(with_prompt))
    while index < limit and with_call[index] == with_prompt[index]:
        index += 1
    output = with_call[index:].strip()

    for marker in TURN_END_MARKERS:
        if output.endswith(marker):
            return output[: -len(marker)].strip()
    return output


@pytest.mark.parametrize("template_name, parser_name", PYTHONIC_PAIRS)
def test_rendered_tool_call_is_readable_by_its_parser(
    template_name: str, parser_name: str
):
    """A tool call rendered into history parses back to the same call."""
    model_output = _model_output(template_name)
    parser = ToolParserManager.get_tool_parser(parser_name)(
        MagicMock(spec=TokenizerLike)
    )
    request = ChatCompletionRequest(model="test-model", messages=[], tools=TOOLS)

    content, tool_calls = run_tool_extraction(parser, model_output, request)

    assert tool_calls, (
        f"{template_name} rendered a tool call that the {parser_name} parser "
        f"does not recognise, so it was returned as content: {content!r}"
    )
    assert tool_calls[0].function.name == "get_weather"
    assert json.loads(tool_calls[0].function.arguments) == ARGUMENTS


def test_arguments_cover_every_json_scalar_kind():
    """Guard the fixture: all-string arguments would hide type corruption."""
    kinds = {type(value) for value in ARGUMENTS.values()}
    assert kinds == {str, bool, int, type(None)}
