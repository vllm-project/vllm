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

import jinja2.ext
import jinja2.sandbox
import pytest
import regex as re

from tests.tool_parsers.utils import run_tool_extraction
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.renderers.chat_utils.chat_message import _postprocess_messages
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

# Scaffolding that trails the assistant turn: a parser sees the model's output,
# so the markers closing the turn, and any header the template opens for the
# next one, have to come off. Stripped as one repeated run, because a template
# can emit several in sequence (toolace writes an entire assistant header).
TRAILING_SCAFFOLDING = re.compile(r"(?:<\|[^|]*\|>|<end_of_turn>|</s>|assistant|\s)+$")

# (template, parser) pairs that ship together for pythonic tool calling.
PYTHONIC_PAIRS = [
    ("tool_chat_template_gemma3_pythonic.jinja", "pythonic"),
    pytest.param(
        "tool_chat_template_llama3.2_pythonic.jinja",
        "pythonic",
        marks=pytest.mark.xfail(
            reason="https://github.com/vllm-project/vllm/issues/66300 renders "
            "string arguments unquoted (city=Paris), which is not a Python "
            "literal, so the parser rejects the whole call",
            strict=True,
        ),
    ),
    pytest.param(
        "tool_chat_template_toolace.jinja",
        "pythonic",
        marks=pytest.mark.xfail(
            reason="https://github.com/vllm-project/vllm/issues/66300 renders "
            "string arguments unquoted (city=Paris), which is not a Python "
            "literal, so the parser rejects the whole call",
            strict=True,
        ),
    ),
    ("tool_chat_template_llama4_pythonic.jinja", "llama4_pythonic"),
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


def _conversation(arguments: dict) -> list[dict]:
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
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        },
    ]
    _postprocess_messages(messages)
    return messages


def _model_output(template_name: str, arguments: dict | None = None) -> str:
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
    messages = _conversation(ARGUMENTS if arguments is None else arguments)
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
    output = TRAILING_SCAFFOLDING.sub("", with_call[index:].strip())

    assert output, f"{template_name} rendered no assistant turn to parse"
    assert "<|" not in output and "<end_of_turn>" not in output, (
        f"scaffolding survived extraction from {template_name}: {output!r}. A "
        f"parser never sees this, so leaving it in would fail the round trip "
        f"for a reason that has nothing to do with the template's format."
    )
    return output


@pytest.mark.parametrize("template_name, parser_name", PYTHONIC_PAIRS)
def test_rendered_tool_call_is_readable_by_its_parser(
    template_name: str, parser_name: str, default_tokenizer: TokenizerLike
):
    """A tool call rendered into history parses back to the same call."""
    model_output = _model_output(template_name)
    parser = ToolParserManager.get_tool_parser(parser_name)(default_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], tools=TOOLS)

    content, tool_calls = run_tool_extraction(parser, model_output, request)

    assert tool_calls, (
        f"{template_name} rendered a tool call that the {parser_name} parser "
        f"does not recognise, so it was returned as content: {content!r}"
    )
    assert tool_calls[0].function.name == "get_weather"
    assert json.loads(tool_calls[0].function.arguments) == ARGUMENTS


def test_string_argument_containing_a_quote_survives_rendering(
    default_tokenizer: TokenizerLike,
):
    """A quote inside a string argument must not break the call it sits in.

    A template that wraps values in literal quote characters without escaping
    them emits ``note="say "hi""``, which ``ast.parse`` rejects, so the whole
    call is dropped rather than just that argument.
    """
    arguments = {"city": "Paris", "note": 'say "hi"'}
    model_output = _model_output("tool_chat_template_llama4_pythonic.jinja", arguments)
    parser = ToolParserManager.get_tool_parser("llama4_pythonic")(default_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], tools=TOOLS)

    content, tool_calls = run_tool_extraction(parser, model_output, request)

    assert tool_calls, (
        f"a quote inside a string argument broke the rendered call, which was "
        f"returned as content instead: {content!r}"
    )
    assert json.loads(tool_calls[0].function.arguments) == arguments
