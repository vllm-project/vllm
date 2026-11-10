# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Round-trip tests for the FunctionGemma chat template.

`FunctionGemmaToolParser._parse_arguments` documents the wire format the
chat template is expected to produce: strings as `<escape>text<escape>` and
everything else bare (`true`/`false`, numbers, `[item,...]` and
`{key:value,...}` with bare keys). A tool call replayed from conversation
history must therefore come back out of the parser with the types and
structure the client sent.
"""

import json
from os.path import commonprefix
from pathlib import Path

import jinja2.sandbox
import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.renderers.chat_utils.chat_message import _postprocess_messages
from vllm.tool_parsers import ToolParserManager

TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "examples"
    / "tool_chat_template_functiongemma.jinja"
)

# One value per JSON type, plus a nested container: the shapes that tell a
# JSON spelling apart from a Python repr.
ARGUMENTS = {
    "city": "Paris",
    "metric": True,
    "off": False,
    "limit": 3,
    "unit": None,
    "addr": {"city": "SF", "zip": 94103},
    "tags": ["a", "b"],
    "nested": {"opts": [True, None]},
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "metric": {"type": "boolean"},
                    "off": {"type": "boolean"},
                    "limit": {"type": "integer"},
                    "unit": {"type": ["string", "null"]},
                    "addr": {"type": "object"},
                    "tags": {"type": "array"},
                    "nested": {"type": "object"},
                },
            },
        },
    }
]


@pytest.fixture(scope="module")
def functiongemma_template():
    """Compile the template in the environment chat templates render in."""
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True
    )
    env.filters["tojson"] = lambda obj, **kwargs: json.dumps(
        obj, ensure_ascii=False, **kwargs
    )
    return env.from_string(TEMPLATE_PATH.read_text())


@pytest.fixture(scope="module")
def assistant_turn(functiongemma_template):
    """What the model itself would have emitted for a tool call.

    Diffing a render of the full conversation against a render that stops at
    the generation prompt cancels the `<start_of_turn>model` header, leaving
    exactly the text a tool parser receives.
    """
    head: list[dict] = [
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "What is the weather in Paris?"},
    ]
    # vLLM hands the template an OpenAI tool call whose arguments have already
    # been turned from a JSON string into a dict.
    messages = head + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps(ARGUMENTS),
                    },
                }
            ],
        }
    ]
    _postprocess_messages(messages)

    full = functiongemma_template.render(
        messages=messages, tools=TOOLS, add_generation_prompt=False
    )
    prompt = functiongemma_template.render(
        messages=head, tools=TOOLS, add_generation_prompt=True
    )
    shared = len(commonprefix([full, prompt]))
    return full[shared:].replace("<end_of_turn>", "").strip()


def test_argument_types_survive_a_round_trip(assistant_turn, default_tokenizer):
    """Replaying a call through the template preserves every argument."""
    request = ChatCompletionRequest(model="functiongemma", messages=[], tools=TOOLS)
    parser = ToolParserManager.get_tool_parser("functiongemma")(
        default_tokenizer, tools=request.tools
    )

    extracted = parser.extract_tool_calls(assistant_turn, request)

    assert extracted.tools_called
    assert extracted.tool_calls[0].function.name == "get_weather"
    assert json.loads(extracted.tool_calls[0].function.arguments) == ARGUMENTS


def test_rendered_values_match_the_parser_wire_format(assistant_turn):
    """Strings are escape-delimited; everything else is bare."""
    assert "city:<escape>Paris<escape>" in assistant_turn
    assert "metric:true" in assistant_turn
    assert "off:false" in assistant_turn
    assert "limit:3" in assistant_turn
    assert "unit:null" in assistant_turn
    # Containers keep their structure, with bare keys.
    assert "addr:{city:<escape>SF<escape>,zip:94103}" in assistant_turn
    assert "tags:[<escape>a<escape>,<escape>b<escape>]" in assistant_turn
    assert "nested:{opts:[true,null]}" in assistant_turn
    # No Python spelling or repr reaches the wire.
    for spelling in ("True", "False", "None", "'"):
        assert spelling not in assistant_turn
