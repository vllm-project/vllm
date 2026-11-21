# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Round-trip test for the FunctionGemma chat template.

`FunctionGemmaToolParser._parse_arguments` documents the wire format the chat
template is expected to produce: strings as `<escape>text<escape>` and
everything else bare (`true`/`false`, numbers, `[item,...]` and
`{key:value,...}` with bare keys). A tool call replayed from conversation
history must come back out of the parser with the types the client sent.
"""

import json
from os.path import commonprefix
from pathlib import Path

import jinja2.sandbox

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.renderers.chat_utils.chat_message import _postprocess_messages
from vllm.tool_parsers import ToolParserManager

TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "examples"
    / "tool_chat_template_functiongemma.jinja"
)

# One value per JSON type, plus nested containers: the shapes that tell a JSON
# spelling apart from a Python repr.
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

# Declared types matter: the parser coerces against this schema, so a round
# trip that holds here holds with coercion active.
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


def test_argument_types_survive_a_round_trip(default_tokenizer):
    """Replaying a tool call through the template preserves every argument."""
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True
    )
    env.filters["tojson"] = lambda obj, **kwargs: json.dumps(
        obj, ensure_ascii=False, **kwargs
    )
    template = env.from_string(TEMPLATE_PATH.read_text())

    head = [{"role": "user", "content": "What is the weather in Paris?"}]
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

    # Diffing the full render against one that stops at the generation prompt
    # cancels the `<start_of_turn>model` header, leaving the text a parser sees.
    full = template.render(messages=messages, tools=TOOLS, add_generation_prompt=False)
    prompt = template.render(messages=head, tools=TOOLS, add_generation_prompt=True)
    model_output = full[len(commonprefix([full, prompt])) :].replace(
        "<end_of_turn>", ""
    )

    request = ChatCompletionRequest(model="functiongemma", messages=[], tools=TOOLS)
    parser = ToolParserManager.get_tool_parser("functiongemma")(
        default_tokenizer, tools=request.tools
    )
    extracted = parser.extract_tool_calls(model_output.strip(), request)

    assert extracted.tools_called
    assert extracted.tool_calls[0].function.name == "get_weather"
    assert json.loads(extracted.tool_calls[0].function.arguments) == ARGUMENTS
