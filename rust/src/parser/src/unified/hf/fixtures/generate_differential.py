# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regenerate `differential.json` from Transformers `parse_response`.

Requires Transformers with `transformers.utils.chat_parsing` (>= 5.13); the
expected values in the checked-in file were produced at commit c6329dd080.

    python generate_differential.py > differential.json
"""

import json

from transformers.utils.chat_parsing import parse_response

GEMMA4 = {
    "defaults": {"role": "assistant"},
    "start_anchor": ["<|turn>model\n", "<tool_response|>"],
    "fields": {
        "thinking": {
            "open": "<|channel>thought\n",
            "close": "<channel|>",
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": r"<\|tool_call>call:(?P<name>\w+)",
            "close": "<tool_call|>",
            "repeats": True,
            "content": "json",
            "content_args": {
                "unquoted_keys": True,
                "string_delims": [['<|"|>', '<|"|>']],
            },
            "transform": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
        "content": {
            "close": ["<turn|>", "<|tool_response>", "<eos>"],
            "content": "text",
        },
    },
}

INVOKE = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|begin|>assistant",
    "fields": {
        "reasoning_content": {
            "open_pattern": r"to=self<\|msg\|>",
            "close": "<|pause|>",
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": r'<invoke\b[^>]*?\bname="(?P<name>[^"]+)">',
            "close": "</invoke>",
            "repeats": True,
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": (
                    r'<param\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>'
                    r"(?P<value>.*?)</param>"
                ),
                "value_parser": {"name": "json", "args": {"allow_non_json": True}},
            },
            "transform": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
        "content": {
            "open_pattern": r"to=(?:user|note)<\|msg\|>",
            "close": ["<|end|>", "<|pause|>"],
            "repeats": True,
            "join": "",
            "content": "text",
        },
    },
}

KV_TOOLS = {
    "start_anchor": "<|im_start|>assistant\n",
    "fields": {
        "thinking": {"open": "<think>", "close": "</think>"},
        "tool_calls": {
            "open_pattern": r"<tool_call>\s*<function=(?P<name>\w+)>\n",
            "close": "</tool_call>",
            "repeats": True,
            "content": "kv-lines",
            "transform": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
        "content": {"close": "<|im_end|>"},
    },
}

# Lookaheads in boundaries and tag patterns: Qwen3.5, Mistral (reasoning
# models), and GLM-5 response templates.
QWEN3_XML = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|im_start|>assistant\n",
    "fields": {
        "reasoning_content": {
            "open": "<think>",
            "close_pattern": r"</think>|(?=<tool_call>)",
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": r"<tool_call>\s*<function=(?P<name>[^>\n]+)>",
            "close_pattern": r"</tool_call>|(?=<tool_call>)",
            "repeats": True,
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": (
                    r"<parameter=(?P<key>[^>\n]+)>\n?(?P<value>.*?)\n?"
                    r"(?:</parameter>|(?=<parameter=)|(?=</function>))"
                )
            },
            "transform": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
        "content": {
            "close": ["<|im_end|>", "<|endoftext|>"],
            "content": "text",
            "repeats": True,
            "join": "",
        },
    },
}

MISTRAL = {
    "defaults": {"role": "assistant"},
    "start_anchor": ["[/INST]", "[/TOOL_RESULTS]"],
    "fields": {
        "reasoning_content": {
            "open": "[THINK]",
            "close": "[/THINK]",
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": r"\[TOOL_CALLS\](?P<name>.+?)\[ARGS\]",
            "close_pattern": r"(?=\[TOOL_CALLS\])|</s>",
            "repeats": True,
            "content": "json",
            "transform": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
        "content": {
            "close": ["[TOOL_CALLS]", "</s>"],
            "content": "text",
            "repeats": True,
            "join": "",
        },
    },
}

GLM = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|assistant|>",
    "fields": {
        "reasoning_content": {
            "open_pattern": r"<think>|(?=</think>)",
            "close_pattern": r"</think>|(?=<tool_call>)",
            "content": "text",
            "content_args": {"strip": True},
            "repeats": True,
            "join": "",
        },
        "tool_calls": {
            "open_pattern": (
                r"\s*<tool_call>\s*(?P<name>[^<]*?)\s*(?=<arg_key>|</tool_call>)"
            ),
            "close": "</tool_call>",
            "repeats": True,
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": (
                    r"<arg_key>\s*(?P<key>.*?)\s*</arg_key>\s*"
                    r"<arg_value>(?P<value>.*?)</arg_value>"
                )
            },
            "transform": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
        "content": {
            "close": ["<|user|>", "<|observation|>", "<|endoftext|>"],
            "content": "text",
            "content_args": {"strip": False},
            "repeats": True,
            "join": "",
        },
    },
}

# Zero-width boundaries after each other: reasoning ends where the answer
# starts, and the answer opens there too.
CHAIN = {
    "start_anchor": "<|assistant|>",
    "fields": {
        "reasoning_content": {
            "open": "<think>",
            "close_pattern": r"</think>|(?=<answer>)",
        },
        "content": {"open_pattern": r"(?=<answer>)", "close": "</answer>"},
    },
}

# Patterns without a literal prefix, from Transformers' tests: Ernie opens
# reasoning at the start of the response, Inkling's openers skip an optional
# header and any author name.
ERNIE = {
    "defaults": {"role": "assistant"},
    "start_anchor": "Assistant:",
    "fields": {
        "reasoning_content": {
            "open_pattern": r"(?:^|<think>\s*)",
            "close": "</think>",
            "content": "text",
        },
        "content": {
            "open": "<response>\n",
            "close_pattern": r"\n?</response>",
            "content": "text",
        },
        "tool_calls": {
            "open": "<tool_call>",
            "close": "</tool_call>",
            "repeats": True,
            "content": "json",
            "transform": {"type": "function", "function": "{content}"},
        },
    },
}

INKLING = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|message_model|>",
    "fields": {
        "reasoning_content": {
            "open_pattern": r"(?:<\|message_model\|>)?[^<]*<\|content_thinking\|>",
            "close": "<|end_message|>",
            "repeats": True,
            "join": "",
            "content_args": {"strip": False},
        },
        "content": {
            "open_pattern": r"(?:<\|message_model\|>)?[^<]*<\|content_text\|>",
            "close": "<|end_message|>",
            "repeats": True,
            "join": "",
            "content_args": {"strip": False},
        },
        "tool_calls": {
            "open_pattern": (
                r"(?:<\|message_model\|>)?[^<]*<\|content_invoke_tool_json\|>"
            ),
            "close": "<|end_message|>",
            "repeats": True,
            "content": "json",
            "transform": {
                "type": "function",
                "function": {"name": "{content.name}", "arguments": "{content.args}"},
            },
        },
    },
}

INKLING_PREFIX = (
    "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
    "<|message_model|>"
)

ALARM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_alarm",
            "parameters": {
                "type": "object",
                "properties": {
                    "hour": {"type": "integer"},
                    "ratio": {"type": "number"},
                    "enabled": {"type": "boolean"},
                    "label": {"type": "string"},
                    "maybe": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                },
            },
        },
    }
]

CASES = [
    ("gemma4", GEMMA4, "", "Hello there.", None),
    ("gemma4", GEMMA4, "", "  \n  Hello\n\n", None),
    (
        "gemma4",
        GEMMA4,
        "",
        "<|channel>thought\n  plan\n<channel|>\n\nAnswer: 天气很好 ",
        None,
    ),
    ("gemma4", GEMMA4, "", "<|channel>thought\nunterminated reasoning", None),
    (
        "gemma4",
        GEMMA4,
        "",
        (
            'Before <|tool_call>call:f{x:<|"|>a, b<|"|>,y:[1,2.5,-3],z:{w:true}}'
            "<tool_call|> after"
        ),
        None,
    ),
    (
        "gemma4",
        GEMMA4,
        "",
        "<|tool_call>call:f{}<tool_call|><|tool_call>call:g{n:1}<tool_call|>",
        None,
    ),
    ("gemma4", GEMMA4, "", '<|tool_call>call:f{a:<|"|>unterminated', None),
    ("gemma4", GEMMA4, "", "text <|tool_call> not a call", None),
    ("gemma4", GEMMA4, "", "<|tool_call>call:f{a:1}", None),
    ("gemma4", GEMMA4, "<|turn>user\nhi<turn|>\n<|turn>model\n", "Hi!", None),
    (
        "gemma4",
        GEMMA4,
        "<|turn>model\n<|tool_call>call:f{}<tool_call|><|tool_response>response:f{}<tool_response|><|channel>thought\n",
        "Done.<channel|>It is sunny.",
        None,
    ),
    (
        "gemma4",
        GEMMA4,
        "<|turn>model\n<|channel>thought\n<channel|>",
        "Plain answer",
        None,
    ),
    (
        "invoke",
        INVOKE,
        "",
        "to=self<|msg|>think<|pause|>to=user<|msg|>A<|pause|>to=note<|msg|> B <|end|>",
        None,
    ),
    (
        "invoke",
        INVOKE,
        "",
        (
            'noise <invoke name="get" id="1"><param name="city">"Paris"</param>'
            '<param name="n">3</param><param name="raw">not json</param></invoke>'
        ),
        None,
    ),
    (
        "invoke",
        INVOKE,
        "",
        (
            '<invoke name="get"><param name="k">1</param>'
            '<param name="k">2</param></invoke>'
        ),
        None,
    ),
    ("invoke", INVOKE, "", "bare text only", None),
    ("invoke", INVOKE, "", "to=user<|msg|>unterminated content", None),
    (
        "invoke",
        INVOKE,
        "<|begin|>user<|end|><|begin|>assistant",
        "to=user<|msg|>hi<|end|>",
        None,
    ),
    (
        "kv_tools",
        KV_TOOLS,
        "",
        (
            "<tool_call>\n<function=set_alarm>\nhour: 7\nratio: 0.5\nenabled: 1\n"
            "label: 7\nmaybe: None\n</tool_call>"
        ),
        ALARM_TOOLS,
    ),
    (
        "kv_tools",
        KV_TOOLS,
        "",
        (
            "<tool_call>\n<function=set_alarm>\nhour: seven\nratio: 1e3\n"
            "enabled: nope\n</tool_call>"
        ),
        ALARM_TOOLS,
    ),
    (
        "kv_tools",
        KV_TOOLS,
        "<|im_start|>assistant\n<think>\n",
        "\n reasoning \n</think>\n\nanswer<|im_end|>",
        None,
    ),
]


QWEN3_THINKING = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
GLM_THINKING = "[gMASK]<sop><|user|>hi<|assistant|><think>"

CASES += [
    (
        "qwen3_xml",
        QWEN3_XML,
        QWEN3_THINKING,
        (
            "Check the weather.\n</think>\n\n<tool_call>\n<function=set_alarm>\n"
            "<parameter=hour>\n7\n</parameter>\n<parameter=label>\nwake up\n"
            "</parameter>\n</function>\n</tool_call>"
        ),
        ALARM_TOOLS,
    ),
    # Reasoning ends where a tool call starts.
    (
        "qwen3_xml",
        QWEN3_XML,
        QWEN3_THINKING,
        (
            "Calling it directly.\n<tool_call>\n<function=get_time>\n</function>\n"
            "</tool_call>"
        ),
        None,
    ),
    # A call ends where the next starts; parameters end where the next
    # parameter or `</function>` starts.
    (
        "qwen3_xml",
        QWEN3_XML,
        QWEN3_THINKING,
        (
            "\n</think>\n\nTwo calls.\n<tool_call>\n<function=a>\n<parameter=x>\n1\n"
            "<parameter=y>\ntwo\n</function>\n<tool_call>\n<function=b>\n"
            "<parameter=z>\nlast\n</function>\n</tool_call>"
        ),
        None,
    ),
    (
        "qwen3_xml",
        QWEN3_XML,
        "<|im_start|>assistant\n",
        "<think>\nshort\n</think>\n\nHello <tool_call world<tool_call>x<|im_end|>",
        None,
    ),
    (
        "mistral",
        MISTRAL,
        "[INST]hi[/INST]",
        (
            '[THINK]Two lookups.[/THINK]Checking.[TOOL_CALLS]get_weather[ARGS]{"city": '
            '"Paris"}[TOOL_CALLS]get_time[ARGS]{}</s>'
        ),
        None,
    ),
    (
        "mistral",
        MISTRAL,
        "[INST]hi[/INST]",
        '[TOOL_CALLS]get_weather[ARGS]{"city": "[TOOL"}',
        None,
    ),
    ("mistral", MISTRAL, "[INST]hi[/INST]", "Just text [TOOL.</s>", None),
    (
        "glm",
        GLM,
        GLM_THINKING,
        (
            "Look it up.</think>Sure.\n<tool_call>set_alarm\n<arg_key>hour</arg_key>\n"
            "<arg_value>7</arg_value>\n</tool_call>"
        ),
        ALARM_TOOLS,
    ),
    # Reasoning ends where a tool call starts; a call without arguments.
    ("glm", GLM, GLM_THINKING, "Calling now.<tool_call>get_time</tool_call>", None),
    # The opener consumes the whitespace before `<tool_call>`.
    (
        "glm",
        GLM,
        GLM_THINKING,
        (
            "</think>Two calls. \n\n<tool_call>a\n<arg_key>x</arg_key>\n"
            "<arg_value>1</arg_value>\n</tool_call>\n<tool_call> b <arg_key>y</arg_key>"
            "<arg_value>two words</arg_value></tool_call>"
        ),
        None,
    ),
    # Without `<think>` in the prompt, `</think>` opens and closes reasoning.
    ("glm", GLM, "<|assistant|>", "Thought text</think>Answer", None),
    (
        "glm",
        GLM,
        GLM_THINKING,
        "</think>Line one\n\n  indented <tool_call\n  ",
        None,
    ),
    ("chain", CHAIN, "", "<think>Thought<answer>Answer</answer>", None),
    ("chain", CHAIN, "", "<think>Thought</think>Skipped<answer>Answer</answer>", None),
    (
        "ernie",
        ERNIE,
        "",
        (
            "The user greets me.\n</think>\n\n"
            "<response>\nHello! How can I help?\n</response>\n</s>"
        ),
        None,
    ),
    (
        "ernie",
        ERNIE,
        "",
        (
            "Check the tool.\n</think>\n\n<tool_call>\n"
            '{"name": "get_current_temperature", "arguments": {"location": "Paris"}}'
            "\n</tool_call>\n</s>"
        ),
        None,
    ),
    (
        "ernie",
        ERNIE,
        "User: hi\nAssistant:",
        "<think> Short.</think><response>\nA < B and </resp\n</response>",
        None,
    ),
    (
        "inkling",
        INKLING,
        INKLING_PREFIX,
        (
            "<|content_thinking|>Consider the weather.<|end_message|>"
            "<|message_model|><|content_thinking|> Tokyo, probably.<|end_message|>"
            "<|message_model|><|content_text|>Checking the weather now.<|end_message|>"
            "<|message_model|>get_weather<|content_invoke_tool_json|>"
            '{"name":"get_weather","args":{"city":"Tokyo","units":"C"}}<|end_message|>'
            "<|content_model_end_sampling|>"
        ),
        None,
    ),
    (
        "inkling",
        INKLING,
        INKLING_PREFIX,
        (
            "<|content_text|>a < b, <i>c</i><|end_message|>"
            "<|message_model|>author <|content_text|> more<|end_message|>"
        ),
        None,
    ),
]


def normalize(message):
    out = {}
    reasoning = message.get("thinking", message.get("reasoning_content"))
    if reasoning:
        out["reasoning"] = reasoning
    if message.get("content"):
        out["content"] = message["content"]
    calls = [
        {"name": call["function"]["name"], "arguments": call["function"]["arguments"]}
        for call in message.get("tool_calls", [])
    ]
    if calls:
        out["tool_calls"] = calls
    return out


def main():
    templates = {}
    cases = []
    for template_name, template, prefix, text, tools in CASES:
        templates[template_name] = template
        try:
            expected = normalize(
                parse_response(text, template, prefix=prefix, tools=tools)
            )
        except Exception as error:  # noqa: BLE001
            expected = {"error": type(error).__name__}
        cases.append(
            {
                "template": template_name,
                "prefix": prefix,
                "text": text,
                "tools": tools,
                "expected": expected,
            }
        )
    print(
        json.dumps(
            {"templates": templates, "cases": cases}, indent=2, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
