# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regenerate `differential.json` from Transformers `parse_response`.

Requires Transformers with `transformers.utils.chat_parsing` (>= 5.13); the
expected values in the checked-in file were produced at commit 6d43ab4008.

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
        'Before <|tool_call>call:f{x:<|"|>a, b<|"|>,y:[1,2.5,-3],z:{w:true}}'
        "<tool_call|> after",
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
        'noise <invoke name="get" id="1"><param name="city">"Paris"</param>'
        '<param name="n">3</param><param name="raw">not json</param></invoke>',
        None,
    ),
    (
        "invoke",
        INVOKE,
        "",
        '<invoke name="get"><param name="k">1</param>'
        '<param name="k">2</param></invoke>',
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
        "<tool_call>\n<function=set_alarm>\nhour: 7\nratio: 0.5\nenabled: 1\n"
        "label: 7\nmaybe: None\n</tool_call>",
        ALARM_TOOLS,
    ),
    (
        "kv_tools",
        KV_TOOLS,
        "",
        "<tool_call>\n<function=set_alarm>\nhour: seven\nratio: 1e3\n"
        "enabled: nope\n</tool_call>",
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
