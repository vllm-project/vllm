# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.tokenizers.registry import TokenizerRegistry, get_tokenizer


def test_rwkv_tokenizer_matches_world_vocab_golden_ids():
    tokenizer = get_tokenizer("BlinkDL/rwkv7-g1", tokenizer_mode="rwkv")

    assert tokenizer.encode("Hello world") == [33155, 40213]
    assert tokenizer.encode("你好") == [10464, 11685]
    assert tokenizer.encode(" 42") == [3515]
    assert tokenizer.decode([33155, 40213]) == "Hello world"
    assert tokenizer.decode([10464, 11685]) == "你好"


def test_rwkv_tokenizer_decode_replaces_invalid_utf8_tokens():
    tokenizer = get_tokenizer("BlinkDL/rwkv7-g1", tokenizer_mode="rwkv")

    assert tokenizer.decode([129]) == "\ufffd"
    assert tokenizer.decode([196]) == "\ufffd"
    assert tokenizer.decode([256]) == "\ufffd"
    assert tokenizer.decode([129, 196, 256]) == "\ufffd\ufffd\ufffd"


def test_rwkv_tokenizer_exposes_cached_metadata():
    tokenizer_cls = TokenizerRegistry.load_tokenizer_cls("rwkv")
    tokenizer = tokenizer_cls.from_pretrained("BlinkDL/rwkv7-g1")

    assert tokenizer.name_or_path == "BlinkDL/rwkv7-g1"
    cached_max_chars = tokenizer.max_chars_per_token
    tokenizer.idx2token.append(b"x" * (cached_max_chars + 1))
    assert tokenizer.max_chars_per_token == cached_max_chars


def test_rwkv_chat_template_renders_basic_dialogue_from_training_template():
    tokenizer = get_tokenizer("BlinkDL/rwkv7-g1", tokenizer_mode="rwkv")

    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "\r\n  You are concise.\r\n\r\nNo fluff.  "},
            {"role": "user", "content": "  Hello\r\n\r\nworld  "},
            {"role": "assistant", "content": "  Hi\r\n\r\nthere  "},
            {"role": "user", "content": "  Continue\r\nplease  "},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert rendered == (
        "System: You are concise.\nNo fluff.\n\n"
        "User: Hello\nworld\n\n"
        "Assistant: Hi\nthere\n\n"
        "User: Continue\nplease\n\n"
        "Assistant:"
    )


def test_rwkv_chat_template_tokenizes_rendered_prompt():
    tokenizer = get_tokenizer("BlinkDL/rwkv7-g1", tokenizer_mode="rwkv")

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        tokenize=True,
        add_generation_prompt=True,
    )

    assert isinstance(token_ids, list)
    assert tokenizer.decode(token_ids) == rendered


def test_rwkv_chat_template_renders_tools_and_tool_outputs_from_training_template():
    tokenizer = get_tokenizer("BlinkDL/rwkv7-g1", tokenizer_mode="rwkv")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "  Use tools carefully.\n\nExplain gaps. "},
            {"role": "user", "content": " Weather in Paris?\r\n\r\nUse Celsius. "},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"temperature": 21, "unit": "celsius"}',
            },
        ],
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )

    assert rendered == (
        "### System\n"
        "Use tools carefully.\n"
        "Explain gaps.\n"
        "### `get_weather`\n"
        "**Description:** Get the weather for a city.\n"
        "**Parameters:**\n"
        "```json\n"
        "{\n"
        '  "type": "object",\n'
        '  "properties": {\n'
        '    "city": {\n'
        '      "type": "string"\n'
        "    }\n"
        "  },\n"
        '  "required": [\n'
        '    "city"\n'
        "  ]\n"
        "}\n"
        "```\n"
        "### User\n"
        "Weather in Paris?\n"
        "Use Celsius.\n"
        "### Assistant\n"
        "**Tool Call:**\n"
        "```json\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "arguments": {\n'
        '    "city": "Paris"\n'
        "  }\n"
        "}\n"
        "```\n"
        "### Tool Output\n"
        "```json\n"
        "{\n"
        '  "temperature": 21,\n'
        '  "unit": "celsius"\n'
        "}\n"
        "```\n"
        "### Assistant\n"
    )
