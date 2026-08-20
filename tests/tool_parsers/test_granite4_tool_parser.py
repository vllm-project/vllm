# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import random
from typing import Any

import pytest
from transformers import AutoTokenizer

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
)
from vllm.tool_parsers.granite4_tool_parser import Granite4ToolParser

MODEL = "ibm-granite/granite-4.0-h-tiny"


def create_complex_input(create_string_args: bool):
    coord_arg: dict | str = {
        "coordinates": [[23.54, 43.1], [-12.2, 54.3], [4, 5]],
        "coordinate_type": "latlong",
    }
    if create_string_args:
        # test granite behavior
        coord_arg = json.dumps(coord_arg)
    return [
        {"name": "find_bbox", "arguments": coord_arg},
        {
            "name": "get_stock_price",
            "arguments": {
                "symbol": "AAPL",
                "start_date": "2021-01-01",
                "end_date": "2021-12-31",
            },
        },
        {"name": "find_bbox", "arguments": coord_arg},
    ]


def random_chunks(s: str, min_len: int, max_len: int):
    chunks = []
    i = 0
    n = len(s)

    while i < n:
        size = random.randint(min_len, max_len)
        chunks.append(s[i : i + size])
        i += size

    return chunks


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


# create a variety of input chunk sizes
@pytest.mark.parametrize(
    "min_chunk, max_chunk",
    [
        (1, 1),
        (1, 2),
        (5, 7),
        (6, 20),
    ],
)
def test_tool_call_parser_complex(min_chunk: int, max_chunk: int, tokenizer):
    input_dicts = create_complex_input(True)

    formatted_tcs = [
        "<tool_call> " + json.dumps(call) + " </tool_call>" for call in input_dicts
    ]

    text_messages = [
        "Here goes the bbox call: \n",
        " Now the stock price call: \n ",
        " Now another bbox call: \n ",
        " See? I'm a helpful assistant.",
    ]

    test_input = (
        text_messages[0]
        + formatted_tcs[0]
        + text_messages[1]
        + formatted_tcs[1]
        + text_messages[2]
        + formatted_tcs[2]
        + text_messages[3]
    )

    any_chat_request = ChatCompletionRequest(
        seed=42,
        model=MODEL,
        messages=[],
    )

    parser = Granite4ToolParser(tokenizer=tokenizer)

    delta_messages = list[DeltaMessage]()
    for text in random_chunks(test_input, min_chunk, max_chunk):
        delta = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="",
            delta_text=text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=any_chat_request,
        )
        if delta is not None:
            delta_messages.append(delta)

    content = ""
    tool_calls = list[dict[str, Any]]()

    current_name = "__start__"
    current_args = ""

    for msg in delta_messages:
        if msg.content:
            content += msg.content
        for tool_call in msg.tool_calls:
            if delta_func := tool_call.function:
                if delta_func.name is not None:
                    if current_name == "__start__":
                        current_name = delta_func.name

                    if delta_func.name != current_name:
                        tool_calls.append(
                            {
                                "name": current_name,
                                "arguments": json.loads(current_args),
                            }
                        )
                        current_name = delta_func.name
                        current_args = ""

                if delta_func.arguments:
                    current_args += delta_func.arguments

    if current_name != "__start__":
        tool_calls.append({"name": current_name, "arguments": json.loads(current_args)})

    assert content == "".join(text_messages)
    assert tool_calls == create_complex_input(False)
