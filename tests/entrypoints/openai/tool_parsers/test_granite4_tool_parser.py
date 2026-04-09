# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import openai
import pytest

from ....utils import RemoteOpenAIServer

MODEL = "ibm-granite/granite-4.0-h-tiny"


@pytest.fixture(scope="module")
def server():
    model = MODEL
    args_for_model = [
        "--enforce-eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "granite4",
        "--tokenizer",
        "ibm-granite/granite-4.0-h-tiny",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "2",
    ]
    with RemoteOpenAIServer(model, args_for_model, max_wait_seconds=480) as server:
        yield server


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_acme_region_name_for_transaction_id",
            "description": "Returns ACME transaction/transaction ID information"
            " including ACME regions\n\nArgs:\n    start_time "
            "(str): Start date and time in datetime format "
            '"%Y-%m-%dT%H:%M:%S.%f"\n    end_time (str): End '
            "date and time in datetime format "
            '"%Y-%m-%dT%H:%M:%S.%f"\n    size (int, optional): '
            "Number of ACME Transaction IDs to return\n    "
            "order (str, optional): Sort by most run "
            "transaction IDs. The value can be 'asc' for "
            "ascending or 'desc' for descending\n    "
            "transaction_id (str, optional): ACME Transaction "
            "ID to filter on\n    acme_region (str, optional): "
            "ACME Region to filter on\nReturns:\n    - A "
            "dictionary containing a list of ACME transaction "
            "ids and the ACME regions they run in:\n        {\n"
            '            "Number of transaction IDs"   : int,\n'
            '            "Total transaction IDs available": int'
            ',\n            "ACME Transaction IDs": [\n        '
            '        {\n                    "Transaction ID": '
            'str,\n                    "Number of runs": int,\n'
            '                    "ACME Regions": [str],\n      '
            "          },\n                ...\n            ],"
            '\n            "Start time"         : datetime,\n '
            '           "End time"           : datetime,\n    '
            '        "Order"              : str\n        }\n  '
            "  - If no ACME region found for transaction id, "
            'returns:\n        {"Success": "No ACME region '
            'found for transaction id."}\n    - If an error '
            'occurs, returns:\n        {"Error": "{exception'
            ' message}"}',
            "parameters": {
                "properties": {
                    "start_time": {},
                    "end_time": {},
                    "size": {"default": 500},
                    "order": {"default": "desc"},
                    "transaction_id": {"default": None},
                    "acme_region": {"default": None},
                },
                "required": ["start_time", "end_time"],
                "type": "object",
            },
        },
    }
]

tools2 = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "description": "The city and state, e.g. San Francisco, CA",
                        "type": "string",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Retrieves the current stock price for a given "
            "ticker symbol. The ticker symbol must be a valid "
            "symbol for a publicly traded company on a major US"
            " stock exchange like NYSE or NASDAQ. The tool will"
            " return the latest trade price in USD. It should "
            "be used when the user asks about the current or "
            "most recent price of a specific stock. It will not"
            " provide any other information about the stock or"
            " company.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "description": "The stock ticker symbol, e.g."
                        " AAPL for Apple Inc.",
                        "type": "string",
                    }
                },
            },
        },
    },
]

messages = [
    {
        "content": "\n\nSystem: You are a helpful, precise, and methodical AI"
        " assistant that uses tool outputs provided inline.\nAlways"
        " assume the current datetime is 2026-01-29T13:59:09.238901"
        "+00:00.\n\nIf you receive a ToolMessage with `tool_call_id"
        '` equal to "get_time_range" (or "time_range_tool"), you '
        "MUST:\n  1. Parse that JSON and use the values `start` and"
        " `end` directly when calling other tools.\n  2. Do not "
        "re-call or re-compute the time range.\n  3. Pass resolved "
        "values (ISO strings) as arguments to any subsequent tool "
        "(do not pass function metadata or placeholders).\n  4. If "
        "a tool requires datetime objects rather than strings, "
        "convert the ISO strings into language-native datetime "
        "objects before invoking.\n\nAlways return fully resolved "
        "arguments in correct types (e.g., ISO datetime strings or"
        " datetime objects) and never include placeholders like "
        '"<start>".\n\n',
        "role": "system",
    },
    {
        "content": "What are the transaction IDs that ran in the"
        " ACME region A9345 over the last two months?",
        "role": "user",
    },
    {
        "content": '["2026-01-26T09: 51: 55.467722Z", "2026-01-27T09: 51: 55.467722Z"]',
        "role": "tool",
        "tool_call_id": "time_range_tool",
    },
]
messages2 = [{"role": "user", "content": "What's stock price for IBM?"}]

messages3 = [{"role": "user", "content": "What's the current weather in New York?"}]


def get_args(client: openai.OpenAI, _tools, _messages, _stop):
    response = client.chat.completions.create(
        model=MODEL,
        messages=_messages,
        temperature=0,
        tools=_tools,
        max_tokens=200,
        stop=_stop,
        tool_choice="auto",
    )

    return response.choices[0].message.tool_calls[0].function.arguments


async def get_args_streaming(
    async_client: openai.AsyncOpenAI, _tools, _messages, _stop
):
    stream = await async_client.chat.completions.create(
        model=MODEL,
        messages=_messages,
        temperature=0,
        tools=_tools,
        max_tokens=200,
        stop=_stop,
        tool_choice="auto",
        stream=True,
    )
    full_call = []
    async for chunk in stream:
        tc = chunk.choices[0].delta.tool_calls
        if tc and tc[0].function.arguments:
            full_call.append(tc[0].function.arguments)
    return "".join(full_call)


async def run_scenario(server: RemoteOpenAIServer, _tools, _messages, _stop):
    non_streaming = get_args(server.get_client(), _tools, _messages, _stop)
    json.loads(non_streaming)  # verify that it is json loadable
    streaming = await get_args_streaming(
        server.get_async_client(), _tools, _messages, _stop
    )
    json.loads(streaming)
    assert non_streaming == streaming, f"{non_streaming=}, {streaming=}"


@pytest.mark.asyncio
async def test_stop_sequence_interference(server: RemoteOpenAIServer):
    print("Testing scenario 1")
    await run_scenario(server, tools, messages, "veroniqueprattyushveroniqueprattyush")

    print("Testing scenario 2")
    await run_scenario(
        server, tools2, messages2, "veroniqueprattyushveroniqueprattyush"
    )

    print("Testing scenario 3")
    await run_scenario(server, tools2, messages3, "prattyush")
