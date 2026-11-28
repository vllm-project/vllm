# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Number arguments in dots tool calls must stay exact and valid JSON."""

import json

import openai
import pytest
import pytest_asyncio
import regex as re

from tests.utils import RemoteOpenAIServer

MODEL = "Qwen/Qwen3-0.6B"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "transfer",
            "parameters": {
                "type": "object",
                "properties": {"amount": {"type": "number"}},
                "required": ["amount"],
            },
        },
    }
]
TOOL_CALL = (
    '<dots_function_call><invoke name="transfer">'
    '<parameter name="amount">{value}</parameter>'
    "</invoke></dots_function_call>"
)


@pytest.fixture(scope="module")
def server():
    args = [
        "--enforce-eager",
        "--max-model-len",
        "1024",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "dots",
        "--max-num-seqs",
        "2",
    ]
    with RemoteOpenAIServer(MODEL, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server: RemoteOpenAIServer):
    async with server.get_async_client() as async_client:
        yield async_client


def _strict_json(arguments: str):
    """Parse like a strict JSON client, which rejects `NaN` and `Infinity`."""

    def reject(constant: str):
        raise ValueError(f"invalid JSON constant: {constant}")

    return json.loads(arguments, parse_constant=reject)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "amount"),
    [
        ("12345678901234567890", 12345678901234567890),
        ("1e400", "1e400"),
        ("-1e400", "-1e400"),
        ("nan", "nan"),
    ],
)
async def test_number_argument_is_exact_valid_json(
    client: openai.AsyncOpenAI, raw: str, amount
):
    """A 20-digit integer keeps every digit; `inf` and `nan` stay raw text."""
    reply = TOOL_CALL.format(value=raw)
    request = dict(
        model=MODEL,
        messages=[{"role": "user", "content": "Hi"}],
        tools=TOOLS,
        max_tokens=80,
        temperature=0,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "structured_outputs": {"regex": re.escape(reply)},
        },
    )

    response = await client.chat.completions.create(**request)
    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls and len(tool_calls) == 1
    assert _strict_json(tool_calls[0].function.arguments) == {"amount": amount}

    stream = await client.chat.completions.create(stream=True, **request)
    arguments = ""
    async for chunk in stream:
        if chunk.choices:
            for tool_call in chunk.choices[0].delta.tool_calls or []:
                arguments += tool_call.function.arguments or ""
    assert _strict_json(arguments) == {"amount": amount}
