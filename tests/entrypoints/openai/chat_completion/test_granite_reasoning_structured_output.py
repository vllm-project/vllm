# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured output with ``--reasoning-parser granite``.

The grammar applies only after the parser reports the reasoning end: after
the response marker with ``thinking`` on, and from the first token with it off.
"""

import json

import jsonschema
import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "ibm-granite/granite-3.2-2b-instruct"
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}
JSON_SCHEMA = {
    "response_format": {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": SCHEMA},
    }
}
THINKING = {"chat_template_kwargs": {"thinking": True}}
CHOICES = ["zebra-17", "yak-42"]
MATH = "What is 17 times 3?"
MAX_TOKENS = 512
CAP_HIT = f"reasoning hit the {MAX_TOKENS} token cap before the response"


@pytest.fixture(scope="module")
def server():
    args = [
        "--reasoning-parser",
        "granite",
        "--max-model-len",
        "4096",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.4",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


def _request(prompt: str, stream: bool = False, **extra_body):
    return dict(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=0,
        stream=stream,
        extra_body=extra_body,
    )


NO_THOUGHTS = (
    "the model skipped its thought section; the grammar applies only after "
    "the response marker, so the content is unconstrained"
)


def _check_reasoning(reasoning: str | None):
    # With thinking on, the grammar starts after the response marker, so the
    # schema holds only when the model wrote a thought section, which the
    # parser ends at that marker.
    assert reasoning, NO_THOUGHTS
    assert "my response:" not in reasoning


@pytest.mark.asyncio
async def test_json_schema_with_thinking(client: openai.AsyncOpenAI):
    response = await client.chat.completions.create(
        **_request(MATH, **JSON_SCHEMA, **THINKING)
    )
    assert response.choices[0].finish_reason == "stop", CAP_HIT
    msg = response.choices[0].message
    _check_reasoning(getattr(msg, "reasoning", None))
    jsonschema.validate(json.loads(msg.content), SCHEMA)


@pytest.mark.asyncio
async def test_json_schema_with_thinking_streaming(client: openai.AsyncOpenAI):
    stream = await client.chat.completions.create(
        **_request(MATH, stream=True, **JSON_SCHEMA, **THINKING)
    )
    reasoning, content = [], []
    finish_reason = None
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning.append(getattr(delta, "reasoning", None) or "")
        content.append(delta.content or "")
        finish_reason = chunk.choices[0].finish_reason or finish_reason
    assert finish_reason == "stop", CAP_HIT
    _check_reasoning("".join(reasoning))
    jsonschema.validate(json.loads("".join(content)), SCHEMA)


@pytest.mark.asyncio
async def test_choice_without_thinking(client: openai.AsyncOpenAI):
    response = await client.chat.completions.create(
        **_request("Say hello.", structured_outputs={"choice": CHOICES})
    )
    # With thinking off the whole output is content, so the grammar covers it.
    assert response.choices[0].message.content in CHOICES
