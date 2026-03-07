# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for beam search with structured output on the online OpenAI server.

Run:
    pytest tests/entrypoints/openai/test_beam_search_structured_output.py -v
"""

import json

import jsonschema
import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

BEAM_WIDTH = 4
MAX_TOKENS = 64

CHOICES = [
    "Python",
    "Java",
    "JavaScript",
    "C++",
    "C#",
    "PHP",
    "TypeScript",
    "Ruby",
    "Swift",
    "Kotlin",
]

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}


@pytest.fixture(scope="module")
def default_server_args():
    return [
        "--dtype",
        "half",
        "--max-model-len",
        "512",
        "--enforce-eager",
        "--structured-outputs-config.backend",
        "xgrammar",
        "--structured-outputs-config.disable_any_whitespace",
        "true",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.3",
    ]


@pytest.fixture(scope="module")
def server(default_server_args):
    with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


# ---- Chat Completions Tests ----


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_beam_search_structured_output_chat_json_schema(
    client: openai.AsyncOpenAI,
    model_name: str,
):
    """Beam search + structured output (json_schema) via chat completions."""
    messages = [
        {
            "role": "user",
            "content": "Generate a JSON object for a person with name and age.",
        }
    ]

    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        n=BEAM_WIDTH,
        max_completion_tokens=MAX_TOKENS,
        extra_body=dict(
            use_beam_search=True,
            structured_outputs={"json": JSON_SCHEMA},
        ),
    )

    assert len(chat_completion.choices) == BEAM_WIDTH
    for choice in chat_completion.choices:
        text = choice.message.content
        assert text is not None and len(text) > 0
        # Strip any trailing special tokens.
        text = text.replace("</s>", "").strip()
        # Locate the JSON object.
        json_start = text.find("{")
        assert json_start != -1, f"No JSON object found in: {text!r}"
        parsed, _ = json.JSONDecoder().raw_decode(text[json_start:])
        jsonschema.validate(instance=parsed, schema=JSON_SCHEMA)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_beam_search_structured_output_chat_response_format(
    client: openai.AsyncOpenAI,
    model_name: str,
):
    """Beam search + structured output via response_format (json_schema)."""
    messages = [
        {
            "role": "user",
            "content": "Generate a JSON object for a person with name and age.",
        }
    ]

    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        n=BEAM_WIDTH,
        max_completion_tokens=MAX_TOKENS,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "person_test",
                "schema": JSON_SCHEMA,
            },
        },
        extra_body=dict(use_beam_search=True),
    )

    assert len(chat_completion.choices) == BEAM_WIDTH
    for choice in chat_completion.choices:
        text = choice.message.content
        assert text is not None and len(text) > 0
        text = text.replace("</s>", "").strip()
        json_start = text.find("{")
        assert json_start != -1, f"No JSON object found in: {text!r}"
        # Use raw_decode to handle trailing characters after the JSON.
        parsed, _ = json.JSONDecoder().raw_decode(text[json_start:])
        jsonschema.validate(instance=parsed, schema=JSON_SCHEMA)


# ---- Completions API Tests ----


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_beam_search_structured_output_completion_json(
    client: openai.AsyncOpenAI,
    model_name: str,
):
    """Beam search + structured output (json) via completions API."""
    completion = await client.completions.create(
        model=model_name,
        prompt="Generate a JSON object for a person with name and age:",
        n=BEAM_WIDTH,
        max_tokens=MAX_TOKENS,
        extra_body=dict(
            use_beam_search=True,
            structured_outputs={"json": JSON_SCHEMA},
        ),
    )

    assert len(completion.choices) == BEAM_WIDTH
    for choice in completion.choices:
        text = choice.text
        assert text is not None and len(text) > 0
        text = text.replace("</s>", "").strip()
        json_start = text.find("{")
        assert json_start != -1, f"No JSON object found in: {text!r}"
        parsed, _ = json.JSONDecoder().raw_decode(text[json_start:])
        jsonschema.validate(instance=parsed, schema=JSON_SCHEMA)


# ---- Edge Cases ----


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_beam_search_without_structured_output(
    client: openai.AsyncOpenAI,
    model_name: str,
):
    """Verify beam search still works without structured output."""
    messages = [
        {
            "role": "user",
            "content": "Tell me a short joke.",
        }
    ]

    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        n=BEAM_WIDTH,
        max_completion_tokens=MAX_TOKENS,
        extra_body=dict(use_beam_search=True),
    )

    assert len(chat_completion.choices) == BEAM_WIDTH
    for choice in chat_completion.choices:
        assert choice.message.content is not None
        assert len(choice.message.content) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_beam_search_structured_output_small_beam_width(
    client: openai.AsyncOpenAI,
    model_name: str,
):
    """Beam search + structured output with beam_width=2."""
    messages = [
        {
            "role": "user",
            "content": "Generate a JSON object for a person with name and age.",
        }
    ]

    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        n=2,
        max_completion_tokens=MAX_TOKENS,
        extra_body=dict(
            use_beam_search=True,
            structured_outputs={"json": JSON_SCHEMA},
        ),
    )

    assert len(chat_completion.choices) == 2
    for choice in chat_completion.choices:
        text = choice.message.content
        assert text is not None and len(text) > 0
        text = text.replace("</s>", "").strip()
        json_start = text.find("{")
        assert json_start != -1, f"No JSON object found in: {text!r}"
        parsed, _ = json.JSONDecoder().raw_decode(text[json_start:])
        jsonschema.validate(instance=parsed, schema=JSON_SCHEMA)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_beam_search_structured_output_choice(
    client: openai.AsyncOpenAI,
    model_name: str,
):
    """Beam search + structured output (choice) via chat completions."""
    messages = [
        {
            "role": "user",
            "content": "The best language for type-safe systems programming is ",
        }
    ]

    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        n=BEAM_WIDTH,
        max_completion_tokens=MAX_TOKENS,
        extra_body=dict(
            use_beam_search=True,
            structured_outputs={"choice": CHOICES},
        ),
    )

    assert len(chat_completion.choices) == BEAM_WIDTH
    for choice in chat_completion.choices:
        text = choice.message.content
        assert text is not None and len(text) > 0
        text = text.replace("</s>", "").strip()
        assert text in CHOICES, f"Expected one of {CHOICES}, got: {text!r}"


# ---- Auto Backend Test (no explicit --structured-outputs-config.backend) ----


@pytest.fixture(scope="module")
def auto_server():
    """Server with default backend='auto' (no explicit backend flag)."""
    args = [
        "--dtype",
        "half",
        "--max-model-len",
        "512",
        "--enforce-eager",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.3",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def auto_client(auto_server):
    async with auto_server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_beam_search_structured_output_auto_backend(
    auto_client: openai.AsyncOpenAI,
    model_name: str,
):
    """Beam search + structured output works with backend='auto' (default)."""
    messages = [
        {
            "role": "user",
            "content": "Generate a JSON object for a person with name and age.",
        }
    ]

    chat_completion = await auto_client.chat.completions.create(
        model=model_name,
        messages=messages,
        n=2,
        max_completion_tokens=MAX_TOKENS,
        extra_body=dict(
            use_beam_search=True,
            structured_outputs={"json": JSON_SCHEMA},
        ),
    )

    assert len(chat_completion.choices) == 2
    for choice in chat_completion.choices:
        text = choice.message.content
        assert text is not None and len(text) > 0
        text = text.replace("</s>", "").strip()
        json_start = text.find("{")
        assert json_start != -1, f"No JSON object found in: {text!r}"
        parsed, _ = json.JSONDecoder().raw_decode(text[json_start:])
        jsonschema.validate(instance=parsed, schema=JSON_SCHEMA)
