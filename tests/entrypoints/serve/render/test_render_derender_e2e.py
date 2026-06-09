# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end render -> derender round trip tests.

These tests exercise the full postprocessing loop of the disaggregated /
RL serving path:

    /v1/chat/completions/render   (prompt messages -> prompt token IDs)
        -> [generation happens on a separate tier]
    /v1/chat/completions/derender (output token IDs -> ChatCompletionResponse)

The `render` tier is GPU less (`vllm launch render`): it only loads the
tokenizer, never the model weights, so these tests run on a system with
no accelerator. Because generation lives on a different (disaggregated) tier
that is not present here, we synthesize the "generated" output tokens in test
by encoding a known assistant string with the same tokenizer the server
uses. That is exactly the contract the real caller fulfils: it already holds
the output token IDs returned by the generate tier and hands them to
`/derender`. Encoding known text lets us assert the true inverse property,
that `derender` reconstructs the chat response we expect.

Two test scenarios:

1. `test_e2e_plain_*`: baseline round trip with no parsers configured.
   The decoded output text passes straight through as `message.content`.
2. `test_e2e_parsed_*`: same round trip but the render tier is launched with
   a reasoning parser, a tool-call parser and auto-tool-choice, and the
   `chat_request` is supplied. This drives the parsing path:
   ``message.reasoning`` and ``message.tool_calls`` are populated
   by vLLM's real serving parser (the single source of truth shared with the
   coupled chat endpoint).

The tokenizer/config for both models is downloaded from the Hugging Face Hub on
first run (weights are not fetched).
"""

import httpx
import pytest
import pytest_asyncio

from tests.utils import RemoteLaunchRenderServer
from vllm.tokenizers import get_tokenizer

# Tiny random model, only its tokenizer is loaded by the render tier.
PLAIN_MODEL = "hmellor/tiny-random-LlamaForCausalLM"

# Small real model used purely for its tokenizer, which defines the
# `<think>`/`</think>` reasoning tokens and the `<tool_call>` tokens the
# parsers below operate on. Weights are NOT downloaded (render is GPU less).
PARSER_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode(tokenizer, text: str) -> list[int]:
    """Encode text into the token IDs a generate tier would have emitted.

    ``add_special_tokens=False`` keeps BOS/EOS out so the round trip recovers
    exactly the supplied string.
    """
    return tokenizer.encode(text, add_special_tokens=False)


def _decoded(tokenizer, token_ids: list[int]) -> str:
    """Decode the way the derender endpoint does (skip_special_tokens=True)."""
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _generate_response(
    token_ids: list[int],
    request_id: str = "chatcmpl-e2e",
    finish_reason: str = "stop",
) -> dict:
    """Build a minimal GenerateResponse body for the derender endpoint."""
    return {
        "request_id": request_id,
        "choices": [
            {
                "index": 0,
                "token_ids": token_ids,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "prompt_logprobs": None,
        "kv_transfer_params": None,
    }


async def _render_chat(client: httpx.AsyncClient, model: str, messages: list) -> dict:
    """Render a chat request. Return the GenerateRequest dict (has token_ids)."""
    resp = await client.post(
        "/v1/chat/completions/render",
        json={"model": model, "messages": messages},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["token_ids"], "render produced no prompt token IDs"
    return body


# ---------------------------------------------------------------------------
# Scenario 1: baseline render -> derender round trip (no parsers)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plain_server():
    with RemoteLaunchRenderServer(PLAIN_MODEL, []) as server:
        yield server


@pytest_asyncio.fixture
async def plain_client(plain_server):
    async with httpx.AsyncClient(
        base_url=plain_server.url_for(""), timeout=60.0
    ) as client:
        yield client


@pytest.fixture(scope="module")
def plain_tokenizer():
    return get_tokenizer(PLAIN_MODEL)


@pytest.mark.asyncio
async def test_e2e_plain_roundtrip(plain_client, plain_tokenizer):
    """Render a prompt, then derender synthesized output tokens.

    With no parser configured the decoded output text is returned verbatim as
    ``message.content`` — i.e. derender is the exact inverse of detokenization.
    """
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    gen_req = await _render_chat(plain_client, PLAIN_MODEL, messages)
    prompt_token_count = len(gen_req["token_ids"])

    # Synthesize what the generate tier would have produced.
    answer = "The capital of France is Paris."
    output_ids = _encode(plain_tokenizer, answer)
    expected_content = _decoded(plain_tokenizer, output_ids)

    resp = await plain_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PLAIN_MODEL,
            "generate_response": _generate_response(output_ids),
            "prompt_tokens": prompt_token_count,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    choice = data["choices"][0]
    assert choice["message"]["role"] == "assistant"
    # The inverse property: decoded output text round-trips into content.
    assert choice["message"]["content"] == expected_content
    assert choice["message"].get("reasoning") is None
    assert not choice["message"].get("tool_calls")
    assert choice["finish_reason"] == "stop"

    usage = data["usage"]
    assert usage["prompt_tokens"] == prompt_token_count
    assert usage["completion_tokens"] == len(output_ids)
    assert usage["total_tokens"] == prompt_token_count + len(output_ids)


# ---------------------------------------------------------------------------
# Test 2: render -> derender with parsers, tools and reasoning
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parser_server():
    args = [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--reasoning-parser",
        "deepseek_r1",
    ]
    with RemoteLaunchRenderServer(PARSER_MODEL, args) as server:
        yield server


@pytest_asyncio.fixture
async def parser_client(parser_server):
    async with httpx.AsyncClient(
        base_url=parser_server.url_for(""), timeout=60.0
    ) as client:
        yield client


@pytest.fixture(scope="module")
def parser_tokenizer():
    return get_tokenizer(PARSER_MODEL)


def _require_markers_survive(tokenizer, text: str, *markers: str) -> list[int]:
    """Encode ``text`` and guard that ``markers`` survive the server's decode.

    The derender endpoint decodes with ``skip_special_tokens=True``. If a
    tokenizer treats a marker as a stripped special token the parser could
    never see it (a tokenize specific quirk, not a code defect), so we skip
    rather than report a misleading failure.
    """
    token_ids = _encode(tokenizer, text)
    decoded = _decoded(tokenizer, token_ids)
    for marker in markers:
        if marker not in decoded:
            pytest.skip(
                f"marker {marker!r} did not survive tokenizer round-trip for "
                f"{PARSER_MODEL}; cannot exercise the parser path"
            )
    return token_ids


@pytest.mark.asyncio
async def test_e2e_parsed_reasoning(parser_client, parser_tokenizer):
    """Reasoning model: derender splits ``<think>...</think>`` into reasoning.

    Mirrors test 1 but the render tier runs the deepseek_r1 reasoning
    parser and the request carries chat_request, so the path segments
    reasoning from content.
    """
    messages = [{"role": "user", "content": "Add 2 and 3."}]
    gen_req = await _render_chat(parser_client, PARSER_MODEL, messages)

    reasoning_text = "The user wants 2 + 3. That is 5."
    answer_text = "The answer is 5."
    output_text = f"<think>{reasoning_text}</think>{answer_text}"
    output_ids = _require_markers_survive(parser_tokenizer, output_text, "</think>")

    resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PARSER_MODEL,
            "generate_response": _generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
            "chat_request": {
                "model": PARSER_MODEL,
                "messages": messages,
                "include_reasoning": True,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    message = resp.json()["choices"][0]["message"]

    assert message["role"] == "assistant"
    assert message["reasoning"] is not None
    assert reasoning_text in message["reasoning"]
    assert message["content"] is not None
    assert answer_text in message["content"]
    # Reasoning markers must not leak into the user-facing content.
    assert "<think>" not in message["content"]
    assert "</think>" not in message["content"]


@pytest.mark.asyncio
async def test_e2e_parsed_reasoning_disabled(parser_client, parser_tokenizer):
    """include_reasoning=False drops reasoning from the derendered message."""
    messages = [{"role": "user", "content": "Add 2 and 3."}]
    gen_req = await _render_chat(parser_client, PARSER_MODEL, messages)

    output_text = "<think>2 + 3 = 5</think>The answer is 5."
    output_ids = _require_markers_survive(parser_tokenizer, output_text, "</think>")

    resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PARSER_MODEL,
            "generate_response": _generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
            "chat_request": {
                "model": PARSER_MODEL,
                "messages": messages,
                "include_reasoning": False,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    message = resp.json()["choices"][0]["message"]
    assert message["reasoning"] is None


@pytest.mark.asyncio
async def test_e2e_parsed_tool_call(parser_client, parser_tokenizer):
    """Auto tool choice: derender extracts a hermes ``<tool_call>`` into
    ``message.tool_calls`` and flags ``finish_reason == 'tool_calls'``."""
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    gen_req = await _render_chat(parser_client, PARSER_MODEL, messages)

    output_text = (
        '<tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "Paris"}}\n</tool_call>'
    )
    output_ids = _require_markers_survive(
        parser_tokenizer, output_text, "<tool_call>", "</tool_call>"
    )

    resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PARSER_MODEL,
            "generate_response": _generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
            "chat_request": {
                "model": PARSER_MODEL,
                "messages": messages,
                "tools": _TOOLS,
                "tool_choice": "auto",
            },
        },
    )
    assert resp.status_code == 200, resp.text
    choice = resp.json()["choices"][0]
    tool_calls = choice["message"]["tool_calls"]

    assert tool_calls, "expected a tool call to be extracted"
    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert call["function"]["name"] == "get_weather"
    assert "Paris" in call["function"]["arguments"]
    assert call["id"], "tool call should have a generated id"
    assert choice["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_e2e_parsed_reasoning_and_tool_call(parser_client, parser_tokenizer):
    """Combined: reasoning block followed by a tool call are both parsed."""
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    gen_req = await _render_chat(parser_client, PARSER_MODEL, messages)

    reasoning_text = "I should look up the weather for Paris."
    output_text = (
        f"<think>{reasoning_text}</think>"
        '<tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "Paris"}}\n</tool_call>'
    )
    output_ids = _require_markers_survive(
        parser_tokenizer, output_text, "</think>", "<tool_call>", "</tool_call>"
    )

    resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PARSER_MODEL,
            "generate_response": _generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
            "chat_request": {
                "model": PARSER_MODEL,
                "messages": messages,
                "tools": _TOOLS,
                "tool_choice": "auto",
                "include_reasoning": True,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    choice = resp.json()["choices"][0]
    message = choice["message"]

    assert message["reasoning"] is not None
    assert reasoning_text in message["reasoning"]
    assert message["tool_calls"], "expected a tool call alongside reasoning"
    assert message["tool_calls"][0]["function"]["name"] == "get_weather"
    assert choice["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_e2e_parsed_matches_plain_when_no_markers(
    parser_client, parser_tokenizer
):
    """A plain answer (no reasoning/tool markers) round trips to content even
    on a parse enabled server which is parity with the test 1."""
    messages = [{"role": "user", "content": "Say hello."}]
    gen_req = await _render_chat(parser_client, PARSER_MODEL, messages)

    answer = "Hello there!"
    output_ids = _encode(parser_tokenizer, answer)
    expected_content = _decoded(parser_tokenizer, output_ids)

    resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PARSER_MODEL,
            "generate_response": _generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
            "chat_request": {"model": PARSER_MODEL, "messages": messages},
        },
    )
    assert resp.status_code == 200, resp.text
    message = resp.json()["choices"][0]["message"]
    assert message["content"] == expected_content
    assert message["reasoning"] is None
    assert not message["tool_calls"]
