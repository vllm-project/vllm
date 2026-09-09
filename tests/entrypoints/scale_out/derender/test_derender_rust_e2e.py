# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end test for the Rust frontend ``/derender`` endpoints.

Boots the disaggregated stack as two separate processes — a real GPU engine
(``vllm serve``, which also mounts the Python ``/derender`` endpoints) and the
GPU-less Rust render server (``vllm-rs render``, mounting the Rust ``/render``
and ``/derender`` endpoints) — and verifies:

1. A full render -> generate -> derender roundtrip across the wire for both
   chat and completions.
2. Python vs Rust derender parity on identical greedy token IDs, mirroring
   ``test_derender_parity.py`` (which pins the coupled path against the Python
   derender). Feeding both derenders the same generated token IDs makes
   generation nondeterminism irrelevant.
3. The Rust streaming derender protocol (client-carried ``stream_state``)
   against real generated tokens: chunked derendering must produce the same
   text as the one-shot derender of the full token list.

Requires the ``vllm-rs`` binary (``cargo build --release -p vllm-cmd``) and a
GPU.
"""

import json

import httpx
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer, RemoteRustRenderServer

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MAX_MODEL_LEN = 4096
GPU_ARGS = [
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "hermes",
    "--reasoning-parser",
    "deepseek_r1",
    "--enforce-eager",
    "--max-model-len",
    str(MAX_MODEL_LEN),
    # 8 GB cards share VRAM with the desktop; leave headroom.
    "--gpu-memory-utilization",
    "0.8",
]
RUST_RENDER_ARGS = [
    "--max-model-len",
    str(MAX_MODEL_LEN),
    "--tool-call-parser",
    "hermes",
    "--reasoning-parser",
    "deepseek_r1",
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]
FORCE_WEATHER_TOOL = {"type": "function", "function": {"name": "get_weather"}}


@pytest.fixture(scope="module")
def gpu_server():
    with RemoteOpenAIServer(MODEL, GPU_ARGS) as server:
        yield server


@pytest.fixture(scope="module")
def rust_render():
    with RemoteRustRenderServer(MODEL, RUST_RENDER_ARGS, seed=None) as server:
        yield server


@pytest_asyncio.fixture
async def gpu_client(gpu_server):
    async with httpx.AsyncClient(
        base_url=gpu_server.url_for(""), timeout=60.0
    ) as client:
        yield client


@pytest_asyncio.fixture
async def rust_client(rust_render):
    async with httpx.AsyncClient(
        base_url=rust_render.url_for(""), timeout=60.0
    ) as client:
        yield client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chat_request(messages: list[dict], **extra) -> dict:
    return {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 128,
        **extra,
    }


async def _render_chat(rust_client: httpx.AsyncClient, chat_request: dict) -> dict:
    resp = await rust_client.post("/v1/chat/completions/render", json=chat_request)
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _generate(gpu_client: httpx.AsyncClient, generate_request: dict) -> dict:
    # The Rust render server serializes unset sampling params as explicit
    # nulls, which the Python /inference/v1/generate msgspec validation
    # rejects (a pre-existing render-side wire incompatibility, unrelated to
    # derender). Strip them before the generate hop.
    stripped = {
        key: (
            {k: v for k, v in value.items() if v is not None}
            if isinstance(value, dict)
            else value
        )
        for key, value in generate_request.items()
    }
    resp = await gpu_client.post("/inference/v1/generate", json=stripped)
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _derender_chat(
    client: httpx.AsyncClient,
    generate_response: dict,
    prompt_tokens: int,
    chat_request: dict,
) -> dict:
    resp = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL,
            "generate_response": generate_response,
            "prompt_tokens": prompt_tokens,
            "chat_request": chat_request,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _tool_sig(choice: dict) -> list[tuple[str, dict]]:
    """[(name, json normalized args)] so key ordering / whitespace don't
    cause false negatives."""
    return [
        (tc["function"]["name"], json.loads(tc["function"]["arguments"]))
        for tc in (choice["message"].get("tool_calls") or [])
    ]


def _assert_derender_parity(python: dict, rust: dict, output_ids: list[int]) -> None:
    """Both derenders saw the same tokens, so they must agree unconditionally."""
    p, r = python["choices"][0], rust["choices"][0]
    assert r["message"]["content"] == p["message"]["content"]
    assert r["message"].get("reasoning") == p["message"].get("reasoning")
    assert _tool_sig(r) == _tool_sig(p)
    assert r["finish_reason"] == p["finish_reason"]
    assert rust["usage"]["prompt_tokens"] == python["usage"]["prompt_tokens"]
    assert rust["usage"]["completion_tokens"] == len(output_ids)


# ---------------------------------------------------------------------------
# Roundtrip tests: render (Rust) -> generate (GPU) -> derender (Rust)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_chat_roundtrip(gpu_client, rust_client):
    """Full disaggregated chat flow across the two servers."""
    chat_request = _chat_request(
        [{"role": "user", "content": "What is 2+2? Answer in one short sentence."}]
    )
    generate_request = await _render_chat(rust_client, chat_request)
    generate_response = await _generate(gpu_client, generate_request)

    derendered = await _derender_chat(
        rust_client,
        generate_response,
        prompt_tokens=len(generate_request["token_ids"]),
        chat_request=chat_request,
    )

    choice = derendered["choices"][0]
    output_ids = generate_response["choices"][0]["token_ids"]
    assert choice["message"]["role"] == "assistant"
    # A reasoning parser is configured, so output may land in `reasoning`.
    message = choice["message"]
    assert message["content"] or message.get("reasoning")
    assert derendered["usage"]["prompt_tokens"] == len(generate_request["token_ids"])
    assert derendered["usage"]["completion_tokens"] == len(output_ids)


@pytest.mark.asyncio
async def test_e2e_completion_roundtrip(gpu_client, rust_client):
    """Full disaggregated completions flow across the two servers."""
    completion_request = {
        "model": MODEL,
        "prompt": "The capital of France is",
        "temperature": 0,
        "max_tokens": 32,
    }
    resp = await rust_client.post("/v1/completions/render", json=completion_request)
    assert resp.status_code == 200, resp.text
    generate_requests = resp.json()
    assert len(generate_requests) == 1

    generate_response = await _generate(gpu_client, generate_requests[0])
    resp = await rust_client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL,
            "generate_responses": [generate_response],
            "prompt_tokens": [len(generate_requests[0]["token_ids"])],
            "completion_request": completion_request,
        },
    )
    assert resp.status_code == 200, resp.text
    derendered = resp.json()

    assert derendered["choices"][0]["text"]
    assert derendered["usage"]["prompt_tokens"] == len(
        generate_requests[0]["token_ids"]
    )
    assert derendered["usage"]["completion_tokens"] == len(
        generate_response["choices"][0]["token_ids"]
    )


# ---------------------------------------------------------------------------
# Parity tests: Python /derender (GPU server) vs Rust /derender (render server)
# ---------------------------------------------------------------------------


async def _run_parity_case(
    gpu_client: httpx.AsyncClient, rust_client: httpx.AsyncClient, chat_request: dict
) -> tuple[dict, dict, list[int]]:
    """One greedy generation, derendered by both implementations.

    Returns (python_derender, rust_derender, output_ids)."""
    resp = await gpu_client.post(
        "/v1/chat/completions", json={**chat_request, "return_token_ids": True}
    )
    assert resp.status_code == 200, resp.text
    coupled = resp.json()
    choice = coupled["choices"][0]
    output_ids = choice["token_ids"]

    generate_response = {
        "request_id": "parity",
        "choices": [
            {
                "index": 0,
                "token_ids": output_ids,
                "finish_reason": choice["finish_reason"],
            }
        ],
    }
    prompt_tokens = coupled["usage"]["prompt_tokens"]
    python = await _derender_chat(
        gpu_client, generate_response, prompt_tokens, chat_request
    )
    rust = await _derender_chat(
        rust_client, generate_response, prompt_tokens, chat_request
    )
    return python, rust, output_ids


@pytest.mark.asyncio
async def test_derender_parity_chat(gpu_client, rust_client):
    """Plain + reasoning content parity between the Python and Rust derender."""
    chat_request = _chat_request(
        [{"role": "user", "content": "What is 17 times 23? Think it through."}],
        include_reasoning=True,
        max_tokens=256,
    )
    python, rust, output_ids = await _run_parity_case(
        gpu_client, rust_client, chat_request
    )
    _assert_derender_parity(python, rust, output_ids)

    if not python["choices"][0]["message"].get("reasoning"):
        pytest.skip("Model did not emit a <think> block")
    assert rust["choices"][0]["message"]["reasoning"]


@pytest.mark.asyncio
async def test_derender_parity_tool_call(gpu_client, rust_client):
    """Tool call name+args parity between the Python and Rust derender."""
    chat_request = _chat_request(
        [{"role": "user", "content": "What's the weather in Paris?"}],
        tools=TOOLS,
        tool_choice=FORCE_WEATHER_TOOL,
        max_tokens=1024,
    )
    python, rust, output_ids = await _run_parity_case(
        gpu_client, rust_client, chat_request
    )
    _assert_derender_parity(python, rust, output_ids)

    if not _tool_sig(python["choices"][0]):
        pytest.skip("Model did not emit a tool call")
    assert _tool_sig(rust["choices"][0])


# ---------------------------------------------------------------------------
# Streaming derender: chunked == one-shot over real generated tokens
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_derender_streaming_roundtrip(gpu_client, rust_client):
    """Rust streaming completions derender: chunked text == one-shot text.

    Uses the completions endpoint because chat streaming derender fails closed
    when a reasoning/tool parser is configured (same as Python's
    ``NotImplementedError`` path).
    """
    completion_request = {
        "model": MODEL,
        "prompt": "The capital of France is",
        "temperature": 0,
        "max_tokens": 32,
    }
    resp = await rust_client.post("/v1/completions/render", json=completion_request)
    assert resp.status_code == 200, resp.text
    generate_response = await _generate(gpu_client, resp.json()[0])
    output_ids: list[int] = generate_response["choices"][0]["token_ids"]
    assert len(output_ids) >= 3, "need a few tokens to chunk the stream"

    # One-shot baseline from the Rust derender.
    resp = await rust_client.post(
        "/v1/completions/derender",
        json={"model": MODEL, "generate_responses": [generate_response]},
    )
    assert resp.status_code == 200, resp.text
    expected_text = resp.json()["choices"][0]["text"]

    # Chunked streaming derender, carrying stream_state between calls.
    third = max(1, len(output_ids) // 3)
    chunks = [
        (output_ids[:third], None),
        (output_ids[third : 2 * third], None),
        (output_ids[2 * third :], "stop"),
    ]
    stream_state = None
    streamed_text = ""
    for chunk_ids, finish_reason in chunks:
        resp = await rust_client.post(
            "/v1/completions/derender",
            json={
                "stream": True,
                "model": MODEL,
                "generate_chunk": {
                    "request_id": generate_response["request_id"],
                    "choices": [
                        {
                            "index": 0,
                            "token_ids": chunk_ids,
                            "finish_reason": finish_reason,
                        }
                    ],
                },
                "stream_state": stream_state,
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        streamed_text += payload["chunk"]["choices"][0]["text"]
        stream_state = payload["stream_state"]

    assert streamed_text == expected_text
