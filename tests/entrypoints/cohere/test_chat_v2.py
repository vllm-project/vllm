# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end integration tests for the ``POST /cohere/v2/chat`` endpoint.

These tests spin up a real ``vllm serve`` process via
:class:`tests.utils.RemoteOpenAIServer` and exercise the Cohere Chat v2
contract over HTTP. They mirror the pattern used by
:mod:`tests.entrypoints.anthropic.test_messages`.

Two layers of integration are covered:

1. Raw HTTP via :mod:`httpx` — always runs, verifies the wire contract.
2. Cohere SDK (``pip install cohere``) — auto-skipped when the optional
   dependency isn't installed, verifies SDK-level interop.

We use a tiny generic chat model (``HuggingFaceTB/SmolLM2-135M-Instruct``,
≈135M params) so the fixture stays runnable on CPU-only laptops. The
Cohere v2 endpoint is model-agnostic and just performs the v2 ↔ OpenAI
chat-completion translation, so the choice of model only matters for
response *shape* — the translation logic itself is unit-tested
elsewhere.
"""

import json

import httpx
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

# Tiny chat-tuned model so the fixture is cheap to boot on CPU-only
# hosts (Mac, CI runners without GPUs). The Cohere v2 translation is
# completely independent of which underlying chat model is loaded.
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
SERVED_MODEL_NAME = "command-r-plus-08-2024"


@pytest.fixture(scope="module")
def server():
    args = [
        # 1k tokens is plenty for the prompts these tests send; trimming
        # it from the default keeps the KV-cache footprint tiny.
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "4",
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        # Advertise a Cohere model name so Cohere SDK ``model=`` calls
        # round-trip without the ``model not found`` check.
        "--served-model-name",
        SERVED_MODEL_NAME,
        # Disable the reasoning-model path so the test doesn't require
        # the conversation to surface a thinking block (SmolLM2 doesn't
        # emit Cohere-style reasoning tokens out of the box).
        "--no-cohere-is-reasoning-model",
    ]

    # Cap the CPU KV-cache pool the vLLM CPU backend reserves at
    # startup. The default (4 GB) trips Mac's RAM check on smaller
    # machines; 1 GB is more than enough for max_model_len=1024 and
    # max_num_seqs=4.
    env_dict = {"VLLM_CPU_KVCACHE_SPACE": "1"}

    with RemoteOpenAIServer(
        MODEL_NAME, args, env_dict=env_dict
    ) as remote_server:
        yield remote_server


# ----------------------------------------------------------------------
# Layer 1: raw HTTP contract (no optional deps)
# ----------------------------------------------------------------------


@pytest_asyncio.fixture
async def httpx_client(server):
    async with httpx.AsyncClient(
        base_url=server.url_root, timeout=httpx.Timeout(120.0)
    ) as client:
        yield client


@pytest.mark.asyncio
async def test_cohere_v2_chat_non_streaming(httpx_client: httpx.AsyncClient):
    resp = await httpx_client.post(
        "/cohere/v2/chat",
        json={
            "model": SERVED_MODEL_NAME,
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 16,
            "stream": False,
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    # The response envelope follows Cohere v2's schema.
    assert "message" in payload
    assert payload["message"]["role"] == "assistant"
    # ``content`` is a list of content blocks; the synthesized
    # ``CohereServingChatV2`` should always emit at least one ``text``
    # block (it falls back to an empty block when the model returned
    # nothing).
    content = payload["message"]["content"]
    assert isinstance(content, list) and len(content) >= 1
    assert content[0]["type"] == "text"
    # ``usage`` is always populated by the translator.
    assert "usage" in payload
    assert "billed_units" in payload["usage"]
    assert "tokens" in payload["usage"]
    # ``finish_reason`` is one of Cohere's enum values.
    assert payload["finish_reason"] in {
        "COMPLETE",
        "MAX_TOKENS",
        "STOP_SEQUENCE",
        "TOOL_CALL",
        "ERROR",
    }


@pytest.mark.asyncio
async def test_cohere_v2_chat_streaming(httpx_client: httpx.AsyncClient):
    """Streaming returns SSE frames in the v2 message-lifecycle shape."""
    events: list[dict] = []
    async with httpx_client.stream(
        "POST",
        "/cohere/v2/chat",
        json={
            "model": SERVED_MODEL_NAME,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8,
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200, await resp.aread()
        assert resp.headers["content-type"].startswith("text/event-stream")
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                events.append({"type": "_DONE_"})
                continue
            events.append(json.loads(data))

    types = [ev["type"] for ev in events]
    # The lifecycle always starts with message-start and ends with [DONE]
    # preceded by message-end.
    assert types[0] == "message-start"
    assert types[-1] == "_DONE_"
    assert types[-2] == "message-end"
    # message-start must carry the chunk/message id.
    assert events[0].get("id")


@pytest.mark.asyncio
async def test_cohere_v2_chat_validation_error_returns_400(
    httpx_client: httpx.AsyncClient,
):
    # Missing required ``model`` field → FastAPI/Pydantic returns 400.
    resp = await httpx_client.post(
        "/cohere/v2/chat",
        json={"messages": []},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_cohere_v2_chat_documents_field_accepted(
    httpx_client: httpx.AsyncClient,
):
    """The v2 endpoint forwards ``documents`` into chat_template_kwargs.

    We only assert the request is accepted and produces a 200 response —
    the renderer-level effect is covered by ``tests/renderers/test_cohere.py``.
    """
    resp = await httpx_client.post(
        "/cohere/v2/chat",
        json={
            "model": SERVED_MODEL_NAME,
            "messages": [{"role": "user", "content": "Summarize."}],
            "documents": [{"id": "d1", "data": {"title": "T", "snippet": "S"}}],
            "max_tokens": 16,
            "stream": False,
        },
    )
    assert resp.status_code == 200, resp.text


# ----------------------------------------------------------------------
# Layer 2: Cohere SDK round-trip (auto-skipped if SDK absent)
# ----------------------------------------------------------------------


@pytest_asyncio.fixture
async def cohere_async_client(server):
    cohere = pytest.importorskip("cohere")
    # The vLLM endpoint is mounted at ``/cohere/v2/chat`` while the
    # cohere SDK targets ``${base_url}/v2/chat``; point base_url at the
    # ``/cohere`` prefix so paths line up.
    client = cohere.AsyncClientV2(
        api_key="dummy",
        base_url=server.url_for("cohere"),
    )
    try:
        yield client
    finally:
        # ``AsyncClientV2`` exposes a sync close; if a future version
        # adds aclose we still close cleanly.
        close = getattr(client, "aclose", None) or getattr(client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result


@pytest.mark.asyncio
async def test_cohere_sdk_non_streaming(cohere_async_client):
    resp = await cohere_async_client.chat(
        model=SERVED_MODEL_NAME,
        messages=[{"role": "user", "content": "Say hi."}],
        max_tokens=16,
    )
    # SDK parses our JSON into typed objects.
    assert resp.message.role == "assistant"
    assert resp.message.content is not None
    assert len(resp.message.content) >= 1
    assert resp.message.content[0].type == "text"
    assert resp.finish_reason in {
        "COMPLETE",
        "MAX_TOKENS",
        "STOP_SEQUENCE",
        "TOOL_CALL",
        "ERROR",
    }


@pytest.mark.asyncio
async def test_cohere_sdk_streaming(cohere_async_client):
    events: list[str] = []
    stream = cohere_async_client.chat_stream(
        model=SERVED_MODEL_NAME,
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=8,
    )
    async for ev in stream:
        events.append(ev.type)

    assert events, "SDK stream yielded no events"
    assert events[0] == "message-start"
    assert events[-1] == "message-end"
