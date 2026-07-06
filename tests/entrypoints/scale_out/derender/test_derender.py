# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the /derender endpoints (postprocessing counterpart to /render)."""

import httpx
import pytest
import pytest_asyncio

from tests.utils import RemoteLaunchRenderServer
from vllm.tokenizers import get_tokenizer

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def server():
    with RemoteLaunchRenderServer(MODEL_NAME, []) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with httpx.AsyncClient(
        base_url=server.url_for(""), timeout=30.0
    ) as http_client:
        yield http_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _render_chat(client: httpx.AsyncClient) -> dict:
    """Render a minimal chat request and return the GenerateRequest dict."""
    resp = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert resp.status_code == 200
    return resp.json()


def _make_generate_response(
    token_ids: list[int] | None,
    request_id: str = "chatcmpl-test-id",
    finish_reason: str = "stop",
    logprobs: dict | None = None,
    prompt_logprobs: list | None = None,
    kv_transfer_params: dict | None = None,
) -> dict:
    choice: dict = {
        "index": 0,
        "token_ids": token_ids,
        "finish_reason": finish_reason,
        "logprobs": logprobs,
    }
    return {
        "request_id": request_id,
        "choices": [choice],
        "prompt_logprobs": prompt_logprobs,
        "kv_transfer_params": kv_transfer_params,
    }


def _make_logprobs_with_placeholders(token_id: int = 1234) -> dict:
    entry = {
        "token": f"token_id:{token_id}",
        "logprob": -1.0,
        "bytes": None,
        "top_logprobs": [
            {"token": f"token_id:{token_id + 1}", "logprob": -2.0, "bytes": None}
        ],
    }
    return {"content": [entry]}


# ---------------------------------------------------------------------------
# Chat derender tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_derender_chat_roundtrip(client):
    """Render then derender: decoded content should be a non-empty string."""
    gen_req = await _render_chat(client)
    # Use the first 5 rendered token IDs as synthetic "generated" tokens.
    synthetic_ids = gen_req["token_ids"][:5]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(synthetic_ids),
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["content"]
    assert data["choices"][0]["message"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_derender_chat_usage(client):
    """Supplied prompt_tokens flows through into usage correctly."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(synthetic_ids),
            "prompt_tokens": 10,
        },
    )
    assert response.status_code == 200
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == len(synthetic_ids)
    assert usage["total_tokens"] == 10 + len(synthetic_ids)


@pytest.mark.asyncio
async def test_derender_chat_usage_default(client):
    """Omitting prompt_tokens gives usage.prompt_tokens == 0."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(synthetic_ids),
        },
    )
    assert response.status_code == 200
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 0


@pytest.mark.asyncio
async def test_derender_chat_logprobs(client):
    """token_id:N placeholders in content.token are resolved to real strings."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]
    token_id = synthetic_ids[0]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(
                synthetic_ids,
                logprobs=_make_logprobs_with_placeholders(token_id),
            ),
        },
    )
    assert response.status_code == 200
    data = response.json()
    logprobs = data["choices"][0]["logprobs"]
    assert logprobs is not None
    content = logprobs["content"]
    assert content is not None and len(content) == 1
    token_str = content[0]["token"]
    assert not token_str.startswith("token_id:"), (
        f"Placeholder was not resolved: {token_str!r}"
    )


@pytest.mark.asyncio
async def test_derender_chat_logprobs_bytes(client):
    """Resolved logprob entries have bytes populated as list[int]."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]
    token_id = synthetic_ids[0]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(
                synthetic_ids,
                logprobs=_make_logprobs_with_placeholders(token_id),
            ),
        },
    )
    assert response.status_code == 200
    content = response.json()["choices"][0]["logprobs"]["content"]
    bytes_field = content[0]["bytes"]
    assert isinstance(bytes_field, list)
    assert len(bytes_field) > 0
    assert all(isinstance(b, int) for b in bytes_field)


@pytest.mark.asyncio
async def test_derender_chat_top_logprobs(client):
    """top_logprobs entries also have their placeholders resolved."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]
    token_id = synthetic_ids[0]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(
                synthetic_ids,
                logprobs=_make_logprobs_with_placeholders(token_id),
            ),
        },
    )
    assert response.status_code == 200
    content = response.json()["choices"][0]["logprobs"]["content"]
    top = content[0]["top_logprobs"]
    assert len(top) == 1
    assert not top[0]["token"].startswith("token_id:"), (
        f"top_logprobs placeholder not resolved: {top[0]['token']!r}"
    )


@pytest.mark.asyncio
async def test_derender_chat_prompt_logprobs_passthrough(client):
    """prompt_logprobs on GenerateResponse passes through unchanged."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]
    # prompt_logprobs is a list[dict[int, Logprob] | None]; use None entries.
    prompt_logprobs = [None, None]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(
                synthetic_ids, prompt_logprobs=prompt_logprobs
            ),
        },
    )
    assert response.status_code == 200
    assert response.json()["prompt_logprobs"] == prompt_logprobs


@pytest.mark.asyncio
async def test_derender_chat_kv_transfer_params_passthrough(client):
    """kv_transfer_params passes through to the ChatCompletionResponse."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]
    kv = {"key": "value"}

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(
                synthetic_ids, kv_transfer_params=kv
            ),
        },
    )
    assert response.status_code == 200
    assert response.json()["kv_transfer_params"] == kv


@pytest.mark.asyncio
async def test_derender_chat_empty_token_ids(client):
    """Empty token_ids list returns 400."""
    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response([]),
        },
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_derender_chat_null_token_ids(client):
    """Null token_ids returns 400."""
    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": _make_generate_response(None),
        },
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_derender_chat_unknown_model(client):
    """Unknown model returns 404."""
    gen_req = await _render_chat(client)
    synthetic_ids = gen_req["token_ids"][:3]

    response = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": "does-not-exist",
            "generate_response": _make_generate_response(synthetic_ids),
        },
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Completion derender tests
# ---------------------------------------------------------------------------


async def _render_completion(client: httpx.AsyncClient, prompt: str) -> dict:
    """Render a completion prompt and return the first GenerateRequest dict."""
    resp = await client.post(
        "/v1/completions/render",
        json={"model": MODEL_NAME, "prompt": prompt},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list) and len(data) >= 1
    return data[0]


def _make_completion_generate_response(
    token_ids: list[int],
    request_id: str,
    kv_transfer_params: dict | None = None,
    logprobs: dict | None = None,
) -> dict:
    return {
        "request_id": request_id,
        "choices": [
            {
                "index": 0,
                "token_ids": token_ids,
                "finish_reason": "stop",
                "logprobs": logprobs,
            }
        ],
        "prompt_logprobs": None,
        "kv_transfer_params": kv_transfer_params,
    }


@pytest.mark.asyncio
async def test_derender_completion_roundtrip(client):
    """Two prompts rendered, two GenerateResponses → two choices with indices 0, 1."""
    gr1 = await _render_completion(client, "Hello world")
    gr2 = await _render_completion(client, "Goodbye world")

    ids1 = gr1["token_ids"][:4]
    ids2 = gr2["token_ids"][:4]

    response = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": [
                _make_completion_generate_response(ids1, gr1["request_id"]),
                _make_completion_generate_response(ids2, gr2["request_id"]),
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "text_completion"
    choices = data["choices"]
    assert len(choices) == 2
    assert choices[0]["index"] == 0
    assert choices[1]["index"] == 1
    assert choices[0]["text"]
    assert choices[1]["text"]


@pytest.mark.asyncio
async def test_derender_completion_usage_aggregation(client):
    """prompt_tokens=[5, 10] is aggregated correctly into usage."""
    gr1 = await _render_completion(client, "Hello")
    gr2 = await _render_completion(client, "World")

    ids1 = gr1["token_ids"][:3]
    ids2 = gr2["token_ids"][:4]

    response = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": [
                _make_completion_generate_response(ids1, gr1["request_id"]),
                _make_completion_generate_response(ids2, gr2["request_id"]),
            ],
            "prompt_tokens": [5, 10],
        },
    )
    assert response.status_code == 200
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 15
    assert usage["completion_tokens"] == len(ids1) + len(ids2)
    assert usage["total_tokens"] == 15 + len(ids1) + len(ids2)


@pytest.mark.asyncio
async def test_derender_completion_prompt_tokens_length_mismatch(client):
    """len(prompt_tokens) != len(generate_responses) returns 400."""
    gr1 = await _render_completion(client, "Hello")
    ids1 = gr1["token_ids"][:3]

    response = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": [
                _make_completion_generate_response(ids1, gr1["request_id"]),
            ],
            "prompt_tokens": [5, 10],
        },
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_derender_completion_empty_generate_responses(client):
    """Empty generate_responses list returns 400."""
    response = await client.post(
        "/v1/completions/derender",
        json={"model": MODEL_NAME, "generate_responses": []},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_derender_completion_logprobs(client):
    """token_id:N placeholders in logprobs are resolved; CompletionLogProbs
    flat-list structure is returned with non-empty tokens and text_offsets."""
    gr1 = await _render_completion(client, "Hello world")
    ids1 = gr1["token_ids"][:3]
    token_id = ids1[0]

    response = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": [
                _make_completion_generate_response(
                    ids1,
                    gr1["request_id"],
                    logprobs=_make_logprobs_with_placeholders(token_id),
                ),
            ],
        },
    )
    assert response.status_code == 200
    logprobs = response.json()["choices"][0]["logprobs"]
    assert logprobs is not None
    tokens = logprobs["tokens"]
    assert len(tokens) == 1
    assert not tokens[0].startswith("token_id:"), (
        f"Placeholder was not resolved: {tokens[0]!r}"
    )
    assert len(logprobs["token_logprobs"]) == 1
    assert isinstance(logprobs["token_logprobs"][0], float)
    assert len(logprobs["text_offset"]) == 1
    assert logprobs["text_offset"][0] == 0


@pytest.mark.asyncio
async def test_derender_completion_kv_transfer_params_passthrough(client):
    """kv_transfer_params passes through to CompletionResponse."""
    gr1 = await _render_completion(client, "Hello")
    ids1 = gr1["token_ids"][:3]
    kv = {"node": "abc"}

    response = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": [
                _make_completion_generate_response(
                    ids1, gr1["request_id"], kv_transfer_params=kv
                ),
            ],
        },
    )
    assert response.status_code == 200
    assert response.json()["kv_transfer_params"] == kv


# ---------------------------------------------------------------------------
# E2E: render -> derender roundtrip with parser (reasoning + tool calls)
# ---------------------------------------------------------------------------

PARSER_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

_E2E_TOOLS = [
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


@pytest.fixture(scope="module")
def parser_server():
    args = [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--reasoning-parser",
        "deepseek_r1",
    ]
    with RemoteLaunchRenderServer(PARSER_MODEL, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def parser_client(parser_server):
    async with httpx.AsyncClient(
        base_url=parser_server.url_for(""), timeout=60.0
    ) as http_client:
        yield http_client


@pytest.fixture(scope="module")
def parser_tokenizer():
    return get_tokenizer(PARSER_MODEL)


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _decoded(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _require_markers_survive(tokenizer, text: str, *markers: str) -> list[int]:
    """Encode text and skip the test if any marker is lost in roundtrip."""
    ids = _encode(tokenizer, text)
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    for m in markers:
        if m not in decoded:
            pytest.skip(f"Marker {m!r} lost in encode->decode roundtrip")
    return ids


async def _e2e_render_chat(
    client: httpx.AsyncClient,
    model: str,
    messages: list[dict],
) -> dict:
    resp = await client.post(
        "/v1/chat/completions/render",
        json={"model": model, "messages": messages},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _e2e_generate_response(
    token_ids: list[int],
    request_id: str = "chatcmpl-e2e-test",
) -> dict:
    return {
        "request_id": request_id,
        "choices": [
            {
                "index": 0,
                "token_ids": token_ids,
                "finish_reason": "stop",
            }
        ],
    }


@pytest.mark.asyncio
async def test_e2e_plain_roundtrip(parser_client, parser_tokenizer):
    """Plain text without reasoning markers roundtrips correctly."""
    messages = [{"role": "user", "content": "What is 2+2?"}]
    gen_req = await _e2e_render_chat(parser_client, PARSER_MODEL, messages)

    answer = "The answer is four."
    output_ids = _encode(parser_tokenizer, answer)
    expected = _decoded(parser_tokenizer, output_ids)

    resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PARSER_MODEL,
            "generate_response": _e2e_generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
        },
    )
    assert resp.status_code == 200, resp.text
    content = resp.json()["choices"][0]["message"]["content"]
    assert content == expected


@pytest.mark.asyncio
async def test_e2e_token_identity(parser_client, parser_tokenizer):
    """encode(derender(token_ids)) == token_ids (RL invariant)."""
    messages = [{"role": "user", "content": "Hi"}]
    gen_req = await _e2e_render_chat(parser_client, PARSER_MODEL, messages)

    answer = "Hello! How can I help?"
    output_ids = _encode(parser_tokenizer, answer)

    resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PARSER_MODEL,
            "generate_response": _e2e_generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
        },
    )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    re_encoded = _encode(parser_tokenizer, content)
    assert output_ids == re_encoded


@pytest.mark.asyncio
async def test_e2e_non_ascii_roundtrip(parser_client, parser_tokenizer):
    """CJK + emoji roundtrip without U+FFFD."""
    messages = [{"role": "user", "content": "Reply in Chinese"}]
    gen_req = await _e2e_render_chat(parser_client, PARSER_MODEL, messages)

    answer = "你好世界 😀"
    output_ids = _encode(parser_tokenizer, answer)

    resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PARSER_MODEL,
            "generate_response": _e2e_generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
        },
    )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert "�" not in content


@pytest.mark.asyncio
async def test_e2e_parsed_reasoning(parser_client, parser_tokenizer):
    """<think>...</think> splits into reasoning + content."""
    messages = [{"role": "user", "content": "What is 2+3?"}]
    gen_req = await _e2e_render_chat(parser_client, PARSER_MODEL, messages)

    reasoning_text = "The user wants 2 plus 3. That is 5."
    answer_text = "The answer is 5."
    output_text = f"<think>{reasoning_text}</think>{answer_text}"
    output_ids = _require_markers_survive(parser_tokenizer, output_text, "</think>")

    resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PARSER_MODEL,
            "generate_response": _e2e_generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
            "chat_request": {
                "model": PARSER_MODEL,
                "messages": messages,
                "include_reasoning": True,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    msg = resp.json()["choices"][0]["message"]
    assert msg["reasoning"] is not None
    assert reasoning_text in msg["reasoning"]
    assert answer_text in msg["content"]
    assert "<think>" not in msg["content"]


@pytest.mark.asyncio
async def test_e2e_parsed_tool_call(parser_client, parser_tokenizer):
    """<tool_call> extracted into tool_calls field."""
    messages = [{"role": "user", "content": "Weather in Paris?"}]
    gen_req = await _e2e_render_chat(parser_client, PARSER_MODEL, messages)

    output_text = (
        "<think>Let me check the weather.</think>"
        '<tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "Paris"}}\n</tool_call>'
    )
    output_ids = _require_markers_survive(
        parser_tokenizer,
        output_text,
        "</think>",
        "<tool_call>",
        "</tool_call>",
    )

    resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PARSER_MODEL,
            "generate_response": _e2e_generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
            "chat_request": {
                "model": PARSER_MODEL,
                "messages": messages,
                "tools": _E2E_TOOLS,
                "tool_choice": "auto",
            },
        },
    )
    assert resp.status_code == 200, resp.text
    choice = resp.json()["choices"][0]
    assert choice["message"]["tool_calls"]
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_e2e_parsed_reasoning_and_tool_call(parser_client, parser_tokenizer):
    """Reasoning + tool call in the same output."""
    messages = [{"role": "user", "content": "Weather in Paris?"}]
    gen_req = await _e2e_render_chat(parser_client, PARSER_MODEL, messages)

    reasoning_text = "I should look up the weather."
    tool_text = (
        '<tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "Paris"}}\n</tool_call>'
    )
    output_text = f"<think>{reasoning_text}</think>{tool_text}"
    output_ids = _require_markers_survive(
        parser_tokenizer, output_text, "</think>", "<tool_call>"
    )

    resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PARSER_MODEL,
            "generate_response": _e2e_generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
            "chat_request": {
                "model": PARSER_MODEL,
                "messages": messages,
                "tools": _E2E_TOOLS,
                "tool_choice": "auto",
                "include_reasoning": True,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    choice = resp.json()["choices"][0]
    assert choice["message"]["reasoning"] is not None
    assert reasoning_text in choice["message"]["reasoning"]
    assert choice["message"]["tool_calls"]


@pytest.mark.asyncio
async def test_e2e_no_chat_request_fallback(parser_client, parser_tokenizer):
    """Without chat_request, derender falls back to plain detokenization."""
    messages = [{"role": "user", "content": "Hello"}]
    gen_req = await _e2e_render_chat(parser_client, PARSER_MODEL, messages)

    answer = "Hi there!"
    output_ids = _encode(parser_tokenizer, answer)

    resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PARSER_MODEL,
            "generate_response": _e2e_generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
        },
    )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert "Hi" in content


# ---------------------------------------------------------------------------
# E2E: HarmonyParser + GPT-OSS
# ---------------------------------------------------------------------------

HARMONY_MODEL = "openai/gpt-oss-20b"


def _ensure_harmony_vocab():
    """Pre-cache the o200k_base BPE file needed by openai-harmony.

    The Rust tiktoken-rs backend downloads from Azure Blob Storage, which
    may be unreachable in some environments.  When the cache is cold we
    fetch the file ourselves and place it in ``/tmp/tiktoken-rs-cache/``
    using the SHA-1(URL) filename that tiktoken-rs expects.
    """
    import hashlib
    import urllib.request
    from pathlib import Path

    url = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
    cache_dir = Path("/tmp/tiktoken-rs-cache")
    cache_key = hashlib.sha1(url.encode()).hexdigest()
    cache_file = cache_dir / cache_key
    if not cache_file.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, cache_file)


@pytest.fixture(scope="module")
def harmony_server():
    _ensure_harmony_vocab()
    args = [
        "--trust-remote-code",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "openai",
        "--reasoning-parser",
        "openai_gptoss",
    ]
    with RemoteLaunchRenderServer(HARMONY_MODEL, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def harmony_client(harmony_server):
    async with httpx.AsyncClient(
        base_url=harmony_server.url_for(""), timeout=60.0
    ) as http_client:
        yield http_client


@pytest.fixture(scope="module")
def harmony_tokenizer():
    return get_tokenizer(HARMONY_MODEL, trust_remote_code=True)


def _harmony_extract_assistant_ids(
    tokenizer, assistant_msg: dict, user_content: str = "test"
) -> list[int]:
    """Extract assistant token IDs via apply_chat_template diff."""
    prompt = [{"role": "user", "content": user_content}]
    full = prompt + [assistant_msg]
    text_prompt = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, tokenize=False
    )
    text_full = tokenizer.apply_chat_template(
        full, add_generation_prompt=False, tokenize=False
    )
    prompt_ids = tokenizer.encode(text_prompt)
    full_ids = tokenizer.encode(text_full)
    assistant_ids = list(full_ids[len(prompt_ids) :])
    if not assistant_ids:
        pytest.skip("Could not extract assistant tokens for Harmony")
    return assistant_ids


@pytest.mark.asyncio
async def test_e2e_harmony_plain_roundtrip(harmony_client, harmony_tokenizer):
    """GPT-OSS content-only roundtrip."""
    messages = [{"role": "user", "content": "What is 2+2?"}]
    gen_req = await _e2e_render_chat(harmony_client, HARMONY_MODEL, messages)

    assistant_msg = {"role": "assistant", "content": "Four."}
    output_ids = _harmony_extract_assistant_ids(harmony_tokenizer, assistant_msg)

    resp = await harmony_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": HARMONY_MODEL,
            "generate_response": _e2e_generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
            "chat_request": {
                "model": HARMONY_MODEL,
                "messages": messages,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    content = resp.json()["choices"][0]["message"]["content"]
    assert content is not None and len(content) > 0
    assert "Four" in content


@pytest.mark.asyncio
async def test_e2e_harmony_reasoning(harmony_client, harmony_tokenizer):
    """GPT-OSS reasoning: analysis channel extracted."""
    messages = [{"role": "user", "content": "Add 2 and 3."}]
    gen_req = await _e2e_render_chat(harmony_client, HARMONY_MODEL, messages)

    reasoning_text = "The user wants 2 plus 3."
    answer_text = "The answer is 5."
    assistant_msg = {
        "role": "assistant",
        "thinking": reasoning_text,
        "content": answer_text,
    }
    output_ids = _harmony_extract_assistant_ids(harmony_tokenizer, assistant_msg)

    decoded = harmony_tokenizer.decode(output_ids)
    if reasoning_text not in decoded:
        pytest.skip("Harmony template did not render thinking")

    resp = await harmony_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": HARMONY_MODEL,
            "generate_response": _e2e_generate_response(output_ids),
            "prompt_tokens": len(gen_req["token_ids"]),
            "chat_request": {
                "model": HARMONY_MODEL,
                "messages": messages,
                "include_reasoning": True,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    msg = resp.json()["choices"][0]["message"]
    assert msg["reasoning"] is not None
    assert reasoning_text in msg["reasoning"]
    assert answer_text in (msg["content"] or "")
