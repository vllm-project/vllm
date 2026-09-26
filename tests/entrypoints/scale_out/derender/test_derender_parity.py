# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Round trip parity CI: render -> generate -> derender -> render == render.

Pins the coupled parsing path (`/v1/chat/completions` on a normal GPU
server which parses generated tokens incrementally as they stream out)
against the disaggregated path (`OnlineDerenderer.derender_chat` via
`/v1/chat/completions/derender` which parses the same tokens all at once,
out of process). A standard `vllm serve` GPU server mounts both, so one
real generation lets both parsers run on identical input.

Both paths consume the same generated token IDs (extracted from the coupled
response's `token_ids` via `return_token_ids=True`), so generation
nondeterminism is irrelevant. The only variable under test is whether the
two parsing code paths agree. Parity is asserted unconditionally as only the
stronger per case assertions (e.g. "a tool call was produced") are gated
behind the marker actually having been emitted since a 1.5B model is not
guaranteed to emit `<think>` / `<tool_call>`.

The streaming cases do the same for `stream=true`. Each coupled chunk's
`token_ids` becomes one `/inference/v1/generate` stream chunk, fed
through the chunked derender path with `stream_state` threaded across calls
the way a client would, so both parsers see the same chunk boundaries.

The text level cases pin `/inference/v1/generate` with `output_mode="text"`
against `/v1/completions/derender` on the same token IDs, batch and streaming,
plus logprob resolution and stop string handling.
"""

import json

import httpx
import pytest
import pytest_asyncio
import regex as re

from tests.entrypoints.scale_out.derender.utils import (
    assemble_stream,
    stream_chat_derender,
)
from tests.utils import RemoteOpenAIServer

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ARGS = [
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "hermes",
    "--reasoning-parser",
    "deepseek_r1",
    "--enable-scale-out",
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
def server():
    with RemoteOpenAIServer(MODEL, ARGS) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with httpx.AsyncClient(
        base_url=server.url_for(""), timeout=60.0
    ) as http_client:
        yield http_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _coupled(client: httpx.AsyncClient, messages: list[dict], **extra) -> dict:
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 128,
            "return_token_ids": True,
            **extra,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _disagg(
    client: httpx.AsyncClient,
    output_ids: list[int],
    prompt_tokens: int,
    finish_reason: str,
    chat_request: dict,
    logprobs: dict | None = None,
) -> dict:
    resp = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL,
            "generate_response": {
                "request_id": "parity",
                "choices": [
                    {
                        "index": 0,
                        "token_ids": output_ids,
                        "finish_reason": finish_reason,
                        "logprobs": logprobs,
                    }
                ],
            },
            "prompt_tokens": prompt_tokens,
            "chat_request": chat_request,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _tool_sig(response_choice: dict) -> list[tuple[str, dict]]:
    """[(name, json normalized args)] so key ordering / whitespace don't
    cause false negatives."""
    return [
        (tc["function"]["name"], json.loads(tc["function"]["arguments"]))
        for tc in (response_choice["message"].get("tool_calls") or [])
    ]


def _assert_parity(coupled: dict, disagg: dict) -> None:
    """Both paths saw the same tokens, so they must agree unconditionally."""
    c, d = coupled["choices"][0], disagg["choices"][0]
    assert d["message"]["content"] == c["message"]["content"]
    assert d["message"].get("reasoning") == c["message"].get("reasoning")
    assert _tool_sig(d) == _tool_sig(c)
    assert d["finish_reason"] == c["finish_reason"]
    assert disagg["usage"]["prompt_tokens"] == coupled["usage"]["prompt_tokens"]
    assert disagg["usage"]["completion_tokens"] == len(c["token_ids"])


async def _run_parity_case(
    client: httpx.AsyncClient, messages: list[dict], **extra
) -> tuple[dict, dict]:
    """Run the coupled request then feed its generated tokens into the
    disaggregated derender endpoint. Returns (coupled, disagg)."""
    coupled = await _coupled(client, messages, **extra)
    ch = coupled["choices"][0]
    chat_request = {"model": MODEL, "messages": messages, **extra}
    disagg = await _disagg(
        client,
        ch["token_ids"],
        coupled["usage"]["prompt_tokens"],
        ch["finish_reason"],
        chat_request,
    )
    return coupled, disagg


# ---------------------------------------------------------------------------
# Parity cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parity_plain(client):
    """Plain detokenization parity. No reasoning/tool markers involved."""
    messages = [
        {"role": "user", "content": "What is 2+2? Answer in one short sentence."}
    ]
    coupled, disagg = await _run_parity_case(client, messages)
    _assert_parity(coupled, disagg)


@pytest.mark.asyncio
async def test_parity_reasoning(client):
    """Reasoning/content split parity for <think>...</think> outputs."""
    messages = [{"role": "user", "content": "What is 17 times 23? Think it through."}]
    coupled, disagg = await _run_parity_case(
        client, messages, include_reasoning=True, max_tokens=256
    )
    _assert_parity(coupled, disagg)

    if not coupled["choices"][0]["message"].get("reasoning"):
        pytest.skip("Model did not emit a <think> block")
    assert disagg["choices"][0]["message"]["reasoning"]


@pytest.mark.asyncio
async def test_parity_tool_call(client):
    """Tool call name+args parity."""
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    coupled, disagg = await _run_parity_case(
        client, messages, tools=TOOLS, tool_choice=FORCE_WEATHER_TOOL, max_tokens=1024
    )
    _assert_parity(coupled, disagg)

    if not _tool_sig(coupled["choices"][0]):
        pytest.skip("Model did not emit a tool call")
    assert _tool_sig(disagg["choices"][0])


@pytest.mark.asyncio
async def test_parity_reasoning_and_tool_call(client):
    """Combined reasoning + tool call parity means the highest drift risk
    since it exercises both parser branches on the same output."""
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    coupled, disagg = await _run_parity_case(
        client,
        messages,
        tools=TOOLS,
        tool_choice=FORCE_WEATHER_TOOL,
        include_reasoning=True,
        max_tokens=1024,
    )
    _assert_parity(coupled, disagg)

    c_msg = coupled["choices"][0]["message"]
    if not (c_msg.get("reasoning") and _tool_sig(coupled["choices"][0])):
        pytest.skip("Model did not emit both a <think> block and a tool call")
    d_msg = disagg["choices"][0]["message"]
    assert d_msg["reasoning"]
    assert _tool_sig(disagg["choices"][0])


@pytest.mark.asyncio
async def test_parity_logprobs(client):
    """token_id:N resolution parity vs. the coupled server's real strings.

    A real disaggregated worker only has token IDs so it emits logprobs
    with `token_id:N` placeholders (`return_tokens_as_token_ids=True`
    reproduces that shape here). `/derender` must resolve those
    placeholders to the same token strings/bytes the coupled server
    resolves them to directly.
    """
    messages = [{"role": "user", "content": "What is 2+2?"}]
    extra = {"logprobs": True, "top_logprobs": 3}

    # What a real GPU less worker would hand to /derender is token IDs plus
    # logprobs still in token_id:N placeholder form
    placeholder = await _coupled(
        client, messages, return_tokens_as_token_ids=True, **extra
    )
    ch = placeholder["choices"][0]
    chat_request = {"model": MODEL, "messages": messages, **extra}
    disagg = await _disagg(
        client,
        ch["token_ids"],
        placeholder["usage"]["prompt_tokens"],
        ch["finish_reason"],
        chat_request,
        logprobs=ch["logprobs"],
    )

    # The coupled server resolving the same greedy generation
    # to real token strings itself.
    resolved = await _coupled(client, messages, **extra)
    assert resolved["choices"][0]["token_ids"] == ch["token_ids"], (
        "greedy (temperature=0) generation was expected to be deterministic "
        "across the two coupled calls used to build this test's fixtures"
    )
    _assert_parity(resolved, disagg)

    r_content = resolved["choices"][0]["logprobs"]["content"]
    d_content = disagg["choices"][0]["logprobs"]["content"]
    assert len(d_content) == len(r_content)
    for d_entry, r_entry in zip(d_content, r_content):
        assert d_entry["token"] == r_entry["token"]
        assert d_entry["bytes"] == r_entry["bytes"]


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


async def _coupled_stream(
    client: httpx.AsyncClient, messages: list[dict], **extra
) -> tuple[list[dict], list[int]]:
    """Stream `/v1/chat/completions` and return the choices that carry
    `token_ids` along with the prompt token IDs.

    The leading role only chunk has no `token_ids` and no counterpart on
    `/inference/v1/generate`, so it is dropped.
    """
    choices: list[dict] = []
    prompt_token_ids: list[int] | None = None
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 128,
            "return_token_ids": True,
            "stream": True,
            **extra,
        },
    ) as resp:
        assert resp.status_code == 200, await resp.aread()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            assert "error" not in chunk, chunk
            if chunk.get("prompt_token_ids") is not None:
                prompt_token_ids = chunk["prompt_token_ids"]
            choices.extend(
                ch for ch in chunk["choices"] if ch.get("token_ids") is not None
            )
    assert prompt_token_ids is not None
    return choices, prompt_token_ids


def _tool_call_id_shape(tool_call_id: str | None) -> str | None:
    """Drop the random suffix so IDs compare by format, e.g. `chatcmpl-tool-`.
    Deterministic IDs such as kimi_k2's `functions.<name>:<idx>` are kept
    whole."""
    if tool_call_id is None:
        return None
    return re.sub(r"[0-9a-f]{16}$", "", tool_call_id)


async def _run_stream_parity_case(
    client: httpx.AsyncClient,
    messages: list[dict],
    finish_only_tail: bool = False,
    **extra,
) -> tuple[dict, list[dict]]:
    """Stream the coupled request, replay its chunks through the streaming
    derender endpoint and assert the assembled messages agree. Returns the
    assembled coupled message and the coupled choices.

    Derender gets the engine's finish reason, since `"tool_calls"` is a chat
    level rewrite of `"stop"` that derender has to reapply itself. With
    `finish_only_tail` it goes on a trailing chunk with no tokens.
    """
    coupled_choices, prompt_token_ids = await _coupled_stream(client, messages, **extra)
    finish_reason = coupled_choices[-1]["finish_reason"]
    disagg = await stream_chat_derender(
        client,
        [tid for ch in coupled_choices for tid in ch["token_ids"]],
        [len(ch["token_ids"]) for ch in coupled_choices]
        + ([0] if finish_only_tail else []),
        {"model": MODEL, "messages": messages, **extra},
        len(prompt_token_ids),
        prompt_token_ids,
        finish_reason="stop" if finish_reason == "tool_calls" else finish_reason,
    )
    coupled = assemble_stream(coupled_choices)
    for message in (coupled, disagg):
        for tc in message["tool_calls"]:
            tc["id"] = _tool_call_id_shape(tc["id"])
    assert disagg == coupled
    return coupled, coupled_choices


# ---------------------------------------------------------------------------
# Streaming parity cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_parity_plain(client):
    """Plain streamed detokenization parity."""
    messages = [
        {"role": "user", "content": "What is 2+2? Answer in one short sentence."}
    ]
    await _run_stream_parity_case(client, messages)


@pytest.mark.asyncio
async def test_stream_parity_reasoning(client):
    """Streamed reasoning/content split parity."""
    messages = [{"role": "user", "content": "What is 17 times 23? Think it through."}]
    coupled, _ = await _run_stream_parity_case(
        client, messages, include_reasoning=True, max_tokens=256
    )

    if not coupled["reasoning"]:
        pytest.skip("Model did not emit a <think> block")


@pytest.mark.asyncio
async def test_stream_parity_tool_call(client):
    """Streamed tool call parity under `tool_choice="auto"`, where a tool
    call also rewrites `finish_reason` to `"tool_calls"`."""
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    coupled, _ = await _run_stream_parity_case(
        client, messages, tools=TOOLS, tool_choice="auto", max_tokens=512
    )

    if not coupled["tool_calls"]:
        pytest.skip("Model did not emit a tool call")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunking", ["engine_steps", "multi_token", "finish_only_tail"]
)
async def test_stream_parity_reasoning_and_tool_call(client, chunking):
    """Streamed reasoning + forced tool call parity across chunk shapes.

    `multi_token` batches tokens per chunk with `stream_interval` the way
    speculative decoding does, so replay has to reuse the original chunk
    boundaries. `finish_only_tail` sends the finish reason on a chunk with no
    tokens, which must still flush buffered tool call arguments.
    """
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    extra: dict = {
        "tools": TOOLS,
        "tool_choice": FORCE_WEATHER_TOOL,
        "include_reasoning": True,
        "max_tokens": 512,
    }
    if chunking == "multi_token":
        extra["stream_interval"] = 4
    coupled, coupled_choices = await _run_stream_parity_case(
        client,
        messages,
        finish_only_tail=chunking == "finish_only_tail",
        **extra,
    )

    if chunking == "multi_token":
        assert any(len(ch["token_ids"]) > 1 for ch in coupled_choices)
    if not (coupled["reasoning"] and coupled["tool_calls"]):
        pytest.skip("Model did not emit both a <think> block and a tool call")


# ---------------------------------------------------------------------------
# Inline text level: generate(output_mode="text") == derender(generate(...))
# ---------------------------------------------------------------------------

TEXT_MESSAGES = [
    {"role": "user", "content": "What is 2+2? Answer in one short sentence."}
]


async def _render_token_ids(
    client: httpx.AsyncClient, messages: list[dict]
) -> list[int]:
    resp = await client.post(
        "/v1/chat/completions/render", json={"model": MODEL, "messages": messages}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["token_ids"]


def _generate_payload(
    token_ids: list[int], output_mode: str, stream: bool = False, **sampling
) -> dict:
    return {
        "model": MODEL,
        "token_ids": token_ids,
        "sampling_params": {"temperature": 0, "max_tokens": 128, **sampling},
        "output_mode": output_mode,
        "stream": stream,
    }


async def _generate(
    client: httpx.AsyncClient, token_ids: list[int], output_mode: str, **sampling
) -> dict:
    resp = await client.post(
        "/inference/v1/generate",
        json=_generate_payload(token_ids, output_mode, **sampling),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["output_mode"] == output_mode
    return data


async def _generate_text_stream(
    client: httpx.AsyncClient, token_ids: list[int], **sampling
) -> list[dict]:
    """Stream generate at the text level and return the choice of every chunk
    that has one."""
    choices: list[dict] = []
    async with client.stream(
        "POST",
        "/inference/v1/generate",
        json=_generate_payload(token_ids, "text", stream=True, **sampling),
    ) as resp:
        assert resp.status_code == 200, await resp.aread()
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: ") :])
            assert "error" not in chunk, chunk
            assert chunk["output_mode"] == "text"
            choices.extend(chunk["choices"])
    return choices


async def _derender_completion_text(
    client: httpx.AsyncClient, output_ids: list[int], finish_reason: str
) -> str:
    resp = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL,
            "generate_responses": [
                {
                    "request_id": "text-parity",
                    "choices": [
                        {
                            "index": 0,
                            "token_ids": output_ids,
                            "finish_reason": finish_reason,
                        }
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["choices"][0]["text"]


async def _derender_completion_stream_text(
    client: httpx.AsyncClient, choices: list[dict]
) -> str:
    """Replay each generate chunk's token IDs through the streaming
    completions derender endpoint with the same chunk boundaries."""
    state = None
    text = ""
    for choice in choices:
        resp = await client.post(
            "/v1/completions/derender",
            json={
                "stream": True,
                "model": MODEL,
                "generate_chunk": {
                    "request_id": "text-parity-stream",
                    "choices": [
                        {
                            "index": 0,
                            "token_ids": choice.get("token_ids") or [],
                            "finish_reason": choice.get("finish_reason"),
                        }
                    ],
                },
                "stream_state": state,
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        state = data["stream_state"]
        text += "".join(ch["text"] for ch in data["chunk"]["choices"])
    return text


@pytest.mark.asyncio
async def test_text_parity_batch(client):
    """Inline text matches /v1/completions/derender on the same token IDs."""
    token_ids = await _render_token_ids(client, TEXT_MESSAGES)
    inline = (await _generate(client, token_ids, "text"))["choices"][0]

    derendered = await _derender_completion_text(
        client, inline["token_ids"], inline["finish_reason"]
    )

    assert inline["text"]
    assert inline["text"] == derendered


@pytest.mark.asyncio
async def test_text_parity_stream(client):
    """Streamed inline text matches the streaming derender replay of the same
    chunks and the non-streaming inline text."""
    token_ids = await _render_token_ids(client, TEXT_MESSAGES)
    choices = await _generate_text_stream(client, token_ids)
    inline = "".join(ch["text"] for ch in choices)

    assert choices[-1]["finish_reason"] is not None
    assert inline == await _derender_completion_stream_text(client, choices)
    batch = (await _generate(client, token_ids, "text"))["choices"][0]
    assert [t for ch in choices for t in ch.get("token_ids") or []] == (
        batch["token_ids"]
    )
    assert inline == batch["text"]


@pytest.mark.asyncio
async def test_text_parity_logprobs(client):
    """Inline logprob tokens and bytes match /derender resolving the
    token_id:N placeholders of the tokens level."""
    token_ids = await _render_token_ids(client, TEXT_MESSAGES)
    sampling = {"logprobs": 3}
    tokens_choice = (await _generate(client, token_ids, "tokens", **sampling))[
        "choices"
    ][0]
    text_choice = (await _generate(client, token_ids, "text", **sampling))["choices"][0]
    assert text_choice["token_ids"] == tokens_choice["token_ids"], (
        "greedy (temperature=0) generation was expected to be deterministic "
        "across the two generate calls"
    )

    disagg = await _disagg(
        client,
        tokens_choice["token_ids"],
        len(token_ids),
        tokens_choice["finish_reason"],
        {"model": MODEL, "messages": TEXT_MESSAGES},
        logprobs=tokens_choice["logprobs"],
    )

    inline_content = text_choice["logprobs"]["content"]
    derender_content = disagg["choices"][0]["logprobs"]["content"]
    assert len(inline_content) == len(derender_content)
    for inline_entry, derender_entry in zip(inline_content, derender_content):
        assert inline_entry["token"] == derender_entry["token"]
        assert inline_entry["bytes"] == derender_entry["bytes"]
        assert [
            (top["token"], top["bytes"]) for top in inline_entry["top_logprobs"]
        ] == [(top["token"], top["bytes"]) for top in derender_entry["top_logprobs"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_text_parity_stop_string(client, stream):
    """Inline text follows the engine's stop handling, like /v1/completions.

    token_ids keep every generated token, so /derender brings the matched
    stop string back. Its text must start with the inline text and only
    differ after it.
    """
    stop = "\n"
    token_ids = await _render_token_ids(
        client,
        [{"role": "user", "content": "Write two short lines about the sea."}],
    )

    resp = await client.post(
        "/v1/completions",
        json={
            "model": MODEL,
            "prompt": token_ids,
            "temperature": 0,
            "max_tokens": 128,
            "stop": [stop],
        },
    )
    assert resp.status_code == 200, resp.text
    coupled = resp.json()["choices"][0]

    if stream:
        choices = await _generate_text_stream(client, token_ids, stop=[stop])
        inline_text = "".join(ch["text"] for ch in choices)
        output_ids = [t for ch in choices for t in ch.get("token_ids") or []]
        finish_reason = choices[-1]["finish_reason"]
    else:
        inline = (await _generate(client, token_ids, "text", stop=[stop]))["choices"][0]
        inline_text = inline["text"]
        output_ids = inline["token_ids"]
        finish_reason = inline["finish_reason"]

    assert inline_text == coupled["text"]
    assert finish_reason == coupled["finish_reason"]
    assert stop not in inline_text

    derendered = await _derender_completion_text(client, output_ids, finish_reason)
    assert derendered.startswith(inline_text)
    if coupled["stop_reason"] != stop:
        pytest.skip("Model did not emit the stop string")
    assert derendered[len(inline_text) :].startswith(stop)
