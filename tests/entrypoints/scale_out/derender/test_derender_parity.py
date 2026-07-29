# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Round trip parity CI: render -> generate -> derender -> render == render.

Pins the coupled parsing path (``/v1/chat/completions`` on a normal GPU
server which parses generated tokens incrementally as they stream out)
against the disaggregated path (``OnlineDerenderer.derender_chat`` via
``/v1/chat/completions/derender`` which parses the same tokens all at once,
out of process). A standard ``vllm serve`` GPU server mounts both, so one
real generation lets both parsers run on identical input.

Both paths consume the same generated token IDs (extracted from the coupled
response's ``token_ids`` via ``return_token_ids=True``), so generation
nondeterminism is irrelevant. The only variable under test is whether the
two parsing code paths agree. Parity is asserted unconditionally as only the
stronger per case assertions (e.g. "a tool call was produced") are gated
behind the marker actually having been emitted since a 1.5B model is not
guaranteed to emit ``<think>`` / ``<tool_call>``.
"""

import json

import httpx
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ARGS = [
    "--enable-auto-tool-choice",
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
    with ``token_id:N`` placeholders (``return_tokens_as_token_ids=True``
    reproduces that shape here). ``/derender`` must resolve those
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
