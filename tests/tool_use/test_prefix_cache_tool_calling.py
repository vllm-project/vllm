# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature-combination gate: prefix caching x tool calling.

Enabling prefix caching must never silently disable tool calling. Long-lived agentic
deployments run both together -- repeated system prompts and tool definitions are
exactly the prefix a cache is meant to reuse -- yet the combination is not exercised
anywhere: the per-parser tests never boot an engine, and the tool-calling e2e tests
never enable prefix caching.

The suite boots the same tool-calling server twice, with prefix caching off and on, and
asserts:

  1. a tool-eliciting request still returns a well-formed tool call in BOTH
     configurations, and
  2. under the caching configuration the cache was *actually exercised*
     (``vllm:prefix_cache_queries`` and ``vllm:prefix_cache_hits`` are non-zero), and is
     untouched without it.

Assertion 2 is what keeps the gate honest: a cache that never engaged would let the
test pass while proving nothing about the combination. The same prompt is sent several
times so the cache has something to hit.
"""

from __future__ import annotations

import json

import pytest
import requests

from tests.utils import RemoteOpenAIServer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

BASE_ARGS: list[str] = [
    "--enforce-eager",
    "--max-model-len",
    "4096",
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "hermes",
]

CONFIGS: dict[str, list[str]] = {
    "prefix_cache_off": BASE_ARGS + ["--no-enable-prefix-caching"],
    "prefix_cache_on": BASE_ARGS + ["--enable-prefix-caching"],
}

# Repeated phrasing so there is a substantial shared prefix across the repeated
# requests.
PROMPT = (
    "The weather in San Francisco. The weather in San Francisco. "
    "What is the weather in San Francisco? Use the get_weather tool for San Francisco."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


@pytest.fixture(scope="module", params=list(CONFIGS))
def prefix_cache_server(request):
    with RemoteOpenAIServer(MODEL, CONFIGS[request.param]) as remote_server:
        yield request.param, remote_server


def _counter(remote_server: RemoteOpenAIServer, name: str) -> float:
    """Sum a Prometheus counter across every matching series.

    Accepts the exposition with or without a ``_total`` suffix and ignores sibling
    series
    such as ``_created``. Sums rather than returning the first match, because a
    multi-engine or tensor-parallel deployment emits one labelled series per engine. The
    value is the first field after the label block: the text format permits an optional
    trailing timestamp that must not be read as the counter.
    """
    metrics = requests.get(remote_server.url_for("metrics"), timeout=30).text
    total = 0.0
    for line in metrics.splitlines():
        if not line.startswith(name):
            continue
        rest = line[len(name) :]
        if rest.startswith("_total"):
            rest = rest[len("_total") :]
        if rest[:1] not in ("{", " "):
            continue
        if rest.lstrip().startswith("{"):
            rest = rest[rest.index("}") + 1 :]
        fields = rest.split()
        if fields:
            total += float(fields[0])
    return total


async def _send(client):
    """Issue the tool-eliciting request. Deliberately asserts nothing: used to warm the
    cache, where a model-side hiccup must not surface as a confusing tool-call failure
    in
    a test whose subject is the cache counters."""
    return await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        tools=TOOLS,
        tool_choice="required",
        temperature=0.0,
        max_completion_tokens=128,
    )


async def _make_tool_call(client) -> list:
    """Send the request and assert it produced a well-formed tool call."""
    chat_completion = await _send(client)
    choice = chat_completion.choices[0]
    assert choice.finish_reason == "tool_calls"
    return choice.message.tool_calls or []


@pytest.mark.asyncio
async def test_tool_calling_survives_prefix_caching(prefix_cache_server):
    """A well-formed tool call must still be produced with prefix caching enabled."""
    config_name, remote_server = prefix_cache_server
    async with remote_server.get_async_client() as client:
        tool_calls = await _make_tool_call(client)

    assert tool_calls, f"{config_name}: no tool call was produced"
    names = [c.function.name for c in tool_calls]
    assert "get_weather" in names, f"{config_name}: unexpected tool(s) {names}"
    call = next(c for c in tool_calls if c.function.name == "get_weather")
    arguments = json.loads(call.function.arguments)
    assert "city" in arguments, (
        f"{config_name}: tool arguments missing 'city': {arguments}"
    )


@pytest.mark.asyncio
async def test_prefix_cache_was_actually_exercised(prefix_cache_server):
    """Calibration: the caching configuration must really query and hit the prefix
    cache.

    Without this the assertion above could pass with an inactive cache, proving nothing
    about
    the feature combination.
    """
    config_name, remote_server = prefix_cache_server
    async with remote_server.get_async_client() as client:
        # repeat so later requests can hit the prefix cached by earlier ones
        for _ in range(3):
            await _make_tool_call(client)

    queries = _counter(remote_server, "vllm:prefix_cache_queries")
    hits = _counter(remote_server, "vllm:prefix_cache_hits")

    if config_name == "prefix_cache_off":
        # Control arm: proves the counters discriminate rather than being always-on.
        assert queries == 0.0 and hits == 0.0, (
            f"prefix cache reported activity while disabled: {queries=} {hits=}"
        )
        pytest.skip("cache-hit check is not applicable with prefix caching disabled")

    assert queries > 0.0, "prefix caching was enabled but the cache was never queried"
    assert hits > 0.0, (
        "prefix caching was enabled and queried but never hit; the prefix caching x "
        "tool-calling combination was not actually exercised"
    )
