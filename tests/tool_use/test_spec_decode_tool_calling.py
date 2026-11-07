# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature-combination gate: speculative decoding x tool calling.

Enabling speculative decoding must never silently disable tool calling. That combination
is
what production agentic deployments actually run, and a regression in it is invisible to
both
the per-parser unit tests (which never boot an engine) and the tool-calling e2e tests
(which
never enable spec-dec) -- so it is only catchable by exercising the two together.

The suite boots the same tool-calling server twice, with speculative decoding OFF and
with
ngram speculative decoding ON (ngram needs no draft model, so this stays cheap), and
asserts:

  1. a tool-eliciting request returns a well-formed tool call in BOTH configurations,
     and
  2. under the spec-dec configuration the drafter was *actually exercised*
     (``vllm:spec_decode_num_draft_tokens_total > 0``).

Assertion 2 is what keeps the gate honest: without it a silently-inactive drafter would
let
the test pass while proving nothing about the combination.
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

NGRAM_SPEC_CONFIG = {
    "method": "ngram",
    "num_speculative_tokens": 4,
    "prompt_lookup_max": 4,
    "prompt_lookup_min": 1,
}

# ngram drafts from repeated n-grams in the context, so the prompt deliberately repeats
# the
# phrasing the answer will reuse -- this is what makes the drafter fire on a short
# request.
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

CONFIGS: dict[str, list[str]] = {
    "spec_decode_off": BASE_ARGS,
    "spec_decode_ngram": BASE_ARGS
    + ["--speculative-config", json.dumps(NGRAM_SPEC_CONFIG)],
}


@pytest.fixture(scope="module", params=list(CONFIGS))
def spec_decode_server(request):
    with RemoteOpenAIServer(MODEL, CONFIGS[request.param]) as remote_server:
        yield request.param, remote_server


# The counter is registered as "vllm:spec_decode_num_draft_tokens"; the text exposition
# conventionally appends "_total" for counters. Accept either spelling so this does not
# depend
# on the exposition detail, while still ignoring sibling series (e.g. "..._created").
_DRAFT_TOKENS_METRIC = "vllm:spec_decode_num_draft_tokens"


def _spec_decode_draft_tokens(remote_server: RemoteOpenAIServer) -> float:
    """Total speculative draft tokens reported by the Prometheus endpoint.

    Sums every matching series: a multi-engine or tensor-parallel deployment emits one
    labelled series per engine, and reading only the first would under-count. The value
    is
    taken as the first field after the label block, because the text format allows an
    optional trailing timestamp that must not be mistaken for the counter.
    """
    metrics = requests.get(remote_server.url_for("metrics"), timeout=30).text
    total = 0.0
    for line in metrics.splitlines():
        if not line.startswith(_DRAFT_TOKENS_METRIC):
            continue
        rest = line[len(_DRAFT_TOKENS_METRIC) :]
        if rest.startswith("_total"):
            rest = rest[len("_total") :]
        if rest[:1] not in ("{", " "):  # e.g. "_created" -> a different series
            continue
        if rest.lstrip().startswith("{"):
            rest = rest[rest.index("}") + 1 :]
        fields = rest.split()
        if fields:
            total += float(fields[0])
    return total


@pytest.mark.asyncio
async def test_tool_calling_survives_speculative_decoding(spec_decode_server):
    """A tool call must still be produced with speculative decoding enabled."""
    config_name, remote_server = spec_decode_server
    async with remote_server.get_async_client() as client:
        chat_completion = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT}],
            tools=TOOLS,
            tool_choice="required",
            temperature=0.0,
            max_completion_tokens=128,
        )

    choice = chat_completion.choices[0]
    tool_calls = choice.message.tool_calls
    assert tool_calls, f"{config_name}: no tool call was produced"
    assert choice.finish_reason == "tool_calls"

    call = tool_calls[0]
    assert call.function.name == "get_weather"
    arguments = json.loads(call.function.arguments)
    assert "city" in arguments, (
        f"{config_name}: tool arguments missing 'city': {arguments}"
    )


@pytest.mark.asyncio
async def test_speculative_decoding_was_actually_exercised(spec_decode_server):
    """Calibration: the spec-dec configuration must really draft tokens.

    Without this the parity assertion above could pass with an inactive drafter, proving
    nothing about the feature combination.
    """
    config_name, remote_server = spec_decode_server
    if config_name == "spec_decode_off":
        # Control arm: proves the counter discriminates rather than being always-on.
        assert _spec_decode_draft_tokens(remote_server) == 0.0
        pytest.skip("drafting check is not applicable without speculative decoding")

    async with remote_server.get_async_client() as client:
        for _ in range(3):
            await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": PROMPT}],
                tools=TOOLS,
                tool_choice="required",
                temperature=0.0,
                max_completion_tokens=128,
            )

    assert _spec_decode_draft_tokens(remote_server) > 0.0, (
        "speculative decoding was configured but drafted no tokens; the "
        "spec-dec x tool-calling combination was not actually exercised"
    )
