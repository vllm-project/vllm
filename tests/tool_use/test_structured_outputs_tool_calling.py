# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature-combination gate: structured outputs (strict tool calling) x tool calling.

When ``VLLM_ENFORCE_STRICT_TOOL_CALLING`` is enabled -- which is the default -- and a
tool
declares ``strict``, vLLM constrains tool-call arguments through the structural-tag path
(``AbstractToolParser.get_structural_tag`` ->
``vllm.tool_parsers.structural_tag_registry``).
That path is exercised by every ``auto`` / ``required`` / named tool call, but nothing
serves a
model across it: the existing structured-output tool tests are unit tests of *request
shaping*
(``adjust_request``, ``get_structural_tag``, request validation, the tag registry), so a
failure
that only appears when a real server takes the constrained path is invisible to them.

This suite boots the same tool-calling server with the strict constraint enabled and
disabled and
asserts a well-formed, schema-conformant tool call in both configurations.

Scope note: unlike the speculative-decoding and prefix-caching gates, there is no
runtime counter
that proves the *constraint itself* engaged -- vLLM exposes no grammar/structural-tag
metric. This
suite therefore asserts the observable outcome (the combination still produces a
schema-conformant
call) rather than claiming to prove constraint engagement. That is still the regression
that
matters in practice: a dependency skew in the structural-tag import path previously made
every
tool call return HTTP 500 while the request-shaping unit tests stayed green.
"""

from __future__ import annotations

import json

import pytest

from tests.utils import RemoteOpenAIServer

# Qwen2.5's shipped chat template already emits Hermes-style tool calls, so `hermes` is
# the
# documented parser for Qwen models (see "Qwen Models" in this page's docs) -- this
# pairing is
# intentional, not a mismatch, and both arms were verified against a live server.
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

BASE_ARGS: list[str] = [
    "--enforce-eager",
    "--max-model-len",
    "2048",
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "hermes",
]

# The strict constraint is selected by an env var, so each configuration is a separate
# server.
CONFIGS: dict[str, str] = {
    "strict_tool_calling_on": "1",
    "strict_tool_calling_off": "0",
}

# `strict` plus an enum-constrained required field: the shape the structural-tag path
# constrains.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city", "unit"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

PROMPT = "What is the weather in San Francisco in celsius? Use the get_weather tool."


@pytest.fixture(scope="module", params=list(CONFIGS))
def strict_mode_server(request):
    strict = CONFIGS[request.param]
    with RemoteOpenAIServer(
        MODEL, BASE_ARGS, env_dict={"VLLM_ENFORCE_STRICT_TOOL_CALLING": strict}
    ) as remote_server:
        yield request.param, remote_server


@pytest.mark.asyncio
async def test_tool_calling_survives_strict_structured_outputs(strict_mode_server):
    """A schema-conformant tool call must be produced with the strict constraint on and
    off."""
    config_name, remote_server = strict_mode_server
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
    assert choice.finish_reason == "tool_calls", (
        f"{config_name}: {choice.finish_reason}"
    )
    tool_calls = choice.message.tool_calls
    assert tool_calls, f"{config_name}: no tool call was produced"

    names = [c.function.name for c in tool_calls]
    assert "get_weather" in names, f"{config_name}: unexpected tool(s) {names}"

    call = next(c for c in tool_calls if c.function.name == "get_weather")
    arguments = json.loads(call.function.arguments)

    # The declared schema: both fields required, `unit` restricted to an enum.
    assert isinstance(arguments.get("city"), str), (
        f"{config_name}: bad city {arguments}"
    )
    assert arguments.get("unit") in ("celsius", "fahrenheit"), (
        f"{config_name}: 'unit' violates the declared enum: {arguments}"
    )
