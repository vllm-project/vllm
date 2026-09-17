# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay fixed chat histories and check per-turn cache hits and metrics."""

import asyncio
import json
from uuid import uuid4

import httpx
import pytest
from openai.types.chat import ChatCompletionMessageParam
from prometheus_client.parser import text_string_to_metric_families

from tests.utils import RemoteOpenAIServer

MODEL = "Qwen/Qwen3.5-0.8B"
NUM_CONVERSATIONS = 8
MAX_TOKENS = 32
# Long first turns leave room for hybrid cache alignment and MTP recomputation.
MIN_PROMPT_TOKENS = 2048
METRICS_TIMEOUT = 30


def _user_turns(conversation: int) -> list[str]:
    records = "\n".join(
        f"Sensor {conversation}-{i}: temperature {20 + i % 10} C, "
        f"humidity {40 + i % 20} percent."
        for i in range(160 + conversation)
    )
    return [
        f"Session {conversation}. Summarize these sensor readings briefly:\n{records}",
        "Which readings should an operator investigate first? Explain briefly.",
        "Suggest two checks the operator should perform next.",
    ]


async def _prefix_cache_counters(
    client: httpx.AsyncClient,
) -> tuple[float, float]:
    response = await client.get("/metrics")
    response.raise_for_status()
    names = ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total")
    counters: dict[str, float] = {}
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            if sample.name in names:
                counters[sample.name] = counters.get(sample.name, 0) + sample.value
    assert all(name in counters for name in names), (
        f"missing prefix-cache metrics: {set(names) - counters.keys()}"
    )
    return counters[names[0]], counters[names[1]]


async def _wait_for_counters(
    client: httpx.AsyncClient, expected: tuple[float, float]
) -> None:
    """Allow engine stats to reach the frontend after completions are returned."""
    deadline = asyncio.get_running_loop().time() + METRICS_TIMEOUT
    while True:
        actual = await _prefix_cache_counters(client)
        if actual == expected:
            return
        assert all(a <= e for a, e in zip(actual, expected)), (
            f"prefix-cache counters exceeded usage: {actual=}, {expected=}"
        )
        assert asyncio.get_running_loop().time() < deadline, (
            f"prefix-cache counters did not match usage: {actual=}, {expected=}"
        )
        await asyncio.sleep(0.1)


@pytest.fixture(scope="module")
def server(request):
    args = ["--enable-prompt-tokens-details"]
    if request.param:
        args += [
            "--speculative-config",
            json.dumps({"method": "mtp", "num_speculative_tokens": 1}),
        ]
    with RemoteOpenAIServer(MODEL, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server, min_hit_rate",
    # MTP re-prefills an extra block; both floors reject one more missed block.
    [pytest.param(False, 0.90, id="base"), pytest.param(True, 0.75, id="mtp")],
    indirect=["server"],
)
async def test_prefix_cache_hit_rate(
    server: RemoteOpenAIServer, min_hit_rate: float
) -> None:
    # Isolate sessions so another conversation cannot hide a cache miss.
    salts = [uuid4().hex for _ in range(NUM_CONVERSATIONS)]
    histories: list[list[ChatCompletionMessageParam]] = [
        [] for _ in range(NUM_CONVERSATIONS)
    ]
    prompt_lengths: list[list[int]] = [[] for _ in range(NUM_CONVERSATIONS)]

    async with (
        server.get_async_client() as client,
        httpx.AsyncClient(base_url=server.url_root, timeout=METRICS_TIMEOUT) as metrics,
    ):

        async def run_conversation(session: int, *, replay: bool) -> tuple[int, int]:
            history = histories[session]
            queries = hits = 0
            for turn, user_text in enumerate(_user_turns(session)):
                if not replay:
                    history.append({"role": "user", "content": user_text})
                # Reuse the original replies, even if replay generates different ones.
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=history[: 2 * turn + 1],
                    temperature=0,
                    max_tokens=MAX_TOKENS,
                    extra_body={"cache_salt": salts[session]},
                )
                usage = response.usage
                assert usage is not None
                assert usage.prompt_tokens_details is not None
                prompt = usage.prompt_tokens
                cached = usage.prompt_tokens_details.cached_tokens
                context = f"{session=}, {turn=}, {replay=}: {cached=}, {prompt=}"
                assert cached is not None, context
                assert 0 <= cached < prompt, context
                assert prompt >= MIN_PROMPT_TOKENS, context
                if not replay and turn == 0:
                    assert cached == 0, context
                else:
                    assert cached / prompt >= min_hit_rate, (
                        f"{context}, {min_hit_rate=}"
                    )
                if replay:
                    assert prompt == prompt_lengths[session][turn], context
                else:
                    prompt_lengths[session].append(prompt)
                    history.append(
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content or "",
                        }
                    )
                queries += prompt
                hits += cached
            return queries, hits

        queries, hits = await _prefix_cache_counters(metrics)
        for replay in (False, True):
            counts = await asyncio.gather(
                *(run_conversation(i, replay=replay) for i in range(NUM_CONVERSATIONS))
            )
            queries += sum(prompt for prompt, _ in counts)
            hits += sum(cached for _, cached in counts)
            await _wait_for_counters(metrics, (queries, hits))
