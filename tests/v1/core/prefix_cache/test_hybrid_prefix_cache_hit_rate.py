# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay fixed chat histories and check per-turn cache hits and metrics."""

import asyncio
import json
from math import lcm
from uuid import uuid4

import httpx
import pytest
from openai.types.chat import ChatCompletionMessageParam
from prometheus_client.parser import text_string_to_metric_families

from tests.utils import RemoteOpenAIServer
from vllm.utils.math_utils import round_down

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


async def _scheduler_block_size(client: httpx.AsyncClient) -> int:
    """Return the coarsest boundary an expected cache hit can land on.

    Mirrors ``scheduler_block_size`` in ``resolve_kv_cache_block_sizes``, which
    the engine keeps internal: the LCM of the resolved group block sizes. A
    finer ``prefix_match_unit`` only adds hits, so this stays conservative.
    """
    response = await client.get("/metrics")
    response.raise_for_status()
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            if sample.name != "vllm:cache_config_info":
                continue
            sizes = [
                int(value)
                for name in ("block_size", "mamba_block_size")
                if (value := sample.labels.get(name, "None")) != "None"
            ]
            assert sizes, f"no block size in {sample.labels}"
            return lcm(*sizes)
    raise AssertionError("missing vllm:cache_config_info metric")


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
    "server, dropped_blocks",
    # MTP's prefill lookahead pollutes the tail block, so EAGLE drops one.
    [pytest.param(False, 0, id="base"), pytest.param(True, 1, id="mtp")],
    indirect=["server"],
)
async def test_prefix_cache_hit_rate(
    server: RemoteOpenAIServer, dropped_blocks: int
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
        block_size = await _scheduler_block_size(metrics)
        print(f"{block_size=}")

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
                if replay:
                    assert prompt == prompt_lengths[session][turn], context
                    # The identical request computed this whole prompt, but at
                    # least one token is always recomputed.
                    computed = prompt - 1
                elif turn:
                    computed = prompt_lengths[session][turn - 1]
                else:
                    assert cached == 0, context
                    computed = 0
                # Every token below the last reachable boundary must be reused.
                floor = max(
                    0, round_down(computed, block_size) - dropped_blocks * block_size
                )
                print(
                    f"{session=} {turn=} {replay=} {prompt=} {cached=} {floor=} "
                    f"hit_rate={cached / prompt:.4f}"
                )
                assert cached >= floor, (
                    f"{context}, {computed=}, {block_size=}, "
                    f"hit_rate={cached / prompt:.4f}, "
                    f"expected_hit_rate={floor / prompt:.4f}"
                )
                if not replay:
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
