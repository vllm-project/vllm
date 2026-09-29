# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DP clients combine per-engine utility results like the Rust client."""

from unittest.mock import AsyncMock, call

import pytest

from vllm.v1.engine.core_client import DPLBAsyncMPClient

pytestmark = pytest.mark.cpu_test

ENGINES = [b"engine-0", b"engine-1"]


def dp_client(results):
    client = object.__new__(DPLBAsyncMPClient)
    client.core_engines = ENGINES
    client._call_utility_async = AsyncMock(side_effect=results)
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,args,utility",
    [
        ("wake_up_async", (["weights"],), "wake_up"),
        ("reset_prefix_cache_async", (False, False), "reset_prefix_cache"),
        ("add_lora_async", ("lora",), "add_lora"),
        ("remove_lora_async", (1,), "remove_lora"),
        ("pin_lora_async", (1,), "pin_lora"),
    ],
)
@pytest.mark.parametrize(
    "results,expected",
    [([True, True], True), ([True, False], False), ([False, True], False)],
)
async def test_succeeds_only_if_every_engine_does(
    method, args, utility, results, expected
):
    client = dp_client(results)
    assert await getattr(client, method)(*args) is expected
    assert client._call_utility_async.await_args_list == [
        call(utility, *args, engine=engine) for engine in ENGINES
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,results",
    [
        ("is_sleeping_async", [True, True]),
        ("is_scheduler_paused_async", [False, False]),
        ("get_weight_version_async", ["v1", "v1"]),
    ],
)
async def test_consensus_returns_the_shared_result(method, results):
    assert await getattr(dp_client(results), method)() == results[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,results",
    [
        ("is_sleeping_async", [True, False]),
        ("is_scheduler_paused_async", [False, True]),
        ("get_weight_version_async", ["v1", "v2"]),
    ],
)
async def test_consensus_rejects_disagreeing_engines(method, results):
    with pytest.raises(RuntimeError, match="different"):
        await getattr(dp_client(results), method)()
