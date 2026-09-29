# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

from vllm.v1.engine.core_client import AsyncMPClient, DPLBAsyncMPClient

pytestmark = pytest.mark.cpu_test


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "results, expected",
    [
        ([True, True], True),
        ([True, False], False),
        ([False, True], False),
        ([False, False], False),
    ],
)
@pytest.mark.parametrize("tags", [None, ["weights"], ["kv_cache", "scheduling"]])
async def test_dp_wake_up_requires_all_engines_awake(results, expected, tags):
    client = SimpleNamespace(
        core_engines=[b"engine-0", b"engine-1"],
        _call_utility_async=AsyncMock(side_effect=results),
    )

    assert await DPLBAsyncMPClient.wake_up_async(client, tags) is expected
    assert client._call_utility_async.await_args_list == [
        call("wake_up", tags, engine=b"engine-0"),
        call("wake_up", tags, engine=b"engine-1"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_result", [None, 0, 1, "true"])
@pytest.mark.parametrize("first_result", [True, False])
async def test_dp_wake_up_rejects_non_bool_from_any_engine(
    invalid_result, first_result
):
    client = SimpleNamespace(
        core_engines=[b"engine-0", b"engine-1"],
        _call_utility_async=AsyncMock(side_effect=[first_result, invalid_result]),
    )

    with pytest.raises(RuntimeError, match="wake_up must return a bool"):
        await DPLBAsyncMPClient.wake_up_async(client)
    assert client._call_utility_async.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_result", [None, 0, 1, "true"])
async def test_single_engine_wake_up_rejects_non_bool(invalid_result):
    client = SimpleNamespace(
        call_utility_async=AsyncMock(return_value=invalid_result),
    )

    with pytest.raises(RuntimeError, match="wake_up must return a bool"):
        await AsyncMPClient.wake_up_async(client)


@pytest.mark.asyncio
async def test_dp_other_utility_keeps_first_engine_result():
    client = SimpleNamespace(
        core_engines=[b"engine-0", b"engine-1"],
        _call_utility_async=AsyncMock(side_effect=[False, True]),
    )

    assert (
        await DPLBAsyncMPClient.call_utility_async(
            client, "reset_prefix_cache", True, False
        )
        is False
    )
    assert client._call_utility_async.await_args_list == [
        call("reset_prefix_cache", True, False, engine=b"engine-0"),
        call("reset_prefix_cache", True, False, engine=b"engine-1"),
    ]
