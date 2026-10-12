# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import AsyncMock, call

import pytest

from vllm.v1.engine.core_client import DPLBAsyncMPClient

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
async def test_dp_wake_up_requires_all_engines_awake(results, expected):
    client = object.__new__(DPLBAsyncMPClient)
    client.core_engines = [b"engine-0", b"engine-1"]
    client._call_utility_async = AsyncMock(side_effect=results)

    assert await client.wake_up_async(["weights"]) is expected
    assert client._call_utility_async.await_args_list == [
        call("wake_up", ["weights"], engine=b"engine-0"),
        call("wake_up", ["weights"], engine=b"engine-1"),
    ]
