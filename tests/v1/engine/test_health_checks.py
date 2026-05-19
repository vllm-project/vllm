# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import pytest

from vllm.v1.engine.core_client import AsyncMPClient
from vllm.v1.engine.exceptions import EngineDeadError


@pytest.mark.asyncio
async def test_async_mp_client_check_health_timeout(monkeypatch: pytest.MonkeyPatch):
    client = object.__new__(AsyncMPClient)

    async def never_returns(*_args, **_kwargs):
        await asyncio.Future()

    client.call_utility_async = never_returns  # type: ignore[method-assign]
    monkeypatch.setattr("vllm.v1.engine.core_client.envs.VLLM_HEALTH_CHECK_TIMEOUT", 1)

    with pytest.raises(EngineDeadError, match="did not respond to health ping"):
        await client.check_health_async()
