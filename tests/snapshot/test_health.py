# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace
from typing import cast

from fastapi import Request

from vllm.entrypoints.serve.instrumentator.health import health
from vllm.v1.engine.exceptions import EngineDeadError


def _request(state: str, engine) -> Request:
    client = SimpleNamespace(
        request=lambda command, timeout=None: {"state": state}
    )
    return cast(
        Request,
        SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    engine_snapshot_client=client,
                    engine_client=engine,
                )
            )
        ),
    )


def test_health_is_alive_while_snapshot_is_not_ready():
    async def run():
        checked = False

        class Engine:
            async def check_health(self) -> None:
                nonlocal checked
                checked = True

        response = await health(_request("HIBERNATED", Engine()))
        assert response.status_code == 200
        assert not checked

    asyncio.run(run())


def test_health_reports_dead_engine_when_snapshot_is_ready():
    async def run():
        class Engine:
            async def check_health(self) -> None:
                raise EngineDeadError("engine died")

        response = await health(_request("READY", Engine()))
        assert response.status_code == 503

    asyncio.run(run())
