# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worked example `vllm.endpoint_plugins` entry point.

Reports scheduler config via `collective_rpc`. Demonstrates the full
contract: `attach_router` registers the route at Phase A (`build_app`) and
`init_state` stashes the `EngineClient` the route handler needs at Phase B
(`init_app_state`).

`required_tasks` is `None`, so this plugin is also eligible on the CPU only
render server which has no `EngineClient`. `init_state` is called with
`engine_client=None` in that case and the route handler returns 503 rather
than reaching for a client that doesn't exist.

It also implements the optional lifecycle capabilities. Its session routes
hold one work owner until explicitly released, exercising admission and final
engine work during drain. `events` records lifecycle and owner ordering.
"""

import asyncio

from fastapi import FastAPI, HTTPException, Request


class DummyAdminEndpointPlugin:
    name = "dummy_admin_endpoint_plugin"
    required_tasks: tuple[str, ...] | None = None

    def __init__(self, events: list[str] | None = None) -> None:
        self.events = [] if events is None else events
        self.owner: asyncio.Task | None = None
        self.finish = asyncio.Event()
        self.drain_entered = asyncio.Event()

    def attach_router(self, app: FastAPI) -> None:
        @app.get("/v1/admin/scheduler_config")
        async def scheduler_config(raw_request: Request):
            engine_client = raw_request.app.state.dummy_engine_client
            if engine_client is None:
                raise HTTPException(
                    status_code=503,
                    detail="scheduler_config requires an engine, which this "
                    "server does not have",
                )
            results = await engine_client.collective_rpc("get_scheduler_config")
            return {"scheduler_config": results}

        @app.post("/plugins/dummy/session")
        async def session(request: Request):
            state = request.app.state
            if getattr(state, "draining", False):
                raise HTTPException(status_code=503, detail="draining")
            if self.owner is not None:
                raise HTTPException(status_code=409, detail="owner already accepted")

            async def work() -> bool:
                await self.finish.wait()
                self.events.append("owner_final_submit")
                if state.dummy_engine_client is not None:
                    await state.dummy_engine_client.collective_rpc("finish_session")
                self.events.append("owner_done")
                return getattr(state, "draining", False)

            self.owner = asyncio.create_task(work())
            return {"accepted": True}

        @app.post("/plugins/dummy/finish")
        async def finish():
            if self.owner is None:
                raise HTTPException(status_code=404, detail="no owner")
            self.finish.set()
            return {"finished_while_draining": await self.owner}

    async def init_state(self, engine_client, state, args) -> None:
        state.dummy_engine_client = engine_client

    async def start(self, state) -> None:
        self.events.append(f"start:{self.name}")

    def ready(self, state) -> bool:
        return True

    async def drain(self, state, timeout: float) -> None:
        if self.owner is not None:
            self.events.append("draining")
            self.drain_entered.set()
            try:
                await self.owner
            finally:
                if not self.owner.done():
                    self.owner.cancel()
                    await asyncio.gather(self.owner, return_exceptions=True)

    async def stop(self, state) -> None:
        if self.owner is not None and not self.owner.done():
            self.owner.cancel()
            await asyncio.gather(self.owner, return_exceptions=True)
        self.events.append(f"stop:{self.name}")
