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
from uuid import uuid4

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
        async def session(request: Request, real_generation: bool = False):
            state = request.app.state
            if getattr(state, "draining", False):
                raise HTTPException(status_code=503, detail="draining")
            if self.owner is not None:
                raise HTTPException(status_code=409, detail="owner already accepted")

            async def work():
                await self.finish.wait()
                self.events.append("owner_final_submit")
                if real_generation:
                    result = await self._generate(state)
                elif state.dummy_engine_client is not None:
                    await state.dummy_engine_client.collective_rpc("finish_session")
                self.events.append("owner_done")
                if real_generation:
                    return {
                        "finished_while_draining": getattr(state, "draining", False),
                        **result,
                    }
                return getattr(state, "draining", False)

            self.owner = asyncio.create_task(work())
            return {"accepted": True}

        @app.post("/plugins/dummy/finish")
        async def finish():
            if self.owner is None:
                raise HTTPException(status_code=404, detail="no owner")
            self.finish.set()
            result = await self.owner
            if isinstance(result, dict):
                return result
            return {"finished_while_draining": result}

        @app.post("/plugins/dummy/generate")
        async def generate(request: Request):
            state = request.app.state
            if getattr(state, "draining", False):
                raise HTTPException(status_code=503, detail="draining")
            return await self._generate(state)

        @app.post("/plugins/dummy/shutdown")
        async def shutdown(request: Request):
            request.app.state.request_shutdown()
            return {"shutdown_requested": True}

    async def _generate(self, state):
        from vllm.sampling_params import SamplingParams

        engine_client = state.dummy_engine_client
        if engine_client is None:
            raise HTTPException(status_code=503, detail="generation requires an engine")
        submitted_while_draining = getattr(state, "draining", False)
        final_output = None
        async for output in engine_client.generate(
            {"prompt_token_ids": [1, 7, 42]},
            SamplingParams(temperature=0, max_tokens=16, ignore_eos=True, seed=0),
            request_id=f"dummy-generation-{uuid4().hex}",
        ):
            final_output = output
        assert final_output is not None
        completion = final_output.outputs[0]
        return {
            "submitted_while_draining": submitted_while_draining,
            "finished": final_output.finished,
            "token_ids": list(completion.token_ids),
            "finish_reason": completion.finish_reason,
        }

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
