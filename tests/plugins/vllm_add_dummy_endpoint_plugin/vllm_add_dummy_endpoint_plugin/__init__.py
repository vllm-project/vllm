# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worked example `vllm.endpoint_plugins` entry point.

Reports scheduler config via `collective_rpc`. Demonstrates the full
contract: `attach_router` registers the route at Phase A (`build_app`) and
`init_state` stashes the `EngineClient` the route handler needs at Phase B
(`init_app_state`).
"""

from fastapi import FastAPI, Request


class DummyAdminEndpointPlugin:
    name = "dummy_admin_endpoint_plugin"
    required_tasks: tuple[str, ...] | None = None

    def attach_router(self, app: FastAPI) -> None:
        @app.get("/v1/admin/scheduler_config")
        async def scheduler_config(raw_request: Request):
            engine_client = raw_request.app.state.dummy_engine_client
            results = await engine_client.collective_rpc("get_scheduler_config")
            return {"scheduler_config": results}

    async def init_state(self, engine_client, state, args) -> None:
        state.dummy_engine_client = engine_client
