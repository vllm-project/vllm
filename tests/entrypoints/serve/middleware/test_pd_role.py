# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from starlette.responses import StreamingResponse

from vllm.entrypoints.serve.pd_role.api_router import router
from vllm.entrypoints.serve.pd_role.middleware import PDRoleMiddleware
from vllm.entrypoints.serve.pd_role.state import PDRoleState


@pytest.mark.asyncio
async def test_role_switch_waits_for_http_and_kv_drain():
    release_stream = asyncio.Event()
    entered_stream = asyncio.Event()
    prepared = asyncio.Event()
    release_kv = asyncio.Event()
    calls = []
    committed_status = []

    async def call_all(method, *args):
        calls.append(method)
        if method == "prepare_pd_role":
            prepared.set()
        committed = method == "commit_pd_role"
        if committed:
            # Observe wait_for's scheduling gap on Python 3.10/3.11.
            asyncio.get_running_loop().call_soon(
                lambda: committed_status.append(state.status())
            )
        return [
            {
                "role": "decode" if committed else "prefill",
                "epoch": int(committed),
                "drained": release_kv.is_set(),
            }
        ] * 2

    state = PDRoleState("prefill", 2, call_all)
    app = FastAPI()
    app.state.pd_role = state
    app.add_middleware(PDRoleMiddleware)

    @app.post("/v1/completions")
    async def completion():
        async def stream():
            entered_stream.set()
            yield b"first token\n"
            await release_stream.wait()
            yield b"last token\n"

        return StreamingResponse(stream())

    headers = {"x-vllm-pd-role": "prefill", "x-vllm-pd-epoch": "0"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        accepted = asyncio.create_task(client.post("/v1/completions", headers=headers))
        await asyncio.wait_for(entered_stream.wait(), 1)
        state.start("decode", 0, 2)
        rejected = await client.post("/v1/completions", headers=headers)
        assert rejected.status_code == 409
        assert state.active_requests == 1
        assert not prepared.is_set()

        release_stream.set()
        response = await accepted
        assert response.text == "first token\nlast token\n"
        await asyncio.wait_for(prepared.wait(), 1)
        assert "commit_pd_role" not in calls
        release_kv.set()
        await state.task
        status = committed_status[0]
        assert status["phase"] != "ready" or status["transition_seconds"] is not None
        assert state.status()["phase"] == "ready"
        assert state.duration is not None
        assert (state.role, state.epoch) == ("decode", 1)
        stale = await client.post("/v1/completions", headers=headers)
        assert stale.status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["drain", "commit", "cancel"])
async def test_role_switch_recovers_only_before_commit(failure):
    calls = []

    async def call_all(method, *args):
        calls.append(method)
        if method == "get_pd_role_status" and failure != "commit":
            raise TimeoutError("delayed KV transfer")
        if method == "commit_pd_role":
            raise RuntimeError("one rank did not acknowledge")
        num_ranks = 1 if method == "cancel_pd_role" and failure == "cancel" else 2
        return [
            {"role": "prefill", "epoch": 0, "drained": True, "pending_role": None}
        ] * num_ranks

    state = PDRoleState("prefill", 2, call_all)
    state.start("decode", 0, 1)
    with pytest.raises(ValueError, match="in progress"):
        state.start("decode", 0, 1)
    await state.task
    assert (state.role, state.epoch) == ("prefill", 0)
    assert state.phase == ("ready" if failure == "drain" else "failed")
    assert ("cancel_pd_role" in calls) == (failure != "commit")


@pytest.mark.asyncio
async def test_http_drain_timeout_keeps_existing_request_and_epoch():
    async def call_all(*args):
        pytest.fail("Engine preparation must wait for admitted HTTP requests")

    state = PDRoleState("decode", 2, call_all)
    state.active_requests = 1
    state.idle.clear()
    state.start("prefill", 0, 0.01)
    await state.task
    assert state.phase == "ready"
    assert state.active_requests == 1
    assert (state.role, state.epoch) == ("decode", 0)
    assert "TimeoutError" in state.error


@pytest.mark.asyncio
async def test_prepare_timeout_cancels_every_prepared_rank():
    """Rollback remains nonterminal and fenced until every rank acknowledges."""
    prepared_roles: list[str | None] = [None, None]
    calls = []
    cancelling = asyncio.Event()
    release_cancel = asyncio.Event()

    async def call_all(method, *args):
        calls.append(method)
        if method == "prepare_pd_role":
            prepared_roles[:] = ["decode", "decode"]
            await asyncio.Event().wait()
        if method == "cancel_pd_role":
            cancelling.set()
            await release_cancel.wait()
            prepared_roles[:] = [None, None]
        return [
            {"role": "prefill", "epoch": 0, "pending_role": role}
            for role in prepared_roles
        ]

    state = PDRoleState("prefill", 2, call_all)
    app = FastAPI()
    app.state.pd_role = state
    app.include_router(router)
    app.add_middleware(PDRoleMiddleware)

    @app.post("/v1/completions")
    async def completion():
        return {"accepted": True}

    headers = {"x-vllm-pd-role": "prefill", "x-vllm-pd-epoch": "0"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        state.start("decode", 0, 1)
        try:
            await asyncio.wait_for(cancelling.wait(), 2)
            status = (await client.get("/v1/pd_role")).json()
            assert status["phase"] == "rolling_back"
            assert (status["role"], status["epoch"]) == ("prefill", 0)
            assert prepared_roles == ["decode", "decode"]
            rejected = await client.post("/v1/completions", headers=headers)
            assert rejected.status_code == 409
            assert rejected.json()["phase"] == "rolling_back"
        finally:
            release_cancel.set()
            await state.task
        status = (await client.get("/v1/pd_role")).json()
        assert status["phase"] == "ready"
        assert (
            await client.post("/v1/completions", headers=headers)
        ).status_code == 200
    assert calls == ["prepare_pd_role", "cancel_pd_role"]
    assert prepared_roles == [None, None]
    assert state.epoch == 0
