# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from starlette.responses import StreamingResponse

from vllm.entrypoints.serve.pd_role.middleware import PDRoleMiddleware
from vllm.entrypoints.serve.pd_role.state import PDRoleState


@pytest.mark.asyncio
async def test_role_switch_waits_for_http_and_kv_drain():
    """An accepted stream and delayed KV ownership both precede role commit."""
    release_stream = asyncio.Event()
    entered_stream = asyncio.Event()
    prepared = asyncio.Event()
    release_kv = asyncio.Event()
    calls = []

    async def call_all(method, *args):
        calls.append(method)
        if method == "prepare_pd_role":
            prepared.set()
        committed = method == "commit_pd_role"
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
        assert state.status()["phase"] == "ready"
        assert (state.role, state.epoch) == ("decode", 1)
        stale = await client.post("/v1/completions", headers=headers)
        assert stale.status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize("partial_commit", [False, True])
async def test_role_switch_recovers_only_before_commit(partial_commit):
    """A drain failure restores admission; an uncertain commit stays fenced."""
    calls = []

    async def call_all(method, *args):
        calls.append(method)
        if method == "get_pd_role_status" and not partial_commit:
            raise TimeoutError("delayed KV transfer")
        if method == "commit_pd_role":
            raise RuntimeError("one rank did not acknowledge")
        return [
            {"role": "prefill", "epoch": 0, "drained": True, "pending_role": None}
        ] * 2

    state = PDRoleState("prefill", 2, call_all)
    state.start("decode", 0, 1)
    with pytest.raises(ValueError, match="in progress"):
        state.start("decode", 0, 1)
    await state.task
    assert (state.role, state.epoch) == ("prefill", 0)
    if partial_commit:
        assert state.phase == "failed"
        assert "cancel_pd_role" not in calls
    else:
        assert state.phase == "ready"
        assert calls[-1] == "cancel_pd_role"


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
async def test_role_switch_requires_every_rank_acknowledgement():
    async def call_all(method, *args):
        return [{"role": "prefill", "epoch": 0, "drained": True}]

    state = PDRoleState("prefill", 2, call_all)
    state.start("decode", 0, 1)
    await state.task
    assert state.phase == "failed"
    assert state.epoch == 0


@pytest.mark.asyncio
async def test_prepare_timeout_cancels_every_prepared_rank():
    prepared_roles: list[str | None] = [None, None]
    calls = []

    async def call_all(method, *args):
        calls.append(method)
        if method == "prepare_pd_role":
            prepared_roles[:] = ["decode", "decode"]
            await asyncio.Event().wait()
        if method == "cancel_pd_role":
            prepared_roles[:] = [None, None]
        return [
            {"role": "prefill", "epoch": 0, "pending_role": role}
            for role in prepared_roles
        ]

    state = PDRoleState("prefill", 2, call_all)
    state.start("decode", 0, 0.01)
    await state.task
    assert calls == ["prepare_pd_role", "cancel_pd_role"]
    assert prepared_roles == [None, None]
    assert state.phase == "ready"
    assert state.epoch == 0
