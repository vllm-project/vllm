# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the /flash_epscale endpoint orchestration.

These tests exercise the front-end state machine with a mocked
EngineClient; they do not require GPUs or a running engine.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import vllm.envs as envs
from vllm.entrypoints.serve.flash_epscale.api_router import attach_router


def _build_client(
    monkeypatch,
    *,
    ep_world_size=4,
    active_ep_size=4,
    current_sleeping=None,
    post_resize_active=None,
    post_resize_sleeping=None,
):
    """Build a TestClient whose engine_client is mocked.

    The mock returns a stable EP state from get_ep_sleep_state until
    resize_sleep_ep_ranks is invoked, after which it returns the
    post-resize state.  This lets us assert orchestration paths without
    standing up a real engine.
    """
    monkeypatch.setattr(envs, "VLLM_SERVER_DEV_MODE", True)
    current_sleeping = list(current_sleeping or [])
    if post_resize_active is None:
        post_resize_active = active_ep_size
    if post_resize_sleeping is None:
        post_resize_sleeping = list(current_sleeping)

    state = {
        "ep_world_size": ep_world_size,
        "active_ep_size": active_ep_size,
        "sleeping_ep_ranks": current_sleeping,
    }
    final_state = {
        "ep_world_size": ep_world_size,
        "active_ep_size": post_resize_active,
        "sleeping_ep_ranks": list(post_resize_sleeping),
    }

    rpc_calls: list[tuple[str, dict]] = []

    async def collective_rpc(method, kwargs=None):
        rpc_calls.append((method, dict(kwargs or {})))
        if method == "get_ep_sleep_state":
            if "resize_sleep_ep_ranks" in [c[0] for c in rpc_calls]:
                return [final_state, final_state]
            return [state, state]
        if method == "resize_sleep_ep_ranks":
            new_sleeping = list(kwargs["sleeping_ep_ranks"])
            final_state["sleeping_ep_ranks"] = new_sleeping
            final_state["active_ep_size"] = ep_world_size - len(new_sleeping)
        return None

    engine = MagicMock()
    engine.collective_rpc = AsyncMock(side_effect=collective_rpc)
    engine.pause_generation = AsyncMock(return_value=None)
    engine.resume_generation = AsyncMock(return_value=None)
    engine.set_active_data_parallel_size = MagicMock()
    engine.wait_for_dp_ranks_to_drain = AsyncMock(return_value=None)

    app = FastAPI()
    app.state.engine_client = engine
    attach_router(app)
    return TestClient(app), engine, rpc_calls


def test_rejects_non_int_ep_size(monkeypatch):
    client, _, _ = _build_client(monkeypatch)
    resp = client.post("/flash_epscale", json={"ep_size": "2"})
    assert resp.status_code == 400
    assert "ep_size must be an integer" in resp.json()["detail"]


def test_rejects_out_of_range_ep_size(monkeypatch):
    client, _, _ = _build_client(monkeypatch, ep_world_size=4)
    resp = client.post("/flash_epscale", json={"ep_size": 8})
    assert resp.status_code == 400
    assert "must be in [1, 4]" in resp.json()["detail"]


def test_rejects_invalid_tags(monkeypatch):
    client, _, _ = _build_client(monkeypatch)
    resp = client.post("/flash_epscale", json={"ep_size": 2, "tags": []})
    assert resp.status_code == 400


def test_rejects_invalid_drain_timeout(monkeypatch):
    client, _, _ = _build_client(monkeypatch)
    resp = client.post(
        "/flash_epscale",
        json={"ep_size": 2, "drain_timeout": -1},
    )
    assert resp.status_code == 400


def test_noop_when_target_equals_active(monkeypatch):
    client, engine, _ = _build_client(
        monkeypatch, ep_world_size=4, active_ep_size=4, current_sleeping=[]
    )
    resp = client.post("/flash_epscale", json={"ep_size": 4})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["changed"] is False
    assert body["action"] == "noop"
    # No pause/resume in noop path.
    engine.pause_generation.assert_not_called()
    engine.resume_generation.assert_not_called()
    # Active DP size is still refreshed even on noop.
    engine.set_active_data_parallel_size.assert_called_once_with(4)


def test_scale_down_runs_drain_before_pause(monkeypatch):
    """scale_down: route + drain happen before pause; pause window only
    contains the wake/resize/sleep RPC sequence."""
    client, engine, rpc_calls = _build_client(
        monkeypatch,
        ep_world_size=4,
        active_ep_size=4,
        current_sleeping=[],
        post_resize_active=2,
        post_resize_sleeping=[2, 3],
    )
    resp = client.post("/flash_epscale", json={"ep_size": 2})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["action"] == "scale_down"
    assert body["changed"] is True
    assert body["active_ep_size"] == 2
    assert body["sleeping_ep_ranks"] == [2, 3]

    # Routing is shrunk and ranks are drained before pause.
    assert engine.set_active_data_parallel_size.call_args_list[0].args == (2,)
    engine.wait_for_dp_ranks_to_drain.assert_awaited_once()
    drain_args = engine.wait_for_dp_ranks_to_drain.await_args
    assert drain_args.args[0] == [2, 3]

    # Pause/resume happen exactly once each.
    engine.pause_generation.assert_awaited_once()
    engine.resume_generation.assert_awaited_once()

    # No wake step when nothing was previously sleeping.
    methods = [m for m, _ in rpc_calls]
    assert "wake_up_ep_ranks" not in methods
    # resize comes before sleep.
    resize_idx = methods.index("resize_sleep_ep_ranks")
    sleep_idx = methods.index("sleep_ep_ranks_by_tags")
    assert resize_idx < sleep_idx


def test_scale_up_opens_routing_after_resume(monkeypatch):
    """scale_up: set_active_data_parallel_size is called only after the
    pause/resume window closes, so requests never reach a not-ready rank."""
    client, engine, rpc_calls = _build_client(
        monkeypatch,
        ep_world_size=4,
        active_ep_size=2,
        current_sleeping=[2, 3],
        post_resize_active=4,
        post_resize_sleeping=[],
    )
    resp = client.post("/flash_epscale", json={"ep_size": 4})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["action"] == "scale_up"
    assert body["active_ep_size"] == 4
    assert body["sleeping_ep_ranks"] == []

    # Order must be: pause → wake → resize → resume → set_active_dp.
    rpc_methods = [m for m, _ in rpc_calls]
    assert rpc_methods.index("wake_up_ep_ranks") < rpc_methods.index(
        "resize_sleep_ep_ranks"
    )

    # Pause/resume pair fired exactly once.
    engine.pause_generation.assert_awaited_once()
    engine.resume_generation.assert_awaited_once()

    # set_active_data_parallel_size was called once with the new size,
    # AFTER resume_generation.
    engine.set_active_data_parallel_size.assert_called_once_with(4)


def test_resume_runs_even_when_inner_step_fails(monkeypatch):
    """If a collective_rpc inside the pause window raises, the engine must
    still be resumed before the HTTP error is returned."""
    client, engine, _ = _build_client(monkeypatch, ep_world_size=4, active_ep_size=4)
    # FastAPI TestClient re-raises unhandled server exceptions by default;
    # disable so the wrapped HTTP 500 propagates through the response.
    client = TestClient(client.app, raise_server_exceptions=False)
    boom = RuntimeError("simulated split failure")

    async def explode(method, kwargs=None):
        if method == "resize_sleep_ep_ranks":
            raise boom
        if method == "get_ep_sleep_state":
            return [
                {
                    "ep_world_size": 4,
                    "active_ep_size": 4,
                    "sleeping_ep_ranks": [],
                },
                {
                    "ep_world_size": 4,
                    "active_ep_size": 4,
                    "sleeping_ep_ranks": [],
                },
            ]
        return None

    engine.collective_rpc.side_effect = explode

    resp = client.post("/flash_epscale", json={"ep_size": 2})
    assert resp.status_code == 500
    # resume_generation must still have run despite the error.
    engine.resume_generation.assert_awaited()


def test_resume_failure_returns_500(monkeypatch):
    """If resume itself fails, the endpoint must not report success."""
    client, engine, _ = _build_client(
        monkeypatch,
        ep_world_size=4,
        active_ep_size=4,
        post_resize_active=2,
        post_resize_sleeping=[2, 3],
    )
    client = TestClient(client.app, raise_server_exceptions=False)
    engine.resume_generation.side_effect = RuntimeError("resume failed")

    resp = client.post("/flash_epscale", json={"ep_size": 2})

    assert resp.status_code == 500
    engine.resume_generation.assert_awaited()


def test_scale_down_restores_routing_when_resize_fails(monkeypatch):
    """After route shrink succeeds, later scale_down failures must restore
    frontend routing to the original active DP size."""
    client, engine, _ = _build_client(monkeypatch, ep_world_size=4, active_ep_size=4)
    client = TestClient(client.app, raise_server_exceptions=False)

    async def explode(method, kwargs=None):
        if method == "resize_sleep_ep_ranks":
            raise RuntimeError("split failed")
        if method == "get_ep_sleep_state":
            return [
                {
                    "ep_world_size": 4,
                    "active_ep_size": 4,
                    "sleeping_ep_ranks": [],
                },
                {
                    "ep_world_size": 4,
                    "active_ep_size": 4,
                    "sleeping_ep_ranks": [],
                },
            ]
        return None

    engine.collective_rpc.side_effect = explode

    resp = client.post("/flash_epscale", json={"ep_size": 2})

    assert resp.status_code == 500
    assert engine.set_active_data_parallel_size.call_args_list[0].args == (2,)
    assert engine.set_active_data_parallel_size.call_args_list[-1].args == (4,)


def test_scale_up_resleeps_current_ranks_when_resize_fails(monkeypatch):
    """After scale_up wakes sleeping ranks, resize failures should restore
    the previous sleep state and keep frontend routing closed."""
    client, engine, _ = _build_client(
        monkeypatch,
        ep_world_size=4,
        active_ep_size=2,
        current_sleeping=[2, 3],
    )
    client = TestClient(client.app, raise_server_exceptions=False)
    rpc_calls: list[tuple[str, dict]] = []

    async def explode(method, kwargs=None):
        rpc_calls.append((method, dict(kwargs or {})))
        if method == "resize_sleep_ep_ranks":
            raise RuntimeError("resize failed")
        if method == "get_ep_sleep_state":
            return [
                {
                    "ep_world_size": 4,
                    "active_ep_size": 2,
                    "sleeping_ep_ranks": [2, 3],
                },
                {
                    "ep_world_size": 4,
                    "active_ep_size": 2,
                    "sleeping_ep_ranks": [2, 3],
                },
            ]
        return None

    engine.collective_rpc.side_effect = explode

    resp = client.post("/flash_epscale", json={"ep_size": 3})

    assert resp.status_code == 500
    methods = [method for method, _ in rpc_calls]
    assert methods == [
        "get_ep_sleep_state",
        "wake_up_ep_ranks",
        "resize_sleep_ep_ranks",
        "sleep_ep_ranks_by_tags",
    ]
    assert rpc_calls[-1][1] == {
        "sleeping_ep_ranks": [2, 3],
        "tags": ["shared_weights", "expert_weights", "kv_cache"],
    }
    engine.set_active_data_parallel_size.assert_not_called()


def test_scale_up_restores_previous_state_when_sleep_fails(monkeypatch):
    """If the final scale_up sleep fails, rollback should resize back and
    re-sleep the originally sleeping ranks."""
    client, engine, _ = _build_client(
        monkeypatch,
        ep_world_size=4,
        active_ep_size=2,
        current_sleeping=[2, 3],
    )
    client = TestClient(client.app, raise_server_exceptions=False)
    rpc_calls: list[tuple[str, dict]] = []

    async def explode(method, kwargs=None):
        kwargs = dict(kwargs or {})
        rpc_calls.append((method, kwargs))
        if method == "sleep_ep_ranks_by_tags" and kwargs["sleeping_ep_ranks"] == [3]:
            raise RuntimeError("sleep failed")
        if method == "get_ep_sleep_state":
            return [
                {
                    "ep_world_size": 4,
                    "active_ep_size": 2,
                    "sleeping_ep_ranks": [2, 3],
                },
                {
                    "ep_world_size": 4,
                    "active_ep_size": 2,
                    "sleeping_ep_ranks": [2, 3],
                },
            ]
        return None

    engine.collective_rpc.side_effect = explode

    resp = client.post("/flash_epscale", json={"ep_size": 3})

    assert resp.status_code == 500
    assert rpc_calls[1:] == [
        (
            "wake_up_ep_ranks",
            {
                "sleeping_ep_ranks": [2, 3],
                "tags": ["shared_weights", "expert_weights", "kv_cache"],
            },
        ),
        ("resize_sleep_ep_ranks", {"sleeping_ep_ranks": [3]}),
        (
            "sleep_ep_ranks_by_tags",
            {
                "sleeping_ep_ranks": [3],
                "tags": ["shared_weights", "expert_weights", "kv_cache"],
            },
        ),
        (
            "wake_up_ep_ranks",
            {
                "sleeping_ep_ranks": [3],
                "tags": ["shared_weights", "expert_weights", "kv_cache"],
            },
        ),
        ("resize_sleep_ep_ranks", {"sleeping_ep_ranks": [2, 3]}),
        (
            "sleep_ep_ranks_by_tags",
            {
                "sleeping_ep_ranks": [2, 3],
                "tags": ["shared_weights", "expert_weights", "kv_cache"],
            },
        ),
    ]
    engine.set_active_data_parallel_size.assert_not_called()


def test_inconsistent_worker_state_is_rejected(monkeypatch):
    """Workers reporting different EP states must fail fast with 500."""
    client, engine, _ = _build_client(monkeypatch)

    async def divergent(method, kwargs=None):
        if method == "get_ep_sleep_state":
            return [
                {
                    "ep_world_size": 4,
                    "active_ep_size": 4,
                    "sleeping_ep_ranks": [],
                },
                {
                    "ep_world_size": 4,
                    "active_ep_size": 3,
                    "sleeping_ep_ranks": [3],
                },
            ]
        return None

    engine.collective_rpc.side_effect = divergent
    resp = client.post("/flash_epscale", json={"ep_size": 2})
    assert resp.status_code == 500
    assert "inconsistent EP sleep state" in resp.json()["detail"]


@pytest.mark.parametrize(
    "state, expected_msg",
    [
        (
            {
                "ep_world_size": 4,
                "active_ep_size": 2,
                "sleeping_ep_ranks": [1, 3],
            },
            "sleeping_ep_ranks must be the suffix",
        ),
        (
            {
                "ep_world_size": 4,
                "active_ep_size": 2,
                "sleeping_ep_ranks": [2, 2],
            },
            "contains duplicates",
        ),
        (
            {
                "ep_world_size": 4,
                "active_ep_size": 0,
                "sleeping_ep_ranks": [0, 1, 2, 3],
            },
            "invalid active_ep_size",
        ),
    ],
)
def test_invalid_worker_state_is_rejected(monkeypatch, state, expected_msg):
    client, engine, _ = _build_client(monkeypatch)

    async def invalid_state(method, kwargs=None):
        if method == "get_ep_sleep_state":
            return [state, state]
        return None

    engine.collective_rpc.side_effect = invalid_state

    resp = client.post("/flash_epscale", json={"ep_size": 2})

    assert resp.status_code == 500
    assert expected_msg in resp.json()["detail"]
    engine.set_active_data_parallel_size.assert_not_called()


def test_router_not_attached_outside_dev_mode(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_SERVER_DEV_MODE", False)
    app = FastAPI()
    attach_router(app)
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/flash_epscale" not in paths


@pytest.mark.parametrize(
    "body, expected_msg",
    [
        ({}, "ep_size must be an integer"),
        ({"ep_size": True}, "ep_size must be an integer"),
        ({"ep_size": 1.5}, "ep_size must be an integer"),
        ({"ep_size": 0}, "must be in [1,"),
        (
            {"ep_size": 2, "drain_timeout": True},
            "drain_timeout must be a positive number",
        ),
    ],
)
def test_param_validation_messages(monkeypatch, body, expected_msg):
    client, _, _ = _build_client(monkeypatch)
    resp = client.post("/flash_epscale", json=body)
    assert resp.status_code == 400
    assert expected_msg in resp.json()["detail"]
