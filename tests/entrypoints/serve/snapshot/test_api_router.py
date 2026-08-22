# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI, HTTPException

from vllm.config import SnapshotConfig
from vllm.entrypoints.serve.snapshot.api_router import (
    attach_router,
    resume,
    router,
    snapshot_health,
    suspend,
)
from vllm.snapshot.monitor import SnapshotMonitor
from vllm.v1.engine.core_client import AsyncMPClient
from vllm.v1.engine.exceptions import EngineDeadError


def _request(
    *,
    data_parallel_size: int = 1,
    data_parallel_size_local: int | None = None,
    local_engines_only: bool = False,
    snapshot_metadata: str | None = None,
    query_params: dict[str, str] | None = None,
) -> Mock:
    request = Mock()
    if data_parallel_size_local is None:
        data_parallel_size_local = data_parallel_size
    engine_client = SimpleNamespace(
        check_health=AsyncMock(),
        suspend=AsyncMock(),
        resume=AsyncMock(),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=data_parallel_size,
                data_parallel_size_local=data_parallel_size_local,
                local_engines_only=local_engines_only,
            )
        ),
    )
    request.app.state = SimpleNamespace(
        args=SimpleNamespace(
            snapshot_config=SnapshotConfig(snapshot_metadata=snapshot_metadata)
        ),
        engine_client=engine_client,
        snapshot_monitor=SnapshotMonitor(),
    )
    request.query_params = query_params or {}
    return request


def test_snapshot_router_owns_lifecycle_endpoints():
    paths = {route.path for route in router.routes}

    assert paths == {
        "/suspend",
        "/resume",
        "/device_unlock",
    }


def test_snapshot_router_is_not_attached_without_config():
    app = FastAPI()
    app.state.args = SimpleNamespace(snapshot_config=None)

    attach_router(app)

    paths = {route.path for route in app.routes}
    assert paths.isdisjoint({"/suspend", "/resume", "/device_unlock"})


@pytest.mark.parametrize("enable_auto_checkpoint", [False, True])
def test_snapshot_health_route_requires_auto_checkpoint(enable_auto_checkpoint):
    app = FastAPI()
    app.state.args = SimpleNamespace(
        snapshot_config=SnapshotConfig(
            snapshot_metadata=(
                "/snapshot/metadata.json" if enable_auto_checkpoint else None
            ),
            enable_auto_checkpoint=enable_auto_checkpoint,
        )
    )

    attach_router(app)

    paths = {route.path for route in app.routes}
    assert ("/snapshot/health" in paths) is enable_auto_checkpoint


def test_snapshot_monitor_tracks_shared_client_state():
    monitor = SnapshotMonitor()

    assert monitor.try_start_suspending()
    assert monitor.is_suspending
    assert not monitor.is_suspend_done
    assert not monitor.try_start_suspending()

    monitor.mark_suspend_done()
    assert not monitor.is_suspending
    assert monitor.is_suspend_done

    assert monitor.try_start_unlocking()
    assert monitor.is_unlocking
    assert not monitor.is_unlock_done
    assert not monitor.try_start_unlocking()

    monitor.mark_unlock_done()
    assert not monitor.is_unlocking
    assert monitor.is_unlock_done

    assert monitor.try_start_resuming()
    assert monitor.is_resuming
    assert not monitor.is_resume_done
    assert not monitor.try_start_resuming()

    monitor.mark_resume_done()
    assert not monitor.is_resuming
    assert monitor.is_resume_done


def test_snapshot_monitor_clears_gates_after_failure():
    monitor = SnapshotMonitor()

    assert monitor.try_start_suspending()
    monitor.mark_suspend_failed()
    assert not monitor.is_suspending
    assert monitor.try_start_suspending()
    monitor.mark_suspend_done()

    assert monitor.try_start_unlocking()
    monitor.mark_unlock_failed()
    assert not monitor.is_unlocking
    assert monitor.try_start_unlocking()
    monitor.mark_unlock_done()

    assert monitor.try_start_resuming()
    monitor.mark_resume_failed()
    assert not monitor.is_resuming
    assert monitor.try_start_resuming()


@pytest.mark.asyncio
async def test_engine_client_owns_snapshot_lifecycle_state():
    client = object.__new__(AsyncMPClient)
    client.snapshot_monitor = monitor = SnapshotMonitor()
    client.call_utility_async = AsyncMock()
    client.wait_for_engines_ready = AsyncMock()
    client._is_tcp_input_transport = False

    async def suspend(*args):
        assert monitor.is_suspending
        assert not monitor.is_suspend_done

    client.call_utility_async.side_effect = suspend
    await client.suspend_async("/model")
    assert not monitor.is_suspending
    assert monitor.is_suspend_done
    await client.suspend_async("/model")
    assert client.call_utility_async.await_count == 1

    async def unlock(*args):
        assert monitor.is_unlocking
        assert not monitor.is_unlock_done

    client.call_utility_async.side_effect = unlock
    await client.device_unlock_async()
    assert not monitor.is_unlocking
    assert monitor.is_unlock_done
    await client.device_unlock_async()
    assert client.call_utility_async.await_count == 2

    async def resume(*args):
        assert monitor.is_resuming
        assert not monitor.is_resume_done

    client.call_utility_async.side_effect = resume
    with patch("vllm.v1.engine.core_client.is_restore", return_value=True):
        await client.resume_async("10.0.0.1", "/model")
    assert not monitor.is_resuming
    assert monitor.is_resume_done
    await client.resume_async("10.0.0.1", "/model")
    assert client.call_utility_async.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler", "query_params"),
    [
        (suspend, {"model_save_path": "/model"}),
        (
            resume,
            {
                "data_parallel_master_ip": "10.0.0.1",
                "model_path": "/model",
            },
        ),
    ],
)
async def test_remote_dp_snapshot_requires_metadata(handler, query_params):
    request = _request(
        data_parallel_size=2,
        data_parallel_size_local=1,
        query_params=query_params,
    )

    with pytest.raises(HTTPException) as exc_info:
        await handler(request)

    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
    assert "snapshot_config.snapshot_metadata is required" in exc_info.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler", "query_params", "method"),
    [
        (suspend, {"model_save_path": "/model"}, "suspend"),
        (
            resume,
            {
                "data_parallel_master_ip": "10.0.0.1",
                "model_path": "/model",
            },
            "resume",
        ),
    ],
)
async def test_local_internal_dp_snapshot_does_not_require_metadata(
    handler, query_params, method
):
    request = _request(
        data_parallel_size=2,
        data_parallel_size_local=2,
        query_params=query_params,
    )

    response = await handler(request)

    assert response.status_code == HTTPStatus.OK
    getattr(request.app.state.engine_client, method).assert_awaited_once()


@pytest.mark.asyncio
async def test_snapshot_health_waits_for_suspend_on_cold_start():
    request = _request()

    with patch(
        "vllm.entrypoints.serve.snapshot.api_router.is_restore",
        return_value=False,
    ):
        response = await snapshot_health(request)
        assert response.status_code == HTTPStatus.ACCEPTED

        request.app.state.snapshot_monitor.mark_suspend_done()
        response = await snapshot_health(request)

    assert response.status_code == HTTPStatus.OK


@pytest.mark.asyncio
async def test_snapshot_health_waits_for_resume_after_restore():
    request = _request()
    request.app.state.snapshot_monitor.mark_suspend_done()

    with patch(
        "vllm.entrypoints.serve.snapshot.api_router.is_restore",
        return_value=True,
    ):
        response = await snapshot_health(request)
        assert response.status_code == HTTPStatus.ACCEPTED

        request.app.state.snapshot_monitor.mark_resume_done()
        response = await snapshot_health(request)

    assert response.status_code == HTTPStatus.OK


@pytest.mark.asyncio
async def test_snapshot_health_reports_dead_engine():
    request = _request()
    request.app.state.engine_client.check_health.side_effect = EngineDeadError()

    response = await snapshot_health(request)

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
