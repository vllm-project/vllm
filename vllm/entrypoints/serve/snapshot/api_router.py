# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import Response

from vllm.engine.protocol import EngineClient
from vllm.snapshot.monitor import SnapshotMonitor
from vllm.snapshot.utils import is_restore
from vllm.v1.engine.exceptions import EngineDeadError

router = APIRouter()
health_router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def snapshot_monitor(request: Request) -> SnapshotMonitor:
    return request.app.state.snapshot_monitor


def ensure_snapshot_metadata_for_remote_dp(request: Request) -> None:
    parallel_config = engine_client(request).vllm_config.parallel_config
    # Internal DP uses TCP only when this client also manages remote engines.
    # After restore, those engines need metadata to reconnect to the new master
    # Pod IP. Local-only internal DP and distributed DP keep using local IPC.
    has_remote_dp_engines = (
        not parallel_config.local_engines_only
        and parallel_config.data_parallel_size_local
        < parallel_config.data_parallel_size
    )
    if (
        has_remote_dp_engines
        and request.app.state.args.snapshot_config.snapshot_metadata is None
    ):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=(
                "snapshot_config.snapshot_metadata is required when data parallel "
                "engines are remote"
            ),
        )


@health_router.get("/snapshot/health", response_class=Response)
async def snapshot_health(raw_request: Request) -> Response:
    try:
        await engine_client(raw_request).check_health()
    except EngineDeadError:
        return Response(status_code=HTTPStatus.SERVICE_UNAVAILABLE)

    monitor = snapshot_monitor(raw_request)
    ready = monitor.is_resume_done if is_restore() else monitor.is_suspend_done
    status_code = HTTPStatus.OK if ready else HTTPStatus.ACCEPTED
    return Response(status_code=status_code)


@router.post("/suspend", response_class=Response)
async def suspend(raw_request: Request) -> Response:
    ensure_snapshot_metadata_for_remote_dp(raw_request)
    model_save_path = raw_request.query_params.get("model_save_path")
    if model_save_path is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Missing required parameter: model_save_path",
        )
    await engine_client(raw_request).suspend(model_save_path=model_save_path)
    return Response(status_code=HTTPStatus.OK)


@router.post("/resume", response_class=Response)
async def resume(raw_request: Request) -> Response:
    ensure_snapshot_metadata_for_remote_dp(raw_request)
    data_parallel_master_ip = raw_request.query_params.get("data_parallel_master_ip")
    model_path = raw_request.query_params.get("model_path")
    if data_parallel_master_ip is None or model_path is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Missing required parameter: data_parallel_master_ip and model_path",
        )
    await engine_client(raw_request).resume(
        data_parallel_master_ip=data_parallel_master_ip,
        model_path=model_path,
    )
    return Response(status_code=HTTPStatus.OK)


@router.post("/device_unlock", response_class=Response)
async def device_unlock(raw_request: Request) -> Response:
    await engine_client(raw_request).device_unlock()
    return Response(status_code=HTTPStatus.OK)


def attach_router(app: FastAPI) -> None:
    if app.state.args.snapshot_config is None:
        return
    app.include_router(router)
    if app.state.args.snapshot_config.enable_auto_checkpoint:
        app.include_router(health_router)
