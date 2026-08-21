# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P/D concurrent dispatch: KV-ready notification endpoint.

The disaggregation router posts the prefill's kv_transfer_params to
``POST /v1/pd_kv_ready`` as soon as the prefill responds, resuming the
decode-side prepare request parked/armed in the scheduler.
"""

import pickle
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _pd_ready_fast_push(
    raw_request: Request, raw_request_id: str, kv_transfer_params: dict[str, Any]
) -> bool:
    """Early-arm fast path: push the KV-ready notification straight to the
    EngineCore fast-KV bridge over the notify ipc socket. Fire-and-forget
    (losses covered by the armed-timeout fallback); False if unavailable."""
    vllm_config = getattr(raw_request.app.state, "vllm_config", None)
    if vllm_config is None:
        return False
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.fast_kv import (
            fast_kv_notify_path,
            fast_notify_enabled,
        )

        ktc = vllm_config.kv_transfer_config
        if (
            ktc is None
            or ktc.kv_connector not in ("NixlConnector", "NixlPullConnector")
            or not fast_notify_enabled(vllm_config)
            or not ktc.get_from_extra_config("pd_early_arm", False)
        ):
            return False
        app_state = raw_request.app.state
        sock = getattr(app_state, "pd_kv_ready_sock", None)
        if sock is None:
            import zmq

            sock = zmq.Context.instance().socket(zmq.PUSH)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(fast_kv_notify_path(vllm_config))
            app_state.pd_kv_ready_sock = sock
        import zmq

        sock.send(
            pickle.dumps(("pd_ready", raw_request_id, kv_transfer_params)),
            zmq.NOBLOCK,
        )
        return True
    except Exception:
        logger.exception("pd_kv_ready fast push failed")
        return False


@router.post("/v1/pd_kv_ready")
async def pd_kv_ready(raw_request: Request):
    """Body: {"request_id": <router request id>,
    "kv_transfer_params": {...}}."""
    body = await raw_request.json()
    raw_request_id = body.get("request_id")
    kv_transfer_params = body.get("kv_transfer_params")
    if not raw_request_id or not isinstance(kv_transfer_params, dict):
        return JSONResponse(
            content={"error": "request_id and kv_transfer_params are required"},
            status_code=HTTPStatus.BAD_REQUEST.value,
        )
    if _pd_ready_fast_push(raw_request, raw_request_id, kv_transfer_params):
        return JSONResponse(content={"queued": True})
    client = engine_client(raw_request)
    engine_core = getattr(client, "engine_core", None)
    call_utility = getattr(engine_core, "call_utility_async", None)
    if call_utility is None:
        return JSONResponse(
            content={"error": "engine does not support pd_kv_ready"},
            status_code=HTTPStatus.NOT_IMPLEMENTED.value,
        )
    matched = await call_utility("update_pd_kv_ready", raw_request_id, kv_transfer_params)
    return JSONResponse(content={"matched": bool(matched)})


def attach_router(app: FastAPI):
    app.include_router(router)
