# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import Response

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()

OCTET_STREAM = "application/octet-stream"


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/v1/kv_connector/rpc")
async def kv_connector_rpc(raw_request: Request):
    """Deliver a generic control RPC to the scheduler-side KV connector.

    The request body is the connector-defined request payload (raw bytes);
    vLLM never parses it. The bytes are routed unchanged to the connector's
    `on_rpc` hook running in the EngineCore scheduler process, and its
    response bytes are returned verbatim.

    Status codes:
        200: The connector handled the RPC. The body is the response bytes
            (an empty body is a valid success/ack).
        404: No KV connector is configured.
        501: The connector does not implement the RPC hook.
        500: The connector's handler raised.
    """
    client = engine_client(raw_request)

    if client.vllm_config.kv_transfer_config is None:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND.value,
            detail="No KV connector is configured.",
        )

    payload = await raw_request.body()
    result = await client.invoke_kv_connector(payload)

    if result is None:
        raise HTTPException(
            status_code=HTTPStatus.NOT_IMPLEMENTED.value,
            detail="The configured KV connector does not implement on_rpc().",
        )

    return Response(content=bytes(result), media_type=OCTET_STREAM)


def attach_router(app: FastAPI):
    app.include_router(router)
