# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from vllm.entrypoints.codec_compression import wrap_streaming_response
from vllm.entrypoints.codec_frame import (
    CONTENT_TYPE,
    PROTO_SCHEMA,
    decode_msgpack,
    decode_protobuf_request,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.orca_metrics import metrics_header
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.utils import (
    load_aware_call,
    with_cancellation,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()
ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"


def completion(request: Request) -> OpenAIServingCompletion | None:
    return request.app.state.openai_serving_completion


@router.post(
    "/v1/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_completion(request: CompletionRequest, raw_request: Request):
    # Codec v0.4 version-negotiation gate. Default-off; only fires when
    # the deployment has CODEC_*_REQUIRED set and the client speaks below
    # the floor. See spec/versions/v0.4.md § Version Compatibility Signaling.
    from vllm.entrypoints.codec_version import (
        make_426_response,
        needs_upgrade,
        parse_client_version,
    )

    client_version = parse_client_version(raw_request)
    if needs_upgrade(client_version):
        return make_426_response(client_version=client_version)

    metrics_header_format = raw_request.headers.get(
        ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL, ""
    )
    handler = completion(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Completions API")

    generator = await handler.create_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
    elif isinstance(generator, CompletionResponse):
        return JSONResponse(
            content=generator.model_dump(),
            headers=metrics_header(metrics_header_format),
        )

    media_type = CONTENT_TYPE.get(request.stream_format, "text/event-stream")
    # Negotiated transport compression for binary streams. JSON SSE keeps
    # whatever compression is applied higher up the stack (proxies / FastAPI
    # middleware) and is unaffected by this codepath.
    if request.stream_format != "json":
        return wrap_streaming_response(
            raw_request.headers.get("accept-encoding", ""),
            generator,
            media_type=media_type,
            stream_format=request.stream_format,
            client_version=client_version,
        )
    return StreamingResponse(content=generator, media_type=media_type)


@router.post(
    "/v1/completions/codec",
    summary="Token-native binary completions (Codec protocol)",
    description=(
        "Bidirectional token-native endpoint. Submit prompt token IDs as a "
        "binary body (msgpack or protobuf) and receive a binary stream of "
        "CodecFrame messages containing generated token IDs — no text "
        "conversion at any point.\n\n"
        "Content-Type of request body sets the input format:\n"
        "  application/x-msgpack  → msgpack dict with keys: prompt_ids, "
        "max_tokens, temperature, stop, stream_format\n"
        "  application/x-protobuf → CodecRequest proto message\n\n"
        "stream_format in the body determines response encoding "
        "('msgpack' or 'protobuf')."
    ),
    responses={
        HTTPStatus.OK.value: {
            "content": {
                "application/x-msgpack": {},
                "application/x-protobuf": {},
            }
        },
    },
)
@with_cancellation
@load_aware_call
async def create_completion_codec(raw_request: Request):
    # Codec v0.4 version-negotiation gate.
    from vllm.entrypoints.codec_version import (
        make_426_response,
        needs_upgrade,
        parse_client_version,
    )

    client_version = parse_client_version(raw_request)
    if needs_upgrade(client_version):
        return make_426_response(client_version=client_version)

    handler = completion(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Completions API")

    content_type = raw_request.headers.get("content-type", "")
    body = await raw_request.body()

    if "protobuf" in content_type:
        params = decode_protobuf_request(body)
    else:
        params = decode_msgpack(body)

    prompt_ids: list[int] = params.get("prompt_ids", [])
    stream_format: str = params.get("stream_format", "msgpack")

    request = CompletionRequest(
        prompt=prompt_ids,
        max_tokens=params.get("max_tokens", 256),
        temperature=params.get("temperature"),
        stop=params.get("stop"),
        stream=True,
        stream_format=stream_format,
    )

    generator = await handler.create_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    media_type = CONTENT_TYPE.get(stream_format, "application/x-msgpack")
    return wrap_streaming_response(
        raw_request.headers.get("accept-encoding", ""),
        generator,
        media_type=media_type,
        stream_format=stream_format,
        client_version=client_version,
    )


@router.get(
    "/codec/schema",
    summary="Codec protobuf schema",
    description="Returns the .proto schema for CodecFrame and CodecRequest.",
)
async def codec_schema():
    return PlainTextResponse(content=PROTO_SCHEMA, media_type="text/plain")


@router.get(
    "/.well-known/codec/version-policy.json",
    summary="Codec v0.4 deployment minimum-version policy",
    description=(
        "Pre-flight discovery doc per spec/WELL_KNOWN_DISCOVERY.md § "
        "Version policy (v0.4+). Returns 404 when the deployment does "
        "NOT mandate any v0.4 feature — that's the normal state for an "
        "unrestricted deployment. Returns 200 with the policy doc when "
        "any CODEC_*_REQUIRED env var is set."
    ),
)
async def well_known_version_policy():
    from fastapi.responses import JSONResponse
    from fastapi.responses import Response as FastResponse

    from vllm.entrypoints.codec_version import version_policy_document

    doc = version_policy_document()
    if doc is None:
        return FastResponse(status_code=404)
    return JSONResponse(doc)


def attach_router(app: FastAPI):
    app.include_router(router)
