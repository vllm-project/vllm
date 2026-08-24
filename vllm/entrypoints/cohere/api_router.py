# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FastAPI router for the Cohere Chat v2 API.

Exposes two routes:

* ``POST /cohere/v2/chat`` - the chat endpoint itself.
* ``POST /cohere/v2/chat/render`` - tokenize a request without running
  generation, mirroring ``POST /v1/chat/completions/render``.

The Cohere v2 protocol models are sourced from the official ``cohere``
Python SDK (``pip install cohere``). To keep that an *optional*
dependency for vLLM, the SDK-dependent imports - and the route handlers
themselves - are gated on a one-shot probe at module load. If the SDK
isn't installed, :func:`attach_router` becomes a no-op (with an info
log) and vLLM continues to boot normally.

Even when the SDK is installed, :func:`attach_router` also requires
``VLLM_ENABLE_COHERE_API=1`` in the environment before it will expose
the routes. This keeps non-Cohere deployments that pull in the SDK for
unrelated reasons (e.g. test dependencies) from accidentally exposing
the api.

Note: the handlers must live at module scope (not inside
``attach_router``) so that FastAPI's ``typing.get_type_hints`` resolves
the ``CohereChatV2Request`` body annotation against the module's
globals. Defining them locally inside ``attach_router`` would hide the
type from ``get_type_hints``, causing FastAPI to silently degrade the
body parameter into a query parameter and reject every request with
422.
"""

import json
from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

import vllm.envs as envs
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.exception_handling.utils import sanitize_message
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)
from vllm.logger import init_logger

_COHERE_PATH_PREFIX = "/cohere/"

logger = init_logger(__name__)


try:
    import cohere  # noqa: F401  -- dependency probe
except ImportError:
    _SDK_AVAILABLE = False
else:
    _SDK_AVAILABLE = True


if _SDK_AVAILABLE:
    from vllm.entrypoints.cohere.protocol import (
        CohereChatV2Request,
        CohereChatV2Response,
        CohereError,
    )
    from vllm.entrypoints.cohere.serving import CohereServingChatV2
    from vllm.entrypoints.scale_out.render.serving import ServingRender
    from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest

    router = APIRouter()

    def _serving(request: Request) -> CohereServingChatV2 | None:
        return getattr(request.app.state, "cohere_serving_chat_v2", None)

    def _serving_render(request: Request) -> ServingRender | None:
        return getattr(request.app.state, "serving_render", None)

    def _request_id(raw_request: Request | None) -> str | None:
        """Best-effort lookup of the active request id.

        Prefers the id the underlying chat handler stamped onto
        ``raw_request.state.request_metadata`` (if it got that far before
        failing), falling back to the ``X-Request-Id`` HTTP header. May
        return ``None`` if neither is available, in which case the field
        is omitted from the response.
        """
        if raw_request is None:
            return None
        meta = getattr(raw_request.state, "request_metadata", None)
        if meta is not None and getattr(meta, "request_id", None):
            return meta.request_id
        return raw_request.headers.get("X-Request-Id")

    def _error_response(
        error: ErrorResponse,
        raw_request: Request | None,
        *,
        fallback_status: int = HTTPStatus.BAD_REQUEST,
    ) -> JSONResponse:
        """Translate vLLM's internal error envelope into Cohere's shape."""
        info = error.error
        status = info.code or fallback_status
        return JSONResponse(
            status_code=status,
            content=CohereError(
                message=sanitize_message(info.message),
                id=_request_id(raw_request),
            ).model_dump(exclude_none=True),
        )

    @router.post(
        "/cohere/v2/chat",
        dependencies=[Depends(validate_json_request)],
        responses={
            HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
            HTTPStatus.BAD_REQUEST.value: {"model": CohereError},
            HTTPStatus.NOT_FOUND.value: {"model": CohereError},
            HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": CohereError},
        },
    )
    @with_cancellation
    @load_aware_call
    async def chat_v2(request: CohereChatV2Request, raw_request: Request):
        handler = _serving(raw_request)
        if handler is None:
            return JSONResponse(
                status_code=HTTPStatus.NOT_IMPLEMENTED.value,
                content=CohereError(
                    message="The model does not support the Cohere v2 chat API.",
                    id=_request_id(raw_request),
                ).model_dump(exclude_none=True),
            )

        try:
            result = await handler.create_chat_v2(request, raw_request)
        except Exception as e:  # noqa: BLE001 - report as 500 for parity
            logger.exception("Error in /cohere/v2/chat: %s", e)
            return JSONResponse(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                content=CohereError(
                    message=sanitize_message(str(e)),
                    id=_request_id(raw_request),
                ).model_dump(exclude_none=True),
            )

        match result:
            case ErrorResponse():
                return _error_response(result, raw_request)
            case CohereChatV2Response():
                return JSONResponse(content=result.model_dump(exclude_none=True))
            case _:
                return StreamingResponse(content=result, media_type="text/event-stream")

    @router.post(
        "/cohere/v2/chat/render",
        dependencies=[Depends(validate_json_request)],
        response_model=GenerateRequest,
        responses={
            HTTPStatus.BAD_REQUEST.value: {"model": CohereError},
            HTTPStatus.NOT_FOUND.value: {"model": CohereError},
            HTTPStatus.NOT_IMPLEMENTED.value: {"model": CohereError},
            HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": CohereError},
        },
    )
    @with_cancellation
    async def chat_v2_render(request: CohereChatV2Request, raw_request: Request):
        """Tokenize a Cohere v2 chat request without running generation.

        The Cohere counterpart to ``POST /v1/chat/completions/render``.
        The v2 body goes through the same conversion
        ``/cohere/v2/chat`` uses and is then handed to the shared
        :class:`ServingRender`, so the returned ``GenerateRequest``
        carries the prompt tokens and sampling params the chat endpoint
        would have sent to the engine.
        """
        handler = _serving(raw_request)
        render_handler = _serving_render(raw_request)
        if handler is None or render_handler is None:
            return JSONResponse(
                status_code=HTTPStatus.NOT_IMPLEMENTED.value,
                content=CohereError(
                    message=(
                        "The model does not support the Cohere v2 chat render API."
                    ),
                    id=_request_id(raw_request),
                ).model_dump(exclude_none=True),
            )

        try:
            chat_request = handler.to_chat_completion_request(request)
            result = await render_handler.render_chat_request(chat_request)
        except Exception as e:  # noqa: BLE001 - report as 500 for parity
            logger.exception("Error in /cohere/v2/chat/render: %s", e)
            return JSONResponse(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                content=CohereError(
                    message=sanitize_message(str(e)),
                    id=_request_id(raw_request),
                ).model_dump(exclude_none=True),
            )

        if isinstance(result, ErrorResponse):
            return _error_response(result, raw_request)

        return JSONResponse(content=result.model_dump())

    class CohereErrorEnvelopeMiddleware(BaseHTTPMiddleware):
        """Rewrite vLLM error bodies into the Cohere ``{message, id}`` shape.

        The endpoint handler above already returns :class:`CohereError` for
        errors it owns, but globally-registered exception handlers (e.g.
        :func:`validation_exception_handler` for pydantic body errors,
        :func:`http_exception_handler`, engine error handlers) fire *before*
        the handler runs and produce vLLM's internal
        ``ErrorResponse`` shape (``{"error": {"message": ...}}``). That
        shape doesn't match the ``CohereError`` schema advertised on the
        route's OpenAPI ``responses``, so clients (and schema-conformance
        tests like ``test_openai_schema.py``) would see a mismatch on
        those paths. This middleware normalises any error body on
        ``/cohere/*`` responses to :class:`CohereError`.
        """

        async def dispatch(self, request: Request, call_next):
            response = await call_next(request)
            if not request.url.path.startswith(_COHERE_PATH_PREFIX):
                return response
            if response.status_code < 400:
                return response
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("application/json"):
                return response

            body = b"".join([chunk async for chunk in response.body_iterator])
            translated = _translate_vllm_error_body(body, request)
            if translated is not None:
                return translated
            passthrough_headers = {
                k: v
                for k, v in response.headers.items()
                if k.lower() != "content-length"
            }
            return Response(
                content=body,
                status_code=response.status_code,
                headers=passthrough_headers,
                media_type=content_type,
            )

    def _translate_vllm_error_body(raw: bytes, request: Request) -> JSONResponse | None:
        """Translate a vLLM ``ErrorResponse`` body to a ``CohereError`` body.

        Returns ``None`` if ``raw`` does not match the vLLM error envelope
        (which signals the middleware to pass the body through unchanged).
        """
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        if not (
            isinstance(data, dict)
            and isinstance(data.get("error"), dict)
            and "message" in data["error"]
        ):
            return None
        try:
            err = ErrorResponse.model_validate(data)
        except Exception:  # noqa: BLE001 - malformed envelope; pass through
            return None
        return _error_response(err, request)


def attach_router(app: FastAPI) -> None:
    """Register the ``/cohere/v2/chat`` routes on ``app``.

    No-op when either:

    * the ``VLLM_ENABLE_COHERE_API`` env var isn't set to ``1``. The
      Cohere v2 endpoints are opt-in because they carry Cohere-specific
      request/response semantics (grounding citations, tool_plan,
      PLAN/THINKING_CONTENT blocks) that are only meaningful when
      serving a Cohere Command-family model.
    * the optional ``cohere`` SDK isn't installed (the v2 protocol
      models live there)

    The two skip paths log at different levels: an operator who set
    ``VLLM_ENABLE_COHERE_API=1`` but forgot to install ``cohere`` sees
    a WARNING (they explicitly asked for the endpoints and they're
    silently absent), whereas the default-off skip logs at debug.
    """
    enabled = envs.VLLM_ENABLE_COHERE_API
    if not enabled:
        logger.debug(
            "VLLM_ENABLE_COHERE_API is not set; /cohere/v2/chat endpoints "
            "disabled. Set VLLM_ENABLE_COHERE_API=1 to enable them."
        )
        return
    if not _SDK_AVAILABLE:
        logger.warning(
            "VLLM_ENABLE_COHERE_API=1 but the `cohere` SDK is not "
            "installed; /cohere/v2/chat will not be exposed. Install "
            "with `pip install cohere` to enable the endpoints."
        )
        return
    app.include_router(router)
    app.add_middleware(CohereErrorEnvelopeMiddleware)
