# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FastAPI router for the Cohere Chat v2 API (``POST /cohere/v2/chat``).

The Cohere v2 protocol models are sourced from the official ``cohere``
Python SDK (``pip install cohere``). To keep that an *optional*
dependency for vLLM, the SDK-dependent imports - and the route handler
itself - are gated on a one-shot probe at module load. If the SDK isn't
installed, :func:`attach_router` becomes a no-op (with an info log) and
vLLM continues to boot normally.

Note: the handler must live at module scope (not inside
``attach_router``) so that FastAPI's ``typing.get_type_hints`` resolves
the ``CohereChatV2Request`` body annotation against the module's
globals. Defining it locally inside ``attach_router`` would hide the
type from ``get_type_hints``, causing FastAPI to silently degrade the
body parameter into a query parameter and reject every request with
422.
"""

from fastapi import FastAPI

from vllm.logger import init_logger

logger = init_logger(__name__)


try:
    import cohere  # noqa: F401  -- dependency probe
except ImportError:
    _SDK_AVAILABLE = False
else:
    _SDK_AVAILABLE = True


if _SDK_AVAILABLE:
    from http import HTTPStatus

    from fastapi import APIRouter, Depends, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    from vllm.entrypoints.cohere.protocol import (
        CohereChatV2Request,
        CohereChatV2Response,
        CohereError,
    )
    from vllm.entrypoints.cohere.serving import CohereServingChatV2
    from vllm.entrypoints.openai.engine.protocol import ErrorResponse
    from vllm.entrypoints.serve.utils.api_utils import (
        load_aware_call,
        validate_json_request,
        with_cancellation,
    )

    router = APIRouter()

    def _serving(request: Request) -> CohereServingChatV2 | None:
        return getattr(request.app.state, "cohere_serving_chat_v2", None)

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
            content=CohereError(message=info.message, id=_request_id(raw_request)).model_dump(exclude_none=True),
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
                    message=str(e),
                    id=_request_id(raw_request),
                ).model_dump(exclude_none=True),
            )

        if isinstance(result, ErrorResponse):
            return _error_response(result, raw_request)

        if isinstance(result, CohereChatV2Response):
            return JSONResponse(content=result.model_dump(exclude_none=True))

        return StreamingResponse(content=result, media_type="text/event-stream")


def attach_router(app: FastAPI) -> None:
    """Register ``POST /cohere/v2/chat`` on ``app``.

    No-op (with an info log) when the optional ``cohere`` SDK isn't
    installed, since the v2 protocol models live there.
    """
    if not _SDK_AVAILABLE:
        logger.info(
            "cohere SDK not installed; /cohere/v2/chat endpoint disabled. "
            "Install with `pip install cohere` to enable it."
        )
        return
    app.include_router(router)
