# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FastAPI router for /v1/files.

Wired into the app in `api_server.py:build_app` only when
`FileUploadConfig.enabled` is True. All endpoints return 404 when the
feature is off, by virtue of the router not being attached at all.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.files.serving import OpenAIServingFiles
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def _files(request: Request) -> OpenAIServingFiles | None:
    return getattr(request.app.state, "openai_serving_files", None)


async def _stream_upload_file(upload: UploadFile) -> AsyncIterator[bytes]:
    """Yield chunks from a FastAPI UploadFile as an async iterator.

    Yields:
        64 KiB byte chunks until the upload is exhausted.
    """
    while True:
        chunk = await upload.read(64 * 1024)
        if not chunk:
            return
        yield chunk


def _not_configured() -> JSONResponse:
    return JSONResponse(
        status_code=HTTPStatus.NOT_IMPLEMENTED.value,
        content={
            "error": {
                "message": "File uploads not enabled on this server",
                "type": "not_implemented_error",
                "code": HTTPStatus.NOT_IMPLEMENTED.value,
            }
        },
    )


def _ok(model) -> JSONResponse:
    return JSONResponse(content=model.model_dump(exclude_none=True))


def _err_response(err: ErrorResponse) -> JSONResponse:
    return JSONResponse(content=err.model_dump(), status_code=err.error.code)


@router.post("/v1/files")
async def create_file(
    raw_request: Request,
    file: Annotated[UploadFile, File()],
    purpose: Annotated[str, Form()] = "user_data",
) -> JSONResponse:
    """Upload a multimodal file and return its capability handle.

    The file is streamed to disk, MIME-validated against the media
    allowlist (video/*, image/*, audio/*), and registered in the upload
    store. Reference the returned `id` as `vllm-file://<id>` in a
    subsequent chat completion.

    Returns:
        A JSON response with the new `FileObject` (200) or an
        OpenAI-compatible error envelope (400/413/503).
    """
    handler = _files(raw_request)
    if handler is None:
        return _not_configured()
    result = await handler.create_file(
        raw_request,
        _stream_upload_file(file),
        file.filename or "upload.bin",
        purpose,
    )
    if isinstance(result, ErrorResponse):
        return _err_response(result)
    return _ok(result)


@router.get("/v1/files")
async def list_files(raw_request: Request) -> JSONResponse:
    """List files visible to the caller.

    Returns:
        A JSON `FileList` (200), or an error envelope (400 on missing
        scope header, 404 when listing is disabled).
    """
    handler = _files(raw_request)
    if handler is None:
        return _not_configured()
    result = await handler.list_files(raw_request)
    if isinstance(result, ErrorResponse):
        return _err_response(result)
    return _ok(result)


@router.get("/v1/files/{file_id}")
async def retrieve_file(raw_request: Request, file_id: str) -> JSONResponse:
    """Return metadata for a single uploaded file.

    Returns:
        A JSON `FileObject` (200), or an error envelope (404 on
        unknown id or scope mismatch).
    """
    handler = _files(raw_request)
    if handler is None:
        return _not_configured()
    result = await handler.retrieve_file(raw_request, file_id)
    if isinstance(result, ErrorResponse):
        return _err_response(result)
    return _ok(result)


@router.get("/v1/files/{file_id}/content")
async def retrieve_content(raw_request: Request, file_id: str) -> Response:
    """Stream the raw bytes of an uploaded file.

    Returns:
        A `StreamingResponse` of the file bytes with the sniffed MIME
        type as `Content-Type` (200), or a JSON error envelope (404).
    """
    handler = _files(raw_request)
    if handler is None:
        return _not_configured()
    result = await handler.retrieve_content(raw_request, file_id)
    if isinstance(result, ErrorResponse):
        return _err_response(result)
    stream, media_type = result
    return StreamingResponse(stream, media_type=media_type)


@router.delete("/v1/files/{file_id}")
async def delete_file(raw_request: Request, file_id: str) -> JSONResponse:
    """Delete an uploaded file and its on-disk bytes.

    Returns:
        A JSON `FileDeleteResponse` (200), or an error envelope (404
        on unknown id or scope mismatch).
    """
    handler = _files(raw_request)
    if handler is None:
        return _not_configured()
    result = await handler.delete_file(raw_request, file_id)
    if isinstance(result, ErrorResponse):
        return _err_response(result)
    return _ok(result)


def attach_router(app: FastAPI) -> None:
    """Mount the `/v1/files` routes on `app`."""
    app.include_router(router)


async def init_files_state(state, args) -> None:
    """Populate app.state.openai_serving_files from CLI args.

    Called from api_server.init_app_state when --enable-file-uploads is set.
    Creates the FileUploadConfig, FileUploadStore, and serving handler, and
    attaches both the store and the handler to app.state so downstream
    consumers (MediaConnector, shutdown hooks) can access them.

    Async so that the FileUploadStore's blocking filesystem setup
    (shutil.rmtree on the upload dir, PID lockfile acquisition) runs in
    the default thread pool instead of stalling the event loop — the
    rmtree can take seconds on NFS or large stale upload trees.
    """
    import asyncio

    from vllm.config import FileUploadConfig
    from vllm.entrypoints.openai.files.serving import OpenAIServingFiles
    from vllm.entrypoints.openai.files.store import FileUploadStore

    if not args.enable_file_uploads:
        state.openai_serving_files = None
        state.file_upload_store = None
        return

    # Reject multi-API-server deployments. Each API server process
    # holds a separate in-memory upload store and a separate
    # _GLOBAL_STORE, so an upload handled by process A returns a
    # vllm-file:// ID that process B cannot resolve. The kernel
    # load-balances HTTP traffic across API server processes, so
    # (N-1)/N of subsequent resolve requests would fail.
    api_count = getattr(args, "api_server_count", None)
    dp_size = getattr(args, "data_parallel_size", None) or 1
    effective_api_count = api_count if api_count is not None else dp_size
    if effective_api_count > 1:
        raise ValueError(
            "--enable-file-uploads is not supported with multiple API "
            f"server processes (detected count={effective_api_count}). "
            "Each process would hold a separate upload store, so "
            "vllm-file:// IDs returned by one process would fail to "
            "resolve on another. Re-run with --api-server-count=1 "
            "(and --data-parallel-size=1 if set), or deploy behind an "
            "external sticky-session proxy with per-backend state."
        )

    config = FileUploadConfig(  # type: ignore[call-arg]
        enabled=True,
        dir=args.file_upload_dir,
        ttl_seconds=args.file_upload_ttl_seconds,
        max_size_mb=args.file_upload_max_size_mb,
        max_total_gb=args.file_upload_max_total_gb,
        max_concurrent=args.file_upload_max_concurrent,
        scope_header=args.file_upload_scope_header,
        disable_listing=args.file_upload_disable_listing,
    )
    # FileUploadStore.__init__ does blocking filesystem I/O (mkdtemp or
    # rmtree+mkdir+lockfile acquisition). Dispatch it to the default
    # thread pool so the event loop stays responsive during startup.
    loop = asyncio.get_running_loop()
    store = await loop.run_in_executor(None, FileUploadStore, config)
    state.file_upload_store = store
    state.openai_serving_files = OpenAIServingFiles(store, config)
    # Expose the store to MediaConnector so vllm-file:// URLs resolve in
    # chat-completion requests.
    import atexit

    from vllm.entrypoints.openai.files.store import register_store

    register_store(store)
    # Clear the process-wide reference on interpreter shutdown so test
    # harnesses and `uvicorn --reload` don't leave a stale store (whose
    # asyncio.Lock is bound to a dead event loop) in module state.
    atexit.register(register_store, None)
