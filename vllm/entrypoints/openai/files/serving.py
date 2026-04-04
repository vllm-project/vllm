# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Business logic for /v1/files.

This layer sits between the FastAPI router and the FileUploadStore. It
extracts scope/identifying metadata from requests, converts store records
to wire-format responses, and maps store exceptions to HTTP error
responses.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.openai.files.protocol import (
    FileDeleteResponse,
    FileList,
    FileObject,
)
from vllm.entrypoints.openai.files.store import (
    ConcurrencyLimitExceeded,
    FileRecord,
    FileStoreError,
    FileTooLarge,
    InvalidMime,
    QuotaExceeded,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from fastapi import Request

    from vllm.config import FileUploadConfig
    from vllm.entrypoints.openai.files.store import FileUploadStore

logger = init_logger(__name__)

# Enforced locally; a Pydantic Literal on the request body would 422 rather
# than the 400 we want.
_ALLOWED_PURPOSES: frozenset[str] = frozenset({"vision", "user_data"})


def _err(status: HTTPStatus, message: str, err_type: str) -> ErrorResponse:
    return ErrorResponse(
        error=ErrorInfo(message=message, type=err_type, code=status.value)
    )


class OpenAIServingFiles:
    """Handlers for /v1/files/*.

    Holds a reference to the upload store and the relevant config
    knobs. One instance per server. All instance state is private;
    the public surface is the async handler methods defined below.
    """

    def __init__(self, store: FileUploadStore, config: FileUploadConfig) -> None:
        self._store = store
        self._config = config
        self._ttl_enabled = config.ttl_seconds >= 0

    # ------------------------------------------------------------------
    # scope extraction
    # ------------------------------------------------------------------

    def _extract_scope(
        self, request: Request
    ) -> tuple[str | None, ErrorResponse | None]:
        """Read the configured scope header from the request.

        Returns:
            A `(scope, err)` tuple. `err` is non-None when the scope
            header is required but absent, in which case the caller
            should return the error directly. When scoping is disabled
            server-side, `scope` is always None.
        """
        header_name = self._config.scope_header
        if not header_name:
            return None, None
        value = request.headers.get(header_name)
        if not value:
            return None, _err(
                HTTPStatus.BAD_REQUEST,
                f"Scope header {header_name!r} required",
                "invalid_request_error",
            )
        return value, None

    def _record_to_object(self, record: FileRecord) -> FileObject:
        if self._ttl_enabled:
            expires_at = int(record.last_accessed) + self._config.ttl_seconds
        else:
            expires_at = None
        return FileObject(
            id=record.id,
            bytes=record.bytes,
            created_at=record.created_at,
            expires_at=expires_at,
            filename=record.filename,
            purpose=record.purpose,
            status="processed",
        )

    # ------------------------------------------------------------------
    # request context
    # ------------------------------------------------------------------

    @staticmethod
    def _client_host(request: Request) -> str | None:
        client = getattr(request, "client", None)
        return getattr(client, "host", None) if client is not None else None

    @staticmethod
    def _request_id(request: Request) -> str | None:
        return request.headers.get("x-request-id")

    # ------------------------------------------------------------------
    # handlers
    # ------------------------------------------------------------------

    async def create_file(
        self,
        request: Request,
        stream: AsyncIterator[bytes],
        filename: str,
        purpose: str,
    ) -> FileObject | ErrorResponse:
        """Upload a new file. Validates purpose + scope header, streams
        bytes into the store, and maps store exceptions to HTTP errors.

        Returns:
            The new `FileObject` on success, or an `ErrorResponse`
            (400/413/503) mapped from the store's validation failures.
        """
        scope, err = self._extract_scope(request)
        if err is not None:
            return err
        if purpose not in _ALLOWED_PURPOSES:
            return _err(
                HTTPStatus.BAD_REQUEST,
                f"purpose must be one of {sorted(_ALLOWED_PURPOSES)}",
                "invalid_request_error",
            )
        try:
            record = await self._store.create_file(
                stream=stream,
                filename=filename,
                purpose=purpose,  # type: ignore[arg-type]
                scope=scope,
                client_host=self._client_host(request),
                request_id=self._request_id(request),
            )
        except ConcurrencyLimitExceeded as e:
            return _err(
                HTTPStatus.SERVICE_UNAVAILABLE,
                str(e),
                "server_error",
            )
        except FileTooLarge as e:
            return _err(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                str(e),
                "invalid_request_error",
            )
        except InvalidMime as e:
            return _err(HTTPStatus.BAD_REQUEST, str(e), "invalid_request_error")
        except QuotaExceeded as e:
            return _err(
                HTTPStatus.INSUFFICIENT_STORAGE,
                str(e),
                "server_error",
            )
        except FileStoreError as e:
            return _err(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), "server_error")
        return self._record_to_object(record)

    async def list_files(self, request: Request) -> FileList | ErrorResponse:
        """Return files visible under the caller's scope (newest first).

        Returns:
            A `FileList` of visible files, or an `ErrorResponse` (400
            on missing scope header, 404 when listing is disabled).
        """
        if self._config.disable_listing:
            return _err(
                HTTPStatus.NOT_FOUND,
                "File listing is disabled on this server",
                "not_found_error",
            )
        scope, err = self._extract_scope(request)
        if err is not None:
            return err
        records = self._store.list(scope)
        return FileList(data=[self._record_to_object(r) for r in records])

    async def retrieve_file(
        self, request: Request, file_id: str
    ) -> FileObject | ErrorResponse:
        """Return metadata for one file, or 404 on scope mismatch.

        Returns:
            The matching `FileObject`, or an `ErrorResponse` (400/404).
        """
        scope, err = self._extract_scope(request)
        if err is not None:
            return err
        record = self._store.get(file_id, scope)
        if record is None:
            return _err(
                HTTPStatus.NOT_FOUND,
                f"File {file_id!r} not found",
                "not_found_error",
            )
        return self._record_to_object(record)

    async def retrieve_content(
        self, request: Request, file_id: str
    ) -> tuple[AsyncIterator[bytes], str] | ErrorResponse:
        """Return a streaming async iterator of the file bytes plus its
        sniffed MIME type, or 404 on scope mismatch.

        Returns:
            A `(stream, mime_type)` tuple on success, or an
            `ErrorResponse` (400/404).
        """
        scope, err = self._extract_scope(request)
        if err is not None:
            return err
        record = self._store.get(file_id, scope)
        if record is None:
            return _err(
                HTTPStatus.NOT_FOUND,
                f"File {file_id!r} not found",
                "not_found_error",
            )
        try:
            stream = await self._store.stream_content(
                file_id,
                scope,
                client_host=self._client_host(request),
                request_id=self._request_id(request),
            )
        except FileNotFoundError:
            # Race: file evicted between get() and stream_content().
            return _err(
                HTTPStatus.NOT_FOUND,
                f"File {file_id!r} not found",
                "not_found_error",
            )
        return stream, record.mime_type

    async def delete_file(
        self, request: Request, file_id: str
    ) -> FileDeleteResponse | ErrorResponse:
        """Remove one file and its on-disk bytes, or 404 on scope mismatch.

        Returns:
            A `FileDeleteResponse` with `deleted=True` on success, or
            an `ErrorResponse` (400/404).
        """
        scope, err = self._extract_scope(request)
        if err is not None:
            return err
        deleted = await self._store.delete(
            file_id,
            scope,
            client_host=self._client_host(request),
            request_id=self._request_id(request),
        )
        if not deleted:
            return _err(
                HTTPStatus.NOT_FOUND,
                f"File {file_id!r} not found",
                "not_found_error",
            )
        return FileDeleteResponse(id=file_id, deleted=True)
