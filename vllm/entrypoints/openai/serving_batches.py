"""Batch processing for the OpenAI-compatible Batch API."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from collections.abc import Callable
from typing import Any

import pydantic
from pydantic import TypeAdapter
from starlette.responses import JSONResponse

from vllm.entrypoints.openai.batch.protocol import (
    BatchError,
    BatchErrors,
    BatchObject,
    BatchRequestCounts,
    BatchRequestInput,
    BatchRequestOutput,
    BatchResponseData,
)
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.openai.serving_files import OpenAIServingFiles
from vllm.logger import init_logger
from vllm.utils import random_uuid

logger = init_logger(__name__)

SUPPORTED_ENDPOINTS = {
    "/v1/chat/completions",
    "/v1/embeddings",
    "/v1/score",
}

# Max concurrent requests per batch to prevent memory exhaustion
_BATCH_CONCURRENCY_LIMIT = 256


class _BatchResultWriter:
    """Appends batch results to output/error JSONL files as they complete,
    so the full result set is never held in memory. Files are created
    lazily on first write and registered with the file store on ``close``.
    """

    def __init__(self, serving_files: OpenAIServingFiles,
                 batch_id: str) -> None:
        self._files = serving_files
        self._batch_id = batch_id
        self._out_handle: Any | None = None
        self._out_id: str | None = None
        self._out_bytes = 0
        self._err_handle: Any | None = None
        self._err_id: str | None = None
        self._err_bytes = 0

    async def write(self, output: BatchRequestOutput) -> None:
        data = (output.model_dump_json() + "\n").encode()
        if output.error is not None:
            if self._err_handle is None:
                self._err_id, path = self._files.begin_file(
                    f"{self._batch_id}_errors.jsonl", "batch_error")
                self._err_handle = await asyncio.to_thread(open, path, "wb")
            await asyncio.to_thread(self._err_handle.write, data)
            self._err_bytes += len(data)
        else:
            if self._out_handle is None:
                self._out_id, path = self._files.begin_file(
                    f"{self._batch_id}_output.jsonl", "batch_output")
                self._out_handle = await asyncio.to_thread(open, path, "wb")
            await asyncio.to_thread(self._out_handle.write, data)
            self._out_bytes += len(data)

    async def close(self) -> tuple[str | None, str | None]:
        out_id = err_id = None
        if self._out_handle is not None:
            await asyncio.to_thread(self._out_handle.close)
            out = await self._files.commit_file(
                self._out_id, f"{self._batch_id}_output.jsonl",
                "batch_output", self._out_bytes)
            out_id = out.id
        if self._err_handle is not None:
            await asyncio.to_thread(self._err_handle.close)
            err = await self._files.commit_file(
                self._err_id, f"{self._batch_id}_errors.jsonl",
                "batch_error", self._err_bytes)
            err_id = err.id
        return out_id, err_id


class OpenAIServingBatches:
    """Manages batch lifecycle: create, process, cancel, cleanup.

    Does not inherit from OpenAIServing — delegates to existing serving
    handlers for actual request processing.
    """

    def __init__(
        self,
        storage_dir: str,
        serving_files: OpenAIServingFiles,
        serving_chat: Any,  # Optional[OpenAIServingChat]
        serving_embedding: Any,  # Optional[OpenAIServingEmbedding]
        serving_score: Any,  # Optional[ServingScores]
        batch_priority: int = 0,
        retention_hours: int = 24,
    ) -> None:
        self.storage_dir = storage_dir
        self._serving_files = serving_files
        self._serving_chat = serving_chat
        self._serving_embedding = serving_embedding
        self._serving_score = serving_score
        self._batch_priority = batch_priority
        self._retention_hours = retention_hours

        self.metadata_dir = os.path.join(storage_dir, "metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)

        self._batches: dict[str, BatchObject] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()
        self._metadata_path = os.path.join(self.metadata_dir, "batches.json")
        self._cleanup_task: asyncio.Task | None = None

        self._load_metadata()
        self._recover_crashed_batches()

        # Started here when constructed inside a running loop, else on the
        # first create_batch.
        self._ensure_cleanup_task()

    def _ensure_cleanup_task(self) -> None:
        if self._retention_hours <= 0 or self._cleanup_task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._cleanup_task = loop.create_task(self._cleanup_loop())

    def _load_metadata(self) -> None:
        if os.path.exists(self._metadata_path):
            with open(self._metadata_path) as f:
                data = json.load(f)
            for item in data:
                bo = BatchObject.model_validate(item)
                self._batches[bo.id] = bo
            logger.info("Loaded %d batches from metadata",
                        len(self._batches))

    def _recover_crashed_batches(self) -> None:
        now = int(time.time())
        for batch in self._batches.values():
            if batch.status in ("validating", "in_progress", "cancelling"):
                batch.status = "failed"
                batch.failed_at = now
                if self._retention_hours > 0:
                    batch.expires_at = now + self._retention_hours * 3600
                if batch.errors is None:
                    batch.errors = BatchErrors(data=[])
                batch.errors.data.append(
                    BatchError(
                        code="server_restart",
                        message="Batch was in progress when server "
                        "stopped. Marked as failed on restart.",
                    ))
                logger.warning("Batch %s marked failed (crash recovery)",
                               batch.id)
        self._save_metadata_sync()

    def _save_metadata_sync(self) -> None:
        data = [bo.model_dump() for bo in self._batches.values()]
        with open(self._metadata_path, "w") as f:
            json.dump(data, f)

    async def _save_metadata(self) -> None:
        self._save_metadata_sync()

    async def create_batch(
        self,
        input_file_id: str,
        endpoint: str,
        completion_window: str,
        metadata: dict[str, str] | None = None,
    ) -> BatchObject | ErrorResponse:
        file_obj = await self._serving_files.get_file(input_file_id)
        if file_obj is None:
            return ErrorResponse(error=ErrorInfo(
                message=f"File {input_file_id} not found",
                type="invalid_request_error",
                code=404,
            ))

        if endpoint not in SUPPORTED_ENDPOINTS:
            return ErrorResponse(error=ErrorInfo(
                message=f"Endpoint {endpoint} is not supported. "
                f"Supported: {', '.join(sorted(SUPPORTED_ENDPOINTS))}",
                type="invalid_request_error",
                code=400,
            ))

        now = int(time.time())
        batch_id = f"batch-{random_uuid()}"
        expires_at = (now + self._retention_hours * 3600
                      if self._retention_hours > 0 else None)

        batch = BatchObject(
            id=batch_id,
            endpoint=endpoint,
            input_file_id=input_file_id,
            completion_window=completion_window,
            status="validating",
            created_at=now,
            expires_at=expires_at,
            request_counts=BatchRequestCounts(
                total=0, completed=0, failed=0),
            metadata=metadata,
        )

        async with self._lock:
            self._batches[batch_id] = batch
            await self._save_metadata()

        self._ensure_cleanup_task()
        cancel_event = asyncio.Event()
        self._cancel_events[batch_id] = cancel_event
        task = asyncio.get_running_loop().create_task(
            self._process_batch(batch_id, cancel_event))
        self._tasks[batch_id] = task

        logger.info("Created batch %s (file=%s, endpoint=%s)",
                     batch_id, input_file_id, endpoint)
        return batch

    def is_file_in_active_batch(self, file_id: str) -> bool:
        """Check if a file is referenced by a non-terminal batch."""
        for batch in self._batches.values():
            if (batch.status in ("validating", "in_progress", "cancelling")
                    and file_id in (batch.input_file_id,
                                    batch.output_file_id,
                                    batch.error_file_id)):
                return True
        return False

    async def get_batch(self, batch_id: str) -> BatchObject | None:
        return self._batches.get(batch_id)

    async def list_batches(
        self,
        limit: int = 20,
        after: str | None = None,
    ) -> tuple[list[BatchObject], bool]:
        all_batches = sorted(
            self._batches.values(),
            key=lambda b: b.created_at,
            reverse=True,
        )

        if after is not None:
            found = False
            filtered = []
            for b in all_batches:
                if found:
                    filtered.append(b)
                if b.id == after:
                    found = True
            all_batches = filtered

        limit = min(limit, 100)
        has_more = len(all_batches) > limit
        return all_batches[:limit], has_more

    async def cancel_batch(
        self, batch_id: str
    ) -> BatchObject | ErrorResponse | None:
        batch = self._batches.get(batch_id)
        if batch is None:
            return None

        if batch.status not in ("validating", "in_progress"):
            return ErrorResponse(error=ErrorInfo(
                message=f"Cannot cancel batch with status '{batch.status}'",
                type="invalid_request_error",
                code=400,
            ))

        batch.status = "cancelling"
        batch.cancelling_at = int(time.time())
        async with self._lock:
            await self._save_metadata()

        cancel_event = self._cancel_events.get(batch_id)
        if cancel_event is not None:
            cancel_event.set()

        return batch

    async def _process_batch(
        self,
        batch_id: str,
        cancel_event: asyncio.Event,
    ) -> None:
        batch = self._batches[batch_id]

        try:
            content = await self._serving_files.get_file_content(
                batch.input_file_id)
            if content is None:
                await self._fail_batch(batch, "Input file not found")
                return

            # Keep only the raw JSON of valid requests (re-parsed when
            # processed) to keep peak memory bounded by the input size.
            valid_lines: list[str] = []
            validation_errors: list[BatchError] = []

            for line_num, line in enumerate(content.decode().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    req = BatchRequestInput.model_validate_json(line)
                    if req.url != batch.endpoint:
                        validation_errors.append(BatchError(
                            code="invalid_request",
                            message=f"Request URL {req.url} does not match "
                            f"batch endpoint {batch.endpoint}",
                            line=line_num,
                        ))
                    else:
                        valid_lines.append(line)
                except Exception as e:
                    validation_errors.append(BatchError(
                        code="invalid_request",
                        message=str(e),
                        line=line_num,
                    ))
            del content

            # Validation is all-or-nothing, matching the OpenAI contract.
            if validation_errors:
                batch.errors = BatchErrors(data=validation_errors)
                await self._fail_batch(
                    batch, "One or more requests failed validation")
                return

            if not valid_lines:
                await self._fail_batch(batch, "Input file contains no requests")
                return

            batch.request_counts.total = len(valid_lines)
            batch.status = "in_progress"
            batch.in_progress_at = int(time.time())
            async with self._lock:
                await self._save_metadata()

            # Process in bounded chunks, streaming results to disk.
            writer = _BatchResultWriter(self._serving_files, batch.id)
            try:
                for start in range(0, len(valid_lines),
                                   _BATCH_CONCURRENCY_LIMIT):
                    if cancel_event.is_set():
                        break
                    chunk = valid_lines[start:start + _BATCH_CONCURRENCY_LIMIT]
                    results = await asyncio.gather(
                        *[self._run_one_line(batch, line, cancel_event)
                          for line in chunk],
                        return_exceptions=True,
                    )
                    for output in results:
                        if not isinstance(output, BatchRequestOutput):
                            continue
                        await writer.write(output)
                        if output.error is not None:
                            batch.request_counts.failed += 1
                        else:
                            batch.request_counts.completed += 1
            finally:
                batch.output_file_id, batch.error_file_id = \
                    await writer.close()

            status = "cancelled" if cancel_event.is_set() else "completed"
            await self._finalize_batch(batch, status=status)

        except Exception as e:
            logger.exception("Batch %s failed with error", batch_id)
            await self._fail_batch(batch, str(e))
        finally:
            self._tasks.pop(batch_id, None)
            self._cancel_events.pop(batch_id, None)

    async def _run_one_line(
        self,
        batch: BatchObject,
        line: str,
        cancel_event: asyncio.Event,
    ) -> BatchRequestOutput | None:
        if cancel_event.is_set():
            return None
        request = BatchRequestInput.model_validate_json(line)
        return await self._run_single_request(batch, request)

    async def _run_single_request(
        self,
        batch: BatchObject,
        request: BatchRequestInput,
    ) -> BatchRequestOutput:
        try:
            if hasattr(request.body, 'stream'):
                request.body.stream = False
            if hasattr(request.body, 'priority'):
                request.body.priority = self._batch_priority

            handler_fn = self._get_handler_fn(request.url)
            if handler_fn is None:
                return BatchRequestOutput(
                    id=f"vllm-{random_uuid()}",
                    custom_id=request.custom_id,
                    response=BatchResponseData(
                        status_code=400,
                        request_id=f"vllm-batch-{random_uuid()}"),
                    error=ErrorResponse(error=ErrorInfo(
                        message=f"No handler for {request.url}",
                        type="invalid_request_error",
                        code=400,
                    )),
                )

            response = await handler_fn(request.body)

            # Pooling handlers return a JSONResponse; normalize it to a
            # response model so the body serializes to the OpenAI shape.
            if isinstance(response, JSONResponse):
                # Lazy import: run_batch pulls in api_server and the engine.
                from vllm.entrypoints.openai.run_batch import AllResponse
                parsed: Any = response
                with contextlib.suppress(pydantic.ValidationError):
                    parsed = TypeAdapter(
                        AllResponse | ErrorResponse).validate_python(
                            json.loads(response.body))
                if isinstance(parsed, JSONResponse):
                    # Could not parse into a known response shape.
                    return BatchRequestOutput(
                        id=f"vllm-{random_uuid()}",
                        custom_id=request.custom_id,
                        response=BatchResponseData(
                            status_code=response.status_code,
                            request_id=f"vllm-batch-{random_uuid()}"),
                        error=ErrorResponse(error=ErrorInfo(
                            message="Handler returned an unparseable "
                            "response",
                            type="server_error",
                            code=response.status_code,
                        )),
                    )
                response = parsed

            if isinstance(response, ErrorResponse):
                return BatchRequestOutput(
                    id=f"vllm-{random_uuid()}",
                    custom_id=request.custom_id,
                    response=BatchResponseData(
                        status_code=response.error.code,
                        request_id=f"vllm-batch-{random_uuid()}"),
                    error=response,
                )
            elif response is not None:
                return BatchRequestOutput(
                    id=f"vllm-{random_uuid()}",
                    custom_id=request.custom_id,
                    response=BatchResponseData(
                        body=response,
                        request_id=f"vllm-batch-{random_uuid()}"),
                    error=None,
                )
            else:
                return BatchRequestOutput(
                    id=f"vllm-{random_uuid()}",
                    custom_id=request.custom_id,
                    response=None,
                    error=ErrorResponse(error=ErrorInfo(
                        message="Unexpected response type",
                        type="server_error",
                        code=500,
                    )),
                )
        except Exception as e:
            return BatchRequestOutput(
                id=f"vllm-{random_uuid()}",
                custom_id=request.custom_id,
                response=None,
                error=ErrorResponse(error=ErrorInfo(
                    message=str(e),
                    type="server_error",
                    code=500,
                )),
            )

    def _get_handler_fn(self, url: str) -> Callable | None:
        # Pooling objects are themselves callables (via __call__), unlike
        # the chat handler; this matches run_batch.py's dispatch.
        if url == "/v1/chat/completions" and self._serving_chat:
            return self._serving_chat.create_chat_completion
        elif url == "/v1/embeddings" and self._serving_embedding:
            return self._serving_embedding
        elif url == "/v1/score" and self._serving_score:
            return self._serving_score
        return None

    async def _fail_batch(self, batch: BatchObject, message: str) -> None:
        now = int(time.time())
        batch.status = "failed"
        batch.failed_at = now
        if self._retention_hours > 0:
            batch.expires_at = now + self._retention_hours * 3600
        if batch.errors is None:
            batch.errors = BatchErrors(data=[])
        batch.errors.data.append(
            BatchError(code="batch_failed", message=message))
        async with self._lock:
            await self._save_metadata()

    async def _finalize_batch(
        self,
        batch: BatchObject,
        status: str,
    ) -> None:
        now = int(time.time())
        batch.finalizing_at = now

        batch.status = status
        if status == "completed":
            batch.completed_at = now
        elif status == "cancelled":
            batch.cancelled_at = now

        if self._retention_hours > 0:
            batch.expires_at = now + self._retention_hours * 3600

        async with self._lock:
            await self._save_metadata()

        logger.info("Batch %s finalized with status=%s "
                     "(completed=%d, failed=%d)",
                     batch.id, status,
                     batch.request_counts.completed,
                     batch.request_counts.failed)

    async def _cleanup_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(15 * 60)  # Every 15 minutes
                await self._cleanup_expired()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Error in batch cleanup loop")

    async def _cleanup_expired(self) -> None:
        now = int(time.time())
        expired_ids = [
            b.id for b in self._batches.values()
            if b.expires_at is not None
            and b.expires_at <= now
            and b.status in ("completed", "failed", "cancelled")
        ]
        for batch_id in expired_ids:
            batch = self._batches[batch_id]
            for file_id in (batch.input_file_id,
                            batch.output_file_id,
                            batch.error_file_id):
                if file_id:
                    await self._serving_files.delete_file(file_id)

            async with self._lock:
                del self._batches[batch_id]
                await self._save_metadata()

            logger.info("Cleaned up expired batch %s", batch_id)

    async def wait_for_batch(
        self, batch_id: str, timeout: float = 60
    ) -> None:
        """Wait for a batch task to complete. Used in tests."""
        task = self._tasks.get(batch_id)
        if task is not None:
            await asyncio.wait_for(task, timeout=timeout)

    async def shutdown(self) -> None:
        """Graceful shutdown: cancel tasks, persist state."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

        for cancel_event in list(self._cancel_events.values()):
            cancel_event.set()

        for task in list(self._tasks.values()):
            try:
                await asyncio.wait_for(task, timeout=30)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()

        now = int(time.time())
        for batch in self._batches.values():
            if batch.status in ("validating", "in_progress", "cancelling"):
                batch.status = "cancelled"
                batch.cancelled_at = now

        async with self._lock:
            await self._save_metadata()

        logger.info("Batch serving shut down")
