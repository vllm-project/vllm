# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import contextlib
import json
import os
import time
from collections.abc import Awaitable, Callable
from http import HTTPStatus
from typing import Any, Literal, TypeAlias

from starlette.responses import Response

from vllm.entrypoints.openai.batches.jsonl_protocol import (
    BatchRequestInput,
    BatchRequestOutput,
)
from vllm.entrypoints.openai.batches.protocol import (
    ACTIVE_BATCH_STATUSES,
    TERMINAL_BATCH_STATUSES,
    BatchCreateRequest,
    BatchError,
    BatchErrors,
    BatchListResponse,
    BatchObject,
)
from vllm.entrypoints.openai.batches.result_writer import BatchResultWriter
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.files.serving import OpenAIServingFiles
from vllm.entrypoints.pooling.embed.serving import ServingEmbedding
from vllm.entrypoints.pooling.scoring.serving import ServingScores
from vllm.entrypoints.serve import create_error_response
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import atomic_writer
from vllm.utils import random_uuid

logger = init_logger(__name__)

BatchHandler: TypeAlias = Callable[..., Awaitable[Any]]

MAX_CONCURRENT_REQUESTS = 256
CLEANUP_INTERVAL_SECONDS = 15 * 60
SHUTDOWN_TIMEOUT_SECONDS = 30


class OpenAIServingBatches:
    """Runs batches by dispatching each request to the matching serving handler."""

    def __init__(
        self,
        storage_dir: str,
        serving_files: OpenAIServingFiles,
        serving_chat: OpenAIServingChat | None,
        serving_embedding: ServingEmbedding | None,
        serving_scores: ServingScores | None,
        batch_priority: int = 0,
        retention_hours: int = 24,
    ) -> None:
        self._serving_files = serving_files
        self._batch_priority = batch_priority
        self._retention_hours = retention_hours

        self._handlers: dict[str, BatchHandler] = {}
        if serving_chat is not None:
            self._handlers["/v1/chat/completions"] = serving_chat.create_chat_completion
        if serving_embedding is not None:
            self._handlers["/v1/embeddings"] = serving_embedding
        if serving_scores is not None:
            self._handlers["/v1/score"] = serving_scores
            self._handlers["/v1/rerank"] = serving_scores

        metadata_dir = os.path.join(storage_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        self._metadata_path = os.path.join(metadata_dir, "batches.json")

        self._batches: dict[str, BatchObject] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._cleanup_task: asyncio.Task | None = None

        self._load_metadata()
        self._fail_interrupted_batches()

    def start(self) -> None:
        """Start the periodic cleanup of expired batches."""
        if self._retention_hours > 0 and self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def shutdown(self) -> None:
        """Stop running batches and persist their final state."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

        for cancel_event in self._cancel_events.values():
            cancel_event.set()
        tasks = list(self._tasks.values())
        if tasks:
            await asyncio.wait(tasks, timeout=SHUTDOWN_TIMEOUT_SECONDS)
        for task in tasks:
            task.cancel()

        now = int(time.time())
        for batch in self._batches.values():
            if batch.status in ACTIVE_BATCH_STATUSES:
                batch.status = "cancelled"
                batch.cancelled_at = now
        self._save_metadata()
        logger.info("Batch serving shut down")

    async def create_batch(
        self, request: BatchCreateRequest
    ) -> BatchObject | ErrorResponse:
        if await self._serving_files.get_file(request.input_file_id) is None:
            return create_error_response(
                f"File {request.input_file_id} not found",
                "NotFoundError",
                HTTPStatus.NOT_FOUND,
            )
        if request.endpoint not in self._handlers:
            return create_error_response(
                f"Endpoint {request.endpoint} is not supported by the served model"
            )

        now = int(time.time())
        batch = BatchObject(
            id=f"batch-{random_uuid()}",
            endpoint=request.endpoint,
            input_file_id=request.input_file_id,
            completion_window=request.completion_window,
            status="validating",
            created_at=now,
            expires_at=self._expires_at(now),
            metadata=request.metadata,
        )
        self._batches[batch.id] = batch
        self._save_metadata()

        cancel_event = asyncio.Event()
        self._cancel_events[batch.id] = cancel_event
        self._tasks[batch.id] = asyncio.create_task(
            self._process_batch(batch, cancel_event)
        )
        logger.info(
            "Created batch %s (file=%s, endpoint=%s)",
            batch.id,
            batch.input_file_id,
            batch.endpoint,
        )
        return batch

    async def get_batch(self, batch_id: str) -> BatchObject | None:
        return self._batches.get(batch_id)

    async def list_batches(
        self, limit: int = 20, after: str | None = None
    ) -> BatchListResponse:
        batches = sorted(
            self._batches.values(), key=lambda b: b.created_at, reverse=True
        )
        if after is not None:
            ids = [b.id for b in batches]
            batches = batches[ids.index(after) + 1 :] if after in ids else []

        page = batches[:limit]
        return BatchListResponse(
            data=page,
            has_more=len(batches) > len(page),
            first_id=page[0].id if page else None,
            last_id=page[-1].id if page else None,
        )

    async def cancel_batch(self, batch_id: str) -> BatchObject | ErrorResponse | None:
        batch = self._batches.get(batch_id)
        if batch is None:
            return None
        if batch.status not in ("validating", "in_progress"):
            return create_error_response(
                f"Cannot cancel batch with status '{batch.status}'"
            )

        batch.status = "cancelling"
        batch.cancelling_at = int(time.time())
        self._save_metadata()

        cancel_event = self._cancel_events.get(batch_id)
        if cancel_event is not None:
            cancel_event.set()
        return batch

    def is_file_in_active_batch(self, file_id: str) -> bool:
        return any(
            batch.status in ACTIVE_BATCH_STATUSES
            and file_id
            in (batch.input_file_id, batch.output_file_id, batch.error_file_id)
            for batch in self._batches.values()
        )

    async def wait_for_batch(self, batch_id: str, timeout: float = 60) -> None:
        task = self._tasks.get(batch_id)
        if task is not None:
            await asyncio.wait_for(task, timeout=timeout)

    async def _process_batch(
        self, batch: BatchObject, cancel_event: asyncio.Event
    ) -> None:
        try:
            lines = await self._load_request_lines(batch)
            if lines is not None:
                await self._run_requests(batch, lines, cancel_event)
        except Exception as e:
            logger.exception("Batch %s failed", batch.id)
            self._fail(batch, str(e))
        finally:
            self._tasks.pop(batch.id, None)
            self._cancel_events.pop(batch.id, None)

    async def _load_request_lines(self, batch: BatchObject) -> list[str] | None:
        """Validate the input file and return its request lines.

        Returns None and fails the batch if any line is invalid.
        """
        content = await self._serving_files.get_file_content(batch.input_file_id)
        if content is None:
            self._fail(batch, "Input file not found")
            return None

        lines: list[str] = []
        errors: list[BatchError] = []
        for line_num, line in enumerate(content.decode().splitlines(), start=1):
            if not line.strip():
                continue
            message = self._validate_line(batch, line)
            if message is None:
                lines.append(line)
            else:
                errors.append(
                    BatchError(code="invalid_request", message=message, line=line_num)
                )

        if errors:
            batch.errors = BatchErrors(data=errors)
            self._fail(batch, "One or more requests failed validation")
            return None
        if not lines:
            self._fail(batch, "Input file contains no requests")
            return None
        return lines

    @staticmethod
    def _validate_line(batch: BatchObject, line: str) -> str | None:
        try:
            request = BatchRequestInput.model_validate_json(line)
        except (ValueError, VLLMValidationError) as e:
            return str(e)
        if request.url != batch.endpoint:
            return (
                f"Request url {request.url} does not match "
                f"batch endpoint {batch.endpoint}"
            )
        return None

    async def _run_requests(
        self, batch: BatchObject, lines: list[str], cancel_event: asyncio.Event
    ) -> None:
        batch.request_counts.total = len(lines)
        batch.status = "in_progress"
        batch.in_progress_at = int(time.time())
        self._save_metadata()

        writer = BatchResultWriter(self._serving_files, batch.id)
        try:
            for start in range(0, len(lines), MAX_CONCURRENT_REQUESTS):
                if cancel_event.is_set():
                    break
                chunk = lines[start : start + MAX_CONCURRENT_REQUESTS]
                outputs = await asyncio.gather(*map(self._run_request, chunk))
                for output in outputs:
                    await writer.write(output)
                    if output.error is None:
                        batch.request_counts.completed += 1
                    else:
                        batch.request_counts.failed += 1
        finally:
            batch.output_file_id, batch.error_file_id = await writer.close()

        self._finish(batch, "cancelled" if cancel_event.is_set() else "completed")

    async def _run_request(self, line: str) -> BatchRequestOutput:
        request = BatchRequestInput.model_validate_json(line)
        request.body.priority = self._batch_priority
        if isinstance(request.body, ChatCompletionRequest):
            request.body.stream = False

        try:
            response = await self._handlers[request.url](request.body)
            if isinstance(response, Response):
                response = self._parse_json_response(response)
        except Exception as e:
            response = create_error_response(e)

        if isinstance(response, ErrorResponse):
            return BatchRequestOutput.failure(request.custom_id, response)
        return BatchRequestOutput.success(request.custom_id, response)

    @staticmethod
    def _parse_json_response(response: Response) -> Any:
        body = json.loads(bytes(response.body))
        if response.status_code != HTTPStatus.OK:
            return ErrorResponse.model_validate(body)
        return body

    def _expires_at(self, now: int) -> int | None:
        if self._retention_hours <= 0:
            return None
        return now + self._retention_hours * 3600

    def _mark_failed(self, batch: BatchObject, code: str, message: str) -> None:
        now = int(time.time())
        batch.status = "failed"
        batch.failed_at = now
        batch.expires_at = self._expires_at(now)
        if batch.errors is None:
            batch.errors = BatchErrors()
        batch.errors.data.append(BatchError(code=code, message=message))

    def _fail(self, batch: BatchObject, message: str) -> None:
        self._mark_failed(batch, "batch_failed", message)
        self._save_metadata()

    def _finish(
        self, batch: BatchObject, status: Literal["completed", "cancelled"]
    ) -> None:
        now = int(time.time())
        batch.status = status
        batch.finalizing_at = now
        if status == "completed":
            batch.completed_at = now
        else:
            batch.cancelled_at = now
        batch.expires_at = self._expires_at(now)
        self._save_metadata()
        logger.info(
            "Batch %s %s (completed=%d, failed=%d)",
            batch.id,
            status,
            batch.request_counts.completed,
            batch.request_counts.failed,
        )

    def _fail_interrupted_batches(self) -> None:
        interrupted = [
            b for b in self._batches.values() if b.status in ACTIVE_BATCH_STATUSES
        ]
        for batch in interrupted:
            self._mark_failed(
                batch,
                "server_restart",
                "Batch was in progress when the server stopped.",
            )
            logger.warning("Batch %s was interrupted and marked failed", batch.id)
        if interrupted:
            self._save_metadata()

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
            try:
                await self._cleanup_expired()
            except Exception:
                logger.exception("Failed to clean up expired batches")

    async def _cleanup_expired(self) -> None:
        now = int(time.time())
        expired = [
            b
            for b in self._batches.values()
            if b.status in TERMINAL_BATCH_STATUSES
            and b.expires_at is not None
            and b.expires_at <= now
        ]
        for batch in expired:
            for file_id in (
                batch.input_file_id,
                batch.output_file_id,
                batch.error_file_id,
            ):
                if file_id is not None and not self.is_file_in_active_batch(file_id):
                    await self._serving_files.delete_file(file_id)
            del self._batches[batch.id]
            logger.info("Cleaned up expired batch %s", batch.id)
        if expired:
            self._save_metadata()

    def _load_metadata(self) -> None:
        if not os.path.exists(self._metadata_path):
            return
        with open(self._metadata_path) as f:
            for item in json.load(f):
                batch = BatchObject.model_validate(item)
                self._batches[batch.id] = batch
        logger.info("Loaded %d batches from metadata", len(self._batches))

    def _save_metadata(self) -> None:
        with atomic_writer(self._metadata_path) as f:
            json.dump([b.model_dump() for b in self._batches.values()], f)
