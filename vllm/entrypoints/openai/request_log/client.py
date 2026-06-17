# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""API-server-side client for the request/response logger subprocess."""

from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import os
import tempfile
import time
from collections.abc import AsyncIterator
from typing import Any

import zmq

import vllm.envs as envs
from vllm.entrypoints.openai.request_log.proto import (
    DEFAULT_SNDHWM,
    FRAME_RECORD,
    FRAME_SHUTDOWN,
    WRITER_READY_TIMEOUT_S,
    WRITER_SHUTDOWN_TIMEOUT_S,
)
from vllm.entrypoints.openai.request_log.writer_proc import writer_main
from vllm.logger import init_logger

logger = init_logger(__name__)


def _resolve_log_path(args, rank: int, total_ranks: int) -> str | None:
    """CLI flag wins over the env var; either being unset disables the hub."""
    path = getattr(args, "request_log_path", None) or envs.VLLM_REQUEST_LOG_PATH
    if not path:
        return None
    # When multiple API server workers are running, suffix per rank so
    # that each writer subprocess owns its own file (no cross-process
    # write races).
    if total_ranks and total_ranks > 1:
        base, ext = os.path.splitext(path)
        path = f"{base}.rank{rank}{ext or '.jsonl'}"
    return path


class RequestLoggerHub:
    """Owns the writer subprocess and a PUSH socket connected to it.

    ``log()`` is non-blocking; if the socket high-water mark is hit the
    record is dropped (and a warning is emitted). This keeps the request
    path immune to disk pressure or a stalled writer.
    """

    def __init__(
        self,
        socket_addr: str,
        process: mp.process.BaseProcess,
        ctx: zmq.Context,
        sock: zmq.Socket,
        output_path: str,
    ) -> None:
        self._socket_addr = socket_addr
        self._process = process
        self._ctx = ctx
        self._sock = sock
        self.output_path = output_path
        self._closed = False

    @classmethod
    def maybe_create(
        cls,
        args,
        rank: int = 0,
        total_ranks: int = 1,
    ) -> RequestLoggerHub | None:
        output_path = _resolve_log_path(args, rank, total_ranks)
        if output_path is None:
            return None

        # Use a per-pid IPC path so multiple workers don't collide.
        sock_name = f"vllm_req_log_{os.getpid()}_{rank}.sock"
        socket_addr = f"ipc://{os.path.join(tempfile.gettempdir(), sock_name)}"

        ready_event = mp.get_context("spawn").Event()
        process = mp.get_context("spawn").Process(
            target=writer_main,
            args=(socket_addr, output_path, ready_event),
            name=f"vllm-request-log-writer-{rank}",
            daemon=True,
        )
        process.start()
        try:
            ready_timeout = float(
                os.environ.get(
                    "VLLM_REQUEST_LOG_WRITER_READY_TIMEOUT_S",
                    WRITER_READY_TIMEOUT_S,
                )
            )
        except ValueError:
            ready_timeout = WRITER_READY_TIMEOUT_S
        if not ready_event.wait(timeout=ready_timeout):
            logger.warning(
                "Request log writer did not become ready within %.1fs; "
                "request logging disabled. The writer subprocess uses "
                "multiprocessing 'spawn', so it has to cold-import vllm; "
                "raise the budget via "
                "VLLM_REQUEST_LOG_WRITER_READY_TIMEOUT_S=<seconds> if "
                "your machine is slow.",
                ready_timeout,
            )
            with contextlib.suppress(Exception):
                process.terminate()
            return None

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PUSH)
        sock.setsockopt(zmq.SNDHWM, DEFAULT_SNDHWM)
        sock.setsockopt(zmq.LINGER, 1000)
        sock.connect(socket_addr)

        logger.info(
            "Request logging enabled; writing to %s (writer pid=%d)",
            output_path,
            process.pid,
        )
        return cls(socket_addr, process, ctx, sock, output_path)

    def log(self, record: dict[str, Any]) -> None:
        """Best-effort, non-blocking push of a single jsonl record."""
        if self._closed:
            return
        try:
            payload = json.dumps(record, ensure_ascii=False, default=str).encode(
                "utf-8"
            )
        except (TypeError, ValueError):
            logger.exception("request log: failed to serialize record")
            return
        try:
            self._sock.send_multipart([FRAME_RECORD, payload], flags=zmq.NOBLOCK)
        except zmq.Again:
            logger.warning(
                "request log: send queue full, dropping record %s",
                record.get("request_id"),
            )
        except zmq.ZMQError as e:
            logger.warning("request log: zmq send failed: %s", e)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._sock.send_multipart([FRAME_SHUTDOWN, b""], flags=zmq.NOBLOCK)
        with contextlib.suppress(Exception):
            self._sock.close(linger=1000)
        if self._process.is_alive():
            self._process.join(timeout=WRITER_SHUTDOWN_TIMEOUT_S)
            if self._process.is_alive():
                logger.warning(
                    "Request log writer did not exit within %.1fs; terminating.",
                    WRITER_SHUTDOWN_TIMEOUT_S,
                )
                with contextlib.suppress(Exception):
                    self._process.terminate()
        # Best-effort cleanup of the ipc socket file.
        if self._socket_addr.startswith("ipc://"):
            path = self._socket_addr[len("ipc://") :]
            with contextlib.suppress(OSError):
                os.unlink(path)


def make_record(
    *,
    raw_request,
    endpoint: str,
    request_obj,
    response: Any,
    received_at: float,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one jsonl record ready to be ``hub.log()``-ed.

    ``request_id`` prefers the inbound ``X-Request-Id`` header so that
    records correlate 1:1 with the caller's own id. If the header is
    absent we fall back to the vLLM-side id stashed on
    ``raw_request.state.request_metadata.request_id`` (e.g.
    ``chatcmpl-…`` / ``cmpl-…``), which is auto-generated per request.
    """
    request_id: str | None = None
    rendered_prompts: list[str | None] | None = None
    client_info: dict[str, Any] | None = None
    if raw_request is not None:
        request_id = raw_request.headers.get("X-Request-Id")
        if not request_id:
            meta = getattr(raw_request.state, "request_metadata", None)
            if meta is not None:
                request_id = getattr(meta, "request_id", None)
        rendered_prompts = getattr(raw_request.state, "rendered_prompts", None)
        client = raw_request.client
        ua = raw_request.headers.get("user-agent")
        if client is not None or ua is not None:
            client_info = {
                "ip": getattr(client, "host", None) if client else None,
                "user_agent": ua,
            }

    if hasattr(request_obj, "model_dump"):
        request_dict = request_obj.model_dump(exclude_none=True)
    else:
        request_dict = request_obj

    return {
        "request_id": request_id,
        "endpoint": endpoint,
        "received_at": received_at,
        "completed_at": time.time(),
        "client": client_info,
        "request": request_dict,
        "rendered_prompts": rendered_prompts,
        "response": response,
        "error": error,
    }


async def stream_logging_wrapper(
    generator: AsyncIterator[str],
    *,
    hub: RequestLoggerHub,
    aggregator,
    raw_request,
    endpoint: str,
    request_obj,
    received_at: float,
) -> AsyncIterator[str]:
    """Wrap an SSE generator so that every chunk is captured for logging.

    The aggregator is called once at the end (whether the stream finished
    cleanly or was cut off by an exception) to produce the response dict.
    """
    chunks: list[str] = []
    try:
        async for chunk in generator:
            chunks.append(chunk)
            yield chunk
    finally:
        try:
            response = aggregator(chunks)
            hub.log(
                make_record(
                    raw_request=raw_request,
                    endpoint=endpoint,
                    request_obj=request_obj,
                    response=response,
                    received_at=received_at,
                )
            )
        except Exception:
            logger.exception("request log: failed to aggregate/push stream")
