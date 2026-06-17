# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""API-server-side client for the request/response logger subprocess."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import tempfile
import time
from collections.abc import AsyncIterator
from typing import Any

import msgspec
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

# Reusable encoder for the hot-path log() call. msgpack is C-implemented
# and on par with pickle for our payload shape, while staying within the
# project's "no pickle" policy and the safer-by-default trust boundary.
_MSGPACK_ENCODER = msgspec.msgpack.Encoder()


def serialize_sampling_params(params: Any) -> dict[str, Any] | None:
    """Convert a ``SamplingParams`` / ``BeamSearchParams`` instance into a
    plain dict suitable for inclusion in a jsonl record.

    The inputs are msgspec.Struct subclasses. We use ``msgspec.to_builtins``
    with an ``enc_hook`` that falls back to ``str`` so unknown / callable
    fields (e.g. ``logits_processors``) don't blow up the whole encode.
    Returns ``None`` if conversion fails for any reason — never raises,
    so a logging hiccup can't impact the request path.
    """
    if params is None:
        return None
    try:
        return msgspec.to_builtins(params, enc_hook=str)
    except Exception:
        try:
            # Last-ditch fallback: msgspec dataclasses expose dataclasses-style
            # __struct_fields__ that we can dump field-by-field.
            return {
                name: getattr(params, name, None)
                for name in getattr(params, "__struct_fields__", ())
            }
        except Exception:
            return None


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
        """Best-effort, non-blocking push of a single jsonl record.

        We serialize with ``msgspec.msgpack`` here — C-implemented, fast
        — and let the writer subprocess do the (slower) ``json.dumps``
        so the cost stays off the API server's event loop.
        """
        if self._closed:
            return
        try:
            payload = _MSGPACK_ENCODER.encode(record)
        except (msgspec.EncodeError, TypeError):
            logger.exception("request log: failed to encode record")
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
    received_at: float,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one jsonl record ready to be ``hub.log()``-ed.

    The record only carries the *raw* input/output (rendered prompt, model
    output before any parser) plus minimal metadata. The original request
    body and the structured response are intentionally NOT persisted —
    they can be reconstructed from the raw text if needed and are too
    expensive to serialize on the hot path.

    ``request_id`` prefers the inbound ``X-Request-Id`` header so records
    correlate 1:1 with the caller's own id. If the header is absent we
    fall back to ``raw_request.state.request_metadata.request_id`` (the
    auto-generated ``chatcmpl-…`` / ``cmpl-…``).
    """
    request_id: str | None = None
    rendered_prompts: list[str | None] | None = None
    raw_output_texts: list[str] | None = None
    sampling_params: dict[str, Any] | None = None
    client_info: dict[str, Any] | None = None
    if raw_request is not None:
        request_id = raw_request.headers.get("X-Request-Id")
        meta = getattr(raw_request.state, "request_metadata", None)
        if meta is not None:
            if not request_id:
                request_id = getattr(meta, "request_id", None)
            raw_output_texts = getattr(meta, "raw_output_texts", None)
            sampling_params = getattr(meta, "sampling_params", None)
        rendered_prompts = getattr(raw_request.state, "rendered_prompts", None)
        client = raw_request.client
        ua = raw_request.headers.get("user-agent")
        if client is not None or ua is not None:
            client_info = {
                "ip": getattr(client, "host", None) if client else None,
                "user_agent": ua,
            }

    return {
        "request_id": request_id,
        "endpoint": endpoint,
        "received_at": received_at,
        "completed_at": time.time(),
        "client": client_info,
        "sampling_params": sampling_params,
        "rendered_prompts": rendered_prompts,
        "raw_output_texts": raw_output_texts,
        "error": error,
    }


async def stream_logging_wrapper(
    generator: AsyncIterator[str],
    *,
    hub: RequestLoggerHub,
    raw_request,
    endpoint: str,
    received_at: float,
) -> AsyncIterator[str]:
    """Wrap an SSE generator so the request log fires once at stream end.

    We do not collect the SSE chunks themselves — the raw model output is
    captured upstream into ``request_metadata.raw_output_texts`` by the
    serving layer, and ``make_record`` reads it back. This wrapper just
    waits for the generator to finish (or be aborted) and writes one
    record.
    """
    try:
        async for chunk in generator:
            yield chunk
    finally:
        try:
            hub.log(
                make_record(
                    raw_request=raw_request,
                    endpoint=endpoint,
                    received_at=received_at,
                )
            )
        except Exception:
            logger.exception("request log: failed to push stream record")
