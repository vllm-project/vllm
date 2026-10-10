# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ASGI middleware giving a live snapshot of in-flight completion requests.

Load it with the ``--middleware`` flag of ``vllm serve``::

    PYTHONPATH=/path/to/this/directory vllm serve <model> \\
        --middleware stream_tracker.StreamTracker

It observes -- never alters -- request and response frames for
``POST /v1/chat/completions`` and ``POST /v1/completions``, and answers
``GET /streams`` with a JSON snapshot of active and recently finished
requests (id, state, age, prompt preview, token count, live tokens/s).
Response bytes are forwarded verbatim and all tracking is best effort: an
internal failure never changes what the server sends.

This complements the aggregate surfaces (``/metrics``, ``/load``) and
OpenTelemetry tracing: those answer "how is the server doing" and "what
happened to a request", while ``/streams`` answers "what is every stream
doing right now" without per-request metric labels.
"""

import asyncio
import contextlib
import json
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

Scope = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

_TRACKED_PATHS = ("/v1/chat/completions", "/v1/completions")
_BODY_CAP = 512 * 1024  # bytes of request body retained for the preview
_PREVIEW_CHARS = 200
_CARRY = 24  # >= len(b"\ndata: [DONE]"); keeps split markers whole
_RECENT_MAX = 50
_ACTIVE_MAX = 256


def _round(value: float | None) -> float | None:
    """Round a float to 2dp; ``None`` is returned unchanged."""
    return None if value is None else round(value, 2)


class _Record:
    """Per-request tracking state. Never touched by more than one task at a
    time because the server runs a single event loop, so no lock is needed."""

    __slots__ = (
        "id",
        "path",
        "state",
        "model",
        "preview",
        "preview_truncated",
        "stream",
        "tokens",
        "status",
        "start",
        "first_chunk",
        "last_chunk",
        "end",
        "chunk_times",
        "carry",
        "sse",
        "body",
        "body_truncated",
        "body_parsed",
        "finalized",
    )

    def __init__(self, rid: int, path: str) -> None:
        self.id = rid
        self.path = path
        self.state = "pending"
        self.model: str | None = None
        self.preview: str | None = None
        self.preview_truncated = False
        self.stream = False
        self.tokens: int | None = None
        self.status: int | None = None
        self.start = time.monotonic()
        self.first_chunk: float | None = None
        self.last_chunk: float | None = None
        self.end: float | None = None
        self.chunk_times: deque[float] = deque(maxlen=48)
        self.carry = b"\n"  # primed so the first field line matches "\ndata:"
        self.sse = False
        self.body = bytearray()
        self.body_truncated = False
        self.body_parsed = False
        self.finalized = False


class StreamTracker:
    """Pure-ASGI middleware tracking concurrent completion streams."""

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app
        self._counter = 0
        self._active: dict[int, _Record] = {}
        self._recent: deque[_Record] = deque(maxlen=_RECENT_MAX)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        path = scope.get("path", "")

        if path == "/streams":
            if method == "GET":
                await self._send_snapshot(send)
                return
            await self.app(scope, receive, send)
            return

        if method == "POST" and path in _TRACKED_PATHS:
            await self._track(scope, receive, send)
            return

        await self.app(scope, receive, send)

    # -- snapshot ----------------------------------------------------------

    async def _send_snapshot(self, send: Send) -> None:
        try:
            body = json.dumps(self._snapshot()).encode("utf-8")
        except Exception:
            body = b'{"active": [], "recent": []}'
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    def _snapshot(self) -> dict[str, list[dict[str, Any]]]:
        now = time.monotonic()
        active = [
            self._entry(r, now, False) for r in reversed(list(self._active.values()))
        ]
        recent = [self._entry(r, now, True) for r in reversed(self._recent)]
        return {"active": active, "recent": recent}

    def _entry(self, r: _Record, now: float, recent: bool) -> dict[str, Any]:
        age = now - r.start
        idle = (now - r.last_chunk) if r.last_chunk is not None else age

        if recent:
            tok_s: float | None = None
            if (
                r.tokens is not None
                and r.first_chunk is not None
                and r.end is not None
                and r.end > r.first_chunk
            ):
                tok_s = r.tokens / (r.end - r.first_chunk)
        else:
            tok_s = None
            if len(r.chunk_times) >= 2:
                span = r.chunk_times[-1] - r.chunk_times[0]
                if span > 0:
                    tok_s = (len(r.chunk_times) - 1) / span

        entry: dict[str, Any] = {
            "id": str(r.id),
            "state": r.state,
            "model": r.model,
            "preview": r.preview,
            "preview_truncated": bool(r.preview_truncated),
            "stream": bool(r.stream),
            "tokens": r.tokens,
            "tok_s": _round(tok_s),
            "idle_s": _round(idle),
            "age_s": _round(age),
            "status": r.status,
        }
        if recent:
            e2e = (r.end - r.start) if r.end is not None else None
            entry["e2e_s"] = _round(e2e)
        return entry

    # -- tracked request lifecycle ----------------------------------------

    async def _track(self, scope: Scope, receive: Receive, send: Send) -> None:
        rec = self._new_record(scope.get("path", ""))

        async def wrapped_receive() -> Message:
            message = await receive()
            # Tracking must never disturb the served request.
            with contextlib.suppress(Exception):
                self._observe_receive(rec, message)
            return message

        async def wrapped_send(message: Message) -> None:
            # Observe first, but forward regardless of any tracking fault.
            with contextlib.suppress(Exception):
                self._observe_send(rec, message)
            await send(message)

        try:
            await self.app(scope, wrapped_receive, wrapped_send)
        except asyncio.CancelledError:
            # Task cancellation is how the server aborts a request whose
            # client disconnected mid-stream -- not an application failure.
            if not rec.finalized:
                self._finalize(rec, "aborted")
            raise
        except BaseException:
            # An exception from the application is a terminal error;
            # re-raise it untouched.
            if not rec.finalized:
                self._finalize(rec, "error")
            raise
        finally:
            # Reached without a final body frame (e.g. client disconnect):
            # the request never completed, so record it as aborted.
            if not rec.finalized:
                self._finalize(rec, "aborted")

    def _new_record(self, path: str) -> _Record:
        self._counter += 1
        rec = _Record(self._counter, path)
        self._active[rec.id] = rec
        if len(self._active) > _ACTIVE_MAX:
            self._active.pop(next(iter(self._active)), None)
        return rec

    def _finalize(self, rec: _Record, state: str) -> None:
        if rec.finalized:
            return
        rec.finalized = True
        rec.state = state
        if rec.end is None:
            rec.end = time.monotonic()
        self._active.pop(rec.id, None)
        self._recent.append(rec)

    # -- frame observers ---------------------------------------------------

    def _observe_receive(self, rec: _Record, message: Message) -> None:
        if message.get("type") != "http.request":
            return
        body = message.get("body", b"") or b""
        if body:
            room = _BODY_CAP - len(rec.body)
            if room > 0:
                rec.body += body[:room]
                if len(body) > room:
                    rec.body_truncated = True
            else:
                rec.body_truncated = True
        if not message.get("more_body", False) and not rec.body_parsed:
            rec.body_parsed = True
            self._parse_body(rec)

    def _parse_body(self, rec: _Record) -> None:
        # All parse failures are swallowed: a preview is a convenience,
        # never a correctness requirement.
        try:
            data = json.loads(bytes(rec.body))
            if isinstance(data, dict):
                model = data.get("model")
                if isinstance(model, str):
                    rec.model = model
                rec.stream = bool(data.get("stream", False))
                preview = _extract_preview(rec.path, data)
                if isinstance(preview, str):
                    if len(preview) > _PREVIEW_CHARS:
                        preview = preview[:_PREVIEW_CHARS]
                        rec.preview_truncated = True
                    rec.preview = preview
        except Exception:
            pass
        if rec.body_truncated:
            rec.preview_truncated = True

    def _observe_send(self, rec: _Record, message: Message) -> None:
        mtype = message.get("type")
        if mtype == "http.response.start":
            rec.status = message.get("status")
            for name, value in message.get("headers") or []:
                if name.lower() == b"content-type" and value.strip().lower().startswith(
                    b"text/event-stream"
                ):
                    rec.sse = True
                    break
        elif mtype == "http.response.body":
            body = message.get("body", b"") or b""
            more = message.get("more_body", False)
            now = time.monotonic()
            if rec.sse and body:
                if rec.first_chunk is None:
                    rec.first_chunk = now
                    rec.state = "streaming"
                # Anchor markers to a preceding newline: raw newlines cannot
                # occur inside JSON string payloads, so this matches SSE
                # framing only and never a "data:" substring within
                # generated content.
                buf = rec.carry + body
                delta = (buf.count(b"\ndata:") - rec.carry.count(b"\ndata:")) - (
                    buf.count(b"\ndata: [DONE]") - rec.carry.count(b"\ndata: [DONE]")
                )
                rec.tokens = (rec.tokens or 0) + delta
                rec.carry = buf[-_CARRY:]
                rec.chunk_times.append(now)
                rec.last_chunk = now
            if not more:
                rec.end = now
                state = (
                    "error"
                    if (rec.status is not None and rec.status >= 400)
                    else "done"
                )
                self._finalize(rec, state)


def _extract_preview(path: str, data: dict[str, Any]) -> str | None:
    """Best-effort prompt preview from a parsed request body."""
    if path == "/v1/chat/completions":
        messages = data.get("messages")
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        parts = [
                            p.get("text")
                            for p in content
                            if isinstance(p, dict) and isinstance(p.get("text"), str)
                        ]
                        return " ".join(parts)
                    return None
        return None

    prompt = data.get("prompt")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], str):
        return prompt[0]
    return None
