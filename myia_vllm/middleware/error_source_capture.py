"""Capture source identity of /v1/* requests (good AND bad traffic).

Diagnostic — created 2026-05-16 after we saw ~50 stale `qwen3.5-35b-a3b`
404s/hour from `172.27.0.1` (Docker gateway) with no way to tell whether the
caller was a host-side process, an OWUI instance, or external traffic reverse-
proxied via IIS. Captures source IP + key forwarding headers + User-Agent +
requested model on every chat-completions request, regardless of status.

User explicitly asked (2026-05-16) to log both the good AND the bad traffic,
so we can characterize the legitimate sources too — not only the 4xx noise.

Filter after the fact with `jq 'select(.status >= 400)' /logs/error_sources.jsonl`
or `jq -s 'group_by(.user_agent) | map({ua: .[0].user_agent, n: length})'`.

Output: /logs/error_sources.jsonl (one JSON line per request).

Body capture: first N + last N bytes of request body (default N=1500, env override
ERROR_SOURCE_BODY_BYTES). Head shows model/system prompt/messages start; tail shows
the latest user turn + sampling params. NOTE: bodies may contain user prompts,
chat history, and (rarely) secrets embedded by clients. Reading /logs/error_sources.jsonl
is privileged.

Wire-up: profile must mount middleware/ and set PYTHONPATH=/middleware, then
  --middleware error_source_capture.ErrorSourceCapture

Disable: remove the --middleware flag and restart.
"""

import asyncio
import json
import os
import time
from pathlib import Path

_BODY_BYTES = int(os.environ.get("ERROR_SOURCE_BODY_BYTES", "1500"))

# Oversize-request guard. Refuse generation requests whose body exceeds this many
# bytes with HTTP 413 BEFORE they reach the engine. Added 2026-06-24 after large
# claudish (Claude Code -> local Qwen) contexts of 200-762 KB were dragging the
# TurboQuant-k8v4 continuous batch down to ~0 tok/s for minutes, timing out the
# small co-scheduled dashboard-condensation calls (OpenAI/JS, <=59 KB).
#
# DEFAULT = 0 (DISABLED). The guard is OPT-IN. A 512 KB default-on guard caused a
# cluster-wide cascade outage the same day (2026-06-24): it hard-failed the slow
# z.ai->qwen fallback overflow instead of letting it through, so a degraded-but-
# working fallback became a hard error storm. The structural fix is elsewhere
# (Genesis P72/P74 batch unlock to 8192 + claudish-side concurrency cap), NOT
# body-size rejection. Set VLLM_MAX_REQUEST_BODY_BYTES=<bytes> to arm it; the
# profile keeps it explicitly =0. The check is Content-Length-based (no body
# buffering) and FAILS OPEN: any error in the guard lets the request through.
_MAX_BODY_BYTES = int(os.environ.get("VLLM_MAX_REQUEST_BODY_BYTES", "0"))

# Response body captured for non-2xx only (default 2 KiB). Empty for successful
# requests, so the field costs nothing on the hot path. 2 KiB is enough for
# vLLM's JSON error envelope, whose `message` field carries the reason
# ("maximum context length", validation detail, ...).
_ERR_BODY_BYTES = int(os.environ.get("ERROR_SOURCE_ERR_BODY_BYTES", "2048"))


class ErrorSourceCapture:
    def __init__(self, app):
        self.app = app
        out_dir = Path(os.environ.get("VLLM_LOG_DIR", "/logs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_path = out_dir / "error_sources.jsonl"
        self._lock = asyncio.Lock()

    def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return self.app(scope, receive, send)
        path = scope.get("path", "")
        if not (path.startswith("/v1/") or path.startswith("/openai/v1/")):
            return self.app(scope, receive, send)
        return self._intercept(scope, receive, send)

    async def _intercept(self, scope, receive, send):
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        client = scope.get("client") or ("-", 0)

        # --- Oversize-request guard (refuse giant requests up-front) ---
        # Use Content-Length to reject without buffering the body. FAIL OPEN:
        # any error here must never break a legitimate request.
        if _MAX_BODY_BYTES > 0:
            try:
                # "/v1/chat/completions" and "/v1/completions" both end this way.
                if scope.get("path", "").endswith("/completions"):
                    cl = headers.get("content-length")
                    if cl is not None and int(cl) > _MAX_BODY_BYTES:
                        await self._reject_oversize(
                            receive, send, scope, headers, client, int(cl)
                        )
                        return
            except Exception:
                pass

        body_chunks = bytearray()
        captured_model = None

        async def receive_wrapper():
            message = await receive()
            if message["type"] == "http.request":
                body_chunks.extend(message.get("body", b""))
                if not message.get("more_body", False):
                    nonlocal captured_model
                    try:
                        captured_model = json.loads(bytes(body_chunks)).get("model")
                    except Exception:
                        captured_model = None
            return message

        status_holder = {"status": 0}
        # Response body of FAILED requests only. Added 2026-10-24 after six HTTP
        # 400s whose cause could not be established: the request side is logged,
        # but vLLM does not print a 4xx reason at this log level, so "context too
        # long" and "malformed request" were indistinguishable after the fact.
        # Buffered only when status >= 400 (the 2xx/SSE path pays nothing) and
        # bounded, so a 200-streaming response never accumulates here.
        err_body = bytearray()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["status"] = message.get("status", 0)
            elif (message["type"] == "http.response.body"
                  and status_holder["status"] >= 400
                  and len(err_body) < _ERR_BODY_BYTES):
                chunk = message.get("body") or b""
                err_body.extend(bytes(chunk)[:_ERR_BODY_BYTES - len(err_body)])
            await send(message)

        await self.app(scope, receive_wrapper, send_wrapper)

        # Log every request — good AND bad. Filter with jq after the fact.
        if status_holder["status"] > 0:
            body_bytes = bytes(body_chunks)
            body_len = len(body_bytes)
            if body_len <= 2 * _BODY_BYTES:
                body_head = body_bytes.decode("utf-8", errors="replace")
                body_tail = ""
                body_truncated = 0
            else:
                body_head = body_bytes[:_BODY_BYTES].decode("utf-8", errors="replace")
                body_tail = body_bytes[-_BODY_BYTES:].decode("utf-8", errors="replace")
                body_truncated = body_len - 2 * _BODY_BYTES
            entry = {
                "ts": time.time(),
                "status": status_holder["status"],
                "path": scope.get("path", ""),
                "client": f"{client[0]}:{client[1]}",
                "model": captured_model,
                "user_agent": headers.get("user-agent", ""),
                "x_forwarded_for": headers.get("x-forwarded-for", ""),
                "x_real_ip": headers.get("x-real-ip", ""),
                "x_forwarded_host": headers.get("x-forwarded-host", ""),
                "host": headers.get("host", ""),
                "referer": headers.get("referer", ""),
                "auth_prefix": (headers.get("authorization", "")[:24] + "...")
                if headers.get("authorization") else "",
                "body_bytes": body_len,
                "body_truncated_bytes": body_truncated,
                "body_head": body_head,
                "body_tail": body_tail,
            }
            if status_holder["status"] >= 400 and err_body:
                # Why the request failed, from the server's own mouth.
                entry["error_body"] = bytes(err_body).decode(
                    "utf-8", errors="replace"
                )
            asyncio.get_running_loop().run_in_executor(
                None, _append_line, self.out_path, entry
            )

    async def _reject_oversize(self, receive, send, scope, headers, client, body_len):
        # Drain the request body so the HTTP/1.1 connection stays clean, then 413.
        # Draining reads ~512 KB+ off the socket (cheap) but never hits the GPU —
        # the expensive prefill/decode of the giant context is what we are avoiding.
        try:
            while True:
                m = await receive()
                if m["type"] == "http.disconnect":
                    break
                if m["type"] == "http.request" and not m.get("more_body", False):
                    break
        except Exception:
            pass

        payload = json.dumps({
            "error": {
                "message": (
                    f"Request body of {body_len} bytes exceeds the "
                    f"{_MAX_BODY_BYTES}-byte limit for this endpoint. "
                    "Reduce the prompt/context length and retry."
                ),
                "type": "invalid_request_error",
                "param": "messages",
                "code": "request_too_large",
            }
        }).encode("utf-8")
        try:
            await send({
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode("latin-1")),
                ],
            })
            await send({"type": "http.response.body", "body": payload})
        except Exception:
            pass

        # Log the rejection (same schema as captured requests, + oversize flag).
        entry = {
            "ts": time.time(),
            "status": 413,
            "path": scope.get("path", ""),
            "client": f"{client[0]}:{client[1]}",
            "model": None,
            "user_agent": headers.get("user-agent", ""),
            "x_forwarded_for": headers.get("x-forwarded-for", ""),
            "x_real_ip": headers.get("x-real-ip", ""),
            "x_forwarded_host": headers.get("x-forwarded-host", ""),
            "host": headers.get("host", ""),
            "referer": headers.get("referer", ""),
            "auth_prefix": (headers.get("authorization", "")[:24] + "...")
            if headers.get("authorization") else "",
            "body_bytes": body_len,
            "body_truncated_bytes": 0,
            "body_head": "",
            "body_tail": "",
            "oversize_rejected": True,
        }
        try:
            asyncio.get_running_loop().run_in_executor(
                None, _append_line, self.out_path, entry
            )
        except Exception:
            pass


def _append_line(path, entry):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
