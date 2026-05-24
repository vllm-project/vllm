# SPDX-License-Identifier: Apache-2.0
"""P50 — cliproxyapi middleware for Genesis response cache (P41).

This module provides a `ResponseCacheMiddleware` ASGI-compatible
middleware that intercepts `POST /v1/chat/completions` and `POST
/v1/completions` at the proxy layer (cliproxyapi on port 8330). On
cache hit the middleware returns the cached response WITHOUT
forwarding the request to the vLLM backend, saving 100% of decode
cost for repeated identical queries.

Why at the proxy layer
----------------------
vLLM does not expose a clean middleware hook for response caching
(its FastAPI app is built internally and plugins cannot reach it).
Our production stack already has cliproxyapi running as a front
proxy (port 8330, see `project_genesis_cliproxyapi_management`
memory entry). Plugging the cache there is:

- zero-code-change in vLLM itself
- trivially revertable (remove the middleware registration)
- observable via cliproxyapi's existing metrics/logging
- cross-process by design (multiple client apps hit one proxy)

Scope (v7.8)
------------
- Exact-match only (uses P41 `ResponseCacheLRU` / `RedisResponseCache`
  under the hood; semantic/interpolation cache P41b is SHELVED per
  user direction — hallucination risk).
- Streaming (`stream=True`) requests are NOT cache-eligible; the
  middleware skips cache lookup AND skips post-response store, so
  they pass through unchanged. SSE buffering + replay was originally
  planned for v7.9 but never shipped — see `build_cache_key_from_request`
  for the explicit early-return on `body.get("stream")`. G-005 audit
  fix 2026-05-02: docstring previously claimed "honours stream=True
  by buffering SSE and replaying on hit" which contradicted the
  actual code.
- Never caches responses for `temperature > 0` unless
  `allow_sampled=True` in the middleware config — non-deterministic
  output must not be resurrected from cache.
- Key includes: `(prompt, model, frozen_sampling_params)`. `stop`
  tokens, `max_tokens`, `top_p`, etc. are all part of the key.

Integration
-----------
The cliproxyapi operator registers the middleware via its FastAPI
app (typical pattern shown below; actual cliproxyapi API may differ
slightly — adapt to its hook):

    from fastapi import FastAPI
    from vllm._genesis.middleware import ResponseCacheMiddleware
    from vllm._genesis.cache.response_cache import get_default_cache

    app = FastAPI()
    cache = get_default_cache()  # None if GENESIS_ENABLE_P41_* unset
    if cache is not None:
        app.add_middleware(
            ResponseCacheMiddleware,
            cache=cache,
            paths=("/v1/chat/completions", "/v1/completions"),
            allow_sampled=False,
        )

Graceful degradation
--------------------
- If `cache is None` (P41 disabled), middleware is a pass-through.
- Redis timeouts / disconnects — cache reports miss, request
  forwards normally. Never blocks the request path.
- Malformed JSON body — middleware skips cache lookup, forwards
  raw body to vLLM.
- Unsupported endpoint (not in `paths`) — forwards unchanged.

Never raises exceptions that reach the client. Cache faults are
always silent relative to the user-visible path.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Status: v7.8 (opt-in; wire in cliproxyapi deployment)
"""
from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Mapping, Optional

log = logging.getLogger("genesis.middleware.response_cache")


# ── key extraction ─────────────────────────────────────────────────

def build_cache_key_from_request(
    body: Mapping[str, Any],
    *,
    allow_sampled: bool = False,
) -> Optional[tuple[str, str, dict]]:
    """Extract `(prompt, model, sampling_params)` triple from an
    OpenAI-compatible request body. Returns None if the request is
    NOT cache-eligible (streaming, sampled, missing model/prompt,
    malformed).

    For `/v1/completions`:
        - prompt: str or list[str] (first element used if list)
        - model: str

    For `/v1/chat/completions`:
        - messages: list[{"role": ..., "content": ...}] — serialised
          deterministically into a single string key
        - model: str
    """
    if not isinstance(body, dict):
        return None

    # Stream requests are NOT cached by default — their semantics
    # require partial delivery which cache replay complicates; we
    # implement streaming-aware caching in v7.9 if needed.
    if body.get("stream", False):
        return None

    # Non-deterministic sampling must not be cached unless operator
    # explicitly opts in (same prompt → different output each call).
    #
    # G-003 fix (audit 2026-05-02): wrap numeric coercion in try/except.
    # Malformed client payload like {"temperature": "abc"} raised
    # ValueError that propagated all the way up through _try_cache_lookup
    # → __call__, contradicting the "never raises / pass-through on
    # malformed body" contract documented at the top of this module.
    # Defensive: any non-coercible value is treated as "not cache-eligible"
    # → fall through to the downstream app which will return a proper
    # 4xx for the malformed input.
    temperature = body.get("temperature", 0.0)
    top_p = body.get("top_p", 1.0)
    top_k = body.get("top_k", -1)
    if not allow_sampled:
        try:
            t_val = float(temperature) if temperature is not None else 0.0
            tk_val = int(top_k) if top_k is not None else -1
            tp_val = float(top_p) if top_p is not None else 1.0
        except (TypeError, ValueError):
            return None
        if t_val > 0.0 or tk_val > 1 or tp_val < 1.0:
            return None

    model = body.get("model")
    if not isinstance(model, str) or not model:
        return None

    # Build prompt string
    if "messages" in body:
        messages = body.get("messages")
        if not isinstance(messages, list):
            return None
        # Deterministic serialisation — same messages → same key.
        try:
            prompt = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            return None
    elif "prompt" in body:
        p = body.get("prompt")
        if isinstance(p, list):
            if not p or not all(isinstance(x, str) for x in p):
                return None
            prompt = p[0] if len(p) == 1 else json.dumps(
                p, ensure_ascii=False,
            )
        elif isinstance(p, str):
            prompt = p
        else:
            return None
    else:
        return None

    # Distil stable sampling params — sort_keys via P41's _stable_key
    # does the heavy lifting downstream.
    sp: dict[str, Any] = {}
    for k in (
        "temperature", "top_p", "top_k", "max_tokens",
        "presence_penalty", "frequency_penalty",
        "repetition_penalty", "stop", "seed",
        "tool_choice", "response_format",
    ):
        if k in body:
            sp[k] = body[k]
    return prompt, model, sp


# ── ASGI middleware ────────────────────────────────────────────────

class ResponseCacheMiddleware:
    """ASGI middleware (Starlette-compatible) for short-circuiting
    cache-hit requests.

    Constructor params
    ------------------
    - `app`: the downstream ASGI app (FastAPI / Starlette).
    - `cache`: a `ResponseCacheLRU` / `RedisResponseCache` instance
      (anything with `.get(prompt, model, params)` +
      `.store(prompt, model, params, response)`).
    - `paths`: iterable of URL paths to intercept. Default:
      `("/v1/chat/completions", "/v1/completions")`.
    - `allow_sampled`: when True, caches even `temperature>0`
      requests. Default False — safer for quality.
    """

    def __init__(
        self,
        app,
        *,
        cache,
        paths: Iterable[str] = (
            "/v1/chat/completions", "/v1/completions",
        ),
        allow_sampled: bool = False,
    ):
        self.app = app
        self.cache = cache
        self.paths = tuple(paths)
        self.allow_sampled = bool(allow_sampled)

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "").upper()
        path = scope.get("path", "")
        if method != "POST" or path not in self.paths:
            await self.app(scope, receive, send)
            return

        # Buffer the request body so we can both parse it for key-
        # extraction AND replay it downstream on cache miss.
        body_chunks: list[bytes] = []
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] != "http.request":
                break
            body_chunks.append(message.get("body", b""))
            more_body = message.get("more_body", False)
        body_bytes = b"".join(body_chunks)

        # Attempt cache lookup
        cached = self._try_cache_lookup(body_bytes)
        if cached is not None:
            # HIT — synthesise a 200 OK response with the cached body.
            #
            # G-004 fix (audit 2026-05-02): _send_cached_response can fail
            # to deliver if the cached entry is non-JSON-serializable
            # (e.g. a corrupt entry written before a serialisation bug
            # was fixed). Returns False in that case; we then fall
            # through to _forward_and_store so the client always gets
            # SOME response. Old behavior left the connection hanging
            # because no http.response.start/body was ever sent.
            sent = await self._send_cached_response(send, cached)
            if sent:
                return
            # else: corrupt cached entry — fall through to MISS path

        # MISS — forward to downstream with buffered body, then
        # intercept the response and store on 2xx.
        await self._forward_and_store(scope, body_bytes, send)

    # ── internals ─────────────────────────────────────────────────

    def _try_cache_lookup(self, body_bytes: bytes) -> Optional[dict]:
        if not body_bytes:
            return None
        try:
            body = json.loads(body_bytes.decode("utf-8", errors="replace"))
        except (ValueError, UnicodeDecodeError):
            return None
        key_tuple = build_cache_key_from_request(
            body, allow_sampled=self.allow_sampled,
        )
        if key_tuple is None:
            return None
        prompt, model, params = key_tuple
        try:
            hit = self.cache.get(prompt, model, params)
        except Exception as e:
            # Cache errors must not propagate — log + miss.
            log.warning(
                "[P50 cache middleware] cache.get failed: %s",
                type(e).__name__,
            )
            return None
        return hit

    async def _send_cached_response(self, send, cached: dict) -> bool:
        """Emit a 200 OK with the cached JSON as body.

        Returns True if the response was sent successfully, False if the
        cached entry could not be serialised (caller must fall through
        to MISS path so the client still gets a response).

        G-004 fix (audit 2026-05-02): previously returned None on
        serialisation failure WITHOUT sending any response, leaving the
        ASGI connection hanging. Comment said "treating as miss" but
        the caller didn't have a way to detect that and re-route to
        _forward_and_store.
        """
        try:
            body = json.dumps(
                cached, ensure_ascii=False, default=str,
            ).encode("utf-8")
        except (TypeError, ValueError) as e:
            log.warning(
                "[P50 cache middleware] cached entry not JSON-serialisable "
                "(%s); treating as miss — caller will forward to downstream",
                type(e).__name__,
            )
            return False
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"application/json"),
                (b"x-genesis-cache", b"HIT"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body,
        })
        return True

    async def _forward_and_store(
        self, scope, body_bytes: bytes, send,
    ) -> None:
        """Replay the buffered body to downstream; intercept the response
        stream; store on 2xx completion."""
        replayed = {"done": False}

        async def _receive_replay():
            if replayed["done"]:
                # Downstream should not ask for more body after our
                # single chunk, but if it does, deliver empty-end.
                return {"type": "http.request", "body": b"", "more_body": False}
            replayed["done"] = True
            return {
                "type": "http.request",
                "body": body_bytes,
                "more_body": False,
            }

        response_status = {"code": 500}
        response_body_chunks: list[bytes] = []
        response_headers_out: list[tuple[bytes, bytes]] = []

        async def _send_intercept(message):
            mtype = message.get("type")
            if mtype == "http.response.start":
                response_status["code"] = int(message.get("status", 500))
                response_headers_out[:] = list(
                    message.get("headers", []),
                )
                # Inject a HIT/MISS marker for diagnostics.
                response_headers_out.append(
                    (b"x-genesis-cache", b"MISS"),
                )
                message = {
                    **message,
                    "headers": response_headers_out,
                }
            elif mtype == "http.response.body":
                response_body_chunks.append(message.get("body", b""))
            await send(message)

        await self.app(scope, _receive_replay, _send_intercept)

        # Store on success only — 2xx + non-empty body + parseable JSON.
        if 200 <= response_status["code"] < 300:
            full_body = b"".join(response_body_chunks)
            if full_body:
                try:
                    parsed = json.loads(
                        full_body.decode("utf-8", errors="replace"),
                    )
                except (ValueError, UnicodeDecodeError):
                    return
                try:
                    body = json.loads(
                        body_bytes.decode("utf-8", errors="replace"),
                    )
                except (ValueError, UnicodeDecodeError):
                    return
                key_tuple = build_cache_key_from_request(
                    body, allow_sampled=self.allow_sampled,
                )
                if key_tuple is None:
                    return
                prompt, model, params = key_tuple
                try:
                    self.cache.store(prompt, model, params, parsed)
                except Exception as e:
                    log.warning(
                        "[P50 cache middleware] cache.store failed: %s",
                        type(e).__name__,
                    )
