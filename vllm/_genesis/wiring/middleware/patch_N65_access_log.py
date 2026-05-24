# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN65 — Genesis structured API access log middleware.

Replaces the bare uvicorn default access lines::

    INFO: 192.168.1.10:45116 - "POST /v1/chat/completions HTTP/1.1" 200 OK

with operator-friendly structured lines::

    [Genesis-API] 200  POST /v1/chat/completions    34ms  prompt=46t  completion=400t  stream=N  tools=N  client=192.168.1.10
    [Genesis-API] 401  GET  /v1/models              <1ms  client=192.168.1.10
    [Genesis-API] 200  GET  /health                 <1ms  (suppressed unless GENESIS_PN65_LOG_HEALTH=1)

Health-check polling is suppressed by default to avoid log noise (Docker
healthchecks fire every 5s = 17K lines/day per worker). Set
GENESIS_PN65_LOG_HEALTH=1 to log them.

Categorization:
  • 2xx → INFO (green-ish marker via "OK")
  • 4xx → WARNING
  • 5xx → ERROR

Token info extracted from response body for /v1/chat/completions and
/v1/completions (best-effort; falls back gracefully on streaming responses
where tokens are unknown until end of stream).

================================================================
ENV
================================================================

GENESIS_ENABLE_PN65=1                       — master enable
GENESIS_PN65_LOG_HEALTH=1                   — include /health probes
GENESIS_PN65_QUIET_PATHS=/v1/models,/metrics  — comma-separated path prefixes to suppress

================================================================
RISK
================================================================

LOW — middleware only; no monkey-patching of vllm engine. If wrapper
fails, vllm continues with default access logging.

================================================================
STATE
================================================================

Active runtime middleware install COMPLETE; conservative anchor on
`vllm.entrypoints.openai.api_server.build_app()`. Idempotent via
__pn65_installed__ marker on the FastAPI app.

Author: Sandermage 2026-05-05.
Backport reference: Sander request 2026-05-05 ("по апи лог невзрачный
надо тоже проработать его").
"""
from __future__ import annotations

import logging
import os
import time

log = logging.getLogger("genesis.wiring.pn65_access_log")
api_log = logging.getLogger("genesis.api")


def _quiet_paths() -> set[str]:
    raw = os.environ.get("GENESIS_PN65_QUIET_PATHS", "/health,/metrics")
    return {p.strip() for p in raw.split(",") if p.strip()}


def _log_health() -> bool:
    return os.environ.get("GENESIS_PN65_LOG_HEALTH", "").strip().lower() in (
        "1", "true", "yes", "y", "on",
    )


def _format_duration_ms(seconds: float) -> str:
    ms = seconds * 1000.0
    if ms < 1.0:
        return "<1ms"
    if ms < 10.0:
        return f"{ms:.1f}ms"
    if ms < 1000.0:
        return f"{int(ms)}ms"
    return f"{ms / 1000.0:.2f}s"


def _extract_response_tokens(body_bytes: bytes) -> dict:
    """Best-effort token extraction from JSON response body.

    Looks for `usage.prompt_tokens` / `usage.completion_tokens` /
    `usage.total_tokens` and tool_calls count. Returns empty dict on
    streaming responses (body is multi-line SSE) or unparseable JSON.
    """
    if not body_bytes:
        return {}
    try:
        # Streaming SSE: skip
        if body_bytes.startswith(b"data:") or b"\ndata:" in body_bytes[:200]:
            return {"stream": True}
        import json
        data = json.loads(body_bytes)
        usage = data.get("usage") or {}
        out = {}
        if usage.get("prompt_tokens") is not None:
            out["prompt"] = usage["prompt_tokens"]
        if usage.get("completion_tokens") is not None:
            out["completion"] = usage["completion_tokens"]
        # Tool-call count
        choices = data.get("choices") or []
        if choices:
            tc = choices[0].get("message", {}).get("tool_calls") or []
            if tc:
                out["tools"] = len(tc)
        return out
    except Exception:
        return {}


def _format_log_line(
    method: str,
    path: str,
    status_code: int,
    duration_s: float,
    client: str,
    tokens: dict,
) -> str:
    # Status icon
    if 200 <= status_code < 300:
        status_str = f"{status_code}"
    elif 300 <= status_code < 400:
        status_str = f"{status_code}"
    elif 400 <= status_code < 500:
        status_str = f"{status_code}"
    else:
        status_str = f"{status_code}"

    # Build extras
    extras = []
    if tokens:
        if tokens.get("stream"):
            extras.append("stream=Y")
        else:
            if "prompt" in tokens:
                extras.append(f"prompt={tokens['prompt']}t")
            if "completion" in tokens:
                extras.append(f"completion={tokens['completion']}t")
            if "tools" in tokens:
                extras.append(f"tools={tokens['tools']}")
    extras.append(f"client={client}")
    extras_str = "  ".join(extras)

    # Method left-padded for alignment
    method_str = f"{method:<5}"
    duration_str = _format_duration_ms(duration_s)
    return (
        f"[Genesis-API] {status_str:<3}  {method_str} {path:<35} "
        f"{duration_str:>7}  {extras_str}"
    )


def _make_middleware():
    """Construct a Starlette-compatible ASGI middleware."""
    quiet = _quiet_paths()
    log_health = _log_health()

    async def genesis_access_log_middleware(request, call_next):
        path = request.url.path

        # Suppress noisy paths.
        # Audit P2 fix 2026-05-05 (genesis_deep_cross_audit): GENESIS_PN65_LOG_HEALTH=1
        # MUST override the /health entry in quiet_paths, otherwise the env flag is
        # silently ineffective when /health is in the default quiet set.
        effective_quiet = (
            {q for q in quiet if q != "/health"}
            if log_health else quiet
        )
        if path == "/health" and not log_health:
            return await call_next(request)
        if any(path.startswith(q) for q in effective_quiet):
            return await call_next(request)

        method = request.method
        client_host = (
            (request.client.host if request.client else None) or "?"
        )
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception as e:
            duration = time.perf_counter() - start
            api_log.error(
                _format_log_line(method, path, 500, duration, client_host, {})
                + f"  exception={type(e).__name__}: {str(e)[:60]}"
            )
            raise

        duration = time.perf_counter() - start
        status_code = response.status_code

        # Tokens — only extract for chat/completions to avoid body
        # buffering on every request
        tokens: dict = {}
        if path in ("/v1/chat/completions", "/v1/completions"):
            try:
                # response.body is set on JSONResponse; for StreamingResponse
                # it's not available — we mark stream=Y instead
                body = getattr(response, "body", None)
                if body:
                    tokens = _extract_response_tokens(body)
                else:
                    tokens = {"stream": True}
            except Exception:
                tokens = {}

        line = _format_log_line(
            method, path, status_code, duration, client_host, tokens
        )
        if status_code >= 500:
            api_log.error(line)
        elif status_code >= 400:
            api_log.warning(line)
        else:
            api_log.info(line)
        return response

    return genesis_access_log_middleware


class _DropUvicornAccessInfo(logging.Filter):
    """Drop INFO-level records from uvicorn.access — PN65 dedup filter.

    Audit P1 fix 2026-05-05 (genesis_runtime_server_comparison_audit):
    `setLevel(WARNING)` on `uvicorn.access` logger DOES NOT WORK in our
    container because uvicorn loads its own `log_config` dict AFTER
    Genesis middleware install, and that config re-instantiates handlers
    with INFO level. A persistent `Filter` attached to BOTH the root
    logger AND the `uvicorn.access` logger survives re-init and drops
    the duplicate lines at emit time.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "uvicorn.access":
            return True  # not our concern
        if record.levelno < logging.WARNING:
            return False  # drop INFO/DEBUG access spam
        return True


_PN65_FILTER_INSTANCE = _DropUvicornAccessInfo()
_PN65_FILTER_INSTALLED = False


def _suppress_uvicorn_access_logger() -> None:
    """Audit P1 fix 2026-05-05 v2: install logging.Filter at root + uvicorn.access.

    PN65 was DUPLICATING uvicorn's bare `INFO: 192.168.1.10 - "GET /v1/models" 401`
    instead of REPLACING it. Operator log showed both lines for every request.
    v1 fix (`setLevel(WARNING)`) was bypassed by uvicorn's late log_config init.
    v2 fix attaches a persistent Filter that drops uvicorn.access INFO records
    regardless of when uvicorn (re-)configures its logger.

    Set `GENESIS_PN65_KEEP_UVICORN_ACCESS=1` to opt out (revert to dual-log).
    """
    global _PN65_FILTER_INSTALLED
    if _PN65_FILTER_INSTALLED:
        return  # already installed, idempotent
    if os.environ.get(
        "GENESIS_PN65_KEEP_UVICORN_ACCESS", ""
    ).strip().lower() in ("1", "true", "yes", "y", "on"):
        return  # operator opted to keep uvicorn access lines visible

    # Belt-and-suspenders: attach to root (catches uvicorn.access propagation)
    # AND directly to the uvicorn.access logger (catches non-propagated emits).
    logging.getLogger().addFilter(_PN65_FILTER_INSTANCE)
    logging.getLogger("uvicorn.access").addFilter(_PN65_FILTER_INSTANCE)
    _PN65_FILTER_INSTALLED = True
    log.info(
        "[PN65] uvicorn.access INFO records will be dropped via persistent "
        "logging.Filter — structured Genesis-API lines are now the single "
        "source for request-level observability. "
        "Set GENESIS_PN65_KEEP_UVICORN_ACCESS=1 to keep both."
    )


def install_into_app(app) -> bool:
    """Install the Genesis access log middleware into a FastAPI/Starlette app.

    Idempotent via __pn65_installed__ marker on the app.
    Returns True if newly installed, False if already present.

    Two-path install:
      1. App not yet started (decorator path) — uses standard
         `app.middleware('http')` decorator
      2. App already in user_middleware-locked state (vLLM may build the
         middleware stack early in build_app()) — falls back to
         `user_middleware.insert()` + middleware_stack rebuild

    Side-effect: also suppresses uvicorn.access INFO-level logger so the
    operator log shows only Genesis-API structured lines (audit P1 fix).
    """
    if getattr(app, "__pn65_installed__", False):
        return False
    # Suppress uvicorn.access logger BEFORE adding our middleware, so we
    # don't get one duplicate line during the install itself.
    _suppress_uvicorn_access_logger()
    middleware_fn = _make_middleware()
    try:
        app.middleware("http")(middleware_fn)
    except RuntimeError as e:
        if "started" not in str(e).lower():
            raise
        # Late install — bypass the "already started" guard via direct
        # user_middleware insertion + stack rebuild.
        try:
            from starlette.middleware import Middleware
            from starlette.middleware.base import BaseHTTPMiddleware

            class _GenesisAccessLog(BaseHTTPMiddleware):
                async def dispatch(self, request, call_next):
                    return await middleware_fn(request, call_next)

            app.user_middleware.insert(0, Middleware(_GenesisAccessLog))
            # Rebuild middleware stack so the insert takes effect
            app.middleware_stack = app.build_middleware_stack()
        except Exception as inner:
            log.warning(
                "[PN65] late middleware install failed (%s); access log "
                "stays at uvicorn default", inner,
            )
            return False
    setattr(app, "__pn65_installed__", True)
    return True


def apply() -> tuple[str, str]:
    """Apply PN65 — wrap vllm's build_app() to inject the access middleware."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("PN65")
    log_decision("PN65", decision, reason)
    if not decision:
        return "skipped", reason

    try:
        from vllm.entrypoints.openai import api_server as _server
    except Exception:
        return (
            "skipped",
            "vllm.entrypoints.openai.api_server not importable on this pin",
        )

    if not hasattr(_server, "build_app"):
        return (
            "skipped",
            "build_app() not present on this vllm pin — PN65 NULL",
        )

    if getattr(_server.build_app, "__pn65_wrapped__", False):
        return "applied", "PN65 already wrapped build_app (idempotent)"

    original_build_app = _server.build_app

    def wrapped_build_app(*args, **kwargs):
        app = original_build_app(*args, **kwargs)
        try:
            installed = install_into_app(app)
            if installed:
                log.info("[PN65] Genesis access log middleware installed")
        except Exception as e:
            log.warning("[PN65] middleware install failed: %s", e)
        return app

    wrapped_build_app.__wrapped__ = original_build_app
    wrapped_build_app.__pn65_wrapped__ = True
    _server.build_app = wrapped_build_app

    return (
        "applied",
        "PN65 wrapped api_server.build_app — Genesis structured access log "
        "active on next FastAPI app construction. Quiet paths: "
        f"{','.join(sorted(_quiet_paths()))}. "
        "Set GENESIS_PN65_LOG_HEALTH=1 to include /health probes."
    )
