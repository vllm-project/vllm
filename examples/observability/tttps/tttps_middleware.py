# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TTTPS Proof-of-Time provenance middleware for vLLM's OpenAI-compatible
server.

TTTPS (OpenTTT) is a cryptographic audit-trail protocol: it timestamps and
hashes model output so a third party can later verify *when* a response was
produced and that it has not been altered, without vLLM needing any built-in
support for it. This example wires it in purely through vLLM's existing
`--middleware` CLI flag (see `vllm/entrypoints/openai/api_server.py`,
`build_app()`): any dotted-path Starlette `BaseHTTPMiddleware` subclass
passed via `--middleware` is added to the FastAPI app with
`app.add_middleware(...)`, the same extension point Prometheus/OpenTelemetry
style add-ons use.

Honest scope note: this attaches a cryptographic timestamp + integrity hash
("was this exact text produced at this time and unmodified since"). It does
NOT certify legal/regulatory compliance (EU AI Act, FDA, etc.) and makes no
claim about output correctness.

Start the server with the middleware attached:

    vllm serve facebook/opt-125m --middleware tttps_middleware.TTTPSMiddleware

See README.md in this directory for the full walkthrough, including how to
get a free API key from the public Provenance API this example calls.
"""

import hashlib
import json
import os
import time

import httpx
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse

# Public self-serve Provenance API. Get a free key with:
#   curl -X POST https://kpp.kenosian.com/v1/keys \
#     -H "Content-Type: application/json" \
#     -d '{"email": "you@example.com", "use_case": "vllm middleware demo"}'
KPP_BASE = os.environ.get("KPP_BASE", "https://kpp.kenosian.com")
KPP_API_KEY = os.environ.get("KPP_API_KEY", "")
KPP_TIMEOUT_S = float(os.environ.get("KPP_TIMEOUT_S", "1.0"))

_client = httpx.Client(timeout=KPP_TIMEOUT_S)


def _anchor(text: str) -> dict:
    """POST /v1/anchor against the public KPP endpoint. Fail-open: any
    error (timeout, quota exhaustion, bad key, network failure) degrades to
    a {"status": "degraded", ...} dict instead of raising, so a Provenance
    API outage never blocks or alters the underlying vLLM response."""
    content_hash = f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
    t0 = time.perf_counter()
    try:
        resp = _client.post(
            f"{KPP_BASE}/v1/anchor",
            json={"content_hash": content_hash, "key": KPP_API_KEY},
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        if resp.status_code == 200:
            body = resp.json()
            return {
                "status": "ok",
                "content_hash": content_hash,
                "receipt_id": body.get("receipt_id"),
                "receipt": body.get("receipt"),
                "time": body.get("time"),
                "time_source": body.get("time_source"),
                "verify_url": body.get("verify_url"),
                "overhead_ms": elapsed_ms,
            }
        reason = {402: "quota_exhausted", 403: "unknown_api_key"}.get(
            resp.status_code, f"http_{resp.status_code}"
        )
        return {"status": "degraded", "reason": reason, "overhead_ms": elapsed_ms}
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {"status": "degraded", "reason": type(e).__name__, "overhead_ms": elapsed_ms}


def _extract_text(body_bytes: bytes) -> str:
    """Best-effort text extraction from an OpenAI-compatible chat/completions
    JSON response body. Returns "" on any structural mismatch or on
    streaming responses (SSE, not JSON) -- nothing gets anchored if empty."""
    try:
        data = json.loads(body_bytes)
        choices = data.get("choices", [])
        if not choices:
            return ""
        choice = choices[0]
        if "message" in choice:
            return choice["message"].get("content", "") or ""
        if "text" in choice:
            return choice["text"] or ""
        return ""
    except Exception:
        return ""


class TTTPSMiddleware(BaseHTTPMiddleware):
    """Wraps `/v1/chat/completions` and `/v1/completions` responses with a
    TTTPS Proof-of-Time anchor, exposed as the `X-TTTPS-Receipt` response
    header. Fail-open: Provenance API errors or timeouts never affect the
    underlying vLLM response body or status code.

    Streaming (`stream=true`, SSE) responses are intentionally left
    untouched -- `BaseHTTPMiddleware` buffers non-streaming bodies only;
    anchoring a live SSE stream requires a different approach and is out of
    scope for this example."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        if request.url.path not in ("/v1/chat/completions", "/v1/completions"):
            return response

        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return response

        try:
            body = b"".join([chunk async for chunk in response.body_iterator])
        except Exception:
            return response

        headers = dict(response.headers)
        text = _extract_text(body)
        if text:
            headers["X-TTTPS-Receipt"] = json.dumps(_anchor(text))

        return StarletteResponse(
            content=body,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
        )
