"""
X-Trust Middleware for vLLM
Annotates requests with human presence score via X-Trust header.
Zero KYC, zero PII. Doctrine AIR: annotate, never block.
github.com/htl-syterme/htl-core
"""
import base64
import hashlib
import hmac
import json
import time
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class XTrustMiddleware(BaseHTTPMiddleware):
    """
    Validates X-Trust header and annotates request state.
    Never blocks — doctrine AIR.
    """

    def __init__(self, app, secret: str, min_score: float = 0.0):
        super().__init__(app)
        self.secret = secret
        self.min_score = min_score

    def _verify(self, token: str) -> Optional[dict]:
        try:
            parts = token.split(".")
            if len(parts) != 3 or parts[0] != "v1":
                return None
            _, payload_b64, sig_b64 = parts

            def b64url_decode(s: str) -> bytes:
                padding = 4 - len(s) % 4
                return base64.b64decode(
                    s.replace("-", "+").replace("_", "/") + "=" * padding
                )

            expected = hmac.new(
                self.secret.encode(),
                payload_b64.encode(),
                hashlib.sha256,
            ).digest()
            sig = b64url_decode(sig_b64)
            if not hmac.compare_digest(expected, sig):
                return None
            payload = json.loads(b64url_decode(payload_b64))
            now = int(time.time())
            if now > payload.get("exp", 0):
                return None
            if now - payload.get("iat", 0) > 120:
                return None
            score = payload.get("score", 0)
            if not (0 <= score <= 1):
                return None
            return payload
        except Exception:
            return None

    async def dispatch(self, request: Request, call_next) -> Response:
        token = request.headers.get("x-trust", "")
        payload = self._verify(token) if token else None
        score = payload["score"] if payload else 0.0
        trusted = payload is not None and score >= self.min_score
        request.state.x_trust = {
            "trusted": trusted,
            "score": score,
            "annotated": True,
        }
        response = await call_next(request)
        return response
