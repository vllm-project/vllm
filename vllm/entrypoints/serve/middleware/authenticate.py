# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import secrets
from collections.abc import Awaitable

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

# The paths that answer without the API key: the liveness and readiness
# probes, and the load and version endpoints. Every other path on the app
# needs a bearer token. This includes the inference routes, /tokenize (it
# renders arbitrary text through the chat template), /metrics and the docs.
# A route that is added later is therefore guarded by default. A scraper
# that cannot send the key belongs on a separate listener, not in this set.
UNGUARDED_PATHS = frozenset({"/health", "/ping", "/load", "/version"})


def _is_cors_preflight(scope: Scope, headers: Headers) -> bool:
    """Return True for a CORS preflight, which browsers send without a token.

    This is the same test that Starlette's CORSMiddleware uses before it
    answers a preflight itself. Both headers are necessary. Without "origin",
    CORSMiddleware sends the request through to the app, and a mount such as
    /metrics answers any method. A bare OPTIONS request must therefore pass
    the token check like every other request.
    """
    return (
        scope.get("method") == "OPTIONS"
        and "origin" in headers
        and "access-control-request-method" in headers
    )


class AuthenticationMiddleware:
    """Pure ASGI middleware that authenticates each request by checking
    if the Authorization Bearer token exists and equals anyof "{api_key}".

    Notes
    -----
    There are two cases in which authentication is skipped:
        1. The request is a CORS preflight: OPTIONS with an Origin header
           and an Access-Control-Request-Method header.
        2. The request path, ignoring a trailing slash, is one of
           UNGUARDED_PATHS (e.g. /health).

    """

    def __init__(self, app: ASGIApp, tokens: list[str]) -> None:
        self.app = app
        self.api_tokens = [hashlib.sha256(t.encode("utf-8")).digest() for t in tokens]

    def verify_token(self, headers: Headers) -> bool:
        authorization_header_value = headers.get("Authorization")
        if not authorization_header_value:
            return False

        scheme, _, param = authorization_header_value.partition(" ")
        if scheme.lower() != "bearer":
            return False

        param_hash = hashlib.sha256(param.encode("utf-8")).digest()

        token_match = False
        for token_hash in self.api_tokens:
            token_match |= secrets.compare_digest(param_hash, token_hash)

        return token_match

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] not in ("http", "websocket"):
            # scope["type"] can be "lifespan" or "startup" for example,
            # in which case we don't need to do anything
            return self.app(scope, receive, send)
        root_path = scope.get("root_path", "")
        url_path = scope["path"].removeprefix(root_path)
        # This middleware runs ahead of the router, so a path that differs from
        # an allowlisted one only by a trailing slash never reaches FastAPI's
        # redirect_slashes: match on the normalized path, or a liveness probe
        # configured as /health/ is answered with 401.
        probe_path = url_path.rstrip("/") or "/"
        headers = Headers(scope=scope)
        if _is_cors_preflight(scope, headers):
            return self.app(scope, receive, send)
        # Type narrow to satisfy mypy.
        if probe_path not in UNGUARDED_PATHS and not self.verify_token(headers):
            response = JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return response(scope, receive, send)
        return self.app(scope, receive, send)
