# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import secrets
from collections.abc import Awaitable, Callable, Sequence

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.routing import BaseRoute, Mount
from starlette.types import ASGIApp, Receive, Scope, Send

from vllm.logger import init_logger

logger = init_logger(__name__)

# Endpoints that stay reachable without an API key when `--api-key` is
# configured: health probes, the SageMaker `/ping` contract, metrics
# scraping and the API docs. Every other route registered on the app is
# authenticated. The guarded set is derived from the app's live route
# table (see `build_guarded_prefixes`), so it cannot drift from the set of
# endpoints the server actually exposes: a newly registered route is
# protected by default.
PUBLIC_ENDPOINTS = (
    "/health",
    "/ping",
    "/load",
    "/version",
    "/metrics",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
    "/openapi.json",
    "/static",
)

# Public endpoints that legitimately match no registered route (e.g. the
# API docs on a server started with `--disable_fastapi_docs`); the stale
# entry warning in `build_guarded_prefixes` stays quiet about them.
OPTIONAL_PUBLIC_ENDPOINTS = frozenset(
    ("/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json", "/static")
)

# Fail-closed floor: even if route introspection yields nothing usable,
# these prefixes stay guarded. Derived prefixes can only extend this set,
# never shrink it.
MIN_GUARDED_PREFIXES = ("/v1", "/v2", "/inference", "/cohere")


def is_public_path(path: str, public: str) -> bool:
    """Check whether `path` is covered by the public endpoint `public`.

    Matching is exact or on path-segment boundaries, so `/healthcheck` is
    not whitened by the public endpoint `/health`.
    """
    return path == public or path.startswith(public.rstrip("/") + "/")


def guard_prefix(path_template: str) -> str | None:
    """Return the longest static prefix of a route path template.

    `/v1/responses/{response_id}/cancel` yields `/v1/responses/`: every
    URL under it is guarded. The over-approximation is deliberate — an
    extra 401 for a URL that matches no route is harmless, a missed 401
    is not.
    """
    cut = path_template.find("{")
    if cut != -1:
        path_template = path_template[:cut]
    return path_template or None


def build_guarded_prefixes(routes: Sequence[BaseRoute]) -> tuple[str, ...]:
    """Derive the guarded URL prefixes from the app's route table.

    This is the single source of truth for what `--api-key` protects: the
    prefixes are computed from the routes actually registered on the app
    (core routers, dev-mode routers and endpoint plugin routes alike), so
    the guard cannot drift from the served API surface. Routes are
    protected by default; a route is exempt only when `PUBLIC_ENDPOINTS`
    covers it.
    """
    prefixes = set(MIN_GUARDED_PREFIXES)
    unmatched_public = set(PUBLIC_ENDPOINTS)
    for route in routes:
        path = getattr(route, "path", None)
        if not isinstance(path, str):
            # e.g. lazy router objects without a materialized path
            continue
        for public in PUBLIC_ENDPOINTS:
            if is_public_path(path, public):
                unmatched_public.discard(public)
                break
        else:
            if isinstance(route, Mount) and path in ("", "/"):
                # Guarding "/" would require an API key for every request,
                # health probes included, so a root-level mount is reported
                # and left out of the derivation instead. Serve sub-apps
                # under a sub-path to have them guarded.
                logger.warning(
                    "Ignoring a root-level mount for authentication: guarding "
                    '"/" would require an API key for every endpoint, health '
                    "checks included. Mount it under a sub-path instead if it "
                    "must be authenticated."
                )
                continue
            guard = guard_prefix(path)
            if guard is None:
                # A template with no static part (e.g. "/{path:path}")
                # matches every URL: guard the whole site rather than
                # leaving it unauthenticated.
                logger.warning(
                    "Route %r has no static prefix; guarding every path. "
                    "Register it under a public endpoint prefix if it is "
                    "meant to be reachable without an API key.",
                    path,
                )
                guard = "/"
            prefixes.add(guard)
    if stale := unmatched_public - OPTIONAL_PUBLIC_ENDPOINTS:
        logger.warning(
            "Public endpoints %s match no registered route: either the "
            "routes are not registered on this server or PUBLIC_ENDPOINTS "
            "is stale.",
            sorted(stale),
        )
    return tuple(sorted(prefixes))


class AuthenticationMiddleware:
    """Pure ASGI middleware that authenticates each request by checking
    if the Authorization Bearer token exists and equals anyof "{api_key}".

    Notes
    -----
    The guarded set is derived from the app's route table (see
    `build_guarded_prefixes`): every registered endpoint requires
    authentication unless `PUBLIC_ENDPOINTS` covers it. There are two
    cases in which authentication is skipped:
        1. The HTTP method is OPTIONS (CORS preflight).
        2. The request path is public (e.g. /health) or matches no
           guarded route prefix.

    """

    def __init__(
        self,
        app: ASGIApp,
        tokens: list[str],
        routes_provider: Callable[[], Sequence[BaseRoute]] | None = None,
    ) -> None:
        self.app = app
        self.api_tokens = [hashlib.sha256(t.encode("utf-8")).digest() for t in tokens]
        self._routes_provider = routes_provider
        self._guarded_prefixes: tuple[str, ...] | None = None
        self._routes_len = -1

    def _guarded(self) -> tuple[str, ...]:
        """Get the guarded prefixes, re-deriving them after route changes.

        Routes are registered through the list returned by
        `routes_provider`, so a route added after this middleware was
        installed changes the list length and triggers a re-derivation.
        Starlette routing lists only grow after startup, so the length is
        a sufficient change signal; the fail-closed floor in
        `build_guarded_prefixes` bounds the impact of a hypothetical
        equal-length swap of route objects.
        """
        if self._routes_provider is None:
            # Instantiated without route introspection (e.g. reused by a
            # downstream project): fall back to the fail-closed floor.
            return MIN_GUARDED_PREFIXES
        routes = self._routes_provider()
        routes_len = len(routes)
        if self._guarded_prefixes is None or routes_len != self._routes_len:
            self._guarded_prefixes = build_guarded_prefixes(routes)
            self._routes_len = routes_len
        return self._guarded_prefixes

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
        if (
            scope["type"] not in ("http", "websocket")
            or scope.get("method") == "OPTIONS"
        ):
            # scope["type"] can be "lifespan" or "startup" for example,
            # in which case we don't need to do anything
            return self.app(scope, receive, send)
        root_path = scope.get("root_path", "")
        url_path = scope["path"].removeprefix(root_path)
        headers = Headers(scope=scope)
        # Type narrow to satisfy mypy.
        if url_path.startswith(self._guarded()) and not self.verify_token(headers):
            response = JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return response(scope, receive, send)
        return self.app(scope, receive, send)
