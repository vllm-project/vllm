# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""API-key authentication for the gRPC render server.

Mirrors the HTTP ``AuthenticationMiddleware``
(``vllm/entrypoints/serve/utils/server_utils.py``) so that
``vllm launch render --server grpc`` enforces the same ``--api-key`` /
``VLLM_API_KEY`` credential as ``--server http``.
"""

import argparse
import hashlib
import secrets

import grpc
import grpc.aio

from vllm import envs

# Method paths are ``/<package>.<Service>/<Rpc>``; see vllm_render.proto
# (``package vllm.grpc.render; service VllmRender``).
_SERVICE_PREFIX = "/vllm.grpc.render.VllmRender/"

# RPCs exempt from authentication. ``HealthCheck`` mirrors the HTTP server
# leaving ``/health`` outside ``GUARDED_PREFIX`` so liveness/readiness probes
# work without credentials.
_UNAUTHENTICATED_METHODS = frozenset({_SERVICE_PREFIX + "HealthCheck"})


def resolve_api_tokens(args: argparse.Namespace) -> list[str]:
    """Resolve API tokens exactly as the HTTP server does.

    ``--api-key`` takes precedence over the ``VLLM_API_KEY`` env var; an empty
    result means authentication is disabled.
    """
    return [key for key in (args.api_key or [envs.VLLM_API_KEY]) if key]


async def _abort_unauthenticated(request, context: grpc.aio.ServicerContext) -> None:
    await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid or missing API key.")


# All VllmRender RPCs are unary-unary, so a single unary-unary deny handler
# covers every guarded method. The handler aborts before the request is read,
# so the request/response (de)serializers are irrelevant.
_UNAUTHENTICATED_HANDLER = grpc.unary_unary_rpc_method_handler(_abort_unauthenticated)


class ApiKeyAuthInterceptor(grpc.aio.ServerInterceptor):
    """Enforce ``Bearer`` API-key auth on guarded RPCs.

    Token verification matches the HTTP ``AuthenticationMiddleware``: each
    token is SHA-256 hashed and compared in constant time against the
    ``authorization: Bearer <token>`` request metadata.
    """

    def __init__(self, tokens: list[str]) -> None:
        self._token_hashes = [
            hashlib.sha256(token.encode("utf-8")).digest() for token in tokens
        ]

    def _verify(self, metadata) -> bool:
        authorization = None
        for key, value in metadata or ():
            if key == "authorization":
                authorization = value
                break
        if not authorization:
            return False

        scheme, _, param = authorization.partition(" ")
        if scheme.lower() != "bearer":
            return False

        param_hash = hashlib.sha256(param.encode("utf-8")).digest()
        token_match = False
        for token_hash in self._token_hashes:
            token_match |= secrets.compare_digest(param_hash, token_hash)
        return token_match

    async def intercept_service(self, continuation, handler_call_details):
        method = handler_call_details.method
        if method in _UNAUTHENTICATED_METHODS:
            return await continuation(handler_call_details)
        if self._verify(handler_call_details.invocation_metadata):
            return await continuation(handler_call_details)
        return _UNAUTHENTICATED_HANDLER


def build_auth_interceptors(
    args: argparse.Namespace,
) -> list[grpc.aio.ServerInterceptor]:
    """Build the gRPC auth interceptors for ``run_launch_grpc``.

    Returns an empty list when no API key is configured, mirroring the HTTP
    server only installing ``AuthenticationMiddleware`` when a token exists.
    """
    tokens = resolve_api_tokens(args)
    if not tokens:
        return []
    return [ApiKeyAuthInterceptor(tokens)]
