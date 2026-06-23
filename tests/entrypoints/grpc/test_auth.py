# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the gRPC render server API-key auth interceptor."""

import argparse
from unittest.mock import AsyncMock, MagicMock

import grpc
import grpc.aio
import pytest

from vllm.entrypoints.grpc.auth import (
    _UNAUTHENTICATED_HANDLER,
    ApiKeyAuthInterceptor,
    build_auth_interceptors,
    resolve_api_tokens,
)

_SERVICE_PREFIX = "/vllm.grpc.render.VllmRender/"
_HEALTH_CHECK = _SERVICE_PREFIX + "HealthCheck"
_RENDER_CHAT = _SERVICE_PREFIX + "RenderChat"


def _handler_call_details(method, metadata=()):
    details = MagicMock()
    details.method = method
    details.invocation_metadata = metadata
    return details


# ---------------------------------------------------------------------------
# resolve_api_tokens
# ---------------------------------------------------------------------------


def test_resolve_tokens_from_args():
    args = argparse.Namespace(api_key=["k1", "k2"])
    assert resolve_api_tokens(args) == ["k1", "k2"]


def test_resolve_tokens_from_env(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "envkey")
    args = argparse.Namespace(api_key=None)
    assert resolve_api_tokens(args) == ["envkey"]


def test_resolve_tokens_args_take_precedence_over_env(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "envkey")
    args = argparse.Namespace(api_key=["clikey"])
    assert resolve_api_tokens(args) == ["clikey"]


def test_resolve_tokens_none(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    args = argparse.Namespace(api_key=None)
    assert resolve_api_tokens(args) == []


# ---------------------------------------------------------------------------
# build_auth_interceptors
# ---------------------------------------------------------------------------


def test_build_interceptors_empty_when_no_key(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    args = argparse.Namespace(api_key=None)
    assert build_auth_interceptors(args) == []


def test_build_interceptors_when_key(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    args = argparse.Namespace(api_key=["secret"])
    interceptors = build_auth_interceptors(args)
    assert len(interceptors) == 1
    assert isinstance(interceptors[0], ApiKeyAuthInterceptor)


# ---------------------------------------------------------------------------
# ApiKeyAuthInterceptor._verify
# ---------------------------------------------------------------------------


def test_verify_valid_bearer():
    interceptor = ApiKeyAuthInterceptor(["secret"])
    assert interceptor._verify((("authorization", "Bearer secret"),)) is True


def test_verify_scheme_is_case_insensitive():
    interceptor = ApiKeyAuthInterceptor(["secret"])
    assert interceptor._verify((("authorization", "bearer secret"),)) is True


def test_verify_wrong_token():
    interceptor = ApiKeyAuthInterceptor(["secret"])
    assert interceptor._verify((("authorization", "Bearer nope"),)) is False


def test_verify_missing_metadata():
    interceptor = ApiKeyAuthInterceptor(["secret"])
    assert interceptor._verify(()) is False


def test_verify_non_bearer_scheme():
    interceptor = ApiKeyAuthInterceptor(["secret"])
    assert interceptor._verify((("authorization", "Basic secret"),)) is False


def test_verify_matches_any_of_multiple_tokens():
    interceptor = ApiKeyAuthInterceptor(["a", "b"])
    assert interceptor._verify((("authorization", "Bearer b"),)) is True


# ---------------------------------------------------------------------------
# ApiKeyAuthInterceptor.intercept_service
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intercept_exempts_health_check():
    """HealthCheck is exempt — continuation runs even without a token."""
    interceptor = ApiKeyAuthInterceptor(["secret"])
    continuation = AsyncMock(return_value="real_handler")

    result = await interceptor.intercept_service(
        continuation, _handler_call_details(_HEALTH_CHECK)
    )

    assert result == "real_handler"
    continuation.assert_awaited_once()


@pytest.mark.asyncio
async def test_intercept_allows_valid_token():
    interceptor = ApiKeyAuthInterceptor(["secret"])
    continuation = AsyncMock(return_value="real_handler")

    result = await interceptor.intercept_service(
        continuation,
        _handler_call_details(_RENDER_CHAT, (("authorization", "Bearer secret"),)),
    )

    assert result == "real_handler"
    continuation.assert_awaited_once()


@pytest.mark.asyncio
async def test_intercept_denies_missing_token():
    interceptor = ApiKeyAuthInterceptor(["secret"])
    continuation = AsyncMock()

    result = await interceptor.intercept_service(
        continuation, _handler_call_details(_RENDER_CHAT)
    )

    assert result is _UNAUTHENTICATED_HANDLER
    continuation.assert_not_awaited()


@pytest.mark.asyncio
async def test_intercept_denies_wrong_token():
    interceptor = ApiKeyAuthInterceptor(["secret"])
    continuation = AsyncMock()

    result = await interceptor.intercept_service(
        continuation,
        _handler_call_details(_RENDER_CHAT, (("authorization", "Bearer wrong"),)),
    )

    assert result is _UNAUTHENTICATED_HANDLER
    continuation.assert_not_awaited()


@pytest.mark.asyncio
async def test_deny_handler_aborts_unauthenticated():
    """The deny handler aborts the RPC with UNAUTHENTICATED."""
    ctx = AsyncMock()
    await _UNAUTHENTICATED_HANDLER.unary_unary(MagicMock(), ctx)

    ctx.abort.assert_awaited_once()
    assert ctx.abort.call_args[0][0] == grpc.StatusCode.UNAUTHENTICATED
