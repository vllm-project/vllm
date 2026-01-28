# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for API server tracer initialization (GitHub issue #21).

These tests verify that the API server process initializes a TracerProvider
when journey tracing is enabled, ensuring API spans have valid SpanContext
for parent-child linkage with engine spans.
"""

import asyncio
import socket
from argparse import Namespace
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest


@pytest.mark.asyncio
async def test_api_server_initializes_tracer_when_journey_tracing_enabled():
    """
    Test that run_server_worker initializes tracer provider in API process.

    This is the fix for GitHub issue #21: API spans need a real TracerProvider
    to have valid SpanContext for traceparent injection.
    """
    from vllm.entrypoints.openai import api_server

    # Create fake engine client with journey tracing enabled
    mock_engine_client = MagicMock()
    mock_vllm_config = MagicMock()
    mock_observability_config = MagicMock()
    mock_observability_config.enable_journey_tracing = True
    mock_observability_config.otlp_traces_endpoint = "http://localhost:4317"
    mock_vllm_config.observability_config = mock_observability_config
    mock_vllm_config.parallel_config._api_process_rank = 0
    mock_engine_client.vllm_config = mock_vllm_config

    # Create minimal args
    args = Namespace()
    args.tool_parser_plugin = None
    args.reasoning_parser_plugin = None
    args.log_config_file = None
    args.host = "localhost"
    args.port = 8000
    args.enable_ssl_refresh = False
    args.uvicorn_log_level = "info"
    args.disable_uvicorn_access_log = False
    args.ssl_keyfile = None
    args.ssl_certfile = None
    args.ssl_ca_certs = None
    args.ssl_cert_reqs = None
    args.ssl_ciphers = None
    args.h11_max_incomplete_event_size = None
    args.h11_max_header_count = None

    # Create dummy socket
    dummy_sock = Mock(spec=socket.socket)
    dummy_sock.close = Mock()

    # Track whether init_tracer was called
    init_tracer_called = False
    init_tracer_args = None

    def mock_init_tracer(instrumenting_module_name, otlp_traces_endpoint):
        nonlocal init_tracer_called, init_tracer_args
        init_tracer_called = True
        init_tracer_args = (instrumenting_module_name, otlp_traces_endpoint)
        return Mock()  # Return mock tracer

    # Mock build_async_engine_client to return our fake client
    @asynccontextmanager
    async def mock_build_async_engine_client(*args, **kwargs):
        yield mock_engine_client

    # Mock build_app to return minimal app
    def mock_build_app(args):
        mock_app = MagicMock()
        mock_app.state = MagicMock()
        return mock_app

    # Mock init_app_state (async)
    async def mock_init_app_state(*args, **kwargs):
        pass

    # Mock serve_http to return an awaitable that completes immediately
    async def mock_serve_http(*args, **kwargs):
        # Return an async coroutine that can be awaited
        async def dummy_task():
            pass
        return asyncio.create_task(dummy_task())

    # Apply all patches
    with patch.object(api_server, "build_async_engine_client",
                      mock_build_async_engine_client):
        with patch.object(api_server, "build_app", mock_build_app):
            with patch.object(api_server, "init_app_state", mock_init_app_state):
                with patch.object(api_server, "serve_http", mock_serve_http):
                    # Patch init_tracer in the api_server module's namespace
                    with patch("vllm.tracing.init_tracer", side_effect=mock_init_tracer):
                        with patch("vllm.tracing.is_otel_available", return_value=True):
                            # Run the server worker
                            await api_server.run_server_worker(
                                listen_address="http://localhost:8000",
                                sock=dummy_sock,
                                args=args,
                            )

    # ASSERTION: init_tracer was called with correct arguments
    assert init_tracer_called, \
        "init_tracer was not called - API tracer not initialized"
    assert init_tracer_args == ("vllm.api", "http://localhost:4317"), \
        f"init_tracer called with wrong args: {init_tracer_args}"


@pytest.mark.asyncio
async def test_api_server_skips_tracer_init_when_journey_tracing_disabled():
    """
    Test that run_server_worker skips tracer initialization when disabled.

    This ensures we don't unnecessarily initialize tracing infrastructure.
    """
    from vllm.entrypoints.openai import api_server

    # Create fake engine client with journey tracing DISABLED
    mock_engine_client = MagicMock()
    mock_vllm_config = MagicMock()
    mock_observability_config = MagicMock()
    mock_observability_config.enable_journey_tracing = False  # DISABLED
    mock_observability_config.otlp_traces_endpoint = "http://localhost:4317"
    mock_vllm_config.observability_config = mock_observability_config
    mock_vllm_config.parallel_config._api_process_rank = 0
    mock_engine_client.vllm_config = mock_vllm_config

    # Create minimal args
    args = Namespace()
    args.tool_parser_plugin = None
    args.reasoning_parser_plugin = None
    args.log_config_file = None
    args.host = "localhost"
    args.port = 8000
    args.enable_ssl_refresh = False
    args.uvicorn_log_level = "info"
    args.disable_uvicorn_access_log = False
    args.ssl_keyfile = None
    args.ssl_certfile = None
    args.ssl_ca_certs = None
    args.ssl_cert_reqs = None
    args.ssl_ciphers = None
    args.h11_max_incomplete_event_size = None
    args.h11_max_header_count = None

    # Create dummy socket
    dummy_sock = Mock(spec=socket.socket)
    dummy_sock.close = Mock()

    # Track init_tracer calls
    mock_init_tracer = Mock()

    # Mock functions
    @asynccontextmanager
    async def mock_build_async_engine_client(*args, **kwargs):
        yield mock_engine_client

    def mock_build_app(args):
        mock_app = MagicMock()
        mock_app.state = MagicMock()
        return mock_app

    async def mock_init_app_state(*args, **kwargs):
        pass

    async def mock_serve_http(*args, **kwargs):
        async def dummy_task():
            pass
        return asyncio.create_task(dummy_task())

    # Apply patches
    with patch.object(api_server, "build_async_engine_client",
                      mock_build_async_engine_client):
        with patch.object(api_server, "build_app", mock_build_app):
            with patch.object(api_server, "init_app_state", mock_init_app_state):
                with patch.object(api_server, "serve_http", mock_serve_http):
                    with patch("vllm.tracing.init_tracer", mock_init_tracer):
                        with patch("vllm.tracing.is_otel_available", return_value=True):
                            # Run the server worker
                            await api_server.run_server_worker(
                                listen_address="http://localhost:8000",
                                sock=dummy_sock,
                                args=args,
                            )

    # ASSERTION: init_tracer was NOT called
    assert not mock_init_tracer.called, \
        "init_tracer should not be called when journey tracing is disabled"


@pytest.mark.asyncio
async def test_api_server_skips_tracer_init_when_no_endpoint():
    """
    Test that run_server_worker skips tracer initialization when no endpoint.

    Even if journey tracing is enabled, we need an endpoint to initialize.
    """
    from vllm.entrypoints.openai import api_server

    # Create fake engine client with journey tracing enabled but no endpoint
    mock_engine_client = MagicMock()
    mock_vllm_config = MagicMock()
    mock_observability_config = MagicMock()
    mock_observability_config.enable_journey_tracing = True
    mock_observability_config.otlp_traces_endpoint = None  # No endpoint
    mock_vllm_config.observability_config = mock_observability_config
    mock_vllm_config.parallel_config._api_process_rank = 0
    mock_engine_client.vllm_config = mock_vllm_config

    # Create minimal args
    args = Namespace()
    args.tool_parser_plugin = None
    args.reasoning_parser_plugin = None
    args.log_config_file = None
    args.host = "localhost"
    args.port = 8000
    args.enable_ssl_refresh = False
    args.uvicorn_log_level = "info"
    args.disable_uvicorn_access_log = False
    args.ssl_keyfile = None
    args.ssl_certfile = None
    args.ssl_ca_certs = None
    args.ssl_cert_reqs = None
    args.ssl_ciphers = None
    args.h11_max_incomplete_event_size = None
    args.h11_max_header_count = None

    # Create dummy socket
    dummy_sock = Mock(spec=socket.socket)
    dummy_sock.close = Mock()

    # Track init_tracer calls
    mock_init_tracer = Mock()

    # Mock functions
    @asynccontextmanager
    async def mock_build_async_engine_client(*args, **kwargs):
        yield mock_engine_client

    def mock_build_app(args):
        mock_app = MagicMock()
        mock_app.state = MagicMock()
        return mock_app

    async def mock_init_app_state(*args, **kwargs):
        pass

    async def mock_serve_http(*args, **kwargs):
        async def dummy_task():
            pass
        return asyncio.create_task(dummy_task())

    # Apply patches
    with patch.object(api_server, "build_async_engine_client",
                      mock_build_async_engine_client):
        with patch.object(api_server, "build_app", mock_build_app):
            with patch.object(api_server, "init_app_state", mock_init_app_state):
                with patch.object(api_server, "serve_http", mock_serve_http):
                    with patch("vllm.tracing.init_tracer", mock_init_tracer):
                        with patch("vllm.tracing.is_otel_available", return_value=True):
                            # Run the server worker
                            await api_server.run_server_worker(
                                listen_address="http://localhost:8000",
                                sock=dummy_sock,
                                args=args,
                            )

    # ASSERTION: init_tracer was NOT called
    assert not mock_init_tracer.called, \
        "init_tracer should not be called when no endpoint configured"
