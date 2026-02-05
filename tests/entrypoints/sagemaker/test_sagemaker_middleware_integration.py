# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Integration test for middleware loader functionality.

Tests that customer middlewares get called correctly with a vLLM server.
"""

import os
import tempfile

import pytest
import requests

from ...utils import RemoteOpenAIServer
from .conftest import (
    MODEL_NAME_SMOLLM,
)


class TestMiddlewareIntegration:
    """Integration test for middleware with vLLM server."""

    def setup_method(self):
        """Setup for each test - simulate fresh server startup."""
        self._clear_caches()

    def _clear_caches(self):
        """Clear middleware registry and function loader cache."""
        try:
            from model_hosting_container_standards.common.fastapi.middleware import (
                middleware_registry,
            )
            from model_hosting_container_standards.common.fastapi.middleware.source.decorator_loader import (  # noqa: E501
                decorator_loader,
            )
            from model_hosting_container_standards.sagemaker.sagemaker_loader import (
                SageMakerFunctionLoader,
            )

            middleware_registry.clear_middlewares()
            decorator_loader.clear()
            SageMakerFunctionLoader._default_function_loader = None
        except ImportError:
            pytest.skip("model-hosting-container-standards not available")

    @pytest.mark.asyncio
    async def test_customer_middleware_with_vllm_server(self):
        """Test that customer middlewares work with actual vLLM server.

        Tests decorator-based middlewares (@custom_middleware, @input_formatter,
        @output_formatter)
        on multiple endpoints (chat/completions, invocations).
        """
        try:
            from model_hosting_container_standards.sagemaker.config import (
                SageMakerEnvVars,
            )
        except ImportError:
            pytest.skip("model-hosting-container-standards not available")

        # Customer writes a middleware script with multiple decorators
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
from model_hosting_container_standards.common.fastapi.middleware import (
    custom_middleware, input_formatter, output_formatter
)

# Global flag to track if input formatter was called
_input_formatter_called = False

@input_formatter
async def customer_input_formatter(request):
    # Process input - mark that input formatter was called
    global _input_formatter_called
    _input_formatter_called = True
    return request

@custom_middleware("throttle")
async def customer_throttle_middleware(request, call_next):
    response = await call_next(request)
    response.headers["X-Customer-Throttle"] = "applied"
    order = response.headers.get("X-Middleware-Order", "")
    response.headers["X-Middleware-Order"] = order + "throttle,"
    return response

@output_formatter
async def customer_output_formatter(response):
    global _input_formatter_called
    response.headers["X-Customer-Processed"] = "true"
    # Since input_formatter and output_formatter are combined into
    # pre_post_process middleware,
    # if output_formatter is called, input_formatter should have been called too
    if _input_formatter_called:
        response.headers["X-Input-Formatter-Called"] = "true"
    order = response.headers.get("X-Middleware-Order", "")
    response.headers["X-Middleware-Order"] = order + "output_formatter,"
    return response
""")
            script_path = f.name

        try:
            script_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path)

            # Set environment variables to point to customer script
            env_vars = {
                SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME: script_name,
            }

            args = [
                "--dtype",
                "bfloat16",
                "--max-model-len",
                "2048",
                "--enforce-eager",
                "--max-num-seqs",
                "32",
            ]

            with RemoteOpenAIServer(
                MODEL_NAME_SMOLLM, args, env_dict=env_vars
            ) as server:
                # Test 1: Middlewares applied to chat/completions endpoint
                chat_response = requests.post(
                    server.url_for("v1/chat/completions"),
                    json={
                        "model": MODEL_NAME_SMOLLM,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 5,
                        "temperature": 0.0,
                    },
                )

                assert chat_response.status_code == 200

                # Verify all middlewares were executed
                assert "X-Customer-Throttle" in chat_response.headers
                assert chat_response.headers["X-Customer-Throttle"] == "applied"
                assert "X-Customer-Processed" in chat_response.headers
                assert chat_response.headers["X-Customer-Processed"] == "true"

                # Verify input formatter was called
                assert "X-Input-Formatter-Called" in chat_response.headers
                assert chat_response.headers["X-Input-Formatter-Called"] == "true"

                # Verify middleware execution order
                execution_order = chat_response.headers.get(
                    "X-Middleware-Order", ""
                ).rstrip(",")
                order_parts = execution_order.split(",") if execution_order else []
                assert "throttle" in order_parts
                assert "output_formatter" in order_parts

                # Test 2: Middlewares applied to invocations endpoint
                invocations_response = requests.post(
                    server.url_for("invocations"),
                    json={
                        "model": MODEL_NAME_SMOLLM,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 5,
                        "temperature": 0.0,
                    },
                )

                assert invocations_response.status_code == 200

                # Verify all middlewares were executed
                assert "X-Customer-Throttle" in invocations_response.headers
                assert invocations_response.headers["X-Customer-Throttle"] == "applied"
                assert "X-Customer-Processed" in invocations_response.headers
                assert invocations_response.headers["X-Customer-Processed"] == "true"

                # Verify input formatter was called
                assert "X-Input-Formatter-Called" in invocations_response.headers
                assert (
                    invocations_response.headers["X-Input-Formatter-Called"] == "true"
                )

        finally:
            os.unlink(script_path)

    @pytest.mark.asyncio
    async def test_middleware_with_ping_endpoint(self):
        """Test that middlewares work with SageMaker ping endpoint."""
        try:
            from model_hosting_container_standards.sagemaker.config import (
                SageMakerEnvVars,
            )
        except ImportError:
            pytest.skip("model-hosting-container-standards not available")

        # Customer writes a middleware script
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
from model_hosting_container_standards.common.fastapi.middleware import (
    custom_middleware
)

@custom_middleware("pre_post_process")
async def ping_tracking_middleware(request, call_next):
    response = await call_next(request)
    if request.url.path == "/ping":
        response.headers["X-Ping-Tracked"] = "true"
    return response
""")
            script_path = f.name

        try:
            script_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path)

            env_vars = {
                SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME: script_name,
            }

            args = [
                "--dtype",
                "bfloat16",
                "--max-model-len",
                "2048",
                "--enforce-eager",
                "--max-num-seqs",
                "32",
            ]

            with RemoteOpenAIServer(
                MODEL_NAME_SMOLLM, args, env_dict=env_vars
            ) as server:
                # Test ping endpoint with middleware
                response = requests.get(server.url_for("ping"))

                assert response.status_code == 200
                assert "X-Ping-Tracked" in response.headers
                assert response.headers["X-Ping-Tracked"] == "true"

        finally:
            os.unlink(script_path)

    @pytest.mark.asyncio
    async def test_middleware_env_var_override(self):
        """Test middleware environment variable overrides."""
        try:
            from model_hosting_container_standards.common.fastapi.config import (
                FastAPIEnvVars,
            )
            from model_hosting_container_standards.sagemaker.config import (
                SageMakerEnvVars,
            )
        except ImportError:
            pytest.skip("model-hosting-container-standards not available")

        # Create a script with middleware functions specified via env vars
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
from fastapi import Request

# Global flag to track if pre_process was called
_pre_process_called = False

async def env_throttle_middleware(request, call_next):
    response = await call_next(request)
    response.headers["X-Env-Throttle"] = "applied"
    return response

async def env_pre_process(request: Request) -> Request:
    # Mark that pre_process was called
    global _pre_process_called
    _pre_process_called = True
    return request

async def env_post_process(response):
    global _pre_process_called
    if hasattr(response, 'headers'):
        response.headers["X-Env-Post-Process"] = "applied"
        # Since pre_process and post_process are combined into
        # pre_post_process middleware,
        # if post_process is called, pre_process should have been called too
        if _pre_process_called:
            response.headers["X-Pre-Process-Called"] = "true"
    return response
""")
            script_path = f.name

        try:
            script_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path)

            # Set environment variables for middleware
            # Use script_name with .py extension as per plugin example
            env_vars = {
                SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME: script_name,
                FastAPIEnvVars.CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE: (
                    f"{script_name}:env_throttle_middleware"
                ),
                FastAPIEnvVars.CUSTOM_PRE_PROCESS: f"{script_name}:env_pre_process",
                FastAPIEnvVars.CUSTOM_POST_PROCESS: f"{script_name}:env_post_process",
            }

            args = [
                "--dtype",
                "bfloat16",
                "--max-model-len",
                "2048",
                "--enforce-eager",
                "--max-num-seqs",
                "32",
            ]

            with RemoteOpenAIServer(
                MODEL_NAME_SMOLLM, args, env_dict=env_vars
            ) as server:
                response = requests.get(server.url_for("ping"))
                assert response.status_code == 200

                # Check if environment variable middleware was applied
                headers = response.headers

                # Verify that env var middlewares were applied
                assert (
                    "X-Env-Throttle" in headers
                ), "Throttle middleware should be applied via env var"
                assert headers["X-Env-Throttle"] == "applied"

                assert (
                    "X-Env-Post-Process" in headers
                ), "Post-process middleware should be applied via env var"
                assert headers["X-Env-Post-Process"] == "applied"

                # Verify that pre_process was called
                assert (
                    "X-Pre-Process-Called" in headers
                ), "Pre-process should be called via env var"
                assert headers["X-Pre-Process-Called"] == "true"

        finally:
            os.unlink(script_path)
