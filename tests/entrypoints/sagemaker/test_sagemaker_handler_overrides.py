# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Integration tests for handler override functionality.

Tests real customer usage scenarios:
- Using @custom_ping_handler and @custom_invocation_handler decorators
  to override handlers
- Setting environment variables for handler specifications
- Writing customer scripts with custom_sagemaker_ping_handler() and
  custom_sagemaker_invocation_handler() functions
- Priority: env vars > decorators > customer script files > framework
  defaults

Note: These tests focus on validating server responses rather than directly calling
get_ping_handler() and get_invoke_handler() to ensure full integration testing.
"""

import os
import tempfile

import pytest
import requests

from ...utils import RemoteOpenAIServer
from .conftest import (
    MODEL_NAME_SMOLLM,
)


class TestHandlerOverrideIntegration:
    """Integration tests simulating real customer usage scenarios.

    Each test simulates a fresh server startup where customers:
    - Use @custom_ping_handler and @custom_invocation_handler decorators
    - Set environment variables (CUSTOM_FASTAPI_PING_HANDLER, etc.)
    - Write customer scripts with custom_sagemaker_ping_handler() and
      custom_sagemaker_invocation_handler() functions
    """

    def setup_method(self):
        """Setup for each test - simulate fresh server startup."""
        self._clear_caches()
        self._clear_env_vars()

    def teardown_method(self):
        """Cleanup after each test."""
        self._clear_env_vars()

    def _clear_caches(self):
        """Clear handler registry and function loader cache."""
        try:
            from model_hosting_container_standards.common.handler import (
                handler_registry,
            )
            from model_hosting_container_standards.sagemaker.sagemaker_loader import (
                SageMakerFunctionLoader,
            )

            handler_registry.clear()
            SageMakerFunctionLoader._default_function_loader = None
        except ImportError:
            pytest.skip("model-hosting-container-standards not available")

    def _clear_env_vars(self):
        """Clear SageMaker environment variables."""
        try:
            from model_hosting_container_standards.common.fastapi.config import (
                FastAPIEnvVars,
            )
            from model_hosting_container_standards.sagemaker.config import (
                SageMakerEnvVars,
            )

            # Clear SageMaker env vars
            for var in [
                SageMakerEnvVars.SAGEMAKER_MODEL_PATH,
                SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME,
            ]:
                os.environ.pop(var, None)

            # Clear FastAPI env vars
            for var in [
                FastAPIEnvVars.CUSTOM_FASTAPI_PING_HANDLER,
                FastAPIEnvVars.CUSTOM_FASTAPI_INVOCATION_HANDLER,
            ]:
                os.environ.pop(var, None)
        except ImportError:
            pass

    @pytest.mark.asyncio
    async def test_customer_script_functions_auto_loaded(self):
        """Test customer scenario: script functions automatically override
        framework defaults."""
        try:
            from model_hosting_container_standards.sagemaker.config import (
                SageMakerEnvVars,
            )
        except ImportError:
            pytest.skip("model-hosting-container-standards not available")

        # Customer writes a script file with ping() and invoke() functions
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
from fastapi import Request

async def custom_sagemaker_ping_handler():
    return {
        "status": "healthy",
        "source": "customer_override", 
        "message": "Custom ping from customer script"
    }

async def custom_sagemaker_invocation_handler(request: Request):
    return {
        "predictions": ["Custom response from customer script"],
        "source": "customer_override"
    }
"""
            )
            script_path = f.name

        try:
            script_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path)

            # Customer sets SageMaker environment variables to point to their script
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
                # Customer tests their server and sees their overrides work
                # automatically
                ping_response = requests.get(server.url_for("ping"))
                assert ping_response.status_code == 200
                ping_data = ping_response.json()

                invoke_response = requests.post(
                    server.url_for("invocations"),
                    json={
                        "model": MODEL_NAME_SMOLLM,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 5,
                    },
                )
                assert invoke_response.status_code == 200
                invoke_data = invoke_response.json()

                # Customer sees their functions are used
                assert ping_data["source"] == "customer_override"
                assert ping_data["message"] == "Custom ping from customer script"
                assert invoke_data["source"] == "customer_override"
                assert invoke_data["predictions"] == [
                    "Custom response from customer script"
                ]

        finally:
            os.unlink(script_path)

    @pytest.mark.asyncio
    async def test_customer_decorator_usage(self):
        """Test customer scenario: using @custom_ping_handler and
        @custom_invocation_handler decorators."""
        try:
            from model_hosting_container_standards.sagemaker.config import (
                SageMakerEnvVars,
            )
        except ImportError:
            pytest.skip("model-hosting-container-standards not available")

        # Customer writes a script file with decorators
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
import model_hosting_container_standards.sagemaker as sagemaker_standards
from fastapi import Request

@sagemaker_standards.custom_ping_handler
async def my_ping():
    return {
        "type": "ping",
        "source": "customer_decorator"
    }

@sagemaker_standards.custom_invocation_handler  
async def my_invoke(request: Request):
    return {
        "type": "invoke", 
        "source": "customer_decorator"
    }
"""
            )
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
                ping_response = requests.get(server.url_for("ping"))
                assert ping_response.status_code == 200
                ping_data = ping_response.json()

                invoke_response = requests.post(
                    server.url_for("invocations"),
                    json={
                        "model": MODEL_NAME_SMOLLM,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 5,
                    },
                )
                assert invoke_response.status_code == 200
                invoke_data = invoke_response.json()

                # Customer sees their handlers are used by the server
                assert ping_data["source"] == "customer_decorator"
                assert invoke_data["source"] == "customer_decorator"

        finally:
            os.unlink(script_path)

    @pytest.mark.asyncio
    async def test_handler_priority_order(self):
        """Test priority: @custom_ping_handler/@custom_invocation_handler
        decorators vs script functions."""
        try:
            from model_hosting_container_standards.sagemaker.config import (
                SageMakerEnvVars,
            )
        except ImportError:
            pytest.skip("model-hosting-container-standards not available")

        # Customer writes a script with both decorator and regular functions
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
import model_hosting_container_standards.sagemaker as sagemaker_standards
from fastapi import Request

# Customer uses @custom_ping_handler decorator (higher priority than script functions)
@sagemaker_standards.custom_ping_handler
async def decorated_ping():
    return {
        "status": "healthy",
        "source": "ping_decorator_in_script", 
        "priority": "decorator"
    }

# Customer also has a regular function (lower priority than
# @custom_ping_handler decorator)
async def custom_sagemaker_ping_handler():
    return {
        "status": "healthy",
        "source": "script_function",
        "priority": "function"
    }

# Customer has a regular invoke function
async def custom_sagemaker_invocation_handler(request: Request):
    return {
        "predictions": ["Script function response"],
        "source": "script_invoke_function",
        "priority": "function"
    }
"""
            )
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
                ping_response = requests.get(server.url_for("ping"))
                assert ping_response.status_code == 200
                ping_data = ping_response.json()

                invoke_response = requests.post(
                    server.url_for("invocations"),
                    json={
                        "model": MODEL_NAME_SMOLLM,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 5,
                    },
                )
                assert invoke_response.status_code == 200
                invoke_data = invoke_response.json()

                # @custom_ping_handler decorator has higher priority than
                # script function
                assert ping_data["source"] == "ping_decorator_in_script"
                assert ping_data["priority"] == "decorator"

                # Script function is used for invoke
                assert invoke_data["source"] == "script_invoke_function"
                assert invoke_data["priority"] == "function"

        finally:
            os.unlink(script_path)

    @pytest.mark.asyncio
    async def test_environment_variable_script_loading(self):
        """Test that environment variables correctly specify script location
        and loading."""
        try:
            from model_hosting_container_standards.sagemaker.config import (
                SageMakerEnvVars,
            )
        except ImportError:
            pytest.skip("model-hosting-container-standards not available")

        # Customer writes a script in a specific directory
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
from fastapi import Request

async def custom_sagemaker_ping_handler():
    return {
        "status": "healthy",
        "source": "env_loaded_script",
        "method": "environment_variable_loading"
    }

async def custom_sagemaker_invocation_handler(request: Request):
    return {
        "predictions": ["Loaded via environment variables"],
        "source": "env_loaded_script",
        "method": "environment_variable_loading"
    }
"""
            )
            script_path = f.name

        try:
            script_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path)

            # Test environment variable script loading
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
                ping_response = requests.get(server.url_for("ping"))
                assert ping_response.status_code == 200
                ping_data = ping_response.json()

                invoke_response = requests.post(
                    server.url_for("invocations"),
                    json={
                        "model": MODEL_NAME_SMOLLM,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 5,
                    },
                )
                assert invoke_response.status_code == 200
                invoke_data = invoke_response.json()

                # Verify that the script was loaded via environment variables
                assert ping_data["source"] == "env_loaded_script"
                assert ping_data["method"] == "environment_variable_loading"
                assert invoke_data["source"] == "env_loaded_script"
                assert invoke_data["method"] == "environment_variable_loading"

        finally:
            os.unlink(script_path)

    @pytest.mark.asyncio
    async def test_framework_default_handlers(self):
        """Test that framework default handlers work when no customer
        overrides exist."""
        args = [
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "2048",
            "--enforce-eager",
            "--max-num-seqs",
            "32",
        ]

        # Explicitly pass empty env_dict to ensure no SageMaker env vars are set
        # This prevents pollution from previous tests
        try:
            from model_hosting_container_standards.common.fastapi.config import (
                FastAPIEnvVars,
            )
            from model_hosting_container_standards.sagemaker.config import (
                SageMakerEnvVars,
            )

            env_dict = {
                SageMakerEnvVars.SAGEMAKER_MODEL_PATH: "",
                SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME: "",
                FastAPIEnvVars.CUSTOM_FASTAPI_PING_HANDLER: "",
                FastAPIEnvVars.CUSTOM_FASTAPI_INVOCATION_HANDLER: "",
            }
        except ImportError:
            env_dict = {}

        with RemoteOpenAIServer(MODEL_NAME_SMOLLM, args, env_dict=env_dict) as server:
            # Test that default ping works
            ping_response = requests.get(server.url_for("ping"))
            assert ping_response.status_code == 200

            # Test that default invocations work
            invoke_response = requests.post(
                server.url_for("invocations"),
                json={
                    "model": MODEL_NAME_SMOLLM,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 5,
                },
            )
            assert invoke_response.status_code == 200

    @pytest.mark.asyncio
    async def test_handler_env_var_override(self):
        """Test CUSTOM_FASTAPI_PING_HANDLER and CUSTOM_FASTAPI_INVOCATION_HANDLER
        environment variable overrides."""
        try:
            from model_hosting_container_standards.common.fastapi.config import (
                FastAPIEnvVars,
            )
            from model_hosting_container_standards.sagemaker.config import (
                SageMakerEnvVars,
            )
        except ImportError:
            pytest.skip("model-hosting-container-standards not available")

        # Create a script with both env var handlers and script functions
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
from fastapi import Request, Response
import json

async def env_var_ping_handler(raw_request: Request) -> Response:
    return Response(
        content=json.dumps({
            "status": "healthy",
            "source": "env_var_ping",
            "method": "environment_variable"
        }),
        media_type="application/json"
    )

async def env_var_invoke_handler(raw_request: Request) -> Response:
    return Response(
        content=json.dumps({
            "predictions": ["Environment variable response"],
            "source": "env_var_invoke",
            "method": "environment_variable"
        }),
        media_type="application/json"
    )

async def custom_sagemaker_ping_handler():
    return {
        "status": "healthy",
        "source": "script_ping",
        "method": "script_function"
    }

async def custom_sagemaker_invocation_handler(request: Request):
    return {
        "predictions": ["Script function response"],
        "source": "script_invoke",
        "method": "script_function"
    }
"""
            )
            script_path = f.name

        try:
            script_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path)

            # Set environment variables to override both handlers
            env_vars = {
                SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME: script_name,
                FastAPIEnvVars.CUSTOM_FASTAPI_PING_HANDLER: (
                    f"{script_name}:env_var_ping_handler"
                ),
                FastAPIEnvVars.CUSTOM_FASTAPI_INVOCATION_HANDLER: (
                    f"{script_name}:env_var_invoke_handler"
                ),
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
                # Test ping handler override
                ping_response = requests.get(server.url_for("ping"))
                assert ping_response.status_code == 200
                ping_data = ping_response.json()

                # Environment variable should override script function
                assert ping_data["method"] == "environment_variable"
                assert ping_data["source"] == "env_var_ping"

                # Test invocation handler override
                invoke_response = requests.post(
                    server.url_for("invocations"),
                    json={
                        "model": MODEL_NAME_SMOLLM,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 5,
                    },
                )
                assert invoke_response.status_code == 200
                invoke_data = invoke_response.json()

                # Environment variable should override script function
                assert invoke_data["method"] == "environment_variable"
                assert invoke_data["source"] == "env_var_invoke"

        finally:
            os.unlink(script_path)

    @pytest.mark.asyncio
    async def test_env_var_priority_over_decorator_and_script(self):
        """Test that environment variables have highest priority over decorators
        and script functions for both ping and invocation handlers."""
        try:
            from model_hosting_container_standards.common.fastapi.config import (
                FastAPIEnvVars,
            )
            from model_hosting_container_standards.sagemaker.config import (
                SageMakerEnvVars,
            )
        except ImportError:
            pytest.skip("model-hosting-container-standards not available")

        # Create a script with all three handler types for both ping and invocation
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
import model_hosting_container_standards.sagemaker as sagemaker_standards
from fastapi import Request, Response
import json

# Environment variable handlers (highest priority)
async def env_priority_ping(raw_request: Request) -> Response:
    return Response(
        content=json.dumps({
            "status": "healthy",
            "source": "env_var",
            "priority": "environment_variable"
        }),
        media_type="application/json"
    )

async def env_priority_invoke(raw_request: Request) -> Response:
    return Response(
        content=json.dumps({
            "predictions": ["Environment variable response"],
            "source": "env_var",
            "priority": "environment_variable"
        }),
        media_type="application/json"
    )

# Decorator handlers (medium priority)
@sagemaker_standards.custom_ping_handler
async def decorator_ping(raw_request: Request) -> Response:
    return Response(
        content=json.dumps({
            "status": "healthy",
            "source": "decorator",
            "priority": "decorator"
        }),
        media_type="application/json"
    )

@sagemaker_standards.custom_invocation_handler
async def decorator_invoke(raw_request: Request) -> Response:
    return Response(
        content=json.dumps({
            "predictions": ["Decorator response"],
            "source": "decorator",
            "priority": "decorator"
        }),
        media_type="application/json"
    )

# Script functions (lowest priority)
async def custom_sagemaker_ping_handler():
    return {
        "status": "healthy",
        "source": "script",
        "priority": "script_function"
    }

async def custom_sagemaker_invocation_handler(request: Request):
    return {
        "predictions": ["Script function response"],
        "source": "script",
        "priority": "script_function"
    }
"""
            )
            script_path = f.name

        try:
            script_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path)

            # Set environment variables to specify highest priority handlers
            env_vars = {
                SageMakerEnvVars.SAGEMAKER_MODEL_PATH: script_dir,
                SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME: script_name,
                FastAPIEnvVars.CUSTOM_FASTAPI_PING_HANDLER: (
                    f"{script_name}:env_priority_ping"
                ),
                FastAPIEnvVars.CUSTOM_FASTAPI_INVOCATION_HANDLER: (
                    f"{script_name}:env_priority_invoke"
                ),
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
                # Test ping handler priority
                ping_response = requests.get(server.url_for("ping"))
                assert ping_response.status_code == 200
                ping_data = ping_response.json()

                # Environment variable has highest priority and should be used
                assert ping_data["priority"] == "environment_variable"
                assert ping_data["source"] == "env_var"

                # Test invocation handler priority
                invoke_response = requests.post(
                    server.url_for("invocations"),
                    json={
                        "model": MODEL_NAME_SMOLLM,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 5,
                    },
                )
                assert invoke_response.status_code == 200
                invoke_data = invoke_response.json()

                # Environment variable has highest priority and should be used
                assert invoke_data["priority"] == "environment_variable"
                assert invoke_data["source"] == "env_var"

        finally:
            os.unlink(script_path)
