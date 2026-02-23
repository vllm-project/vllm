# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import openai
import pytest


def pytest_addoption(parser):
    """Add TT-specific command line options."""
    parser.addoption(
        "--tt-server-url",
        action="store",
        required=True,
        help="URL of the running vLLM server (e.g., http://localhost:8000)",
    )
    parser.addoption(
        "--tt-model-name",
        action="store",
        required=True,
        help="Model name served by the server (e.g., meta-llama/Llama-3.1-8B)",
    )
    parser.addoption(
        "--tt-max-num-seqs",
        action="store",
        type=int,
        default=32,
        help="Max batch size for testing (default: 32)",
    )


@pytest.fixture(scope="session")
def tt_server_url(request):
    """Returns the server URL."""
    return request.config.getoption("--tt-server-url")


@pytest.fixture(scope="session")
def tt_model_name(request):
    """Returns the model name being tested."""
    return request.config.getoption("--tt-model-name")


@pytest.fixture(scope="session")
def max_batch_size(request):
    """Returns the max batch size for testing."""
    return request.config.getoption("--tt-max-num-seqs")


@pytest.fixture(scope="session")
def tt_server(tt_server_url):
    """
    Returns a simple object with get_async_client() method
    to match the interface expected by tests.
    """

    class ServerWrapper:
        def __init__(self, base_url: str):
            self.base_url = base_url.rstrip("/")

        def get_async_client(self):
            return openai.AsyncOpenAI(
                base_url=f"{self.base_url}/v1",
                api_key="dummy",  # vLLM doesn't require a real key
            )

        def get_client(self):
            return openai.OpenAI(
                base_url=f"{self.base_url}/v1",
                api_key="dummy",
            )

    return ServerWrapper(tt_server_url)
