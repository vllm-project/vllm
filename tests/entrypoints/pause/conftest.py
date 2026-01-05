# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import requests

DEFAULT_SERVER_URL = "http://localhost:8000"


def pytest_addoption(parser):
    parser.addoption(
        "--server-url",
        action="store",
        default=DEFAULT_SERVER_URL,
        help="vLLM server URL for integration tests",
    )


def get_server_url() -> str:
    return os.environ.get("VLLM_TEST_SERVER_URL", DEFAULT_SERVER_URL)


@pytest.fixture
def server_url(request) -> str:
    url = request.config.getoption("--server-url", default=None)
    if url is None:
        url = get_server_url()
    return url


@pytest.fixture
def skip_if_no_server(server_url):
    """Skip test if server is not reachable."""
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code != 200:
            pytest.skip(f"Server at {server_url} not healthy")
    except requests.exceptions.RequestException:
        pytest.skip(f"Server at {server_url} not reachable")

