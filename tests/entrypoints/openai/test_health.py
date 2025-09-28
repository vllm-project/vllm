# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from http import HTTPStatus

import pytest
import requests

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="class")
def server():
    args = [
        "--enforce-eager", "--max-model-len", "100",
        "--gpu-memory-utilization", "0.8"
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


class TestHealth:

    def test_health_basic(self, server: RemoteOpenAIServer):
        """Test basic health check endpoint."""
        response = requests.get(server.url_for("health"))
        assert response.status_code == HTTPStatus.OK

    def test_health_with_generate(self, server: RemoteOpenAIServer):
        """Test health check with generate parameter."""
        response = requests.get(server.url_for("health"),
                                params={"generate": "true"})
        assert response.status_code == HTTPStatus.OK

    def test_health_with_running_query(self, server: RemoteOpenAIServer):
        generation_errors: list[Exception] = []
        start_event = threading.Event()
        done_event = threading.Event()

        def _run_generate() -> None:
            try:
                client = server.get_client()
                start_event.set()
                client.completions.create(
                    model=MODEL_NAME,
                    prompt="Ping health endpoint",
                    max_tokens=50,
                    temperature=0.0,
                )
            except Exception as e:
                generation_errors.append(e)
            finally:
                done_event.set()

        generate_thread = threading.Thread(target=_run_generate, daemon=True)
        generate_thread.start()

        time.sleep(1)  # Ensure the generation has started
        response = requests.get(server.url_for("health"),
                                params={"generate": "true"})
        assert response.status_code == HTTPStatus.OK

        assert start_event.wait(
            timeout=10), "Generation thread failed to start"
        assert done_event.wait(timeout=300), "Generation thread did not finish"
        generate_thread.join(timeout=0)
        if generation_errors:
            raise generation_errors[0]