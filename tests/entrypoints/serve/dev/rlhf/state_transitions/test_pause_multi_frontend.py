# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The engine enforces pause admission, so it holds for every API server
process in front of it, not just the one that received /pause."""

from concurrent.futures import ThreadPoolExecutor

import requests

from tests.entrypoints.serve.dev.rlhf.conftest import pause, resume, server

# Fresh connections, so the kernel spreads them across both API servers.
NUM_REQUESTS = 32


def _statuses(url: str) -> list[int]:
    def status(_: int) -> int:
        return requests.post(
            f"{url}/v1/completions",
            json={"model": "m", "prompt": "Hello", "max_tokens": 4},
            timeout=60,
        ).status_code

    with ThreadPoolExecutor(NUM_REQUESTS) as pool:
        return list(pool.map(status, range(NUM_REQUESTS)))


def test_pause_rejects_on_every_api_server():
    with server(
        extra_args=["--api-server-count", "2"], port=8771, dummy_weights=True
    ) as url:
        assert pause(url, mode="abort") == 200
        assert _statuses(url) == [503] * NUM_REQUESTS

        assert resume(url) == 200
        assert _statuses(url) == [200] * NUM_REQUESTS
