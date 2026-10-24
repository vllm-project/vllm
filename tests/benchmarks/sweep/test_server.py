# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.benchmarks.sweep.server import ServerProcess


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], "http://localhost:8000"),
        (["--host", "127.0.0.2", "--port", "8001"], "http://127.0.0.2:8001"),
        (["--host=127.0.0.2", "--port=8001"], "http://127.0.0.2:8001"),
        (["--host", "127.0.0.2", "-p", "8001"], "http://127.0.0.2:8001"),
        (["--port", "8001", "--port", "8002"], "http://localhost:8002"),
        (["-p", "8001", "--port", "8002"], "http://localhost:8002"),
        (["--host", "::1", "--port", "8001"], "http://[::1]:8001"),
        (["--host=::1", "--port=8001"], "http://[::1]:8001"),
        (["--port", "8001", "--port-other", "9000"], "http://localhost:8001"),
    ],
)
def test_server_address(args, expected):
    server = ServerProcess(["vllm", "serve", "model", *args], [], show_stdout=False)
    assert server._get_vllm_server_address() == expected
