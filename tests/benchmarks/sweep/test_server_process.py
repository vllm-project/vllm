# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
import signal
import socket
import sys

import pytest

from vllm.benchmarks.sweep.server import ServerProcess

pytestmark = [
    pytest.mark.skip_global_cleanup,
    pytest.mark.skipif(not hasattr(os, "fork"), reason="Requires POSIX process groups"),
]


@pytest.mark.parametrize("parent_exits", [False, True])
def test_stop_cleans_up_server_process_group(parent_exits: bool):
    """Sweep cleanup must terminate workers even if the server already exited."""
    script = """
import os
import signal
import socket
import sys

with socket.create_connection(("127.0.0.1", int(sys.argv[1]))) as connection:
    if os.fork() == 0:
        connection.sendall(b"R")
        connection.recv(1)
    elif sys.argv[2] == "True":
        os._exit(1)
    else:
        signal.pause()
"""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(10)
        server = ServerProcess(
            [
                sys.executable,
                "-c",
                script,
                str(listener.getsockname()[1]),
                str(parent_exits),
            ],
            [],
            show_stdout=True,
        )
        try:
            with server:
                process = server._server_process
                connection, _ = listener.accept()
                with contextlib.closing(connection):
                    connection.settimeout(10)
                    assert connection.recv(1) == b"R"
                    if parent_exits:
                        assert process.wait(timeout=10) == 1
                    server.stop()
                    assert connection.recv(1) == b"", "Server worker is still alive"
                    assert process.returncode is not None, "Server was not reaped"
        finally:
            # Keep failing regressions from leaking the deliberately orphaned worker.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(server._server_process.pid, signal.SIGKILL)
            server._server_process.wait(timeout=10)
