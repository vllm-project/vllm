# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`--uds` socket files: a leftover file from a previous server must not block a
restart, a path a live server still listens on must not be taken over, and the
file is removed again on shutdown."""

import errno
import socket
from pathlib import Path

import pytest

from vllm.entrypoints.launchers.launcher import (
    create_server_unix_socket,
    remove_unix_socket_file,
)


@pytest.fixture
def sock_path(tmp_path: Path) -> str:
    return str(tmp_path / "vllm.sock")


def _leave_stale_socket_file(path: str) -> None:
    # Bind, listen and close: this is what a stopped server leaves behind.
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(path)
    sock.listen(1)
    sock.close()
    assert Path(path).exists()


def test_stale_socket_file_is_replaced(sock_path: str):
    _leave_stale_socket_file(sock_path)

    sock = create_server_unix_socket(sock_path)
    try:
        sock.listen(1)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(sock_path)
        client.close()
    finally:
        sock.close()


def test_live_listener_is_not_taken_over(sock_path: str):
    live = create_server_unix_socket(sock_path)
    live.listen(1)
    try:
        with pytest.raises(OSError) as exc_info:
            create_server_unix_socket(sock_path)
        assert exc_info.value.errno == errno.EADDRINUSE
        # The live server is untouched and still reachable.
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(sock_path)
        client.close()
    finally:
        live.close()


def test_regular_file_at_path_is_left_alone(sock_path: str):
    Path(sock_path).write_text("not a socket")

    with pytest.raises(OSError):
        create_server_unix_socket(sock_path)
    assert Path(sock_path).read_text() == "not a socket"


def test_remove_unix_socket_file(sock_path: str):
    _leave_stale_socket_file(sock_path)
    remove_unix_socket_file(sock_path)
    assert not Path(sock_path).exists()

    # Missing file: nothing to do, no error.
    remove_unix_socket_file(sock_path)

    # A regular file at the path is not ours to delete.
    Path(sock_path).write_text("not a socket")
    remove_unix_socket_file(sock_path)
    assert Path(sock_path).exists()
