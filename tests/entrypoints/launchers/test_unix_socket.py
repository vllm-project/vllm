# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`--uds` socket files: a leftover file from a previous server must not block a
restart, a path a live server still listens on must not be taken over, and the
file is removed again on shutdown."""

import errno
import os
import socket
import sys
from pathlib import Path

import pytest

from vllm.entrypoints.launchers.launcher import (
    create_server_unix_socket,
    remove_unix_socket_file,
    unix_socket_file_identity,
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
    identity = unix_socket_file_identity(sock_path)
    assert identity is not None
    remove_unix_socket_file(sock_path, identity)
    assert not Path(sock_path).exists()

    # Missing file: nothing to do, no error.
    remove_unix_socket_file(sock_path, identity)

    # A regular file at the path is not ours to delete.
    Path(sock_path).write_text("not a socket")
    assert unix_socket_file_identity(sock_path) is None
    remove_unix_socket_file(sock_path, identity)
    assert Path(sock_path).exists()


def test_cleanup_keeps_a_replacement_socket(sock_path: str):
    # Server A binds the path; the file is removed from under it and server B
    # binds the same path. A's shutdown must not delete B's file.
    old = create_server_unix_socket(sock_path)
    identity = unix_socket_file_identity(sock_path)
    Path(sock_path).unlink()
    replacement = create_server_unix_socket(sock_path)
    try:
        replacement.listen(1)
        old.close()
        remove_unix_socket_file(sock_path, identity)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(sock_path)
        client.close()
    finally:
        replacement.close()


def test_cleanup_keeps_a_symlink(sock_path: str):
    owner = create_server_unix_socket(sock_path)
    identity = unix_socket_file_identity(sock_path)
    target = sock_path + ".target"
    try:
        Path(sock_path).rename(target)
        Path(sock_path).symlink_to(target)
        remove_unix_socket_file(sock_path, identity)
        assert Path(sock_path).is_symlink()
        assert Path(target).exists()
    finally:
        owner.close()


@pytest.mark.skipif(sys.platform != "linux", reason="abstract namespace is Linux-only")
def test_abstract_namespace_socket_has_no_file():
    path = f"\0vllm-uds-test-{os.getpid()}"
    sock = create_server_unix_socket(path)
    try:
        assert unix_socket_file_identity(path) is None
        remove_unix_socket_file(path, None)
    finally:
        sock.close()
