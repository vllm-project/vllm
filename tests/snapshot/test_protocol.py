# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, cast

import pytest

from vllm.snapshot.protocol import (
    _MAX_MESSAGE_BYTES,
    SnapshotControlClient,
    SnapshotControlError,
    _read_message,
)


class MessageSocket:
    def __init__(self, *chunks: bytes):
        self.chunks = list(chunks)

    def recv(self, size: int) -> bytes:
        return self.chunks.pop(0)


def test_read_message_rejects_oversized_terminated_chunk():
    chunks = [b"x" * 65536] * (_MAX_MESSAGE_BYTES // 65536)
    sock = MessageSocket(*chunks, b"x\n")

    with pytest.raises(SnapshotControlError, match="too large"):
        _read_message(cast(Any, sock))


def test_read_message_requires_newline_terminator():
    sock = MessageSocket(b'{"ok": true}', b"")

    with pytest.raises(SnapshotControlError, match="newline terminator"):
        _read_message(cast(Any, sock))


def test_read_message_normalizes_invalid_json():
    sock = MessageSocket(b"not-json\n")

    with pytest.raises(SnapshotControlError) as exc_info:
        _read_message(cast(Any, sock))

    assert exc_info.value.code == "invalid_message"


def test_control_client_normalizes_connection_errors(monkeypatch):
    class FailingSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def settimeout(self, timeout):
            pass

        def connect(self, path):
            raise ConnectionRefusedError(path)

    monkeypatch.setattr(
        "vllm.snapshot.protocol.socket.socket", lambda *args: FailingSocket()
    )

    with pytest.raises(SnapshotControlError) as exc_info:
        SnapshotControlClient("control.sock").request("status")

    assert exc_info.value.code == "connection_error"
