# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


_MAX_MESSAGE_BYTES = 1024 * 1024
_READ_CHUNK_BYTES = 65536


class SnapshotControlError(RuntimeError):
    def __init__(self, message: str, *, code: str = "snapshot_error") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SnapshotControlClient:
    path: str
    timeout: float = 1800.0

    def request(
        self, command: str, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        request = {"command": command, "payload": dict(payload or {})}
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout)
                sock.connect(self.path)
                sock.sendall(json.dumps(request).encode() + b"\n")
                response = _read_message(sock)
        except SnapshotControlError:
            raise
        except TimeoutError as exc:
            raise SnapshotControlError(
                f"snapshot control request timed out after {self.timeout} seconds",
                code="timeout",
            ) from exc
        except OSError as exc:
            raise SnapshotControlError(
                f"snapshot control connection failed: {exc}",
                code="connection_error",
            ) from exc
        if not response.get("ok", False):
            raise SnapshotControlError(
                response.get("error", "snapshot control request failed"),
                code=response.get("code", "snapshot_error"),
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise SnapshotControlError(
                "invalid snapshot control response",
                code="invalid_message",
            )
        return result


def _read_message(sock: socket.socket) -> dict[str, Any]:
    chunks = bytearray()
    while True:
        try:
            chunk = sock.recv(_READ_CHUNK_BYTES)
        except TimeoutError as exc:
            raise SnapshotControlError(
                "snapshot control message timed out",
                code="timeout",
            ) from exc
        except OSError as exc:
            raise SnapshotControlError(
                f"snapshot control connection failed: {exc}",
                code="connection_error",
            ) from exc
        if not chunk:
            if not chunks:
                raise SnapshotControlError(
                    "snapshot control connection closed",
                    code="connection_error",
                )
            raise SnapshotControlError(
                "snapshot control message is missing a newline terminator",
                code="invalid_message",
            )
        chunks.extend(chunk)
        newline = chunks.find(b"\n")
        if newline >= 0:
            if newline > _MAX_MESSAGE_BYTES:
                raise SnapshotControlError(
                    "snapshot control message is too large",
                    code="invalid_message",
                )
            break
        if len(chunks) > _MAX_MESSAGE_BYTES:
            raise SnapshotControlError(
                "snapshot control message is too large",
                code="invalid_message",
            )
    try:
        message = json.loads(bytes(chunks[:newline]))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotControlError(
            "invalid snapshot control JSON message",
            code="invalid_message",
        ) from exc
    if not isinstance(message, dict):
        raise SnapshotControlError(
            "snapshot control message must be a JSON object",
            code="invalid_message",
        )
    return message
