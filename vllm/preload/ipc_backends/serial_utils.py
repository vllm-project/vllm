# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pickle framing and descriptor attachments for trusted local IPC sockets."""

import array
import io
import os
import pickle
import socket
import struct
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any


class SharedFileDescriptor:
    """Own one descriptor; transfer it with `send_msg`, never as an integer."""

    def __init__(self, fd: int):
        self._fd = fd

    def fileno(self) -> int:
        if self._fd < 0:
            raise ValueError("File descriptor is closed")
        return self._fd

    def close(self) -> None:
        fd, self._fd = self._fd, -1
        if fd >= 0:
            os.close(fd)

    def __reduce__(self):
        raise TypeError("File descriptors must be sent with the weight-cache protocol")

    def __del__(self, _close=os.close, _suppress=suppress):
        fd, self._fd = self._fd, -1
        if fd >= 0:
            with _suppress(OSError):
                _close(fd)


_FD_BATCH_SIZE = 128
MAX_FILE_DESCRIPTORS = 1 << 16


@dataclass
class _DescriptorMessage:
    payload: bytes
    count: int


def send_descriptors(sock: socket.socket, descriptors: list[int]) -> None:
    """Pass descriptors in bounded SCM_RIGHTS batches, each carrying one byte."""
    for start in range(0, len(descriptors), _FD_BATCH_SIZE):
        batch = array.array("i", descriptors[start : start + _FD_BATCH_SIZE])
        if sock.sendmsg([b"F"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, batch)]) != 1:
            raise ConnectionError("Could not send weight-cache file descriptors")


def receive_descriptors(sock: socket.socket, count: int) -> list[SharedFileDescriptor]:
    if not isinstance(count, int) or not 0 < count <= MAX_FILE_DESCRIPTORS:
        raise ValueError("Invalid weight-cache descriptor count")
    descriptors: list[SharedFileDescriptor] = []
    itemsize = array.array("i").itemsize
    flags = getattr(socket, "MSG_CMSG_CLOEXEC", 0)
    try:
        while len(descriptors) < count:
            data, ancillary, message_flags, _ = sock.recvmsg(
                1, socket.CMSG_SPACE(_FD_BATCH_SIZE * itemsize), flags
            )
            batch: list[int] = []
            for level, kind, payload in ancillary:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    values = array.array("i")
                    values.frombytes(payload[: len(payload) // itemsize * itemsize])
                    batch.extend(values)
            descriptors.extend(SharedFileDescriptor(fd) for fd in batch)
            if (
                data != b"F"
                or not batch
                or message_flags & socket.MSG_CTRUNC
                or len(descriptors) > count
            ):
                raise ConnectionError("Invalid or truncated weight-cache descriptors")
            if not flags:
                for fd in batch:
                    os.set_inheritable(fd, False)
        return descriptors
    except BaseException:
        for descriptor in descriptors:
            descriptor.close()
        raise


_LEN_STRUCT = struct.Struct("!Q")
# Sanity bound for a single message. IPC handles are tiny; only small
# non-CUDA tensors are ever shipped by value.
MAX_MSG_SIZE = 1 << 34


def encode_message(obj: Any) -> tuple[bytes, list[int]]:
    """Keep legacy framing unchanged when a message has no descriptor attachments."""
    descriptors: list[int] = []
    indices: dict[int, int] = {}

    class DescriptorPickler(pickle.Pickler):
        def persistent_id(self, value):
            if not isinstance(value, SharedFileDescriptor):
                return None
            fd = value.fileno()
            if fd not in indices:
                indices[fd] = len(descriptors)
                descriptors.append(fd)
            return "vllm-fd", indices[fd]

    stream = io.BytesIO()
    DescriptorPickler(stream, protocol=pickle.HIGHEST_PROTOCOL).dump(obj)
    payload = stream.getvalue()
    if descriptors:
        if len(descriptors) > MAX_FILE_DESCRIPTORS:
            raise ValueError("Too many weight-cache file descriptors")
        payload = pickle.dumps(
            _DescriptorMessage(payload, len(descriptors)),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return payload, descriptors


def decode_message(sock: socket.socket, payload: bytes) -> Any:
    message = pickle.loads(payload)
    if not isinstance(message, _DescriptorMessage):
        return message
    descriptors = receive_descriptors(sock, message.count)

    class DescriptorUnpickler(pickle.Unpickler):
        def persistent_load(self, value):
            if (
                not isinstance(value, tuple)
                or len(value) != 2
                or value[0] != "vllm-fd"
                or not isinstance(value[1], int)
                or not 0 <= value[1] < len(descriptors)
            ):
                raise pickle.UnpicklingError(
                    "Invalid weight-cache descriptor reference"
                )
            return descriptors[value[1]]

    try:
        return DescriptorUnpickler(io.BytesIO(message.payload)).load()
    except BaseException:
        for descriptor in descriptors:
            descriptor.close()
        raise


def send_msg(sock: socket.socket, obj: Any) -> None:
    payload, descriptors = encode_message(obj)
    if len(payload) > MAX_MSG_SIZE:
        raise ValueError(f"Message size {len(payload)} exceeds limit {MAX_MSG_SIZE}")
    sock.sendall(_LEN_STRUCT.pack(len(payload)))
    sock.sendall(payload)
    send_descriptors(sock, descriptors)


def recv_msg(sock: socket.socket) -> Any:
    (length,) = _LEN_STRUCT.unpack(_recv_exact(sock, _LEN_STRUCT.size))
    if length > MAX_MSG_SIZE:
        raise ValueError(f"Message size {length} exceeds limit {MAX_MSG_SIZE}")
    return decode_message(sock, _recv_exact(sock, length))


def _recv_exact(
    sock: socket.socket, num_bytes: int, deadline: float | None = None
) -> bytes:
    buf = bytearray()
    while len(buf) < num_bytes:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Peer did not send its message in time")
            sock.settimeout(remaining)
        chunk = sock.recv(num_bytes - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while receiving message")
        buf.extend(chunk)
    return bytes(buf)
