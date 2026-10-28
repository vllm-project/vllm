# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy-based stream transport shared by the control and data planes.

This is the ``inline`` transport: control frames carry encoded
MessageEnvelopes and data frames carry raw tensor bytes tagged with a slot
index, multiplexed on one stream connection. It needs only the standard
library and msgspec, and is correct rather than fast; zero-copy transports
(CUDA IPC, pinned-host ring) replace only the data-plane half and keep the
control plane as is.

Frame layout: ``kind:u8 | length:u32 big-endian | body``. Data-plane
payloads are in native byte order, which is acceptable because remote
placement is same-host only.
"""

import array
import contextlib
import ipaddress
import math
import os
import selectors
import socket
import struct
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

import msgspec

INLINE_TRANSPORT = "inline"

FRAME_CONTROL = 0x01
FRAME_DATA = 0x02

MAX_FRAME_BYTES = 1 << 30
MAX_CONTROL_FRAME_BYTES = 1 << 20
MAX_HANDSHAKE_BYTES = 1 << 16

_HEADER = struct.Struct("!BI")
_RECV_CHUNK = 1 << 16
_RECV_MAX = 1 << 24
# torch dtype name -> (array typecode, itemsize) for the torch-free integer
# helpers used on the server side.
_INT_TYPECODES = {"int32": ("i", 4), "int64": ("q", 8)}
_DTYPE_ITEMSIZE = {
    "int32": 4,
    "int64": 8,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}


class TransportError(Exception):
    """Failure on the stream connection."""


class TransportTimeout(TransportError):
    """A frame could not be sent or received before the deadline."""


class ConnectionClosed(TransportError):
    """The peer closed the connection or the socket failed."""


class DataFrame(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    """One tensor on the data plane.

    ``dtype`` is the torch dtype name (``"int32"``, ``"bfloat16"``) and
    ``payload`` holds the contiguous row-major bytes in native byte order.
    """

    slot: int
    dtype: str
    shape: tuple[int, ...]
    payload: bytes


@dataclass(frozen=True)
class Frame:
    """One raw frame off the wire."""

    kind: int
    body: bytes


_frame_encoder = msgspec.msgpack.Encoder()
_frame_decoder = msgspec.msgpack.Decoder(DataFrame)


def check_data_frame_size(
    slot: int,
    dtype: str,
    shape: tuple[int, ...],
    *,
    payload_bytes: int | None = None,
) -> int:
    """Check and return encoded size without allocating/copying the payload."""
    if payload_bytes is None:
        try:
            itemsize = _DTYPE_ITEMSIZE[dtype]
        except KeyError:
            raise TransportError(f"unknown dtype {dtype!r} in data frame") from None
        if any(size < 0 for size in shape):
            raise TransportError(f"data frame {slot} has negative shape {shape}")
        payload_bytes = math.prod(shape) * itemsize
    if payload_bytes > MAX_FRAME_BYTES:
        raise TransportError(
            f"frame payload of {payload_bytes} bytes exceeds {MAX_FRAME_BYTES}"
        )
    # Empty bytes encode with a two-byte bin8 header. Account for the wider
    # bin16/bin32 length when replacing them with the actual payload.
    header_growth = 0 if payload_bytes < 256 else 1 if payload_bytes < 65536 else 3
    metadata = _frame_encoder.encode(DataFrame(slot, dtype, shape, b""))
    encoded_size = len(metadata) + header_growth + payload_bytes
    if encoded_size > MAX_FRAME_BYTES:
        raise TransportError(f"frame of {encoded_size} bytes exceeds {MAX_FRAME_BYTES}")
    return encoded_size


def decode_data_frame(body: bytes) -> DataFrame:
    """Decode the body of a FRAME_DATA frame."""
    try:
        return _frame_decoder.decode(body)
    except msgspec.DecodeError as e:
        raise TransportError(f"malformed data frame: {e}") from e


def ints_to_frame(
    slot: int,
    values: Sequence[int],
    *,
    dtype: str = "int32",
    shape: tuple[int, ...] | None = None,
) -> DataFrame:
    """Build an integer DataFrame without torch."""
    typecode, itemsize = _int_typecode(dtype)
    if shape is None:
        shape = (len(values),)
    elif math.prod(shape) != len(values):
        raise ValueError(f"shape {shape} does not hold {len(values)} values")
    data = array.array(typecode, values)
    if data.itemsize != itemsize:
        raise TransportError(f"array typecode {typecode!r} is not {dtype} here")
    return DataFrame(slot=slot, dtype=dtype, shape=tuple(shape), payload=data.tobytes())


def frame_to_ints(frame: DataFrame) -> list[int]:
    """Read an integer DataFrame without torch (row-major, flattened)."""
    typecode, _ = _int_typecode(frame.dtype)
    validate_integer_frame(frame)
    data = array.array(typecode)
    data.frombytes(frame.payload)
    return data.tolist()


def validate_integer_frame(frame: DataFrame) -> None:
    """Validate an integer frame's dtype and payload size without decoding it."""
    _int_typecode(frame.dtype)
    validate_frame(frame)


def validate_frame(frame: DataFrame) -> None:
    """Validate a frame's dtype, shape, and payload size without decoding it."""
    try:
        itemsize = _DTYPE_ITEMSIZE[frame.dtype]
    except KeyError:
        raise TransportError(f"unknown dtype {frame.dtype!r} in data frame") from None
    if any(size < 0 for size in frame.shape):
        raise TransportError(
            f"data frame {frame.slot} has negative shape {frame.shape}"
        )
    expected = math.prod(frame.shape) * itemsize
    if len(frame.payload) != expected:
        raise TransportError(
            f"data frame {frame.slot} has {len(frame.payload)} payload bytes, "
            f"expected {expected} for {frame.dtype} shape {frame.shape}"
        )


def _int_typecode(dtype: str) -> tuple[str, int]:
    try:
        return _INT_TYPECODES[dtype]
    except KeyError:
        raise TransportError(
            f"dtype {dtype!r} is not an integer frame dtype ({sorted(_INT_TYPECODES)})"
        ) from None


def parse_endpoint(endpoint: str) -> tuple[int, tuple[str, int] | str]:
    """Parse ``tcp://host:port`` or ``unix:///path`` into a socket address."""
    parts = urlsplit(endpoint)
    if parts.scheme == "tcp":
        try:
            port = parts.port
        except ValueError as e:
            raise ValueError(f"invalid tcp port in {endpoint!r}") from e
        if not parts.hostname or port is None:
            raise ValueError(f"tcp endpoint needs host and port: {endpoint!r}")
        if ":" in parts.hostname:
            return socket.AF_INET6, (parts.hostname, port)
        return socket.AF_INET, (parts.hostname, port)
    if parts.scheme == "unix":
        path = parts.path or parts.netloc
        if not path:
            raise ValueError(f"unix endpoint needs a path: {endpoint!r}")
        family = getattr(socket, "AF_UNIX", None)
        if family is None:
            raise ValueError("unix sockets are not supported on this platform")
        return family, path
    raise ValueError(
        f"unsupported endpoint scheme {parts.scheme!r} in {endpoint!r}; "
        "use tcp://host:port or unix:///path"
    )


def is_local_endpoint(endpoint: str) -> bool:
    """Whether only processes on this host can reach ``endpoint``."""
    family, address = parse_endpoint(endpoint)
    if not isinstance(address, tuple):
        return True
    host = address[0]
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class FramedConnection:
    """Blocking, single-reader framed stream with per-call deadlines.

    Receive timeouts leave partial frames buffered for retries. A send
    whose deadline passed before the frame started leaves the stream
    intact; any other send failure closes it because a partial write cannot
    be resumed.
    ``close`` may be called from another thread to unblock pending I/O.
    """

    def __init__(self, sock: socket.socket) -> None:
        sock.settimeout(None)
        self._sock = sock
        self._buffer = bytearray()
        self._closed = False
        self._max_control_bytes = MAX_CONTROL_FRAME_BYTES
        self._max_data_bytes = MAX_FRAME_BYTES
        self._max_buffered_bytes = MAX_FRAME_BYTES + MAX_CONTROL_FRAME_BYTES
        self._scan_offset = 0
        self._scan_end: int | None = None
        self._partial_started: float | None = None
        self._partial_frame_timeout: float | None = None

    def set_receive_limits(
        self,
        *,
        max_control_bytes: int = MAX_CONTROL_FRAME_BYTES,
        max_data_bytes: int = MAX_FRAME_BYTES,
        max_buffered_bytes: int = MAX_FRAME_BYTES + MAX_CONTROL_FRAME_BYTES,
        partial_frame_timeout: float | None = None,
    ) -> None:
        """Bound both normal reads and replies buffered during duplex sends.

        Zero data bytes forbids data frames (useful before HELLO). A started
        frame has a finite lifetime independent of an otherwise idle connection.
        Only the connection's single I/O owner may change these limits.
        """
        if not 0 < max_control_bytes < (1 << 32):
            raise ValueError("invalid control frame byte limit")
        if not 0 <= max_data_bytes < (1 << 32):
            raise ValueError("invalid data frame byte limit")
        if max_buffered_bytes < _HEADER.size:
            raise ValueError("invalid buffered byte limit")
        if partial_frame_timeout is not None and (
            not math.isfinite(partial_frame_timeout) or partial_frame_timeout <= 0
        ):
            raise ValueError("partial frame timeout must be finite and positive")
        self._max_control_bytes = min(max_control_bytes, MAX_FRAME_BYTES)
        self._max_data_bytes = min(max_data_bytes, MAX_FRAME_BYTES)
        self._max_buffered_bytes = max_buffered_bytes
        self._partial_frame_timeout = partial_frame_timeout
        # Revalidate unread frames if a handshake narrowed the contract.
        partial_started = self._partial_started
        self._scan_offset = 0
        self._scan_end = None
        self._scan_buffer()
        if self._scan_end is not None or self._scan_offset < len(self._buffer):
            self._partial_started = partial_started

    def _receive_error(self, detail: str) -> None:
        with contextlib.suppress(OSError):
            self.close()
        raise TransportError(detail)

    def _scan_buffer(self) -> None:
        if len(self._buffer) > self._max_buffered_bytes:
            self._receive_error("buffered receive bytes exceed the connection budget")
        while True:
            if self._scan_end is None:
                if len(self._buffer) - self._scan_offset < _HEADER.size:
                    return
                kind, length = _HEADER.unpack_from(self._buffer, self._scan_offset)
                if kind not in (FRAME_CONTROL, FRAME_DATA):
                    self._receive_error(f"unknown frame kind {kind:#x}")
                limit = (
                    self._max_control_bytes
                    if kind == FRAME_CONTROL
                    else self._max_data_bytes
                )
                if length > limit or (kind == FRAME_DATA and limit == 0):
                    self._receive_error(f"frame length {length} exceeds {limit}")
                end = self._scan_offset + _HEADER.size + length
                if end > self._max_buffered_bytes:
                    self._receive_error(
                        "buffered receive bytes exceed the connection budget"
                    )
                self._scan_end = end
            if len(self._buffer) < self._scan_end:
                return
            self._scan_offset = self._scan_end
            self._scan_end = None
            self._partial_started = None

    def _receive_size(self, maximum: int) -> int:
        # Read a header before its body, including when draining replies while
        # writing. Oversized advertised bodies are never read speculatively.
        boundary = self._scan_end or self._scan_offset + _HEADER.size
        capacity = self._max_buffered_bytes - len(self._buffer)
        if capacity <= 0:
            self._receive_error("buffered receive bytes exceed the connection budget")
        return min(maximum, boundary - len(self._buffer), capacity)

    def _append_received(self, chunk: bytes) -> None:
        if self._partial_started is None:
            self._partial_started = time.monotonic()
        self._buffer.extend(chunk)
        self._scan_buffer()

    def _receive_deadline(self, deadline: float | None) -> float | None:
        if self._partial_started is None or self._partial_frame_timeout is None:
            return deadline
        partial_deadline = self._partial_started + self._partial_frame_timeout
        if time.monotonic() >= partial_deadline:
            with contextlib.suppress(OSError):
                self.close()
            raise TransportTimeout("started frame exceeded its receive timeout")
        return partial_deadline if deadline is None else min(deadline, partial_deadline)

    @property
    def closed(self) -> bool:
        return self._closed

    def send_control(
        self,
        data: bytes,
        *,
        deadline: float | None = None,
        drain_incoming: bool = False,
    ) -> None:
        self._send(
            FRAME_CONTROL, data, deadline=deadline, drain_incoming=drain_incoming
        )

    def send_data(
        self,
        frame: DataFrame,
        *,
        deadline: float | None = None,
        drain_incoming: bool = False,
    ) -> None:
        check_data_frame_size(
            frame.slot, frame.dtype, frame.shape, payload_bytes=len(frame.payload)
        )
        self._send(
            FRAME_DATA,
            _frame_encoder.encode(frame),
            deadline=deadline,
            drain_incoming=drain_incoming,
        )

    def _send(
        self,
        kind: int,
        body: bytes,
        *,
        deadline: float | None = None,
        drain_incoming: bool = False,
    ) -> None:
        if self._closed:
            raise ConnectionClosed("connection is closed")
        if len(body) > MAX_FRAME_BYTES:
            raise TransportError(
                f"frame of {len(body)} bytes exceeds {MAX_FRAME_BYTES}"
            )
        if deadline is not None and deadline <= time.monotonic():
            raise TransportTimeout("deadline elapsed before sending a frame")
        parts = (_HEADER.pack(kind, len(body)), body)
        try:
            if drain_incoming:
                self._send_duplex(parts, deadline)
                return
            for part in parts:
                if not part:
                    continue
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError
                self._sock.settimeout(remaining)
                self._sock.sendall(part)
        except MemoryError:
            # A write or duplex read may already have consumed stream bytes.
            with contextlib.suppress(OSError):
                self.close()
            raise
        except TransportError:
            with contextlib.suppress(OSError):
                self.close()
            raise
        except TimeoutError:
            with contextlib.suppress(OSError):
                self.close()
            raise TransportTimeout("timed out sending a frame") from None
        except OSError as e:
            with contextlib.suppress(OSError):
                self.close()
            raise ConnectionClosed(f"send failed: {e}") from e

    def _send_duplex(self, parts: tuple[bytes, bytes], deadline: float | None) -> None:
        """Buffer replies while writing so pipelined peers cannot block each other."""
        self._sock.setblocking(False)
        with selectors.DefaultSelector() as selector:
            try:
                selector.register(
                    self._sock, selectors.EVENT_READ | selectors.EVENT_WRITE
                )
            except ValueError:
                if self._closed:
                    raise OSError("connection is closed") from None
                raise
            for part in parts:
                pending = memoryview(part)
                while pending:
                    if self._closed:
                        raise OSError("connection is closed")
                    receive_deadline = self._receive_deadline(deadline)
                    remaining = (
                        None
                        if receive_deadline is None
                        else receive_deadline - time.monotonic()
                    )
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError
                    # Closing a socket can remove it from epoll without waking
                    # the selector. Periodically check close() even without a deadline.
                    events = selector.select(
                        0.1 if remaining is None else min(remaining, 0.1)
                    )
                    for _, mask in events:
                        if mask & selectors.EVENT_READ:
                            try:
                                chunk = self._sock.recv(self._receive_size(_RECV_CHUNK))
                            except BlockingIOError:
                                pass
                            else:
                                if not chunk:
                                    raise OSError("peer closed the connection")
                                self._append_received(chunk)
                        if mask & selectors.EVENT_WRITE:
                            try:
                                sent = self._sock.send(pending)
                            except BlockingIOError:
                                continue
                            if not sent:
                                raise OSError("peer closed the connection")
                            pending = pending[sent:]

    def recv(
        self, timeout: float | None = None, *, max_frame_bytes: int | None = None
    ) -> Frame:
        """Receive one frame, waiting at most ``timeout`` seconds."""
        if self._closed:
            raise ConnectionClosed("connection is closed")
        if max_frame_bytes is None:
            max_frame_bytes = MAX_FRAME_BYTES
        if not 0 < max_frame_bytes <= MAX_FRAME_BYTES:
            raise ValueError("invalid receive frame byte limit")
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            self._fill(_HEADER.size, deadline)
            kind, length = _HEADER.unpack_from(self._buffer)
            if length > max_frame_bytes:
                self._receive_error(f"frame length {length} exceeds {max_frame_bytes}")
            end = _HEADER.size + length
            self._fill(end, deadline)
            body = bytes(self._buffer[_HEADER.size : end])
            del self._buffer[:end]
            self._scan_offset -= end
            if self._scan_end is not None:
                self._scan_end -= end
            return Frame(kind, body)
        except MemoryError:
            # Bytes consumed from the socket or buffer may not be recoverable.
            with contextlib.suppress(OSError):
                self.close()
            raise

    def _fill(self, size: int, deadline: float | None) -> None:
        while len(self._buffer) < size:
            if self._closed:
                raise ConnectionClosed("connection is closed")
            receive_deadline = self._receive_deadline(deadline)
            remaining = None
            if receive_deadline is not None:
                remaining = receive_deadline - time.monotonic()
                if remaining <= 0:
                    raise TransportTimeout("timed out waiting for a frame")
            want = self._receive_size(min(size - len(self._buffer), _RECV_MAX))
            try:
                self._sock.settimeout(remaining)
                chunk = self._sock.recv(want)
            except TimeoutError:
                self._receive_deadline(deadline)
                raise TransportTimeout("timed out waiting for a frame") from None
            except OSError as e:
                raise ConnectionClosed(f"recv failed: {e}") from e
            if not chunk:
                raise ConnectionClosed("peer closed the connection")
            self._append_received(chunk)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(OSError):
            self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()


class Listener:
    """Bound server socket handing out FramedConnections."""

    def __init__(
        self,
        sock: socket.socket,
        endpoint: str,
        unix_path: str | None,
        unix_identity: tuple[int, int] | None,
    ) -> None:
        self._sock = sock
        self._unix_path = unix_path
        self._unix_identity = unix_identity
        self.endpoint = endpoint
        """Resolved endpoint, with the actual port for ``tcp://host:0``."""

    def accept(self, timeout: float | None = None) -> FramedConnection:
        try:
            self._sock.settimeout(timeout)
            conn, _ = self._sock.accept()
        except TimeoutError:
            raise TransportTimeout("no connection before the deadline") from None
        except OSError as e:
            raise ConnectionClosed(f"listener closed: {e}") from e
        _tune(conn)
        return FramedConnection(conn)

    def close(self) -> None:
        unix_identity = self._unix_identity
        self._unix_identity = None
        try:
            self._sock.close()
        finally:
            if self._unix_path is not None and unix_identity is not None:
                _unlink_if_owned(self._unix_path, unix_identity)


def _tune(sock: socket.socket) -> None:
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def _unlink_if_owned(path: str, identity: tuple[int, int]) -> None:
    with contextlib.suppress(OSError):
        info = os.stat(path, follow_symlinks=False)
        if (info.st_dev, info.st_ino) == identity:
            os.unlink(path)


def connect(endpoint: str, timeout: float | None = None) -> FramedConnection:
    """Open a client connection to ``endpoint``."""
    family, address = parse_endpoint(endpoint)
    try:
        if family in (socket.AF_INET, socket.AF_INET6):
            assert isinstance(address, tuple)
            sock = socket.create_connection(address, timeout=timeout)
        else:
            assert isinstance(address, str)
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(address)
    except TimeoutError:
        raise TransportTimeout(f"connect to {endpoint} timed out") from None
    except OSError as e:
        raise ConnectionClosed(f"connect to {endpoint} failed: {e}") from e
    _tune(sock)
    return FramedConnection(sock)


def listen(endpoint: str, backlog: int = 16) -> Listener:
    """Bind ``endpoint`` and start listening."""
    family, address = parse_endpoint(endpoint)
    sock = socket.socket(family, socket.SOCK_STREAM)
    unix_path: str | None = None
    unix_identity: tuple[int, int] | None = None
    try:
        if family in (socket.AF_INET, socket.AF_INET6):
            if sys.platform != "win32":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(address)
            host, port = sock.getsockname()[:2]
            if family == socket.AF_INET6:
                host = f"[{host}]"
            resolved = f"tcp://{host}:{port}"
        else:
            assert isinstance(address, str)
            unix_path = address
            sock.bind(unix_path)
            info = os.stat(unix_path, follow_symlinks=False)
            unix_identity = (info.st_dev, info.st_ino)
            resolved = f"unix://{unix_path}"
        sock.listen(backlog)
    except OSError as e:
        sock.close()
        if unix_path is not None and unix_identity is not None:
            _unlink_if_owned(unix_path, unix_identity)
        raise TransportError(f"cannot listen on {endpoint}: {e}") from e
    return Listener(sock, resolved, unix_path, unix_identity)
