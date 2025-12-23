# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import ipaddress
import os
import socket
import sys
import warnings
from collections.abc import (
    Iterator,
    Sequence,
)
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import psutil
import zmq
import zmq.asyncio

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# Module-level tracking of reserved port sockets to avoid pydantic serialization issues
# Maps port number -> socket object
_reserved_port_sockets: dict[int, socket.socket] = {}


@dataclass
class ReservedPort:
    """A port reservation that holds the socket open until explicitly released.

    This prevents race conditions where a port is discovered as free but then
    claimed by another process before it can be used. The socket remains bound
    until release() is called or the context manager exits.

    See GitHub issue #28498 for details on the race condition this solves.
    """

    port: int
    _socket: socket.socket | None = None

    def release(self) -> int:
        """Release the port reservation and return the port number."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        return self.port

    def __enter__(self) -> "ReservedPort":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __del__(self) -> None:
        if self._socket is not None:
            with contextlib.suppress(Exception):
                self._socket.close()


def close_sockets(sockets: Sequence[zmq.Socket | zmq.asyncio.Socket]):
    for sock in sockets:
        if sock is not None:
            sock.close(linger=0)


def get_ip() -> str:
    host_ip = envs.VLLM_HOST_IP
    if "HOST_IP" in os.environ and "VLLM_HOST_IP" not in os.environ:
        logger.warning(
            "The environment variable HOST_IP is deprecated and ignored, as"
            " it is often used by Docker and other software to"
            " interact with the container's network stack. Please "
            "use VLLM_HOST_IP instead to set the IP address for vLLM processes"
            " to communicate with each other."
        )
    if host_ip:
        return host_ip

    # IP is not set, try to get it from the network interface

    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
            return s.getsockname()[0]
    except Exception:
        pass

    # try ipv6
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as s:
            # Google's public DNS server, see
            # https://developers.google.com/speed/public-dns/docs/using#addresses
            s.connect(("2001:4860:4860::8888", 80))  # Doesn't need to be reachable
            return s.getsockname()[0]
    except Exception:
        pass

    warnings.warn(
        "Failed to get the IP address, using 0.0.0.0 by default."
        "The value can be set by the environment variable"
        " VLLM_HOST_IP or HOST_IP.",
        stacklevel=2,
    )
    return "0.0.0.0"


def test_loopback_bind(address: str, family: int) -> bool:
    try:
        s = socket.socket(family, socket.SOCK_DGRAM)
        s.bind((address, 0))  # Port 0 = auto assign
        s.close()
        return True
    except OSError:
        return False


def get_loopback_ip() -> str:
    loopback_ip = envs.VLLM_LOOPBACK_IP
    if loopback_ip:
        return loopback_ip

    # VLLM_LOOPBACK_IP is not set, try to get it based on network interface

    if test_loopback_bind("127.0.0.1", socket.AF_INET):
        return "127.0.0.1"
    elif test_loopback_bind("::1", socket.AF_INET6):
        return "::1"
    else:
        raise RuntimeError(
            "Neither 127.0.0.1 nor ::1 are bound to a local interface. "
            "Set the VLLM_LOOPBACK_IP environment variable explicitly."
        )


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def split_host_port(host_port: str) -> tuple[str, int]:
    # ipv6
    if host_port.startswith("["):
        host, port = host_port.rsplit("]", 1)
        host = host[1:]
        port = port.split(":")[1]
        return host, int(port)
    else:
        host, port = host_port.split(":")
        return host, int(port)


def join_host_port(host: str, port: int) -> str:
    if is_valid_ipv6_address(host):
        return f"[{host}]:{port}"
    else:
        return f"{host}:{port}"


def get_distributed_init_method(ip: str, port: int) -> str:
    return get_tcp_uri(ip, port)


def get_tcp_uri(ip: str, port: int) -> str:
    if is_valid_ipv6_address(ip):
        return f"tcp://[{ip}]:{port}"
    else:
        return f"tcp://{ip}:{port}"


def get_open_zmq_ipc_path() -> str:
    base_rpc_path = envs.VLLM_RPC_BASE_PATH
    return f"ipc://{base_rpc_path}/{uuid4()}"


def get_open_zmq_inproc_path() -> str:
    return f"inproc://{uuid4()}"


def get_open_port() -> int:
    """
    Get an open port for the vLLM process to listen on.
    An edge case to handle, is when we run data parallel,
    we need to avoid ports that are potentially used by
    the data parallel master process.
    Right now we reserve 10 ports for the data parallel master
    process. Currently it uses 2 ports.
    """
    if "VLLM_DP_MASTER_PORT" in os.environ:
        dp_master_port = envs.VLLM_DP_MASTER_PORT
        reserved_port_range = range(dp_master_port, dp_master_port + 10)
        while True:
            candidate_port = _get_open_port()
            if candidate_port not in reserved_port_range:
                return candidate_port
    return _get_open_port()


def get_open_ports_list(count: int = 5) -> list[int]:
    """Get a list of open ports."""
    ports = set[int]()
    while len(ports) < count:
        ports.add(get_open_port())
    return list(ports)


def get_reserved_port() -> ReservedPort:
    """Get a reserved port that stays held until explicitly released.

    Unlike get_open_port(), this function returns a ReservedPort object
    that keeps the underlying socket bound, preventing other processes
    from claiming the port until release() is called.
    """
    if "VLLM_DP_MASTER_PORT" in os.environ:
        dp_master_port = envs.VLLM_DP_MASTER_PORT
        reserved_port_range = range(dp_master_port, dp_master_port + 10)
        while True:
            reserved = _get_reserved_port()
            if reserved.port not in reserved_port_range:
                return reserved
            reserved.release()
    return _get_reserved_port()


def get_reserved_ports_list(count: int = 5) -> list[ReservedPort]:
    """Get a list of reserved ports."""
    reservations: list[ReservedPort] = []
    seen_ports: set[int] = set()
    while len(reservations) < count:
        reserved = get_reserved_port()
        if reserved.port not in seen_ports:
            seen_ports.add(reserved.port)
            reservations.append(reserved)
        else:
            reserved.release()
    return reservations


def get_reserved_ports_as_int_list(count: int = 5) -> list[int]:
    """Get a list of reserved port numbers.

    Unlike get_reserved_ports_list(), this returns plain integers.
    The sockets are tracked internally and must be released via
    release_reserved_port(). This is designed for use with pydantic
    dataclasses that cannot serialize socket objects.
    """
    ports: list[int] = []
    seen_ports: set[int] = set()
    while len(ports) < count:
        reserved = get_reserved_port()
        if reserved.port not in seen_ports:
            seen_ports.add(reserved.port)
            ports.append(reserved.port)
            # Transfer socket ownership to module-level tracking
            if reserved._socket is not None:
                _reserved_port_sockets[reserved.port] = reserved._socket
                reserved._socket = None  # Prevent double-close
        else:
            reserved.release()
    return ports


def release_reserved_port(port: int) -> int:
    """Release a port reserved via get_reserved_ports_as_int_list().

    Args:
        port: The port number to release.

    Returns:
        The port number that was released.
    """
    sock = _reserved_port_sockets.pop(port, None)
    if sock is not None:
        with contextlib.suppress(Exception):
            sock.close()
    return port


def _get_reserved_port() -> ReservedPort:
    """Internal function to get a reserved port with socket held open."""
    port_env = envs.VLLM_PORT
    if port_env is not None:
        port = port_env
        while True:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("", port))
                return ReservedPort(port=port, _socket=sock)
            except OSError:
                port += 1
                logger.info("Port %d is already in use, trying port %d", port - 1, port)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        return ReservedPort(port=port, _socket=sock)
    except OSError:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        return ReservedPort(port=port, _socket=sock)


def _get_open_port() -> int:
    port = envs.VLLM_PORT
    if port is not None:
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                port += 1  # Increment port number if already in use
                logger.info("Port %d is already in use, trying port %d", port - 1, port)
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def find_process_using_port(port: int) -> psutil.Process | None:
    # TODO: We can not check for running processes with network
    # port on macOS. Therefore, we can not have a full graceful shutdown
    # of vLLM. For now, let's not look for processes in this case.
    # Ref: https://www.florianreinhard.de/accessdenied-in-psutil/
    if sys.platform.startswith("darwin"):
        return None

    our_pid = os.getpid()
    for conn in psutil.net_connections():
        if conn.laddr.port == port and (conn.pid is not None and conn.pid != our_pid):
            try:
                return psutil.Process(conn.pid)
            except psutil.NoSuchProcess:
                return None
    return None


def split_zmq_path(path: str) -> tuple[str, str, str]:
    """Split a zmq path into its parts."""
    parsed = urlparse(path)
    if not parsed.scheme:
        raise ValueError(f"Invalid zmq path: {path}")

    scheme = parsed.scheme
    host = parsed.hostname or ""
    port = str(parsed.port or "")

    if scheme == "tcp" and not all((host, port)):
        # The host and port fields are required for tcp
        raise ValueError(f"Invalid zmq path: {path}")

    if scheme != "tcp" and port:
        # port only makes sense with tcp
        raise ValueError(f"Invalid zmq path: {path}")

    return scheme, host, port


def make_zmq_path(scheme: str, host: str, port: int | None = None) -> str:
    """Make a ZMQ path from its parts.

    Args:
        scheme: The ZMQ transport scheme (e.g. tcp, ipc, inproc).
        host: The host - can be an IPv4 address, IPv6 address, or hostname.
        port: Optional port number, only used for TCP sockets.

    Returns:
        A properly formatted ZMQ path string.
    """
    if port is None:
        return f"{scheme}://{host}"
    if is_valid_ipv6_address(host):
        return f"{scheme}://[{host}]:{port}"
    return f"{scheme}://{host}:{port}"


# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L783 # noqa: E501
def make_zmq_socket(
    ctx: zmq.asyncio.Context | zmq.Context,  # type: ignore[name-defined]
    path: str,
    socket_type: Any,
    bind: bool | None = None,
    identity: bytes | None = None,
    linger: int | None = None,
) -> zmq.Socket | zmq.asyncio.Socket:  # type: ignore[name-defined]
    """Make a ZMQ socket with the proper bind/connect semantics."""

    mem = psutil.virtual_memory()
    socket = ctx.socket(socket_type)

    # Calculate buffer size based on system memory
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    # For systems with substantial memory (>32GB total, >16GB available):
    # - Set a large 0.5GB buffer to improve throughput
    # For systems with less memory:
    # - Use system default (-1) to avoid excessive memory consumption
    buf_size = int(0.5 * 1024**3) if total_mem > 32 and available_mem > 16 else -1

    if bind is None:
        bind = socket_type not in (zmq.PUSH, zmq.SUB, zmq.XSUB)

    if socket_type in (zmq.PULL, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type in (zmq.PUSH, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    if identity is not None:
        socket.setsockopt(zmq.IDENTITY, identity)

    if linger is not None:
        socket.setsockopt(zmq.LINGER, linger)

    if socket_type == zmq.XPUB:
        socket.setsockopt(zmq.XPUB_VERBOSE, True)

    # Determine if the path is a TCP socket with an IPv6 address.
    # Enable IPv6 on the zmq socket if so.
    scheme, host, _ = split_zmq_path(path)
    if scheme == "tcp" and is_valid_ipv6_address(host):
        socket.setsockopt(zmq.IPV6, 1)

    if bind:
        socket.bind(path)
    else:
        socket.connect(path)

    return socket


@contextlib.contextmanager
def zmq_socket_ctx(
    path: str,
    socket_type: Any,
    bind: bool | None = None,
    linger: int = 0,
    identity: bytes | None = None,
) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    ctx = zmq.Context()  # type: ignore[attr-defined]
    try:
        yield make_zmq_socket(ctx, path, socket_type, bind=bind, identity=identity)
    except KeyboardInterrupt:
        logger.debug("Got Keyboard Interrupt.")

    finally:
        ctx.destroy(linger=linger)
