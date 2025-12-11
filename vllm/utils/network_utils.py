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
from typing import Any, Literal, overload
from urllib.parse import urlparse
from uuid import uuid4

import psutil
import zmq
import zmq.asyncio

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


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


def is_wildcard_addr(addr: str) -> bool:
    """Check if an address is a TCP address with wildcard port requiring late binding.

    A wildcard port address has port 0, which tells the OS to assign an available
    port. The host can be specific (e.g., "tcp://192.168.1.5:0") or wildcard
    (e.g., "tcp://*:0").

    Args:
        addr: Address string to check

    Returns:
        True if the address is a TCP address with wildcard port (:0)

    Examples:
        >>> is_wildcard_addr("tcp://*:0")
        True
        >>> is_wildcard_addr("tcp://192.168.1.5:0")
        True
        >>> is_wildcard_addr("tcp://127.0.0.1:8080")
        False
        >>> is_wildcard_addr("ipc:///tmp/socket")
        False
    """
    return addr.startswith("tcp://") and ":0" in addr


def bind_zmq_socket_and_get_address(
    ctx: zmq.asyncio.Context | zmq.Context,  # type: ignore[name-defined]
    wildcard_addr: str,
    socket_type: Any,
    **socket_opts: Any,
) -> tuple[zmq.Socket | zmq.asyncio.Socket, str]:  # type: ignore[name-defined]
    """
    Bind a ZMQ socket to an address and return the actual bound address.

    For TCP wildcard addresses like "tcp://*:0", binds to let the OS assign
    a port, then discovers the actual port via socket.last_endpoint.

    For IPC addresses, binds directly without modification.

    This eliminates port race conditions by binding immediately and discovering
    the actual assigned port, rather than pre-allocating a port that could be
    stolen before binding.

    Note: This is a convenience wrapper around make_zmq_socket() with
    return_address=True. Prefer using make_zmq_socket() directly for new code.

    Args:
        ctx: ZMQ context (async or sync)
        wildcard_addr: Address to bind (e.g., "tcp://*:0" or "ipc:///tmp/path")
        socket_type: ZMQ socket type constant (zmq.ROUTER, zmq.PULL, etc.)
        **socket_opts: Additional options passed to make_zmq_socket
                      (identity, linger, etc.)

    Returns:
        (socket, actual_address) tuple where:
        - socket: The bound ZMQ socket (caller must keep alive)
        - actual_address: Real address with OS-assigned port

    Example:
        >>> ctx = zmq.Context()
        >>> sock, addr = bind_zmq_socket_and_get_address(ctx, "tcp://*:0", zmq.ROUTER)
        >>> print(addr)  # "tcp://127.0.0.1:54321"
    """
    # Use make_zmq_socket with return_address=True to handle both wildcard
    # and non-wildcard addresses uniformly
    return make_zmq_socket(
        ctx, wildcard_addr, socket_type, bind=True, return_address=True, **socket_opts
    )


# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L783 # noqa: E501
@overload
def make_zmq_socket(
    ctx: zmq.asyncio.Context | zmq.Context,  # type: ignore[name-defined]
    path: str,
    socket_type: Any,
    bind: bool | None = ...,
    identity: bytes | None = ...,
    linger: int | None = ...,
    *,
    return_address: Literal[True],
) -> tuple[zmq.Socket | zmq.asyncio.Socket, str]: ...  # type: ignore[name-defined]


@overload
def make_zmq_socket(
    ctx: zmq.asyncio.Context | zmq.Context,  # type: ignore[name-defined]
    path: str,
    socket_type: Any,
    bind: bool | None = ...,
    identity: bytes | None = ...,
    linger: int | None = ...,
    *,
    return_address: Literal[False] = ...,
) -> zmq.Socket | zmq.asyncio.Socket: ...  # type: ignore[name-defined]


def make_zmq_socket(
    ctx: zmq.asyncio.Context | zmq.Context,  # type: ignore[name-defined]
    path: str,
    socket_type: Any,
    bind: bool | None = None,
    identity: bytes | None = None,
    linger: int | None = None,
    return_address: bool = False,
) -> zmq.Socket | zmq.asyncio.Socket | tuple[zmq.Socket | zmq.asyncio.Socket, str]:  # type: ignore[name-defined]
    """Make a ZMQ socket with the proper bind/connect semantics.

    Args:
        ctx: ZMQ context
        path: Socket path/address to bind or connect to
        socket_type: ZMQ socket type
        bind: Whether to bind (True) or connect (False). If None, auto-determined.
        identity: Optional socket identity
        linger: Optional linger value
        return_address: If True, return (socket, actual_address) tuple.
                       For wildcard addresses, returns the discovered address.
                       For non-wildcard addresses, returns the input path.

    Returns:
        Socket if return_address=False (default for backward compatibility)
        (socket, actual_address) tuple if return_address=True
    """
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
        # For wildcard port addresses, discover the actual bound address.
        if return_address and is_wildcard_addr(path):
            # last_endpoint is bytes like b"tcp://192.168.1.5:54321" or b"tcp://[::1]:54321"
            actual_endpoint = socket.last_endpoint.decode("utf-8")

            # Parse the endpoint to extract host and port
            # Handle both IPv4 and IPv6 formats
            scheme, host, port_str = split_zmq_path(actual_endpoint)
            if scheme != "tcp":
                # Shouldn't happen for wildcard TCP addresses, but fallback safely
                actual_address = actual_endpoint
            else:
                # Preserve the host from the bound endpoint
                actual_address = make_zmq_path(scheme, host, int(port_str))
        else:
            actual_address = path
    else:
        socket.connect(path)
        actual_address = path

    return (socket, actual_address) if return_address else socket


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
