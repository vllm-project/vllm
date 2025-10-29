# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import ipaddress
import os
import socket
import sys
import warnings
from collections.abc import (
    Iterable,
    Iterator,
    Sequence,
)
from typing import Any
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


def test_loopback_bind(address, family):
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


def _get_open_ports(ports_to_try: Iterable[int] = (0,), max_count: int = 1) -> set[int]:
    """
    Find a maximum of `max_count` open ports from the `ports_to_try` list.
    `ports_to_try` can contain zeros, meaning any available port
    returned by the os.

    Note: asking for port "0" lets the OS chose a random open port, see
    https://man7.org/linux/man-pages/man7/ip.7.html

    Try to open a port either with IPv4 or IPv6.
    """
    open_ports: set[int] = set()
    for port in ports_to_try:
        for family in (socket.AF_INET, socket.AF_INET6):
            try:
                with socket.socket(family, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    port = s.getsockname()[1]
            except OSError:
                continue

            open_ports.add(port)
            if len(open_ports) >= max_count:
                return open_ports
            break

    return open_ports


def get_open_ports_list(max_count: int = 5) -> list[int]:
    """
    Find a maximum of `max_count` open ports. The range of acceptable port numbers
    if inferred from environment variables: VLLM_DP_MASTER_PORT and VLLM_PORT.

    Otherwise try random valid port numbers.

    If no ports could be opened, raise OSError.
    """
    starting_port = 0
    if "VLLM_DP_MASTER_PORT" in os.environ:
        starting_port = envs.VLLM_DP_MASTER_PORT
    elif "VLLM_PORT" in os.environ:
        starting_port = envs.VLLM_PORT
        # assertion holds because "VLLM_PORT" in os.environ
        assert starting_port is not None

    # give ourselves some room for failure / already taken ports
    num_tries = max(2 * max_count, 10)

    if starting_port != 0:
        ports_to_try = range(starting_port, starting_port + num_tries)
    else:
        # "0" meaning let the os choose a random available port
        ports_to_try = (0 for _ in range(num_tries))

    open_port = _get_open_ports(ports_to_try, max_count)
    if len(open_port) != max_count:
        if starting_port == 0:
            err = f"Could not get {max_count} random ports after {num_tries} tries."
        else:
            err = (
                f"Could not get {max_count} ports in the range "
                f"[{starting_port}, {starting_port + num_tries})"
            )
        raise OSError(err)

    return list(open_port)


def get_open_port() -> int:
    """
    Special case of get_open_port_list with a single port
    """
    return get_open_ports_list(max_count=1)[0]


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
