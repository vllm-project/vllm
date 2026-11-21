# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preflight check for the data parallel RPC port.

The rank 0 frontend binds `tcp://<master ip>:<rpc port>` for the data parallel
startup handshake. When a previous engine did not exit cleanly, that stale
process keeps the port (and the accelerator devices) alive, so the next engine
started in the same container fails with a bare:

    zmq.error.ZMQError: Address already in use (addr='tcp://<node-ip>:29550>')

The bind happens in the frontend process, so the error carries no hint about
the stale process that has to be stopped. This module detects the conflict
before the engine starts and reports it with the information needed to recover
from it.
"""

import errno
import socket
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def _resolve_ipv4(host: str) -> set[str]:
    """Return the IPv4 addresses that `host` resolves to."""
    try:
        infos = socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError):
        return set()
    return {str(info[4][0]) for info in infos}


def _address_matches(listen_ip: str, host: str) -> bool:
    """Whether a socket bound to `listen_ip` also owns `host`."""
    if listen_ip in ("0.0.0.0", ""):
        # A wildcard listener owns every local address of the port.
        return True
    return listen_ip in _resolve_ipv4(host)


def _endpoint_family(host: str) -> int:
    """Return the address family the engine binds the endpoint with."""
    address = host.strip("[]")
    if not address:
        # An empty host stands for the wildcard address of the IPv4 family.
        return socket.AF_INET
    if ":" in address:
        return socket.AF_INET6
    try:
        infos = socket.getaddrinfo(address, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError):
        return socket.AF_INET
    for info in infos:
        if info[0] in (socket.AF_INET, socket.AF_INET6):
            return info[0]
    return socket.AF_INET


def _is_endpoint_free(host: str, port: int) -> bool:
    """Return whether the endpoint is free, that is, not held by a socket.

    The socket is created with the same options as the handshake socket libzmq
    binds, so the answer is the one the engine gets when it binds the endpoint.
    An error other than "address already in use" means that the endpoint cannot
    be bound at all, which is not a conflict with another process and therefore
    reported as free.
    """
    family = _endpoint_family(host)
    address = host.strip("[]")
    try:
        sock = socket.socket(family, socket.SOCK_STREAM)
    except OSError:
        # The address family is not available on this host.
        return True
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            # Windows: libzmq asks for an exclusive address, so a port that is
            # in use cannot be taken over.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            # libzmq allows the reuse of the address before it binds.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if family == socket.AF_INET6 and hasattr(socket, "IPV6_V6ONLY"):
            # libzmq enables the IPv4 mapping on its IPv6 sockets.
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        sock.bind((address, port))
    except OSError as exc:
        # Only "address already in use" means that the endpoint is taken, any
        # other error means that it cannot be bound at all.
        return exc.errno != errno.EADDRINUSE
    finally:
        sock.close()
    return True


def _process_name(psutil: Any, pid: int) -> str:
    """Return the name of the process, or `unknown` when it is gone."""
    try:
        return psutil.Process(pid).name()
    except psutil.Error:
        return "unknown"


def find_tcp_listener(host: str, port: int) -> tuple[int, str] | None:
    """Return the `(pid, process name)` that holds `host:port`.

    `None` is returned when the endpoint is free, when the owner cannot be
    identified (for example when the socket belongs to a process owned by
    another user) or when the listeners of the port do not really conflict with
    the endpoint.
    """
    try:
        import psutil
    except ImportError:
        return None

    try:
        connections = psutil.net_connections(kind="tcp")
    except (psutil.Error, OSError):
        return None

    for conn in connections:
        if conn.status != psutil.CONN_LISTEN or conn.pid is None:
            continue
        laddr = conn.laddr
        if laddr is None or laddr.port != port:
            continue
        if not _address_matches(str(laddr.ip), host):
            continue
        return conn.pid, _process_name(psutil, conn.pid)
    return None


def check_dp_rpc_port_available(host: str, port: int) -> None:
    """Raise a `RuntimeError` when the data parallel RPC port is taken.

    The port is only checked where it is really bound, so this must only be
    called on the rank 0 frontend of a deployment whose data parallel engines
    are not all local.
    """
    if _is_endpoint_free(host, port):
        return

    holder = find_tcp_listener(host, port)
    if holder is None:
        raise RuntimeError(
            f"The data parallel RPC port {host}:{port} is already in use by "
            "another process, but the process holding it could not be "
            "identified. Stop the stale engine process that holds the port, or "
            "restart the container, and try again."
        )

    pid, name = holder
    raise RuntimeError(
        f"The data parallel RPC port {host}:{port} is already in use by PID "
        f"{pid} ({name}). This is usually a stale engine process that did not "
        f"exit cleanly. Stop it (for example with `kill -9 {pid}`), or restart "
        "the container on that node, and try again."
    )


def binds_dp_rpc_port(
    handshake_local_only: bool,
    local_engines_only: bool,
    dp_rank: int,
) -> bool:
    """Whether this process binds the TCP data parallel handshake address.

    The rank 0 frontend binds it, but only when the data parallel engines are
    not all local: otherwise the local IPC transport is used instead of the
    TCP handshake address. The other ranks connect to the master instead of
    binding the address.
    """
    return not handshake_local_only and not (local_engines_only and dp_rank > 0)
