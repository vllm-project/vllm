# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Socket binding helpers shared by the API server launchers."""

from __future__ import annotations

import argparse
import contextlib
import errno
import os
import socket

from vllm.logger import init_logger
from vllm.utils.network_utils import find_process_using_port, is_valid_ipv6_address

from ..launcher import create_server_socket, create_server_unix_socket

logger = init_logger(__name__)


def _describe_port_holder(port: int) -> str:
    """Best-effort description of the process currently bound to ``port``."""
    process = find_process_using_port(port)
    if process is None:
        return "unknown process"

    try:
        name = process.name()
        cmdline = " ".join(process.cmdline())
    except Exception:
        name = cmdline = "<unavailable>"

    return f"pid={process.pid} name={name!r} cmd={cmdline!r}"


def cleanup_listen_socket(sock: socket.socket, uds_path: str | None = None) -> None:
    """Release the listen socket bound by ``setup_listen_address``."""
    with contextlib.suppress(OSError):
        sock.close()
    if uds_path:
        with contextlib.suppress(FileNotFoundError, OSError):
            os.unlink(uds_path)


def setup_listen_address(
    args: argparse.Namespace, *, reuse_port: bool
) -> tuple[str, socket.socket]:
    """Bind the HTTP listen socket and return ``(listen_address, sock)``."""

    # Bind before the engine starts to avoid race conditions with ray.
    # See https://github.com/vllm-project/vllm/issues/8204
    if args.uds:
        sock = create_server_unix_socket(args.uds)
        listen_address = f"unix:{args.uds}"
    else:
        host = args.host or ""
        port = args.port
        sock_addr = (host, port)

        try:
            sock = create_server_socket(sock_addr, reuse_port=reuse_port)
        except OSError as e:
            # Surface an actionable message for the two common failure modes.
            if e.errno == errno.EADDRINUSE:
                holder = _describe_port_holder(port)
                raise OSError(
                    f"Port {port} is already in use (holder: {holder}). "
                    f"Free the port or pass a different --port. "
                    f"If you intend to share the port, enable SO_REUSEPORT "
                    f"in the launcher."
                ) from e
            if e.errno == errno.EACCES:
                raise OSError(
                    f"Permission denied binding to "
                    f"{host or '0.0.0.0'}:{port}. "
                    f"Ports below 1024 typically require elevated privileges."
                ) from e
            raise

        addr, port = sock_addr
        is_ssl = bool(args.ssl_keyfile and args.ssl_certfile)
        host_part = f"[{addr}]" if is_valid_ipv6_address(addr) else (addr or "0.0.0.0")
        listen_address = f"http{'s' if is_ssl else ''}://{host_part}:{port}"

    return listen_address, sock
