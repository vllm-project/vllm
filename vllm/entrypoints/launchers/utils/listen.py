# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import contextlib
import errno
import os
import socket

from vllm.entrypoints.launchers.launcher import (
    create_server_socket,
    create_server_unix_socket,
)
from vllm.utils.network_utils import find_process_using_port, is_valid_ipv6_address


def setup_listen_address(
    args: argparse.Namespace, *, reuse_port: bool
) -> tuple[str, socket.socket]:
    """Bind the serve listen socket and return ``(listen_address, sock)``."""
    # Bind before the engine starts to avoid race conditions with ray.
    # See https://github.com/vllm-project/vllm/issues/8204
    if args.uds:
        sock = create_server_unix_socket(args.uds)
        return f"unix:{args.uds}", sock

    host = args.host or ""

    try:
        sock = create_server_socket((host, args.port), reuse_port=reuse_port)
    except OSError as e:
        # Surface an actionable message for the two common failure modes.
        if e.errno == errno.EADDRINUSE:
            holder = _describe_port_holder(args.port)
            raise OSError(
                f"Port {args.port} is already in use (holder: {holder}). "
                f"Free the port or pass a different --port. "
                f"If you intend to share the port, enable SO_REUSEPORT "
                f"in the launcher."
            ) from e
        if e.errno == errno.EACCES:
            raise OSError(
                f"Permission denied binding to "
                f"{host or '0.0.0.0'}:{args.port}. "
                f"Ports below 1024 typically require elevated privileges."
            ) from e
        raise

    # Only the port needs to come from the kernel (so ``--port 0`` reports the
    # real port). Keep the user-supplied host in the address so ``--host
    # localhost`` still prints ``localhost`` instead of ``127.0.0.1``.
    bound_addr, bound_port = sock.getsockname()[:2]
    display_addr = host or bound_addr or "0.0.0.0"
    is_ssl = bool(
        getattr(args, "ssl_keyfile", None) and getattr(args, "ssl_certfile", None)
    )
    host_part = (
        f"[{display_addr}]" if is_valid_ipv6_address(display_addr) else display_addr
    )
    listen_address = f"http{'s' if is_ssl else ''}://{host_part}:{bound_port}"

    return listen_address, sock


def cleanup_listen_socket(sock: socket.socket, uds_path: str | None = None) -> None:
    """Release the serve listen socket bound by ``setup_listen_address``.

    For AF_UNIX sockets the path is inferred from the socket itself when
    ``uds_path`` is not provided, so callers don't have to remember to pass
    ``args.uds``. Idempotent: safe to call multiple times (e.g. the shutdown
    path may already have closed the socket).
    """
    if uds_path is None and sock.family == socket.AF_UNIX:
        try:
            name = sock.getsockname()
            if isinstance(name, str) and name:
                uds_path = name
        except OSError:
            pass

    with contextlib.suppress(OSError):
        sock.close()
    if uds_path:
        with contextlib.suppress(FileNotFoundError, OSError):
            os.unlink(uds_path)


def _describe_port_holder(port: int) -> str:
    """Best-effort description of the process currently bound to ``port``."""
    try:
        process = find_process_using_port(port)
    except Exception:
        return "unknown process (query failed)"
    if process is None:
        return "unknown process"

    try:
        name = process.name()
        cmdline = " ".join(process.cmdline())
    except Exception:
        name = cmdline = "<unavailable>"

    return f"pid={process.pid} name={name!r} cmd={cmdline!r}"
