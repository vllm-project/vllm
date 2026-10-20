# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
    host: str | None = None,
    port: int | None = None,
    uds: str | None = None,
    *,
    reuse_port: bool,
    ssl_keyfile: str | None = None,
    ssl_certfile: str | None = None,
) -> tuple[str, socket.socket]:
    """Bind the serve listen socket and return ``(listen_address, sock)``."""
    # Bind before the engine starts to avoid race conditions with ray.
    # See https://github.com/vllm-project/vllm/issues/8204
    if uds:
        try:
            sock = create_server_unix_socket(uds)
        except OSError as e:
            # `create_server_unix_socket` does not unlink a pre-existing
            # socket file, so a crashed prior run leaves a stale path that
            # surfaces here as EADDRINUSE. Port-oriented hints ("free the
            # port") would be confusing for UDS, so give path-specific ones.
            if e.errno == errno.EADDRINUSE:
                raise OSError(
                    f"Unix socket {uds!r} already exists. Another "
                    f"process may be listening on it, or a previous run left "
                    f"a stale socket file. Remove the file (rm {uds}) "
                    f"if no process is using it, or pass a different --uds "
                    f"path."
                ) from e
            raise
        return f"unix:{uds}", sock

    host = host or ""

    try:
        sock = create_server_socket((host, port), reuse_port=reuse_port)
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            holder = _describe_port_holder(port)
            if reuse_port:
                hint = (
                    "The holder does not share the port via SO_REUSEPORT "
                    "(possibly a non-vLLM process or a different user). "
                    "Free the port or pass a different --port."
                )
            else:
                hint = (
                    "Free the port, pass a different --port, or share it "
                    "via SO_REUSEPORT by running multiple API servers "
                    "(--api-server-count > 1)."
                )
            raise OSError(
                f"Port {port} is already in use (holder: {holder}). {hint}"
            ) from e
        if e.errno == errno.EACCES:
            raise OSError(
                f"Permission denied binding to "
                f"{host or '0.0.0.0'}:{port}. "
                f"Ports below 1024 typically require elevated privileges."
            ) from e
        if e.errno == errno.ENOPROTOOPT and reuse_port:
            raise OSError(
                "SO_REUSEPORT is not supported on this platform. "
                "Cannot use --api-server-count > 1 here; use a single "
                "API server."
            ) from e
        raise

    # Only the port needs to come from the kernel (so ``--port 0`` reports the
    # real port). Keep the user-supplied host in the address so ``--host
    # localhost`` still prints ``localhost`` instead of ``127.0.0.1``.
    bound_addr, bound_port = sock.getsockname()[:2]
    display_addr = host or bound_addr or "0.0.0.0"
    is_ssl = bool(ssl_keyfile and ssl_certfile)
    host_part = (
        f"[{display_addr}]" if is_valid_ipv6_address(display_addr) else display_addr
    )
    listen_address = f"http{'s' if is_ssl else ''}://{host_part}:{bound_port}"

    return listen_address, sock


def setup_listen_address_from_args(args, *, reuse_port):
    return setup_listen_address(
        host=args.host,
        port=args.port,
        uds=args.uds,
        reuse_port=reuse_port,
        ssl_keyfile=getattr(args, "ssl_keyfile", None),
        ssl_certfile=getattr(args, "ssl_certfile", None),
    )


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
