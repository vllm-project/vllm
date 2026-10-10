# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
import socket


def cleanup_listen_socket(sock: socket.socket, uds_path: str | None = None) -> None:
    """Release the listen socket bound by ``setup_listen_address``."""
    with contextlib.suppress(OSError):
        sock.close()
    if uds_path:
        with contextlib.suppress(FileNotFoundError, OSError):
            os.unlink(uds_path)
