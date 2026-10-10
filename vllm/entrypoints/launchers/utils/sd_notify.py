# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal sd_notify(3) client, so that a ``Type=notify`` systemd unit learns
when the API server is actually ready instead of when the process was spawned.

Only the standard library is used: systemd hands the socket path to the
service in ``NOTIFY_SOCKET`` and expects newline-separated ``KEY=VALUE``
datagrams on it. When the variable is unset (not running under systemd, or
``NotifyAccess`` does not allow it) every call is a no-op.
"""

import os
import socket

from vllm.logger import init_logger

logger = init_logger(__name__)

NOTIFY_SOCKET_ENV = "NOTIFY_SOCKET"


def sd_notify(state: str) -> bool:
    """Send ``state`` (e.g. ``"READY=1"``) to the systemd notify socket.

    Returns ``True`` when a datagram was sent, ``False`` when there is no
    notify socket or it could not be reached. Never raises: a missing or
    broken socket must not take the server down.
    """
    address = os.environ.get(NOTIFY_SOCKET_ENV)
    if not address:
        return False
    # A leading "@" denotes the Linux abstract socket namespace, which the
    # socket API spells with a leading NUL byte.
    if address.startswith("@"):
        address = "\0" + address[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.sendto(state.encode(), address)
    except OSError as e:
        logger.warning("Could not send %r to %s: %s", state, NOTIFY_SOCKET_ENV, e)
        return False
    return True
