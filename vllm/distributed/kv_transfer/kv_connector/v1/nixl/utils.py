# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared constants, lazy imports and helpers for the NIXL connector."""

import contextlib
from collections.abc import Iterator
from typing import Any

import zmq

from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_socket

# Supported platforms and types of kv transfer buffer.
# {device: tuple of supported kv buffer types}
_NIXL_SUPPORTED_DEVICE = {
    "cuda": (
        "cuda",
        "cpu",
    ),
    "tpu": ("cpu",),
    "xpu": (
        "cpu",
        "xpu",
    ),
    "cpu": ("cpu",),
}
# support for oot platform by providing mapping in current_platform
_NIXL_SUPPORTED_DEVICE.update(current_platform.get_nixl_supported_devices())


# TODO: merge with vllm.utils.network_utils.zmq_socket_ctx
@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ):
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        yield make_zmq_socket(
            ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER
        )
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)
