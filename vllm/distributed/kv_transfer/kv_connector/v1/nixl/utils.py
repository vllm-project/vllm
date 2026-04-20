# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared constants, lazy imports and helpers for the NIXL connector."""

import contextlib
import os
import sys
from collections.abc import Iterator
from typing import Any

import zmq

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_socket

logger = init_logger(__name__)


# Lazy import nixl_wrapper to avoid loading nixl_bindings if nixl is not used
try:
    if "UCX_RCACHE_MAX_UNRELEASED" not in os.environ:
        # avoid a memory leak in UCX when using NIXL on some models
        # see: https://github.com/vllm-project/vllm/issues/24264
        if "nixl" in sys.modules or "rixl" in sys.modules:
            logger.warning(
                "NIXL was already imported, we can't reset UCX_RCACHE_MAX_UNRELEASED. "
                "Please set it to '1024' manually."
            )
        else:
            logger.info(
                "Setting UCX_RCACHE_MAX_UNRELEASED to '1024' to avoid a rare "
                "memory leak in UCX when using NIXL."
            )
            os.environ["UCX_RCACHE_MAX_UNRELEASED"] = "1024"

    if not current_platform.is_rocm():
        from nixl._api import nixl_agent as NixlWrapper
        from nixl._bindings import nixlXferTelemetry
    else:
        from rixl._api import nixl_agent as NixlWrapper
        from rixl._bindings import nixlXferTelemetry

    logger.info("NIXL is available")
except ImportError:
    logger.warning("NIXL is not available")
    NixlWrapper = None
    nixlXferTelemetry = None


try:
    if not current_platform.is_rocm():
        from nixl._api import nixl_agent_config
    else:
        from rixl._api import nixl_agent_config
except ImportError:
    nixl_agent_config = None
    logger.warning("NIXL agent config is not available")

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
