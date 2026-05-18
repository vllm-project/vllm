# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared constants, lazy imports and helpers for the NIXL connector."""

import contextlib
import hashlib
from collections.abc import Iterator
from typing import Any

import regex as re
import zmq

from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.kv_cache_interface import KVCacheSpec, UniformTypeKVCacheSpecs

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


def get_representative_spec_type(spec: KVCacheSpec) -> type[KVCacheSpec]:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        # All inner specs are the same type; pick any.
        inner = next(iter(spec.kv_cache_specs.values()))
        return type(inner)
    return type(spec)


# Compiled regex to extract a standard UUID from vLLM request IDs.
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)


def get_base_request_id(request_id: str) -> str:
    """Extract the core UUID from a vLLM request ID.
    If the ID is already a bare UUID, it is returned as-is.
    """
    m = _UUID_RE.search(request_id)
    return m.group(0) if m else request_id


def push_trigger_addr(engine_id: str, tp_rank: int = 0) -> str:
    """Return a tcp:// address unique to this engine+rank for push triggers.

    The port is derived from a deterministic
    hash of the engine_id (not Python's randomized hash()) to ensure
    scheduler and worker processes agree on the same port.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        PUSH_TRIGGER_BASE_PORT,
    )

    h = int(hashlib.md5(engine_id.encode()).hexdigest(), 16)
    port = PUSH_TRIGGER_BASE_PORT + (h % 1000) + tp_rank

    # TODO: add remote node ip and port details for xPyD deployments
    return f"tcp://127.0.0.1:{port}"
