# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared constants, lazy imports and helpers for the NIXL connector."""

import contextlib
from collections.abc import Iterator
from typing import Any

import regex as re
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import TransferId
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.kv_cache_interface import KVCacheSpec, UniformTypeKVCacheSpecs

logger = init_logger(__name__)

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
    """Context manager for a ZMQ socket."""
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


# Trailing 8-hex randomization suffix appended by
# ``input_processor.assign_request_id`` as ``-{random_uuid():.8}``.
_RANDOM_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$", re.IGNORECASE)


def get_base_request_id(request_id: str) -> str:
    """Strip the per-request ``-<8 hex>`` randomization suffix, if present."""
    return _RANDOM_SUFFIX_RE.sub("", request_id)


def get_transfer_id(kv_transfer_params: dict[str, Any]) -> TransferId | None:
    """Id the router sets on the prefill and decode request of one dispatch."""
    transfer_id = kv_transfer_params.get("transfer_id")
    if isinstance(transfer_id, str) and transfer_id:
        return transfer_id
    logger.warning_once(
        "No valid transfer_id in kv_transfer_params; pairing P and D by base "
        "request id, which can pair the wrong requests if ids repeat. This "
        "fallback is deprecated and will be removed in a future release. "
        "Please use a router that sets transfer_id for NIXL push mode."
    )
    return None
