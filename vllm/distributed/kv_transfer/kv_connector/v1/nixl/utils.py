# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared constants, lazy imports and helpers for the NIXL connector."""

import contextlib
from collections.abc import Iterator, Sequence
from typing import Any

import regex as re
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
)
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


# Per-region NixlAgentMetadata lists, index-aligned with kv_caches_base_addr.
_REGION_FIELDS = (
    "kv_caches_base_addr",
    "block_lens",
    "block_strides",
    "region_num_blocks",
    "region_group_ids",
    "region_names",
    "region_mem_types",
)


def select_remote_regions(meta: NixlAgentMetadata, indices: Sequence[int]) -> None:
    """Keep only the regions at ``indices`` in every per-region field, in order."""
    region_lists = [getattr(meta, name) for name in _REGION_FIELDS]
    assert all(
        values is None or len(values) == len(meta.kv_caches_base_addr)
        for values in region_lists
    ), "Remote region metadata lengths disagree"
    for name, values in zip(_REGION_FIELDS, region_lists):
        if values is not None:
            setattr(meta, name, [values[i] for i in indices])


def align_remote_regions_by_layer(
    meta: NixlAgentMetadata,
    layer_names: Sequence[str],
    local_region_indices: Sequence[int],
    local_block_lens: Sequence[int],
) -> None:
    """Select the remote regions backing ``layer_names``, in that order.

    ``local_region_indices[i]`` is the local region of ``layer_names[i]``, whose
    block length a remote packed layer's page must match.
    """
    remote_region_layers = meta.region_members
    # Region-index routing here would silently transfer stale KV.
    assert remote_region_layers, "Remote advertised no region_members"
    assert len(meta.kv_caches_base_addr) == len(remote_region_layers), (
        "Remote region metadata lengths disagree"
    )

    remote_region_by_layer: dict[str, int] = {}
    for region_idx, region_layers in enumerate(remote_region_layers):
        for layer_name in region_layers:
            assert layer_name not in remote_region_by_layer, (
                f"Remote advertised layer {layer_name!r} in multiple regions"
            )
            remote_region_by_layer[layer_name] = region_idx

    missing = [name for name in layer_names if name not in remote_region_by_layer]
    assert not missing, f"Remote is missing locally owned layers: {missing}"

    layouts = meta.packed_member_layouts
    packed_regions = {remote_region_by_layer[name] for name in layouts}
    remote_regions = [remote_region_by_layer[name] for name in layer_names]
    select_remote_regions(meta, remote_regions)
    if meta.region_names is not None:
        meta.region_names = list(layer_names)
    # A packed remote row is addressed at each layer's page inside the block;
    # the remote whole-row stride is kept as is.
    transfer_layers = zip(
        layer_names, remote_regions, local_region_indices, strict=True
    )
    for i, (layer_name, remote_region, local_region) in enumerate(transfer_layers):
        if remote_region not in packed_regions:
            continue
        assert layer_name in layouts, (
            f"Remote packed layer {layer_name!r} has no layout"
        )
        offset, page_size = layouts[layer_name]
        assert 0 <= offset < offset + page_size <= meta.block_lens[i], (
            f"Remote packed layer {layer_name!r} escapes its block"
        )
        assert page_size == local_block_lens[local_region], (
            f"Packed MLA page sizes must match for layer {layer_name!r}"
        )
        meta.kv_caches_base_addr[i] += offset
        meta.block_lens[i] = page_size
    meta.packed_member_layouts = {}
    # One layer per region keeps a second alignment pass a no-op.
    meta.region_members = [[name] for name in layer_names]
