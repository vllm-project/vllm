# SPDX-License-Identifier: Apache-2.0
"""MUSA ops backend: the torch baseline with one native override.

:class:`MusaDeviceOps` overrides :meth:`multi_layer_block_kv_transfer` to
try the native MUSA path first (when inputs are tensor-backed) and fall
back to the torch baseline otherwise.  Every other op inherits the
baseline via MRO.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# Third Party
import torch

# First Party
from lmcache.v1.platform import torch_ops
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.ops_types import (
    EngineKVFormat,
    PageBufferShapeDesc,
    TransferDirection,
)


def _tensor_list(value: object) -> list[torch.Tensor] | None:
    """Return ``value`` as ``list[torch.Tensor]`` when it is tensor-backed."""
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, torch.Tensor) for item in value):
        return None
    return value


def _musa_multi_layer_block_kv_transfer(
    paged_buffer_ptrs_tensor: torch.Tensor | list,
    lmcache_objects_ptrs: list[int] | list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    device: torch.device | str,
    direction: TransferDirection,
    shape_desc: PageBufferShapeDesc,
    lmcache_chunk_size: int,
    engine_kv_format: EngineKVFormat,
    skip_prefix_n_blocks: int,
) -> None:
    """Native MUSA block transfer when tensor-backed; else torch baseline."""
    # First Party
    from lmcache.v1.platform.musa.native_kv_transfer import (
        try_native_multi_layer_block_kv_transfer,
    )

    object_tensors = _tensor_list(lmcache_objects_ptrs)
    if object_tensors is not None and try_native_multi_layer_block_kv_transfer(
        paged_layers=paged_buffer_ptrs_tensor,
        object_tensors=object_tensors,
        block_ids=block_ids,
        direction=direction,
        shape_desc=shape_desc,
        lmcache_chunk_size=lmcache_chunk_size,
        engine_kv_format=engine_kv_format,
        skip_prefix_n_blocks=skip_prefix_n_blocks,
    ):
        return

    torch_ops.multi_layer_block_kv_transfer(
        paged_buffer_ptrs_tensor,
        lmcache_objects_ptrs,
        block_ids,
        device,
        direction,
        shape_desc,
        lmcache_chunk_size,
        engine_kv_format,
        skip_prefix_n_blocks,
    )


class MusaDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "musa"

    def multi_layer_block_kv_transfer(self, *args, **kwargs) -> None:
        _musa_multi_layer_block_kv_transfer(*args, **kwargs)
