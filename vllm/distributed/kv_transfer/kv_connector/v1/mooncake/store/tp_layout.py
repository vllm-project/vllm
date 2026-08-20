# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU layout conversion for heterogeneous-TP Mooncake store objects."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton

_DEFAULT_STAGING_BYTES = 64 * 1024 * 1024
_COPY_BLOCK_BYTES = 1024


@dataclass(frozen=True)
class TPSharedTensorLayout:
    """Byte geometry of one packed KV cache tensor."""

    base_addr: int
    block_stride: int
    head_stride: int
    token_stride: int
    content_bytes: int
    is_nhd: bool


@triton.jit
def _copy_tp_shared_layout_kernel(
    staging_ptr,
    layer_base_addrs_ptr,
    layer_block_strides_ptr,
    layer_head_strides_ptr,
    layer_token_strides_ptr,
    layer_content_bytes_ptr,
    layer_offsets_ptr,
    layer_sizes_ptr,
    block_ids_ptr,
    head_starts_ptr,
    staging_slots_ptr,
    BLOCK_TOKENS: tl.constexpr,
    OBJECT_BYTES: tl.constexpr,
    COPY_TO_STAGING: tl.constexpr,
    COPY_BLOCK_BYTES: tl.constexpr,
):
    object_idx = tl.program_id(0)
    layer_idx = tl.program_id(1)
    chunk_idx = tl.program_id(2)

    byte_offsets = chunk_idx.to(tl.int64) * COPY_BLOCK_BYTES + tl.arange(
        0, COPY_BLOCK_BYTES
    ).to(tl.int64)
    layer_size = tl.load(layer_sizes_ptr + layer_idx)
    mask = byte_offsets < layer_size

    content_bytes = tl.load(layer_content_bytes_ptr + layer_idx).to(tl.int64)
    head_bytes = BLOCK_TOKENS * content_bytes
    head_idx = byte_offsets // head_bytes
    head_offset = byte_offsets - head_idx * head_bytes
    token_idx = head_offset // content_bytes
    content_offset = head_offset - token_idx * content_bytes

    base_addr = tl.load(layer_base_addrs_ptr + layer_idx)
    block_stride = tl.load(layer_block_strides_ptr + layer_idx).to(tl.int64)
    head_stride = tl.load(layer_head_strides_ptr + layer_idx).to(tl.int64)
    token_stride = tl.load(layer_token_strides_ptr + layer_idx).to(tl.int64)
    block_id = tl.load(block_ids_ptr + object_idx).to(tl.int64)
    head_start = tl.load(head_starts_ptr + object_idx).to(tl.int64)
    local_addr = (
        base_addr
        + block_id * block_stride
        + (head_start + head_idx) * head_stride
        + token_idx * token_stride
        + content_offset
    )
    local_ptr = tl.cast(local_addr, tl.pointer_type(tl.uint8))

    staging_slot = tl.load(staging_slots_ptr + object_idx).to(tl.int64)
    layer_offset = tl.load(layer_offsets_ptr + layer_idx).to(tl.int64)
    staging_offsets = staging_slot * OBJECT_BYTES + layer_offset + byte_offsets
    if COPY_TO_STAGING:
        values = tl.load(local_ptr, mask=mask, other=0)
        tl.store(staging_ptr + staging_offsets, values, mask=mask)
    else:
        values = tl.load(staging_ptr + staging_offsets, mask=mask, other=0)
        tl.store(local_ptr, values, mask=mask)


class TPSharedStagingBuffer:
    """Reusable contiguous arena for canonical HND store objects."""

    def __init__(
        self,
        layouts: Sequence[TPSharedTensorLayout],
        block_size: int,
        heads_per_store_shard: int,
        first_store_shard: int,
        device: torch.device,
        target_bytes: int = _DEFAULT_STAGING_BYTES,
    ) -> None:
        if not layouts:
            raise ValueError("TP-shared staging requires at least one KV tensor")

        self.block_size = block_size
        self.heads_per_store_shard = heads_per_store_shard
        self.first_store_shard = first_store_shard
        layer_sizes = [
            heads_per_store_shard * block_size * layout.content_bytes
            for layout in layouts
        ]
        layer_offsets: list[int] = []
        offset = 0
        for size in layer_sizes:
            layer_offsets.append(offset)
            offset += size
        self.object_nbytes = offset
        self.capacity_objects = max(1, target_bytes // self.object_nbytes)
        self.nbytes = self.capacity_objects * self.object_nbytes
        self.buffer = torch.empty(self.nbytes, dtype=torch.uint8, device=device)
        self.stream = torch.cuda.Stream(device=device)

        def device_tensor(values: Sequence[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int64, device=device)

        self._layer_base_addrs = device_tensor([layout.base_addr for layout in layouts])
        self._layer_block_strides = device_tensor(
            [layout.block_stride for layout in layouts]
        )
        self._layer_head_strides = device_tensor(
            [layout.head_stride for layout in layouts]
        )
        self._layer_token_strides = device_tensor(
            [layout.token_stride for layout in layouts]
        )
        self._layer_content_bytes = device_tensor(
            [layout.content_bytes for layout in layouts]
        )
        self._layer_offsets = device_tensor(layer_offsets)
        self._layer_sizes = device_tensor(layer_sizes)
        self._block_ids = torch.empty(
            self.capacity_objects, dtype=torch.int64, device=device
        )
        self._head_starts = torch.empty_like(self._block_ids)
        self._staging_slots = torch.empty_like(self._block_ids)
        self._block_ids_host = torch.empty(
            self.capacity_objects, dtype=torch.int64, pin_memory=True
        )
        self._head_starts_host = torch.empty(
            self.capacity_objects, dtype=torch.int64, pin_memory=True
        )
        self._staging_slots_host = torch.empty(
            self.capacity_objects, dtype=torch.int64, pin_memory=True
        )
        self._num_layers = len(layouts)
        self._max_layer_nbytes = max(layer_sizes)

    @property
    def data_ptr(self) -> int:
        return self.buffer.data_ptr()

    def transfer_descriptors(
        self, count: int
    ) -> tuple[list[list[int]], list[list[int]]]:
        if count > self.capacity_objects:
            raise ValueError(
                f"staging batch has {count} objects, capacity is "
                f"{self.capacity_objects}"
            )
        addrs = [[self.data_ptr + index * self.object_nbytes] for index in range(count)]
        sizes = [[self.object_nbytes] for _ in range(count)]
        return addrs, sizes

    def pack(self, block_ids: Sequence[int], store_shard_ids: Sequence[int]) -> None:
        self._copy(block_ids, store_shard_ids, range(len(block_ids)), True)

    def unpack(
        self,
        block_ids: Sequence[int],
        store_shard_ids: Sequence[int],
        staging_slots: Sequence[int],
    ) -> None:
        self._copy(block_ids, store_shard_ids, staging_slots, False)

    def _copy(
        self,
        block_ids: Sequence[int],
        store_shard_ids: Sequence[int],
        staging_slots: Sequence[int],
        copy_to_staging: bool,
    ) -> None:
        count = len(block_ids)
        if not (
            count == len(store_shard_ids) == len(staging_slots)
            and count <= self.capacity_objects
        ):
            raise ValueError("TP-shared staging metadata has inconsistent lengths")
        if count == 0:
            return

        head_starts = [
            (shard_id - self.first_store_shard) * self.heads_per_store_shard
            for shard_id in store_shard_ids
        ]
        for index, (block_id, head_start, staging_slot) in enumerate(
            zip(block_ids, head_starts, staging_slots, strict=True)
        ):
            self._block_ids_host[index] = block_id
            self._head_starts_host[index] = head_start
            self._staging_slots_host[index] = staging_slot
        with torch.cuda.stream(self.stream):
            self._block_ids[:count].copy_(
                self._block_ids_host[:count], non_blocking=True
            )
            self._head_starts[:count].copy_(
                self._head_starts_host[:count], non_blocking=True
            )
            self._staging_slots[:count].copy_(
                self._staging_slots_host[:count], non_blocking=True
            )
            grid = (
                count,
                self._num_layers,
                triton.cdiv(self._max_layer_nbytes, _COPY_BLOCK_BYTES),
            )
            _copy_tp_shared_layout_kernel[grid](
                self.buffer,
                self._layer_base_addrs,
                self._layer_block_strides,
                self._layer_head_strides,
                self._layer_token_strides,
                self._layer_content_bytes,
                self._layer_offsets,
                self._layer_sizes,
                self._block_ids,
                self._head_starts,
                self._staging_slots,
                BLOCK_TOKENS=self.block_size,
                OBJECT_BYTES=self.object_nbytes,
                COPY_TO_STAGING=copy_to_staging,
                COPY_BLOCK_BYTES=_COPY_BLOCK_BYTES,
            )
        self.stream.synchronize()
