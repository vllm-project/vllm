# SPDX-License-Identifier: Apache-2.0
"""CXLMemSim/LegoMem backend for vLLM KV cache blocks."""

from __future__ import annotations

import ctypes
import os
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.simple_kv_offload.cpu_copy_backend import ImmediateEvent

logger = init_logger(__name__)


class LegoMemBackend:
    """Synchronously stage vLLM CPU KV blocks through CXLMemSim.

    Each model-parallel rank owns the usable data area of the next distributed
    CXLMemSim node in a ring. This guarantees that every KV store/load crosses
    Soft-RoCE instead of being satisfied from the frontend node's local DRAM.
    The nominal node stride is retained because the final cache line of every
    node contains CXLMemSim's backing-store header.
    """

    def __init__(self) -> None:
        self._active: dict[str, torch.Tensor] = {}
        self._buffer: torch.Tensor | None = None
        self._block_bytes = 0
        self._segment_base = 0
        self._segment_bytes = 0
        self._handle: int | None = None
        self._lib: ctypes.CDLL | None = None
        self.bytes_read = 0
        self.bytes_written = 0

    def init(
        self,
        active_caches: dict[str, torch.Tensor],
        library_path: str,
        host: str,
        port: int,
        rank: int,
        num_nodes: int,
        node_capacity_bytes: int,
        usable_bytes_per_rank: int,
    ) -> None:
        if not active_caches:
            raise ValueError("LegoMemBackend requires at least one KV tensor")
        if any(t.device.type != "cpu" for t in active_caches.values()):
            raise ValueError("LegoMemBackend currently requires CPU KV tensors")
        if not 0 <= rank < num_nodes:
            raise ValueError(f"rank {rank} is outside the {num_nodes}-node pool")
        if usable_bytes_per_rank > node_capacity_bytes - 64:
            raise ValueError("usable rank capacity overlaps the CXLMemSim header line")

        self._active = active_caches
        self._block_bytes = sum(t[0].numel() * t.element_size() for t in active_caches.values())
        self._buffer = torch.empty(self._block_bytes, dtype=torch.uint8, device="cpu")
        target_node = (rank + 1) % num_nodes
        self._segment_base = target_node * node_capacity_bytes
        self._segment_bytes = usable_bytes_per_rank

        self._lib = ctypes.CDLL(os.path.expanduser(library_path))
        self._lib.legomem_client_open.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._lib.legomem_client_open.restype = ctypes.c_void_p
        self._lib.legomem_client_read.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self._lib.legomem_client_read.restype = ctypes.c_int
        self._lib.legomem_client_write.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self._lib.legomem_client_write.restype = ctypes.c_int
        self._lib.legomem_client_close.argtypes = [ctypes.c_void_p]
        self._lib.legomem_client_close.restype = None

        handle = self._lib.legomem_client_open(host.encode(), port)
        if not handle:
            raise ConnectionError(f"cannot connect LegoMem client to {host}:{port}")
        self._handle = int(handle)
        logger.info(
            "LegoMem KV backend connected: rank=%d/%d target_node=%d segment=[%d,%d) "
            "block_bytes=%d slots=%d",
            rank,
            num_nodes,
            target_node,
            self._segment_base,
            self._segment_base + self._segment_bytes,
            self._block_bytes,
            self._segment_bytes // self._block_bytes,
        )

    def _address(self, slot: int) -> int:
        address = self._segment_base + slot * self._block_bytes
        if slot < 0 or address + self._block_bytes > self._segment_base + self._segment_bytes:
            raise IndexError(f"LegoMem KV slot {slot} exceeds rank segment")
        return address

    def _pack(self, block: int) -> None:
        assert self._buffer is not None
        offset = 0
        for tensor in self._active.values():
            row = tensor[block].view(torch.uint8).reshape(-1)
            size = row.numel()
            self._buffer[offset : offset + size].copy_(row)
            offset += size

    def _unpack(self, block: int) -> None:
        assert self._buffer is not None
        offset = 0
        for tensor in self._active.values():
            row = tensor[block].view(torch.uint8).reshape(-1)
            size = row.numel()
            row.copy_(self._buffer[offset : offset + size])
            offset += size

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, Any]],
        wait_event: Any | None = None,
    ) -> None:
        del wait_event
        assert self._lib is not None and self._handle is not None and self._buffer is not None
        pointer = ctypes.c_void_p(self._buffer.data_ptr())
        handle = ctypes.c_void_p(self._handle)
        for src_block, dst_block in zip(src_blocks, dst_blocks, strict=True):
            if is_store:
                self._pack(src_block)
                status = self._lib.legomem_client_write(
                    handle, self._address(dst_block), pointer, self._block_bytes
                )
                self.bytes_written += self._block_bytes
            else:
                status = self._lib.legomem_client_read(
                    handle, self._address(src_block), pointer, self._block_bytes
                )
                self._unpack(dst_block)
                self.bytes_read += self._block_bytes
            if status != 0:
                operation = "write" if is_store else "read"
                raise OSError(f"LegoMem KV {operation} failed with status {status}")
        events_list.append((event_idx, ImmediateEvent()))

    def shutdown(self) -> None:
        if self._lib is not None and self._handle is not None:
            logger.info(
                "LegoMem KV backend closing: bytes_written=%d bytes_read=%d",
                self.bytes_written,
                self.bytes_read,
            )
            self._lib.legomem_client_close(ctypes.c_void_p(self._handle))
            self._handle = None
