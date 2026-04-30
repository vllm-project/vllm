# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ExOffloadingStorageKVCacheConfig:
    kv_caches: dict[str, torch.Tensor]
    split_k_and_v: bool
    is_block_first: bool


class ExOffloadingStorage:
    def __init__(self, config):
        self.config = config

    @classmethod
    def parse_uri(cls, uri: str) -> tuple[dict[str, str], str]:
        raise NotImplementedError

    @abstractmethod
    def register_kvcache(self, config: ExOffloadingStorageKVCacheConfig) -> None: ...

    @abstractmethod
    async def load(self, filepath: str, offset: int, block_ids: list[int]) -> None: ...

    @abstractmethod
    async def save(self, filepath: str, offset: int, block_ids: list[int]) -> None: ...


def build_mem_regions(
    kvcache_config: ExOffloadingStorageKVCacheConfig,
) -> tuple[list[tuple[int, int]], list[int]]:
    assert not kvcache_config.is_block_first, (
        "Explicit offloading does not support block first layout"
    )

    split_k_and_v = kvcache_config.split_k_and_v

    seen_base_addrs = []
    mem_regions_per_layer = []
    block_bytes_per_layer = []
    num_blocks = 0

    tensor_size_bytes = None
    for _, cache_or_caches in kvcache_config.kv_caches.items():
        cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
        for cache in cache_list:
            cache_addr = cache.data_ptr()
            if cache_addr in seen_base_addrs:
                continue

            seen_base_addrs.append(cache_addr)

            cache_bytes = cache.nbytes
            if tensor_size_bytes is None:
                tensor_size_bytes = cache_bytes
                num_blocks = cache.shape[0]
            assert cache.shape[0] == num_blocks, (
                "All kv cache tensors must have the same number of blocks"
            )
            assert cache_bytes % num_blocks == 0, (
                "Explicit offloading expects each kv cache tensor size to be "
                "divisible by the number of blocks."
            )

            block_bytes_per_layer.append(cache_bytes // num_blocks)
            mem_regions_per_layer.append((cache_addr, cache_bytes))

    if split_k_and_v:
        mem_regions_per_layer = mem_regions_per_layer[::2] + mem_regions_per_layer[1::2]
        block_bytes_per_layer = block_bytes_per_layer[::2] + block_bytes_per_layer[1::2]

    return mem_regions_per_layer, block_bytes_per_layer


def group_block_contiguous(block_ids: list[int]) -> list[list[int]]:
    if len(block_ids) == 0:
        return []

    brk = np.where(np.diff(block_ids) != 1)[0] + 1
    groups = np.split(block_ids, brk)
    return [g.tolist() for g in groups]


def get_mem_regions(
    mem_regions_per_layer: list[tuple[int, int]],
    block_bytes_per_layer: list[int],
    block_ids: list[int],
) -> list[tuple[int, int]]:
    group_block_ids = group_block_contiguous(block_ids)
    mem_regions = []

    for layer_idx, (addr, bytes) in enumerate(mem_regions_per_layer):
        block_bytes = block_bytes_per_layer[layer_idx]

        for group in group_block_ids:
            group_addr = addr + group[0] * block_bytes
            group_size = len(group) * block_bytes

            if group_addr + group_size > addr + bytes:
                raise ValueError(
                    f"memory region [{group_addr}, {group_addr + group_size}] "
                    "is out of bound"
                )

            mem_regions.append((group_addr, group_size))

    return mem_regions


def get_mem_tensors(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[int],
    split_k_and_v: bool,
) -> list[torch.Tensor]:
    group_block_ids = group_block_contiguous(block_ids)
    k_mem_tensors: list[torch.Tensor] = []
    v_mem_tensors: list[torch.Tensor] = []

    for kv_cache in kv_caches.values():
        for group in group_block_ids:
            if split_k_and_v:
                k_mem_tensors.append(
                    kv_cache[0, group[0] : (group[0] + len(group)), :, :, :]
                )
                v_mem_tensors.append(
                    kv_cache[1, group[0] : (group[0] + len(group)), :, :, :]
                )
            else:
                k_mem_tensors.append(kv_cache[group[0] : (group[0] + len(group)), :, :])

    return k_mem_tensors + v_mem_tensors


def copy_data_h2d(host_data: torch.Tensor, data: list[torch.Tensor]):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        off = 0
        for d in data:
            df = d.flatten()
            len = df.numel()
            df.copy_(host_data[off : off + len], non_blocking=True)
            off += len
    stream.synchronize()


def copy_data_d2h(data: list[torch.Tensor], host_data: torch.Tensor):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        off = 0
        for d in data:
            df = d.flatten()
            len = df.numel()
            host_data[off : off + len].copy_(df, non_blocking=True)
            off += len
    stream.synchronize()


def tensors_total_numel(tensors: list[torch.Tensor]) -> int:
    return sum(tensor.numel() for tensor in tensors)
