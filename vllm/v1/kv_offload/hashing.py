# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Iterator
from typing import overload

from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_block_hash_extra_keys,
    hash_block_tokens,
)
from vllm.v1.request import Request


class RequestBlockHashList:
    """Compute request block hashes directly at an arbitrary block size.

    Unlike ``Request.block_hashes``, this does not depend on the scheduler's
    single ``hash_block_size``. It is used for offload paths that need a
    different granularity than EngineCore prefix-caching.
    """

    def __init__(
        self,
        request: Request,
        block_size: int,
        hash_function: Callable[[object], bytes],
    ):
        self.request = request
        self.block_size = block_size
        self.hash_function = hash_function
        self._hashes: list[BlockHash] = []
        self._next_token_idx = 0
        self._curr_mm_idx = 0
        self._prev_block_hash: BlockHash | None = None

    def __len__(self) -> int:
        return self.request.num_tokens // self.block_size

    @overload
    def __getitem__(self, idx: int) -> BlockHash: ...

    @overload
    def __getitem__(self, idx: slice) -> list[BlockHash]: ...

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._get_value_at(idx)

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self._get_value_at(i) for i in range(start, stop, step)]

        raise TypeError(f"Invalid index type: {type(idx)!r}")

    def __iter__(self) -> Iterator[BlockHash]:
        for i in range(len(self)):
            yield self._get_value_at(i)

    def _get_value_at(self, idx: int) -> BlockHash:
        self._ensure_computed_through(idx)
        return self._hashes[idx]

    def _ensure_computed_through(self, idx: int) -> None:
        while len(self._hashes) <= idx:
            end_token_idx = self._next_token_idx + self.block_size
            if end_token_idx > self.request.num_tokens:
                raise IndexError(idx)

            extra_keys, self._curr_mm_idx = generate_block_hash_extra_keys(
                self.request,
                self._next_token_idx,
                end_token_idx,
                self._curr_mm_idx,
            )
            block_tokens = self.request.all_token_ids[self._next_token_idx:end_token_idx]
            block_hash = hash_block_tokens(
                self.hash_function,
                self._prev_block_hash,
                block_tokens,
                extra_keys,
            )
            self._hashes.append(block_hash)
            self._next_token_idx = end_token_idx
            self._prev_block_hash = block_hash


class HybridChunkBlockHashList:
    """Compose a logical offload hash from per-group block hashes.

    Each logical chunk boundary advances by ``logical_chunk_size`` tokens.
    For each group, we use the most recent full group-sized block hash that
    fits under that chunk boundary, then hash the tuple of group hashes into a
    single offload key.
    """

    def __init__(
        self,
        request: Request,
        group_block_sizes: tuple[int, ...],
        logical_chunk_size: int,
        hash_function: Callable[[object], bytes],
    ):
        self.request = request
        self.group_block_sizes = group_block_sizes
        self.logical_chunk_size = logical_chunk_size
        self.hash_function = hash_function
        self.first_hashable_chunk_idx = max(
            (block_size + logical_chunk_size - 1) // logical_chunk_size
            for block_size in group_block_sizes
        ) - 1
        self.group_hashes = tuple(
            RequestBlockHashList(request, block_size, hash_function)
            for block_size in group_block_sizes
        )

    def __len__(self) -> int:
        return max(
            0,
            self.request.num_tokens // self.logical_chunk_size
            - self.first_hashable_chunk_idx,
        )

    @overload
    def __getitem__(self, idx: int) -> BlockHash: ...

    @overload
    def __getitem__(self, idx: slice) -> list[BlockHash]: ...

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._get_value_at(idx)

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self._get_value_at(i) for i in range(start, stop, step)]

        raise TypeError(f"Invalid index type: {type(idx)!r}")

    def __iter__(self) -> Iterator[BlockHash]:
        for i in range(len(self)):
            yield self._get_value_at(i)

    def _get_value_at(self, idx: int) -> BlockHash:
        chunk_end = (
            idx + 1 + self.first_hashable_chunk_idx
        ) * self.logical_chunk_size
        component_hashes: list[BlockHash] = []
        for block_size, group_hashes in zip(self.group_block_sizes, self.group_hashes):
            num_full_blocks = chunk_end // block_size
            component_hashes.append(group_hashes[num_full_blocks - 1])
        return BlockHash(self.hash_function(tuple(component_hashes)))
