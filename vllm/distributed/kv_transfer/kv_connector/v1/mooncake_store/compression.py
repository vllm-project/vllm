# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class CompressedStore:
    """Keep Mooncake I/O and request scheduling outside the KVServe runtime."""

    def __init__(self, store: Any, block_size: int):
        self.store = store
        self.block_size = block_size
        self.codec: Any = None
        self._registered: list[int] = []
        self._lock = threading.Lock()

    def __getattr__(self, name: str):
        return getattr(self.store, name)

    def bind(
        self,
        runtime: Any,
        kv_caches: dict[str, torch.Tensor],
        group_indices: dict[str, int],
    ) -> None:
        from kvserve_v1.integrations.mooncake_store import (
            MooncakeStoreCompressionAdapter,
        )

        codec = MooncakeStoreCompressionAdapter(runtime)
        self.codec = codec
        try:
            codec.bind_kv_caches(kv_caches, group_indices)
            if runtime.fa_group_indices != {0}:
                raise ValueError(
                    "Compressed Mooncake Store requires one attention group"
                )
            self._device = next(iter(kv_caches.values())).device
            for ptr, length in codec.registered_memory():
                if self.store.register_buffer(ptr, length) != 0:
                    raise RuntimeError("Mooncake could not register a codec buffer")
                self._registered.append(ptr)
        except Exception:
            self.close()
            raise

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        assert self.codec is not None
        objects: list[str] = []
        for base_key in keys:
            for part in (0, 1):
                key = self.codec.key_for(base_key, part)
                objects.extend((key, f"{key}:manifest"))
        if not objects:
            return []
        result = self.store.batch_is_exist(objects)
        if len(result) != len(objects):
            raise RuntimeError("Mooncake returned an incomplete lookup result")
        matches = []
        for i in range(0, len(result), 4):
            group = result[i : i + 4]
            status = next((value for value in group if value < 0), 0)
            matches.append(1 if all(value == 1 for value in group) else status)
        return matches

    def put(
        self, keys: list[str], blocks: list[int], replicate: Any
    ) -> tuple[list[int], int]:
        return self._run_batch(keys, blocks, replicate)

    def get(self, keys: list[str], blocks: list[int]) -> tuple[list[int], int]:
        return self._run_batch(keys, blocks, None)

    def _run_batch(
        self, keys: list[str], blocks: list[int], replicate: Any
    ) -> tuple[list[int], int]:
        if len(keys) != len(blocks):
            raise ValueError("Mooncake Store requires one block per compressed key")
        results = []
        transferred_bytes = 0
        with self._lock, torch.accelerator.device_index(self._device.index):
            assert self.codec is not None
            for key, block in zip(keys, blocks, strict=True):
                slot = self.codec.acquire_slot()
                status = 0
                try:
                    for part in (0, 1):
                        if replicate is None:
                            transferred_bytes += self._load(slot, key, block, part)
                        else:
                            status, size = self._save(slot, key, block, part, replicate)
                            transferred_bytes += size
                            if status < 0:
                                break
                except Exception:
                    self.codec.runtime.streams[slot].synchronize()
                    logger.exception("KVServe Store failed for %s", key)
                    status = -1
                self.codec.release_slot(slot)
                results.append(status)
        return results, transferred_bytes

    def _save(
        self, slot: int, key: str, block: int, part: int, replicate: Any
    ) -> tuple[int, int]:
        value = self.codec.compress_for_store(
            slot,
            key,
            [[block]],
            part,
            valid_tokens_by_group={0: self.block_size},
        )
        sizes = [[b.nbytes for b in value.buffers], [value.header_buffer.nbytes]]
        result = self.store.batch_put_from_multi_buffers(
            [value.key, value.header_key],
            [[b.address for b in value.buffers], [value.header_buffer.address]],
            sizes,
            replicate,
        )
        if len(result) != 2:
            raise RuntimeError(f"Mooncake compressed PUT failed: {result}")
        status = next((value for value in result if value < 0), 0)
        written = sum(
            sum(lengths)
            for lengths, code in zip(sizes, result, strict=True)
            if code >= 0
        )
        return status, written

    def _load(self, slot: int, key: str, block: int, part: int) -> int:
        object_key = self.codec.key_for(key, part)
        header = self.codec.header_read_target(slot)
        result = self.store.batch_get_into_multi_buffers(
            [f"{object_key}:manifest"],
            [[header.address]],
            [[header.nbytes]],
        )
        if result != [header.nbytes]:
            raise RuntimeError(f"Mooncake manifest GET failed: {result}")
        descriptor = self.codec.read_descriptor(slot)
        value = self.codec.payload_read_plan(slot, key, part, descriptor)
        sizes = [buffer.nbytes for buffer in value.buffers]
        result = self.store.batch_get_into_multi_buffers(
            [value.key],
            [[buffer.address for buffer in value.buffers]],
            [sizes],
        )
        if result != [sum(sizes)]:
            raise RuntimeError(f"Mooncake payload GET failed: {result}")
        self.codec.decompress_from_store(
            slot,
            descriptor,
            [[block]],
            part,
            valid_tokens_by_group={0: self.block_size},
        ).synchronize()
        return header.nbytes + result[0]

    def close(self) -> None:
        with self._lock:
            for ptr in self._registered:
                if self.store.unregister_buffer(ptr) != 0:
                    raise RuntimeError("Mooncake could not unregister a codec buffer")
            self._registered.clear()
            if self.codec is not None:
                self.codec.close()
                self.codec = None
            self.store.close()
