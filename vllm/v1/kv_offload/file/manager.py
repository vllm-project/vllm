# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileOffloadingManager: Manages KV cache offloading to file storage.
"""

import ctypes
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path

from vllm.logger import init_logger
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
)
from vllm.v1.kv_offload.file.load_store_spec import FileLoadStoreSpec

logger = init_logger(__name__)


class FileBlockStatus(ctypes.Structure):
    """
    Metadata for a file-stored block.
    """

    _fields_ = [
        ("ref_cnt", ctypes.c_int32),
        ("block_id", ctypes.c_int64),
        ("is_ready", ctypes.c_bool),
    ]

    def __init__(self, block_id: int):
        super().__init__()
        self.ref_cnt = -1  # -1 = not ready yet
        self.block_id = block_id
        self.is_ready = False


class FileOffloadingManager(OffloadingManager):
    """
    An OffloadingManager that stores KV blocks as files on disk.

    File layout:
        {storage_dir}/
            {key_hex_0}.bin
            {key_hex_1}.bin
            ...

    Each block is stored as a separate file for simplicity.
    The manager tracks which blocks are stored and manages eviction.
    """

    def __init__(
        self,
        storage_dir: str,
        num_blocks: int,
        block_size_bytes: int,
        enable_events: bool = False,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._num_blocks = num_blocks
        self.block_size_bytes = block_size_bytes
        self._num_allocated_blocks = 0
        self._free_list: list[int] = []

        # key -> FileBlockStatus
        self._blocks: OrderedDict[OffloadKey, FileBlockStatus] = OrderedDict()
        self.events: list[OffloadingEvent] | None = [] if enable_events else None

    def _get_num_free_blocks(self) -> int:
        return len(self._free_list) + self._num_blocks - self._num_allocated_blocks

    def _allocate_blocks(self, keys: list[OffloadKey]) -> list[FileBlockStatus]:
        num_fresh = min(len(keys), self._num_blocks - self._num_allocated_blocks)
        num_reused = len(keys) - num_fresh

        blocks: list[FileBlockStatus] = []
        for _ in range(num_fresh):
            blocks.append(FileBlockStatus(self._num_allocated_blocks))
            self._num_allocated_blocks += 1

        for _ in range(num_reused):
            blocks.append(FileBlockStatus(self._free_list.pop()))
        return blocks

    def _free_block(self, block: FileBlockStatus) -> None:
        self._free_list.append(block.block_id)

    def _key_to_path(self, key: OffloadKey) -> Path:
        return self.storage_dir / f"{key.hex()}.bin"

    def _delete_file(self, key: OffloadKey) -> None:
        path = self._key_to_path(key)
        try:
            if path.exists():
                path.unlink()
        except OSError as e:
            logger.warning("Failed to delete file %s: %r", path, e)

    # --- OffloadingManager interface ---

    def lookup(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext,
    ) -> int | None:
        hit_count = 0
        for key in keys:
            block = self._blocks.get(key)
            if block is None or not block.is_ready:
                break
            hit_count += 1
        return hit_count

    def prepare_load(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        keys_list = list(keys)
        file_paths: list[str] = []
        block_offsets: list[int] = []

        for key in keys_list:
            block = self._blocks.get(key)
            assert block is not None, f"Block {key!r} not found in file cache"
            assert block.is_ready, f"Block {key!r} is not ready for reading"
            block.ref_cnt += 1
            file_paths.append(str(self._key_to_path(key)))
            # Each file contains one block's data, so offset is always 0
            block_offsets.append(0)

        return FileLoadStoreSpec(file_paths, block_offsets)

    def touch(self, keys: Iterable[OffloadKey]) -> None:
        for key in reversed(list(keys)):
            if key in self._blocks:
                self._blocks.move_to_end(key)

    def complete_load(self, keys: Iterable[OffloadKey]) -> None:
        for key in keys:
            block = self._blocks.get(key)
            assert block is not None, f"Block {key!r} not found"
            assert block.ref_cnt > 0, f"Block {key!r} ref_cnt is already 0"
            block.ref_cnt -= 1

    def prepare_store(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        keys_list = list(keys)

        # Filter out blocks already stored
        keys_to_store = [k for k in keys_list if k not in self._blocks]

        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=FileLoadStoreSpec([], []),
                evicted_keys=[],
            )

        num_blocks_to_evict = len(keys_to_store) - self._get_num_free_blocks()

        to_evict: list[OffloadKey] = []
        if num_blocks_to_evict > 0:
            protected = set(keys_list)
            candidates: list[tuple[OffloadKey, FileBlockStatus]] = []
            for key, block in self._blocks.items():
                if block.ref_cnt == 0 and key not in protected:
                    candidates.append((key, block))
                    if len(candidates) == num_blocks_to_evict:
                        break

            if len(candidates) < num_blocks_to_evict:
                return None  # Cannot evict enough blocks

            for key, _ in candidates:
                self._blocks.pop(key)
                self._free_block(self._blocks.get(key) or FileBlockStatus(-1))
                self._delete_file(key)
                to_evict.append(key)

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(keys=to_evict, medium="FILE", removed=True)
            )

        # Allocate blocks for new entries
        new_blocks = self._allocate_blocks(keys_to_store)
        for key, block in zip(keys_to_store, new_blocks):
            self._blocks[key] = block

        # Build store spec with file paths
        # Each key maps to one file, so offset is always 0
        file_paths = [str(self._key_to_path(k)) for k in keys_to_store]
        block_offsets = [0] * len(keys_to_store)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=FileLoadStoreSpec(file_paths, block_offsets),
            evicted_keys=to_evict,
        )

    def complete_store(self, keys: Iterable[OffloadKey], success: bool = True) -> None:
        stored_keys: list[OffloadKey] = []

        if success:
            for key in keys:
                block = self._blocks.get(key)
                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    block.is_ready = True
                    stored_keys.append(key)
        else:
            for key in keys:
                block = self._blocks.get(key)
                if block is not None and not block.is_ready:
                    self._free_block(block)
                    self._blocks.pop(key)
                    self._delete_file(key)

        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(keys=stored_keys, medium="FILE", removed=False)
            )

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    def shutdown(self) -> None:
        # Clean up all files on shutdown
        for key in list(self._blocks.keys()):
            self._delete_file(key)
        self._blocks.clear()
        self._free_list.clear()
        logger.info("FileOffloadingManager shutdown complete")
