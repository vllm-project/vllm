# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.primary import (
    ECPrimaryTierManager,
)


class _FakeBlocks:
    def __init__(self) -> None:
        self._blocks = [bytearray(b"abcd") for _ in range(4)]

    def data_ptr(self) -> int:
        return 1234

    def __getitem__(self, index: int):
        return _FakeBlock(self._blocks[index])


class _FakeBlock:
    def __init__(self, data: bytearray) -> None:
        self._data = data

    def numpy(self) -> bytearray:
        return self._data


class _FakeRegion:
    num_blocks = 4
    block_size_bytes = 64
    blocks = _FakeBlocks()

    def __init__(self) -> None:
        self.cleanup_called = False

    def cleanup(self) -> None:
        self.cleanup_called = True


def test_primary_manager_owns_entry_lifecycle_and_region_geometry() -> None:
    region = _FakeRegion()
    primary = ECPrimaryTierManager(region)

    entry = primary.alloc("item", 2, (2, 16))
    assert entry is not None
    assert not entry.ready
    assert primary.get("item") is entry
    assert primary.num_blocks == 4
    assert primary.block_size_bytes == 64
    assert primary.total_size_bytes == 256
    assert primary.base_ptr == 1234
    assert [bytes(view) for view in primary.block_views((0, 1))] == [
        b"abcd",
        b"abcd",
    ]

    primary.mark_ready("item")
    primary.pin("item")
    assert primary.has_held_entries()
    primary.unpin("item")
    assert not primary.has_held_entries()

    primary.shutdown()
    assert region.cleanup_called
