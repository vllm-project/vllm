# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.primary import (
    ECPrimaryTierManager,
)


class _FakeBlocks:
    def __init__(self) -> None:
        self._data = bytearray(b"a" * 64 + b"b" * 64 + b"c" * 64 + b"d" * 64)

    def data_ptr(self) -> int:
        return 1234

    def numpy(self) -> bytearray:
        return self._data


class _FakeRegion:
    num_blocks = 4
    block_size_bytes = 64

    def __init__(self) -> None:
        self.blocks = _FakeBlocks()
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
        b"a" * 64 + b"b" * 64,
    ]

    primary.mark_ready("item")
    primary.pin("item")
    assert primary.has_held_entries()
    primary.unpin("item")
    assert not primary.has_held_entries()

    primary.shutdown()
    assert region.cleanup_called


@pytest.mark.parametrize(
    ("block_ids", "run_count"),
    [((), 0), ((0,), 1), ((0, 1, 2, 3), 1), ((0, 2, 3), 2), ((2, 0, 1), 2)],
)
def test_block_runs_preserve_full_block_bytes_and_alias_storage(
    block_ids, run_count
) -> None:
    region = _FakeRegion()
    primary = ECPrimaryTierManager(region)
    views = primary.block_views(block_ids)

    assert len(views) == run_count
    assert b"".join(views) == b"".join(
        bytes([ord("a") + block_id]) * 64 for block_id in block_ids
    )
    if block_ids:
        views[0][0] = ord("z")
        assert region.blocks._data[block_ids[0] * 64] == ord("z")


def test_coalesced_views_round_trip_full_blocks_through_filesystem(tmp_path) -> None:
    from vllm.distributed.ec_transfer.ec_connector.tiering.io import (
        load_entry,
        store_entry,
    )

    region = _FakeRegion()
    primary = ECPrimaryTierManager(region)
    block_ids = (2, 0, 1)
    expected = b"c" * 64 + b"a" * 64 + b"b" * 64
    path = tmp_path / "entry.ec"
    store_entry(path, tmp_path / "entry.tmp", primary.block_views(block_ids), 192)
    assert path.read_bytes() == expected

    region.blocks._data[:] = bytes(256)
    load_entry(path, primary.block_views(block_ids), 192)
    assert region.blocks._data == b"a" * 64 + b"b" * 64 + b"c" * 64 + bytes(64)
