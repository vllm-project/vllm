# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.embedding_cache import (
    EmbeddingCache,
)
from vllm.distributed.ec_transfer.ec_connector.tiering.primary import (
    ECCPUPrimaryAdapter,
)

pytestmark = pytest.mark.cpu_test


class _FakeBlocks:
    def __init__(self) -> None:
        self._data = bytearray(b"a" * 64 + b"b" * 64 + b"c" * 64 + b"d" * 64)

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


@pytest.mark.parametrize(
    ("block_ids", "run_count"),
    [((), 0), ((0,), 1), ((0, 1, 2, 3), 1), ((0, 2, 3), 2), ((2, 0, 1), 2)],
)
def test_block_runs_preserve_full_block_bytes_and_alias_storage(
    block_ids, run_count
) -> None:
    region = _FakeRegion()
    cache = EmbeddingCache(region.num_blocks)
    primary = ECCPUPrimaryAdapter(region, cache)
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
    cache = EmbeddingCache(region.num_blocks)
    primary = ECCPUPrimaryAdapter(region, cache)
    block_ids = (2, 0, 1)
    expected = b"c" * 64 + b"a" * 64 + b"b" * 64
    path = tmp_path / "entry.ec"
    store_entry(path, tmp_path / "entry.tmp", primary.block_views(block_ids), 192)
    assert path.read_bytes() == expected

    region.blocks._data[:] = bytes(256)
    load_entry(path, primary.block_views(block_ids), 192)
    assert region.blocks._data == b"a" * 64 + b"b" * 64 + b"c" * 64 + bytes(64)


def test_tiering_borrows_cpu_scheduler_objects_and_leaves_cleanup_to_owner(monkeypatch):
    from tests.v1.ec_connector.unit.utils import create_ec_vllm_config
    from vllm.distributed.ec_transfer.ec_connector.cpu import scheduler as cpu
    from vllm.distributed.ec_transfer.ec_connector.tiering.scheduler import (
        ECTieringScheduler,
    )

    region = _FakeRegion()
    cleanups = []
    monkeypatch.setattr(region, "cleanup", lambda: cleanups.append(True))
    monkeypatch.setattr(cpu, "create_ec_shared_region", lambda config: region)
    scheduler = ECTieringScheduler(create_ec_vllm_config())
    try:
        entry = scheduler._cache.alloc("shared", 2, (2, 16))
        scheduler._cache.mark_ready("shared")
        assert scheduler._primary.get("shared") is entry
        scheduler._primary.pin("shared")
        assert scheduler._cache.alloc("other", 3) is None
        scheduler._primary.unpin("shared")
        assert scheduler._cache.alloc("other", 3) is not None
        assert cleanups == []
    finally:
        scheduler.shutdown()
    assert cleanups == [True]
