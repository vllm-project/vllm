# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.distributed.ec_transfer.ec_connector.tiering.filesystem import (
    ECReadResult,
    ECWriteResult,
)
from vllm.distributed.ec_transfer.ec_connector.tiering.manager import (
    ECTieringManager,
)
from vllm.distributed.ec_transfer.ec_connector.tiering.layout import (
    build_ec_cache_key,
)

_NAMESPACE_DIGEST = "a" * 64


class _FakePrimary:
    block_size_bytes = 4

    def __init__(self) -> None:
        self.entries = {
            "item": SimpleNamespace(block_ids=(0, 1), ready=True, shape=None)
        }
        self.blocks = [bytearray(4), bytearray(4)]
        self.pin_count = 0
        self.alloc_enabled = True

    def get(self, key: str):
        return self.entries.get(key)

    def alloc(self, key: str, n_blocks: int, shape=None):
        if key in self.entries:
            return self.entries[key]
        if not self.alloc_enabled:
            return None
        entry = SimpleNamespace(
            block_ids=tuple(range(n_blocks)), ready=False, shape=shape
        )
        self.entries[key] = entry
        return entry

    def mark_ready(self, key: str) -> None:
        self.entries[key].ready = True

    def discard(self, key: str) -> None:
        self.entries.pop(key, None)

    def pin(self, key: str) -> None:
        self.pin_count += 1

    def unpin(self, key: str) -> None:
        self.pin_count -= 1

    def block_views(self, block_ids):
        return tuple(memoryview(self.blocks[block_id]) for block_id in block_ids)


class _FakeSecondary:
    def __init__(self, *, accept: bool = True) -> None:
        self.accept = accept
        self.entries: set[str] = set()
        self.write_submissions: list[tuple[str, int]] = []
        self.read_submissions: list[str] = []
        self.write_completions: list[ECWriteResult] = []
        self.read_completions: list[ECReadResult] = []
        self.shutdown_called = False

    def contains(self, cache_key: str) -> bool:
        return cache_key in self.entries

    def submit_write(self, cache_key, sources, file_size_bytes):
        self.write_submissions.append((cache_key, file_size_bytes))
        return self.accept

    def submit_read(self, cache_key, destinations):
        self.read_submissions.append(cache_key)
        return self.accept

    def poll_completed_reads(self):
        completions, self.read_completions = self.read_completions, []
        return completions

    def poll_completed_writes(self):
        completions, self.write_completions = self.write_completions, []
        return completions

    def shutdown(self) -> None:
        self.shutdown_called = True


def test_write_through_holds_cpu_pin_until_store_completion() -> None:
    primary = _FakePrimary()
    secondary = _FakeSecondary()
    manager = ECTieringManager(primary, secondary, _NAMESPACE_DIGEST)

    manager.register_cpu_save("item", "image", 2)
    manager.on_cpu_save_completed("item")

    assert primary.pin_count == 1
    assert len(secondary.write_submissions) == 1
    cache_key, file_size_bytes = secondary.write_submissions[0]
    assert file_size_bytes == 8
    assert manager.has_pending()

    secondary.write_completions.append(
        ECWriteResult(cache_key=cache_key, success=True, file_size_bytes=8)
    )
    assert not manager.has_pending()
    assert primary.pin_count == 0


def test_write_through_skips_existing_secondary_entry_without_pin() -> None:
    primary = _FakePrimary()
    secondary = _FakeSecondary()
    manager = ECTieringManager(primary, secondary, _NAMESPACE_DIGEST)
    manager.register_cpu_save("item", "image", 2)
    secondary.entries.add(
        build_ec_cache_key(_NAMESPACE_DIGEST, "item", "image", 2, 8)
    )

    manager.on_cpu_save_completed("item")

    assert primary.pin_count == 0
    assert secondary.write_submissions == []


def test_write_through_releases_pin_when_store_is_not_accepted() -> None:
    primary = _FakePrimary()
    secondary = _FakeSecondary(accept=False)
    manager = ECTieringManager(primary, secondary, _NAMESPACE_DIGEST)
    manager.register_cpu_save("item", "image", 2)

    manager.on_cpu_save_completed("item")

    assert primary.pin_count == 0
    assert not manager.has_pending()


def test_secondary_hit_is_promoted_once_and_published_on_completion() -> None:
    primary = _FakePrimary()
    secondary = _FakeSecondary()
    manager = ECTieringManager(primary, secondary, _NAMESPACE_DIGEST)
    cache_key = build_ec_cache_key(
        _NAMESPACE_DIGEST, "loaded", "image", 2, 8
    )
    secondary.entries.add(cache_key)

    assert not manager.ensure_primary_entry(
        "loaded", "image", 2, (2, 1), 4
    )
    assert not manager.ensure_primary_entry(
        "loaded", "image", 2, (2, 1), 4
    )
    assert secondary.read_submissions == [cache_key]
    assert not primary.entries["loaded"].ready

    secondary.read_completions.append(
        ECReadResult(cache_key=cache_key, success=True, file_size_bytes=8)
    )

    assert manager.ensure_primary_entry(
        "loaded", "image", 2, (2, 1), 4
    )
    assert primary.entries["loaded"].ready
    assert not manager.has_pending()


def test_secondary_miss_allows_recomputation_without_cpu_allocation() -> None:
    primary = _FakePrimary()
    secondary = _FakeSecondary()
    manager = ECTieringManager(primary, secondary, _NAMESPACE_DIGEST)

    assert manager.ensure_primary_entry("missing", "image", 2, (2, 1), 4)
    assert "missing" not in primary.entries
    assert secondary.read_submissions == []


def test_promotion_retries_when_cpu_space_becomes_available() -> None:
    primary = _FakePrimary()
    primary.alloc_enabled = False
    secondary = _FakeSecondary()
    manager = ECTieringManager(primary, secondary, _NAMESPACE_DIGEST)
    cache_key = build_ec_cache_key(
        _NAMESPACE_DIGEST, "waiting", "image", 2, 8
    )
    secondary.entries.add(cache_key)

    assert not manager.ensure_primary_entry(
        "waiting", "image", 2, (2, 1), 4
    )
    assert secondary.read_submissions == []

    primary.alloc_enabled = True
    assert not manager.ensure_primary_entry(
        "waiting", "image", 2, (2, 1), 4
    )
    assert secondary.read_submissions == [cache_key]


def test_failed_promotion_discards_cpu_allocation_and_allows_fallback() -> None:
    primary = _FakePrimary()
    secondary = _FakeSecondary()
    manager = ECTieringManager(primary, secondary, _NAMESPACE_DIGEST)
    cache_key = build_ec_cache_key(
        _NAMESPACE_DIGEST, "broken", "image", 2, 8
    )
    secondary.entries.add(cache_key)
    assert not manager.ensure_primary_entry(
        "broken", "image", 2, (2, 1), 4
    )

    secondary.entries.remove(cache_key)
    secondary.read_completions.append(
        ECReadResult(
            cache_key=cache_key,
            success=False,
            file_size_bytes=8,
            error="missing",
        )
    )

    assert manager.ensure_primary_entry("broken", "image", 2, (2, 1), 4)
    assert "broken" not in primary.entries
    assert not manager.has_pending()
