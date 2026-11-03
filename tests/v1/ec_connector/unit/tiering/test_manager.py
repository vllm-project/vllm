# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.distributed.ec_transfer.ec_connector.tiering.filesystem import (
    ECReadResult,
    ECWriteResult,
)
from vllm.distributed.ec_transfer.ec_connector.tiering.layout import (
    build_ec_cache_key,
)
from vllm.distributed.ec_transfer.ec_connector.tiering.manager import (
    ECTieringManager,
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
    secondary.entries.add(build_ec_cache_key(_NAMESPACE_DIGEST, "item", "image", 2, 8))

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
    cache_key = build_ec_cache_key(_NAMESPACE_DIGEST, "loaded", "image", 2, 8)
    secondary.entries.add(cache_key)

    assert not manager.ensure_primary_entry("loaded", "image", 2, (2, 1), 4)
    assert not manager.ensure_primary_entry("loaded", "image", 2, (2, 1), 4)
    assert secondary.read_submissions == [cache_key]
    assert not primary.entries["loaded"].ready

    secondary.read_completions.append(
        ECReadResult(cache_key=cache_key, success=True, file_size_bytes=8)
    )

    assert manager.ensure_primary_entry("loaded", "image", 2, (2, 1), 4)
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
    cache_key = build_ec_cache_key(_NAMESPACE_DIGEST, "waiting", "image", 2, 8)
    secondary.entries.add(cache_key)

    assert not manager.ensure_primary_entry("waiting", "image", 2, (2, 1), 4)
    assert secondary.read_submissions == []

    primary.alloc_enabled = True
    assert not manager.ensure_primary_entry("waiting", "image", 2, (2, 1), 4)
    assert secondary.read_submissions == [cache_key]


def test_failed_promotion_discards_cpu_allocation_and_allows_fallback() -> None:
    primary = _FakePrimary()
    secondary = _FakeSecondary()
    manager = ECTieringManager(primary, secondary, _NAMESPACE_DIGEST)
    cache_key = build_ec_cache_key(_NAMESPACE_DIGEST, "broken", "image", 2, 8)
    secondary.entries.add(cache_key)
    assert not manager.ensure_primary_entry("broken", "image", 2, (2, 1), 4)

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


def test_ready_hit_leaves_store_completion_to_lifecycle_poll() -> None:
    primary = _FakePrimary()
    secondary = _FakeSecondary()
    manager = ECTieringManager(primary, secondary, _NAMESPACE_DIGEST)
    manager.register_cpu_save("item", "image", 2)
    manager.on_cpu_save_completed("item")
    cache_key, _ = secondary.write_submissions[0]
    secondary.write_completions.append(
        ECWriteResult(cache_key=cache_key, success=True, file_size_bytes=8)
    )

    assert manager.ensure_primary_entry("item", "image", 2, (2, 1), 4)
    assert primary.pin_count == 1
    assert not manager.has_pending()
    assert primary.pin_count == 0


@pytest.mark.parametrize("raise_error", [False, True])
def test_rejected_promotion_can_be_retried_without_stale_pending_state(
    raise_error, monkeypatch
) -> None:
    primary = _FakePrimary()
    secondary = _FakeSecondary(accept=False)
    manager = ECTieringManager(primary, secondary, _NAMESPACE_DIGEST)
    cache_key = build_ec_cache_key(_NAMESPACE_DIGEST, "loaded", "image", 2, 8)
    secondary.entries.add(cache_key)
    original_views = primary.block_views
    if raise_error:

        def broken_views(block_ids):
            raise ValueError("invalid destination")

        monkeypatch.setattr(primary, "block_views", broken_views)

    assert manager.ensure_primary_entry("loaded", "image", 2, (2, 1), 4)
    assert "loaded" not in primary.entries
    assert not manager.has_pending()

    monkeypatch.setattr(primary, "block_views", original_views)
    secondary.accept = True
    assert not manager.ensure_primary_entry("loaded", "image", 2, (2, 1), 4)
    assert not manager.ensure_primary_entry("loaded", "image", 2, (2, 1), 4)
    assert secondary.read_submissions.count(cache_key) == (1 if raise_error else 2)
    secondary.read_completions.append(
        ECReadResult(cache_key=cache_key, success=True, file_size_bytes=8)
    )
    assert manager.ensure_primary_entry("loaded", "image", 2, (2, 1), 4)
    assert not manager.has_pending()


def _bounded_prefetch_manager():
    from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.primary import (
        ECPrimaryTierManager,
    )

    storage = bytearray(8)
    primary = ECPrimaryTierManager(
        SimpleNamespace(
            num_blocks=2,
            block_size_bytes=4,
            blocks=SimpleNamespace(numpy=lambda: storage),
        )
    )
    secondary = _FakeSecondary()
    for identifier in ("loaded", "other"):
        secondary.entries.add(
            build_ec_cache_key(_NAMESPACE_DIGEST, identifier, "image", 2, 8)
        )
    return primary, secondary, ECTieringManager(primary, secondary, _NAMESPACE_DIGEST)


def _complete_prefetch(secondary, *, success=True):
    secondary.read_completions.append(
        ECReadResult(
            cache_key=secondary.read_submissions[-1], success=success, file_size_bytes=8
        )
    )


def test_prefetch_deduplicates_waiters_and_holds_until_last_release():
    primary, secondary, manager = _bounded_prefetch_manager()
    for request_id in ("r1", "r1", "r2"):
        manager.prefetch_entry(request_id, "loaded", "image", 2, (2, 1), 4)
    assert len(secondary.read_submissions) == 1
    _complete_prefetch(secondary)
    manager.poll()
    assert primary.get("loaded").ready
    assert primary.alloc("other", 2) is None
    manager.release_prefetch("r1")
    assert primary.alloc("other", 2) is None
    manager.release_prefetch("r2")
    assert primary.alloc("other", 2) is not None


def test_cancelled_prefetch_keeps_blocks_until_io_completes():
    primary, secondary, manager = _bounded_prefetch_manager()
    manager.prefetch_entry("r1", "loaded", "image", 2, (2, 1), 4)
    manager.release_prefetch("r1")
    assert primary.alloc("other", 2) is None
    _complete_prefetch(secondary)
    manager.poll()
    assert primary.alloc("other", 2) is not None


def test_prefetch_handoff_preserves_existing_h2d_pin():
    primary, secondary, manager = _bounded_prefetch_manager()
    manager.prefetch_entry("r1", "loaded", "image", 2, (2, 1), 4)
    _complete_prefetch(secondary)
    manager.poll()
    primary.pin("loaded")  # Acquired by the existing CPU scheduler for H2D.
    manager.release_prefetch("r1", "loaded")
    assert primary.alloc("other", 2) is None
    primary.unpin("loaded")  # Existing H2D completion acknowledgement.
    assert primary.alloc("other", 2) is not None


def test_demand_load_reclaims_speculative_holds_when_cpu_is_full():
    primary, secondary, manager = _bounded_prefetch_manager()
    manager.prefetch_entry("r1", "loaded", "image", 2, (2, 1), 4)
    _complete_prefetch(secondary)
    manager.poll()
    manager.prefetch_entry("r2", "other", "image", 2, (2, 1), 4)
    assert len(secondary.read_submissions) == 1  # Early prefetch cannot steal holds.
    assert not manager.ensure_primary_entry("other", "image", 2, (2, 1), 4)
    assert len(secondary.read_submissions) == 2
    manager.release_prefetch("r1")  # Its reclaimed pin must not be released twice.
    _complete_prefetch(secondary)
    assert manager.ensure_primary_entry("other", "image", 2, (2, 1), 4)
    assert primary.get("loaded") is None


def test_failed_prefetch_releases_waiters_and_allows_a_new_read():
    primary, secondary, manager = _bounded_prefetch_manager()
    manager.prefetch_entry("r1", "loaded", "image", 2, (2, 1), 4)
    _complete_prefetch(secondary, success=False)
    manager.poll()
    assert primary.get("loaded") is None
    manager.release_prefetch("r1")
    manager.prefetch_entry("r1", "loaded", "image", 2, (2, 1), 4)
    assert len(secondary.read_submissions) == 2
    _complete_prefetch(secondary)
    manager.poll()
    manager.release_prefetch("r1")
    assert primary.alloc("other", 2) is not None


def test_shutdown_drains_prefetch_and_releases_ready_reservations():
    primary, secondary, manager = _bounded_prefetch_manager()
    manager.prefetch_entry("r1", "loaded", "image", 2, (2, 1), 4)
    _complete_prefetch(secondary)
    manager.shutdown()
    assert secondary.shutdown_called
    assert primary.alloc("other", 2) is not None


def test_scheduler_prefetch_publishes_before_admission_and_hands_off_to_h2d(
    monkeypatch,
):
    from vllm.distributed.ec_transfer.ec_connector.tiering import scheduler as module

    primary, secondary, manager = _bounded_prefetch_manager()
    scheduler = object.__new__(module.ECTieringScheduler)
    scheduler._tiering_manager = manager
    scheduler._primary = primary
    scheduler._is_consumer = True
    scheduler._is_producer = True
    scheduler._nixl_enabled = False
    scheduler._vllm_config = object()
    scheduler._dtype = SimpleNamespace(itemsize=4)
    scheduler._pending_loads = {}
    scheduler._pending_saves = {}
    scheduler._load_acks = {}
    scheduler._next_transfer_id = 0
    scheduler._expected_load_reports = 1
    monkeypatch.setattr(module, "get_encoder_output_width", lambda *args: 1)
    feature = SimpleNamespace(
        identifier="loaded",
        modality="image",
        mm_position=SimpleNamespace(offset=0, length=2, get_num_embeds=lambda: 2),
    )
    request = SimpleNamespace(request_id="r1", mm_features=[feature])

    scheduler.on_new_request(request)
    assert len(secondary.read_submissions) == 1  # Before ensure_cache_available.
    _complete_prefetch(secondary)
    scheduler.on_schedule_begin()
    assert primary.get("loaded").ready
    scheduler.update_state_after_alloc(request, 0)
    assert "loaded" in scheduler._pending_loads
    assert primary.alloc("other", 2) is None
    primary.unpin("loaded")  # Existing worker's completion, not a prefetch pin.
    assert primary.alloc("other", 2) is not None
