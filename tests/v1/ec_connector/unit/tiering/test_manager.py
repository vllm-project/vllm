# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.embedding_cache import (
    EmbeddingCache,
)
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
pytestmark = pytest.mark.cpu_test


class _FakeSecondary:
    def __init__(self, *, accept: bool = True) -> None:
        self.accept = accept
        self.entries: set[str] = set()
        self.write_submissions: list[tuple[str, int]] = []
        self.read_submissions: list[str] = []
        self.write_completions: list[ECWriteResult] = []
        self.read_completions: list[ECReadResult] = []
        self.shutdown_called = False
        self.lookup_pending = False

    def contains(self, cache_key: str) -> bool:
        return cache_key in self.entries

    def lookup(self, cache_key, file_size_bytes, request_id):
        if self.lookup_pending:
            return None
        return self.contains(cache_key)

    def release_request(self, request_id):
        pass

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


@pytest.mark.parametrize("store_state", ["accepted", "existing", "rejected"])
def test_write_through_releases_blocks_after_completion_or_skip(store_state) -> None:
    primary, secondary, manager = _bounded_promotion_manager()
    primary.alloc("item", 2, (2, 1))
    primary.mark_ready("item")
    cache_key = build_ec_cache_key(_NAMESPACE_DIGEST, "item", "image", 2, 8)
    if store_state == "existing":
        secondary.entries.add(cache_key)
    secondary.accept = store_state != "rejected"
    manager.register_cpu_save("item", "image", 2)
    manager.on_cpu_save_completed("item")

    assert secondary.write_submissions == (
        [] if store_state == "existing" else [(cache_key, 8)]
    )
    if store_state == "accepted":
        assert primary.alloc("other", 2) is None
        secondary.write_completions.append(
            ECWriteResult(cache_key=cache_key, success=True, file_size_bytes=8)
        )
        assert manager.ensure_primary_entry("item", "image", 2, (2, 1), 4)
        assert primary.alloc("other", 2) is None
    assert not manager.has_pending()
    assert primary.alloc("other", 2) is not None


def test_secondary_miss_allows_recomputation_without_cpu_allocation() -> None:
    primary, secondary, manager = _bounded_promotion_manager()

    assert manager.ensure_primary_entry("missing", "image", 2, (2, 1), 4)
    assert primary.get("missing") is None
    assert secondary.read_submissions == []


@pytest.mark.parametrize("raise_error", [False, True])
def test_rejected_promotion_can_be_retried_without_stale_pending_state(
    raise_error, monkeypatch
) -> None:
    primary, secondary, manager = _bounded_promotion_manager()
    secondary.accept = False
    cache_key = build_ec_cache_key(_NAMESPACE_DIGEST, "loaded", "image", 2, 8)
    original_submit = secondary.submit_read
    if raise_error:

        def failed_submit(cache_key, destinations):
            raise OSError("read submission failed")

        monkeypatch.setattr(secondary, "submit_read", failed_submit)

    assert manager.ensure_primary_entry("loaded", "image", 2, (2, 1), 4)
    assert primary.get("loaded") is None
    assert not manager.has_pending()

    monkeypatch.setattr(secondary, "submit_read", original_submit)
    secondary.accept = True
    assert not manager.ensure_primary_entry("loaded", "image", 2, (2, 1), 4)
    assert not manager.ensure_primary_entry("loaded", "image", 2, (2, 1), 4)
    assert secondary.read_submissions.count(cache_key) == (1 if raise_error else 2)
    secondary.read_completions.append(
        ECReadResult(cache_key=cache_key, success=True, file_size_bytes=8)
    )
    assert manager.ensure_primary_entry("loaded", "image", 2, (2, 1), 4)
    assert not manager.has_pending()


def _bounded_promotion_manager():
    from vllm.distributed.ec_transfer.ec_connector.tiering.primary import (
        ECCPUPrimaryAdapter,
    )

    storage = bytearray(8)
    region = SimpleNamespace(
        num_blocks=2,
        block_size_bytes=4,
        blocks=SimpleNamespace(numpy=lambda: storage),
    )
    primary = ECCPUPrimaryAdapter(region, EmbeddingCache(region.num_blocks))
    secondary = _FakeSecondary()
    for identifier in ("loaded", "other"):
        secondary.entries.add(
            build_ec_cache_key(_NAMESPACE_DIGEST, identifier, "image", 2, 8)
        )
    return primary, secondary, ECTieringManager(primary, secondary, _NAMESPACE_DIGEST)


def _complete_promotion(secondary, *, success=True):
    secondary.read_completions.append(
        ECReadResult(
            cache_key=secondary.read_submissions[-1], success=success, file_size_bytes=8
        )
    )


@pytest.mark.parametrize("arrival_prefetch", [False, True])
@pytest.mark.parametrize("load_reports", [1, 2])
def test_scheduler_admission_publishes_completion_and_preserves_h2d_pin(
    monkeypatch,
    load_reports,
    arrival_prefetch,
):
    from vllm.distributed.ec_transfer.ec_connector.tiering import scheduler as module

    primary, secondary, manager = _bounded_promotion_manager()
    scheduler = object.__new__(module.ECTieringScheduler)
    scheduler._tiering_manager = manager
    scheduler._primary = primary
    scheduler._region = primary._region
    scheduler._cache = primary._cache
    scheduler._is_consumer = True
    scheduler._is_producer = True
    scheduler._nixl_enabled = False
    scheduler._vllm_config = object()
    scheduler._dtype = SimpleNamespace(itemsize=4)
    scheduler._pending_loads = {}
    scheduler._pending_saves = {}
    scheduler._load_acks = {}
    scheduler._next_transfer_id = 0
    scheduler._expected_load_reports = load_reports
    monkeypatch.setattr(module, "get_encoder_output_width", lambda *args: 1)
    feature = SimpleNamespace(
        identifier="loaded",
        modality="image",
        mm_position=SimpleNamespace(offset=0, length=2, get_num_embeds=lambda: 2),
    )
    request = SimpleNamespace(request_id="r1", mm_features=[feature])

    if arrival_prefetch:
        scheduler.on_new_request(request)
        assert len(secondary.read_submissions) == 1
    assert not scheduler.ensure_cache_available(request, 0)
    assert not scheduler.ensure_cache_available(request, 0)
    assert len(secondary.read_submissions) == 1
    _complete_promotion(secondary)
    assert scheduler.ensure_cache_available(request, 0)
    assert primary.get("loaded").ready
    scheduler.update_state_after_alloc(request, 0)
    assert "loaded" in scheduler._pending_loads
    assert primary.alloc("other", 2) is None
    metadata = module.ECCPUWorkerMetadata(completed_loads=[0])
    output = SimpleNamespace(ec_connector_worker_meta=metadata)
    for _ in range(load_reports - 1):
        scheduler.update_connector_output(output)
        assert primary.alloc("other", 2) is None
    scheduler.update_connector_output(output)
    scheduler.update_connector_output(output)  # A stale ack must not unpin twice.
    assert primary.alloc("other", 2) is not None


def test_prefetch_deduplicates_waiters_and_holds_until_last_release():
    primary, secondary, manager = _bounded_promotion_manager()
    for request_id in ("r1", "r1", "r2"):
        manager.prefetch_entry(request_id, "loaded", "image", 2, (2, 1), 4)
    assert len(secondary.read_submissions) == 1
    _complete_promotion(secondary)
    manager.poll()
    assert primary.get("loaded").ready
    assert primary.alloc("other", 2) is None
    manager.release_prefetch("r1")
    assert primary.alloc("other", 2) is None
    manager.release_prefetch("r2")
    assert primary.alloc("other", 2) is not None


def test_cpu_save_during_lookup_keeps_shared_prefetch_protection():
    primary, secondary, manager = _bounded_promotion_manager()
    secondary.lookup_pending = True
    for request_id in ("first", "second"):
        manager.prefetch_entry(request_id, "loaded", "image", 2, (2, 1), 4)
    primary.alloc("loaded", 2, (2, 1))
    manager.poll()
    primary.mark_ready("loaded")
    manager.poll()
    assert secondary.read_submissions == []
    assert primary.alloc("other", 2) is None
    manager.request_finished("first")
    assert primary.alloc("other", 2) is None
    manager.request_finished("second")
    assert primary.alloc("other", 2) is not None


def test_cancelled_prefetch_keeps_blocks_until_io_completes():
    primary, secondary, manager = _bounded_promotion_manager()
    manager.prefetch_entry("r1", "loaded", "image", 2, (2, 1), 4)
    manager.release_prefetch("r1")
    assert primary.alloc("other", 2) is None
    _complete_promotion(secondary)
    manager.poll()
    assert primary.alloc("other", 2) is not None


def test_demand_allocation_miss_preserves_other_requests_prefetch():
    primary, secondary, manager = _bounded_promotion_manager()
    manager.prefetch_entry("r1", "loaded", "image", 2, (2, 1), 4)
    _complete_promotion(secondary)
    manager.poll()
    manager.prefetch_entry("r2", "other", "image", 2, (2, 1), 4)
    assert len(secondary.read_submissions) == 1  # Early prefetch cannot steal holds.
    assert manager.ensure_primary_entry("other", "image", 2, (2, 1), 4)
    assert len(secondary.read_submissions) == 1
    assert primary.get("other") is None
    assert primary.get("loaded").ready
    assert primary.alloc("other", 2) is None
    manager.release_prefetch("r1")
    assert not manager.ensure_primary_entry("other", "image", 2, (2, 1), 4)
    assert len(secondary.read_submissions) == 2
    _complete_promotion(secondary)
    assert manager.ensure_primary_entry("other", "image", 2, (2, 1), 4)
    assert primary.get("loaded") is None


def test_failed_prefetch_releases_waiters_and_allows_a_new_read():
    primary, secondary, manager = _bounded_promotion_manager()
    manager.prefetch_entry("r1", "loaded", "image", 2, (2, 1), 4)
    _complete_promotion(secondary, success=False)
    manager.poll()
    assert primary.get("loaded") is None
    manager.release_prefetch("r1")
    manager.prefetch_entry("r1", "loaded", "image", 2, (2, 1), 4)
    assert len(secondary.read_submissions) == 2
    _complete_promotion(secondary)
    manager.poll()
    manager.release_prefetch("r1")
    assert primary.alloc("other", 2) is not None


def test_shutdown_drains_prefetch_and_releases_ready_reservations():
    primary, secondary, manager = _bounded_promotion_manager()
    manager.prefetch_entry("r1", "loaded", "image", 2, (2, 1), 4)
    _complete_promotion(secondary)
    manager.shutdown()
    assert secondary.shutdown_called
    assert primary.alloc("other", 2) is not None


@pytest.fixture
def indexed_scheduler(tmp_path, request):
    """Real mmap, index, executor and scheduler; only reuse the config builder."""
    from tests.v1.ec_connector.unit.utils import create_ec_vllm_config
    from vllm.distributed.ec_transfer.ec_connector.tiering.io import store_entry
    from vllm.distributed.ec_transfer.ec_connector.tiering.layout import (
        get_ec_entry_path,
        get_ec_temporary_path,
    )
    from vllm.distributed.ec_transfer.ec_connector.tiering.scheduler import (
        ECTieringScheduler,
    )

    config = create_ec_vllm_config(
        encoder_output_widths=getattr(request, "param", None)
    )
    config.model_config.model = "test-encoder"
    config.model_config.revision = None
    config.ec_transfer_config.ec_connector_extra_config = {
        "ec_cpu_bytes": 1 << 20,
        "secondary_tiers": [
            {
                "type": "fs",
                "root_dir": str(tmp_path),
                "capacity_bytes": 1 << 20,
                "n_read_threads": 1,
                "n_write_threads": 1,
            }
        ],
    }
    digest = ECTieringScheduler._build_namespace(config).digest()
    file_size = 2 * config.ec_transfer_config.mm_encoder_output_widths["image"] * 2
    for identifier in ("A", "B"):
        key = build_ec_cache_key(digest, identifier, "image", 2, file_size)
        path = get_ec_entry_path(tmp_path, digest, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        store_entry(
            path,
            get_ec_temporary_path(path, "seed"),
            [memoryview(bytes(file_size))],
            file_size,
        )
    scheduler = ECTieringScheduler(config)
    try:
        yield scheduler
    finally:
        scheduler.shutdown()


def image_request(*identifiers, request_id="gpu-hint"):
    from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    return Request(
        request_id=request_id,
        prompt_token_ids=[0] * (2 * len(identifiers)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        mm_features=[
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier=identifier,
                mm_position=PlaceholderRange(offset=2 * i, length=2),
            )
            for i, identifier in enumerate(identifiers)
        ],
    )


def test_mixed_images_prefetch_both_and_gpu_hits_bypass_pending_reads(
    indexed_scheduler,
):
    scheduler = indexed_scheduler
    request = image_request("A", "B")
    scheduler.on_new_request(request, local_cache_hashes={"A"})
    assert len(scheduler._tiering_manager._pending_reads) == 2
    # A concurrent request may have populated B on GPU while its prefetch runs.
    assert scheduler.ensure_cache_available(request, 0, {"A", "B"})
    assert scheduler._primary.get("A") is not None


def test_gpu_cache_snapshot_is_read_only_and_expires_after_eviction():
    from vllm.v1.core.encoder_cache_manager import EncoderCacheManager

    cache = EncoderCacheManager(cache_size=2)
    resident = image_request("A")
    assert cache.can_allocate(resident, 0, 2, 0)
    cache.allocate(resident, 0)
    cache.free(resident)
    references = {key: set(refs) for key, refs in cache.cached.items()}
    reclaimable = cache.num_freeable_slots
    incoming = image_request("A", "B")
    assert cache.get_cached_mm_hashes(incoming) == frozenset({"A"})
    assert cache.cached == references
    assert cache.num_freeable_slots == reclaimable
    other = image_request("B")
    assert cache.can_allocate(other, 0, 2, 0)
    cache.allocate(other, 0)
    assert cache.get_cached_mm_hashes(incoming) == frozenset({"B"})


def test_gpu_snapshot_does_not_hide_incompatible_embedding_count():
    from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
    from vllm.v1.request import Request

    cache = EncoderCacheManager(cache_size=4)
    resident = image_request("A")
    cache.allocate(resident, 0)
    incompatible = Request(
        request_id="different-shape",
        prompt_token_ids=[0] * 3,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        mm_features=[
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier="A",
                mm_position=PlaceholderRange(offset=0, length=3),
            )
        ],
    )
    assert cache.get_cached_mm_hashes(incompatible) == frozenset()


def test_cpu_ready_gpu_miss_loads_without_filesystem_read(indexed_scheduler):
    scheduler = indexed_scheduler
    request = image_request("A")
    scheduler._primary.alloc("A", 2, (2, 32))
    scheduler._primary.mark_ready("A")
    scheduler.on_new_request(request, local_cache_hashes=set())
    assert scheduler.ensure_cache_available(request, 0, set())
    assert not scheduler._tiering_manager._pending_reads
    scheduler.update_state_after_alloc(request, 0)
    assert "A" in scheduler._pending_loads


@pytest.mark.parametrize("indexed_scheduler", [{"image": 128}], indirect=True)
def test_deepstack_promotion_sizes_blocks_like_eccpu(indexed_scheduler):
    scheduler = indexed_scheduler
    scheduler.on_new_request(image_request("A"))
    entry = scheduler._primary.get("A")
    assert entry is not None
    assert entry.shape == (2, 128)
    assert len(entry.block_ids) == scheduler._num_blocks(entry.shape) == 8
    assert len(entry.block_ids) * scheduler._primary.block_size_bytes == 512


@pytest.mark.parametrize("copies", [1, 2])
def test_arrival_holds_ready_cpu_copy_even_with_gpu_hit(indexed_scheduler, copies):
    scheduler = indexed_scheduler
    primary = scheduler._primary
    primary.alloc("A", 2, (2, 32))
    primary.mark_ready("A")
    primary.alloc("busy", primary._region.num_blocks - 2)
    request = image_request(*(["A"] * copies))
    scheduler.on_new_request(request, local_cache_hashes={"A"})
    scheduler.on_new_request(request, local_cache_hashes={"A"})
    assert primary.alloc("B", 2) is None
    assert not scheduler._tiering_manager._pending_reads
    assert scheduler.ensure_cache_available(request, 0, {"A"})
    assert primary.alloc("B", 2) is None
    if copies == 2:
        assert scheduler.ensure_cache_available(request, 2, {"A"})
        assert primary.alloc("B", 2) is None
    assert scheduler.ensure_cache_available(request, 2 * copies, set())
    scheduler._tiering_manager.release_prefetch(request.request_id)
    assert primary.alloc("B", 2) is not None


def test_cpu_full_is_feature_miss_but_submitted_read_stays_pending(
    indexed_scheduler, monkeypatch
):
    from threading import Event

    from vllm.distributed.ec_transfer.ec_connector.tiering import executor

    scheduler = indexed_scheduler
    primary = scheduler._primary
    release = Event()
    load_entry = executor.load_entry

    def blocked_load(*args, **kwargs):
        assert release.wait(timeout=10)
        return load_entry(*args, **kwargs)

    monkeypatch.setattr(executor, "load_entry", blocked_load)
    try:
        scheduler.on_new_request(image_request("A"))
        primary.alloc("busy", primary._region.num_blocks - 2)
        assert not scheduler.ensure_cache_available(image_request("A"), 0, set())
        assert scheduler.ensure_cache_available(image_request("B"), 0, set())
        assert not scheduler.has_cache_item("B")
        assert primary.get("A") is not None
        assert primary.get("B") is None
        assert len(scheduler._tiering_manager._pending_reads) == 1
    finally:
        release.set()


def test_async_arrival_prefetch_survives_one_of_two_request_cancellations(
    indexed_scheduler,
):
    from threading import Event
    from time import monotonic

    from vllm.distributed.ec_transfer.ec_connector.tiering.io import store_entry
    from vllm.distributed.ec_transfer.ec_connector.tiering.layout import (
        get_ec_entry_path,
        get_ec_temporary_path,
    )

    scheduler = indexed_scheduler
    manager = scheduler._tiering_manager
    filesystem = scheduler._filesystem
    key = build_ec_cache_key(manager._namespace_digest, "late", "image", 2, 128)
    path = get_ec_entry_path(filesystem._root_dir, manager._namespace_digest, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    store_entry(
        path, get_ec_temporary_path(path, "late"), [memoryview(bytes(128))], 128
    )
    first = image_request("late", request_id="first")
    second = image_request("late", request_id="second")
    scheduler.on_new_request(first, {"late"})
    scheduler.on_new_request(second, {"late"})
    manager.request_finished(first.request_id)
    deadline = monotonic() + 5
    while not ((entry := scheduler._primary.get("late")) is not None and entry.ready):
        assert monotonic() < deadline
        manager.poll()
        Event().wait(0.001)
    assert not entry.evictable
    assert scheduler.ensure_cache_available(second, 0, set())
    scheduler.update_state_after_alloc(second, 0)
    assert "late" in scheduler._pending_loads
    manager.request_finished(second.request_id)
