# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for FileSystemTierManager.

These tests use real disk I/O to verify the filesystem tier implementation.
The tier manager writes KV cache blocks to disk and reads them back, verifying
data integrity throughout the process.
"""

import mmap
import os
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.distributed.kv_events import MEDIUM_FS
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    OffloadingEvent,
    OffloadingKVEventsConfig,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.tiering.base import JobMetadata
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.fs.manager import (
    FileSystemTierManager,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BLOCK_ELEMENTS = 128 * mmap.PAGESIZE  # 2MB per block for pagesize 4096.
_DTYPE: torch.dtype = torch.float32
_CTX = ReqContext(req_id="test")


def _make_offloading_spec(enable_kv_cache_events: bool) -> MagicMock:
    """Mock spec with an explicit global KV events flag."""
    spec = MagicMock()
    spec.config = OffloadingConfig(
        groups=(),
        worker_kv_bytes_per_block=0,
        enable_kv_cache_events=enable_kv_cache_events,
        extra_config={},
        engine_id="test-engine",
        model=OffloadingModelConfig(name="test-model", dtype="float32"),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=1,
            tp_size=1,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            is_parallelism_agnostic=False,
        ),
    )
    spec.blocks_per_chunk = 1
    spec.kv_events_config = OffloadingKVEventsConfig(
        enable_kv_cache_events=enable_kv_cache_events,
        self_describing_kv_events=False,
    )
    return spec


_MOCK_OFFLOADING_SPEC = _make_offloading_spec(enable_kv_cache_events=False)


def key(n: int) -> OffloadKey:
    return make_offload_key(n.to_bytes(8, "big"), 0)


def make_job(
    job_id: int,
    keys: list[OffloadKey],
    block_ids: list[int] | None = None,
    is_promotion: bool = False,
) -> JobMetadata:
    if block_ids is None:
        block_ids = list(range(len(keys)))
    return JobMetadata(
        job_id=job_id,
        keys=keys,
        block_ids=np.array(block_ids, dtype=np.int64),
        is_promotion=is_promotion,
        req_context=_CTX,
    )


def drain(tier: FileSystemTierManager) -> list:
    """Block until all in-flight jobs finish, then collect results."""
    tier.drain_jobs()
    return list(tier.get_finished_jobs())


def lookup_and_wait(
    tier: FileSystemTierManager,
    keys: list[OffloadKey],
    ctx: ReqContext = _CTX,
    timeout: float = 1.0,
) -> list[LookupResult]:
    """Perform a full async lookup cycle and return resolved results."""
    for k in keys:
        tier.lookup(k, ctx)
    tier.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=()))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not tier._lookup_manager._pending_results.empty():
            break
        time.sleep(0.01)
    return [tier.lookup(k, ctx) for k in keys]


def _page_aligned_zero_tensor(
    num_blocks: int, block_elements: int, dtype: torch.dtype = _DTYPE
) -> torch.Tensor:
    page_size = mmap.PAGESIZE
    dtype_num_bytes = torch.tensor([], dtype=dtype).element_size()

    num_bytes = num_blocks * block_elements * dtype_num_bytes
    num_bytes_aligned = num_bytes + page_size
    t = torch.zeros(num_bytes_aligned, dtype=torch.uint8)

    ptr = t.data_ptr()
    alignment_offset = ptr % page_size
    # Move tensor to next page regardless.
    shift = page_size - alignment_offset
    t = t[shift : shift + num_bytes]
    return t.view(dtype).view(num_blocks, block_elements)


def _page_aligned_rand_tensor(
    num_blocks: int, block_elements: int, dtype: torch.dtype = _DTYPE
) -> torch.Tensor:
    rand_tensor = _page_aligned_zero_tensor(num_blocks, block_elements)
    rand_tensor[:] = torch.rand(num_blocks, block_elements, dtype=dtype)
    return rand_tensor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fs_tier(tmp_path):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    mock_view = memoryview(tensor.numpy())
    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=mock_view,
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=4,
        n_write_threads=4,
    )
    yield tier, tensor
    tier.shutdown()


@pytest.fixture
def fs_tier_with_events(tmp_path):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    mock_view = memoryview(tensor.numpy())
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=True),
        primary_kv_view=mock_view,
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=4,
        n_write_threads=4,
        enable_kv_events=True,
        locality="LOCAL",
    )
    yield tier
    tier.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_lookup_empty_tier(fs_tier):
    tier, _ = fs_tier
    results = lookup_and_wait(tier, [key(1), key(2)])
    assert results == [LookupResult.MISS, LookupResult.MISS]


def test_store_creates_file_and_lookup_succeeds(fs_tier):
    tier, _ = fs_tier
    job = make_job(1, [key(1)], [0])
    tier.submit_store(job)
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    assert lookup_and_wait(tier, [key(1)]) == [LookupResult.HIT]
    dest = tier.file_mapper.get_file_name(key(1))
    assert os.path.exists(dest), f"Expected file at {dest}"


def test_store_then_load_roundtrip(fs_tier):
    tier, _ = fs_tier
    job_s = make_job(1, [key(1), key(2)], [0, 1])
    tier.submit_store(job_s)
    store_results = drain(tier)
    assert all(r.success for r in store_results)

    assert lookup_and_wait(tier, [key(1), key(2)]) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]

    job_l = make_job(2, [key(1), key(2)], [2, 3], is_promotion=True)
    tier.submit_load(job_l)
    load_results = drain(tier)
    assert all(r.success for r in load_results)
    # Blocks stay on disk after load
    assert lookup_and_wait(tier, [key(1), key(2)]) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]


def test_restore_heals_wrong_size_file(fs_tier):
    """A wrong-size/corrupt on-disk file must self-heal on re-store. Size-aware
    dedup falls through the skip-if-exists short-circuit and the atomic replace
    overwrites the bad file. A bare existence check would poison the key
    forever: size-validated lookup MISSes the bad file, but a re-store would
    short-circuit and never overwrite it, leaving the key un-cacheable."""
    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert all(r.success for r in drain(tier))
    path = tier.file_mapper.get_file_name(key(1))
    good_size = os.path.getsize(path)
    assert good_size == tier._block_size

    # Externally corrupt/truncate the stored file to the wrong size.
    with open(path, "wb") as f:
        f.write(b"x" * 10)
    assert os.path.getsize(path) == 10

    # Size-validated lookup now misses the corrupt file.
    miss_ctx = ReqContext(req_id="corrupt-miss")
    assert lookup_and_wait(tier, [key(1)], ctx=miss_ctx) == [LookupResult.MISS]
    # Release that request's cached negative verdict so a later lookup re-runs
    # a fresh disk check (the lookup cache is per-key, independent of stores).
    tier.on_request_finished(miss_ctx)

    # Re-store the same key: the file must heal back to the correct size (the
    # atomic replace overwrites it) instead of short-circuiting on existence.
    tier.submit_store(make_job(3, [key(1)], [0]))
    assert all(r.success for r in drain(tier))
    assert os.path.getsize(path) == good_size

    # ...and a fresh lookup is a HIT again (self-healed).
    healed_ctx = ReqContext(req_id="healed")
    assert lookup_and_wait(tier, [key(1)], ctx=healed_ctx) == [LookupResult.HIT]


def test_invalid_path_raises_at_construction():
    """Construction must fail immediately when the config file cannot be written."""
    tensor = _page_aligned_zero_tensor(32, _BLOCK_ELEMENTS)
    mock_view = memoryview(tensor.numpy())

    with pytest.raises(OSError):
        FileSystemTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=mock_view,
            tier_type="fs",
            root_dir="/dev/null/invalid_path",
        )


@pytest.mark.parametrize("locality", ["local", ""])
def test_invalid_locality_raises_at_construction(tmp_path, locality):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)

    with pytest.raises(ValueError, match="Locality"):
        FileSystemTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=memoryview(tensor.numpy()),
            tier_type="fs",
            root_dir=str(tmp_path),
            locality=locality,
        )


def test_factory_forwards_locality_to_fs_tier(tmp_path):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    tier = SecondaryTierFactory.create_secondary_tier(
        {
            "type": "fs",
            "root_dir": str(tmp_path),
            "n_read_threads": 1,
            "n_write_threads": 1,
            "locality": "LOCAL",
        },
        memoryview(tensor.numpy()),
        _MOCK_OFFLOADING_SPEC,
    )
    try:
        assert isinstance(tier, FileSystemTierManager)
        assert tier.locality is Locality.LOCAL
    finally:
        tier.shutdown()


def test_failed_load_missing_file(fs_tier):
    """Test that loading a block whose file does not exist results in a failed job."""
    tier, _ = fs_tier
    job = make_job(1, [key(99)], [0], is_promotion=True)
    tier.submit_load(job)
    results = drain(tier)
    assert len(results) == 1
    assert not results[0].success


def test_multiple_jobs_tracked_independently(fs_tier):
    tier, _ = fs_tier
    job1 = make_job(1, [key(1)], [0])
    job2 = make_job(2, [key(2)], [1])
    tier.submit_store(job1)
    tier.submit_store(job2)
    results = drain(tier)
    job_ids = {r.job_id for r in results}
    assert job_ids == {1, 2}
    assert lookup_and_wait(tier, [key(1), key(2)]) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]


def test_multi_block_job_partial_failure(fs_tier):
    """A load job where one block file is missing yields a single failed JobResult."""
    tier, _ = fs_tier
    # Store two of three keys
    tier.submit_store(make_job(1, [key(10), key(11)], [0, 1]))
    assert all(r.success for r in drain(tier))

    # Load all three — key(99) was never stored
    tier.submit_load(
        make_job(2, [key(10), key(11), key(99)], [0, 1, 2], is_promotion=True)
    )
    results = drain(tier)

    assert len(results) == 1
    assert results[0].job_id == 2
    assert not results[0].success


def test_shutdown_discards_pending_tasks(fs_tier):
    """Shutdown clears both queues and stops all worker threads without draining."""
    tier, _ = fs_tier
    # Submit many tasks to ensure some remain pending
    for i in range(10):
        tier.submit_store(make_job(i, [key(i)], [i % 4]))

    # Shutdown immediately without draining
    tier.shutdown()

    # Verify queues are cleared and threads stopped
    assert len(tier._pool._load_q) == 0
    assert len(tier._pool._store_q) == 0
    assert all(not t.is_alive() for t in tier._pool._threads)


def test_store_load_data_integrity(fs_tier):
    """Data written by store must be exactly recovered by load."""
    tier, tensor = fs_tier
    # Populate tensor with random data
    tensor[:] = _page_aligned_rand_tensor(4, _BLOCK_ELEMENTS)

    # Store first 2 blocks
    num_store = 2
    expected = tensor[:num_store].clone()

    store_ids = list(range(num_store))
    keys = [key(i) for i in range(num_store)]

    tier.submit_store(make_job(1, keys, store_ids))
    results = drain(tier)
    assert all(r.success for r in results)

    # Overwrite source blocks to prove data is read from disk
    tensor[:num_store] = 0.0

    # Load into last 2 blocks
    load_ids = [2, 3]
    tier.submit_load(make_job(2, keys, load_ids, is_promotion=True))
    results = drain(tier)
    assert all(r.success for r in results)

    for i, bid in enumerate(load_ids):
        assert torch.allclose(tensor[bid], expected[i]), (
            f"Block {bid} data mismatch after store+load"
        )


def test_wait_idle_blocks_until_tasks_complete():
    """wait_idle must not return while a task is still in flight."""
    pool = DualQueueThreadPool(n_read_threads=1, n_write_threads=1)
    gate = threading.Event()
    pool.enqueue_store(job_id=1, n_tasks=1, tasks=[lambda: gate.wait(timeout=5.0)])

    waiter = threading.Thread(target=pool.wait_idle)
    waiter.start()
    try:
        waiter.join(timeout=0.2)
        assert waiter.is_alive(), "wait_idle returned before task completed"
        gate.set()
        waiter.join(timeout=5.0)
        assert not waiter.is_alive(), "wait_idle did not unblock"
    finally:
        gate.set()
        pool.shutdown(wait=True)
        waiter.join(timeout=5.0)


def test_batch_lookup_c_extension(tmp_path):
    """Validates batch_lookup_C: it now returns st_size per path (-1 for
    anything it cannot stat), so the caller can compare against the expected
    block size. Covers empty, single, sized/empty/missing, mixed ordering,
    and input type validation."""
    try:
        from vllm.fs_io_C import batch_lookup as batch_lookup_C
    except ImportError:
        pytest.skip("fs_io_C extension not built")

    empty = str(tmp_path / "empty.bin")
    open(empty, "w").close()
    sized = str(tmp_path / "sized.bin")
    with open(sized, "wb") as f:
        f.write(b"\x00" * 4096)
    missing = str(tmp_path / "missing.bin")

    # Empty list
    assert batch_lookup_C([]) == []

    # Sizes are reported exactly; a zero-byte file is 0, not -1.
    assert batch_lookup_C([sized]) == [4096]
    assert batch_lookup_C([empty]) == [0]
    # Un-stat'able paths are -1 (never a valid block size -> a clean miss).
    assert batch_lookup_C([missing]) == [-1]

    # Mixed — verifies index ordering is preserved.
    assert batch_lookup_C([sized, missing, empty]) == [4096, -1, 0]

    # size == expected is the exact predicate FsAsyncLookupManager applies.
    assert [s == 4096 for s in batch_lookup_C([sized, empty, missing])] == [
        True,
        False,
        False,
    ]

    # Input validation: non-list argument
    with pytest.raises(TypeError):
        batch_lookup_C(("/tmp/foo",))
    with pytest.raises(TypeError):
        batch_lookup_C(None)

    # Input validation: non-str elements in list
    with pytest.raises(TypeError):
        batch_lookup_C([None])
    with pytest.raises(TypeError):
        batch_lookup_C([b"/tmp/foo"])
    with pytest.raises(TypeError):
        batch_lookup_C([42])
    with pytest.raises(TypeError):
        batch_lookup_C([sized, None])  # valid first, invalid mid-list


def test_batch_lookup_prefers_c_ext_when_available(fs_tier, monkeypatch):
    """FsAsyncLookupManager.batch_lookup routes through batch_lookup_C when the
    extension is present, comparing each returned size against the block size
    (independent of whether the extension is actually built in this env)."""
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    tier, _ = fs_tier
    expected = tier._block_size
    calls = {}

    def fake_batch_lookup(paths):
        calls["paths"] = list(paths)
        # first path the right size (HIT), second truncated, third missing
        return [expected, expected - 1, -1]

    monkeypatch.setattr(mgr_mod, "_HAS_BATCH_LOOKUP_C", True)
    monkeypatch.setattr(mgr_mod, "batch_lookup_C", fake_batch_lookup, raising=False)

    results = list(tier._lookup_manager.batch_lookup([key(1), key(2), key(3)], _CTX))
    assert results == [True, False, False]
    assert len(calls["paths"]) == 3


def test_batch_lookup_validates_size(fs_tier):
    """Lookups validate file size, not bare existence: a truncated or
    foreign-layout file is an up-front MISS instead of a HIT that fails
    fatally at load time."""
    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert all(r.success for r in drain(tier))

    results = lookup_and_wait(tier, [key(1), key(2)])
    assert results == [LookupResult.HIT, LookupResult.MISS]

    # Truncate the stored file: still exists, wrong size -> MISS once the
    # cached verdict is released and a fresh batch_lookup runs. (The cache
    # is per-key: while any request holds the entry, even NEW requests are
    # served the stale verdict - which is what the tier's self-invalidation
    # on a failed load breaks; see test_failed_load_invalidates_cached_verdict.)
    path = tier.file_mapper.get_file_name(key(1))
    with open(path, "wb") as f:
        f.write(b"x" * 10)
    tier.on_request_finished(_CTX)  # release the cached entry
    fresh = ReqContext(req_id="fresh-after-truncate")
    assert lookup_and_wait(tier, [key(1)], ctx=fresh) == [LookupResult.MISS]


def test_failed_load_invalidates_cached_verdict(fs_tier):
    """Regression test for the failed-load livelock (#49176): a cached HIT
    must not survive a failed load of the same key. The tier self-invalidates
    its own stale verdict from get_finished_jobs() (drained here); previously
    the cached verdict was served for the life of the requesting request while
    the backing file was already gone, so the scheduler re-initiated the same
    doomed promotion every step."""
    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert all(r.success for r in drain(tier))

    ctx = ReqContext(req_id="livelock-req")
    assert lookup_and_wait(tier, [key(1)], ctx=ctx) == [LookupResult.HIT]

    # The file goes bad AFTER the verdict is cached, then the promotion the
    # HIT triggered fails (short read). draining get_finished_jobs() surfaces
    # the failure and the tier self-invalidates the stale verdict on the
    # scheduler thread.
    path = tier.file_mapper.get_file_name(key(1))
    with open(path, "wb") as f:
        f.write(b"x" * 10)
    tier.submit_load(make_job(2, [key(1)], [0], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1 and not results[0].success

    # After the failed promotion the SAME request's lookup must re-resolve to
    # MISS (size-validated) instead of serving the stale HIT.
    assert lookup_and_wait(tier, [key(1)], ctx=ctx) == [LookupResult.MISS]


def test_successful_load_keeps_cached_verdict(fs_tier):
    """A successful promotion must NOT invalidate the cached HIT: the block is
    still on disk and a subsequent lookup for the same request stays a HIT."""
    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert all(r.success for r in drain(tier))

    ctx = ReqContext(req_id="ok-req")
    assert lookup_and_wait(tier, [key(1)], ctx=ctx) == [LookupResult.HIT]

    tier.submit_load(make_job(2, [key(1)], [0], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1 and results[0].success

    # Verdict intact; the block is still present.
    assert lookup_and_wait(tier, [key(1)], ctx=ctx) == [LookupResult.HIT]


def test_short_read_leaves_file_untouched(fs_tier):
    """A short-read load failure must not modify or delete the file: with
    size-validated lookup a truncated file already misses at lookup, so there
    is nothing to quarantine. The file (and its wrong-size content) survives,
    and the next lookup treats it as a clean MISS."""
    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert all(r.success for r in drain(tier))
    path = tier.file_mapper.get_file_name(key(1))
    with open(path, "wb") as f:
        f.write(b"x" * 10)

    tier.submit_load(make_job(2, [key(1)], [0], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1 and not results[0].success

    # The file is left exactly as it was — never quarantined or removed.
    assert os.path.exists(path)
    with open(path, "rb") as f:
        assert f.read() == b"x" * 10
    assert not os.path.exists(path + ".bad")

    # A fresh request's size-validated lookup treats the wrong-size file as a
    # miss.
    fresh = ReqContext(req_id="fresh-after-short-read")
    assert lookup_and_wait(tier, [key(1)], ctx=fresh) == [LookupResult.MISS]


def test_transient_load_failure_leaves_file(fs_tier, monkeypatch):
    """A transient host error (fd exhaustion) fails the job but must NOT
    destroy the block file: the content was never proven wrong, and the
    old delete-on-any-failure behavior turned hiccups into data loss."""
    import errno as _errno

    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert all(r.success for r in drain(tier))
    path = tier.file_mapper.get_file_name(key(1))
    with open(path, "rb") as f:
        original = f.read()

    def _raise_emfile(*args, **kwargs):
        raise OSError(_errno.EMFILE, "Too many open files")

    monkeypatch.setattr("vllm.v1.kv_offload.tiering.fs.io.os.readv", _raise_emfile)
    tier.submit_load(make_job(2, [key(1)], [0], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1 and not results[0].success
    monkeypatch.undo()

    assert os.path.exists(path)
    with open(path, "rb") as f:
        assert f.read() == original


# ---------------------------------------------------------------------------
# KV events
# ---------------------------------------------------------------------------


def test_successful_store_emits_stored_event(fs_tier_with_events):
    """A completed store job emits one stored event with the job's keys."""
    tier = fs_tier_with_events
    keys = [key(1), key(2)]
    tier.submit_store(make_job(1, keys, [0, 1]))
    assert all(r.success for r in drain(tier))

    events = list(tier.take_events())
    assert len(events) == 1
    assert events[0].keys == keys
    # Literal medium pins the wire contract, not just the constant choice.
    assert events[0].medium == "FS"
    assert events[0].locality is Locality.LOCAL
    assert not events[0].removed
    # take_events drains the buffer.
    assert list(tier.take_events()) == []


@pytest.mark.parametrize(
    ("locality", "expected"),
    [(None, None), ("REMOTE", Locality.REMOTE)],
)
def test_store_event_uses_configured_locality(tmp_path, locality, expected):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    locality_config = {} if locality is None else {"locality": locality}
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=True),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        enable_kv_events=True,
        **locality_config,
    )
    try:
        tier.submit_store(make_job(1, [key(1)], [0]))
        assert all(r.success for r in drain(tier))

        events = list(tier.take_events())
        assert len(events) == 1
        assert events[0].locality is expected
    finally:
        tier.shutdown()


def test_load_job_emits_no_event(fs_tier_with_events):
    tier = fs_tier_with_events
    tier.submit_store(make_job(1, [key(1)], [0]))
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    list(tier.take_events())

    tier.submit_load(make_job(2, [key(1)], [1], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    assert list(tier.take_events()) == []


def test_mixed_job_results_emit_event_only_for_successful_job(
    fs_tier_with_events, monkeypatch
):
    """With a failed and a successful store job in flight, exactly one event
    is emitted and its keys belong to the successful job."""
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    tier = fs_tier_with_events
    failing_path = tier.file_mapper.get_file_name(key(1))
    original_store_block = mgr_mod.store_block

    def flaky_store_block(dest_path, *args, **kwargs):
        if dest_path == failing_path:
            raise OSError("injected store failure")
        return original_store_block(dest_path, *args, **kwargs)

    monkeypatch.setattr(mgr_mod, "store_block", flaky_store_block)

    tier.submit_store(make_job(1, [key(1)], [0]))
    tier.submit_store(make_job(2, [key(2)], [1]))
    results = drain(tier)
    assert len(results) == 2
    by_id = {r.job_id: r for r in results}
    assert not by_id[1].success
    assert by_id[2].success

    events = list(tier.take_events())
    assert len(events) == 1
    assert events[0].keys == [key(2)]


def test_partially_failed_store_emits_no_event(fs_tier_with_events, monkeypatch):
    """A store job with any failed block emits no event for the whole job."""
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    tier = fs_tier_with_events
    failing_path = tier.file_mapper.get_file_name(key(2))
    original_store_block = mgr_mod.store_block

    def flaky_store_block(dest_path, *args, **kwargs):
        if dest_path == failing_path:
            raise OSError("injected store failure")
        return original_store_block(dest_path, *args, **kwargs)

    monkeypatch.setattr(mgr_mod, "store_block", flaky_store_block)

    tier.submit_store(make_job(1, [key(1), key(2)], [0, 1]))
    results = drain(tier)
    assert len(results) == 1
    assert not results[0].success
    assert list(tier.take_events()) == []
    assert tier._store_job_keys == {}


def test_events_disabled_by_default(fs_tier):
    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    assert tier.events is None
    assert tier._store_job_keys == {}
    assert list(tier.take_events()) == []


def test_events_require_global_kv_events_flag(tmp_path):
    """Tier-level opt-in alone is not enough; the global flag gates events."""
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=False),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        enable_kv_events=True,
    )
    try:
        assert tier.events is None
        tier.submit_store(make_job(1, [key(1)], [0]))
        results = drain(tier)
        assert len(results) == 1
        assert results[0].success
        assert list(tier.take_events()) == []
        assert tier._store_job_keys == {}
    finally:
        tier.shutdown()


def test_cascade_store_emits_fs_event_through_tiering_manager(tmp_path):
    """A GPU->CPU->fs cascade surfaces the tier-owned FS stored event via the
    TieringOffloadingManager's aggregated take_events()."""
    from vllm.v1.kv_offload.tiering.manager import (
        CPUPrimaryTierOffloadingManager,
        TieringOffloadingManager,
    )

    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    view = memoryview(tensor.numpy())
    mock_region = MagicMock()
    mock_region.create_kv_memoryview.return_value = view
    primary = CPUPrimaryTierOffloadingManager(num_blocks=4, mmap_region=mock_region)
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=True),
        primary_kv_view=primary.get_kv_memoryview(),
        tier_type="fs",
        root_dir=str(tmp_path),
        enable_kv_events=True,
    )
    manager = TieringOffloadingManager(primary_tier=primary, secondary_tiers=[tier])
    try:
        keys = [key(1), key(2)]
        manager.on_new_request(_CTX)
        assert manager.prepare_store(keys, _CTX) is not None
        manager.complete_store(keys, _CTX)  # cascades to the fs tier

        events: list[OffloadingEvent] = []
        ctx = ScheduleEndContext(new_req_ids=[], preempted_req_ids=())
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not events:
            manager.on_schedule_end(ctx)
            events.extend(manager.take_events())
            time.sleep(0.01)

        fs_events = [e for e in events if e.medium == MEDIUM_FS]
        assert len(fs_events) == 1
        assert set(fs_events[0].keys) == set(keys)
        assert not fs_events[0].removed
    finally:
        tier.shutdown()
