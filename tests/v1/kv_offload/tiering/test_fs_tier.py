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

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.tiering.base import JobMetadata
from vllm.v1.kv_offload.tiering.fs.common import FileSystemLoadStoreSpec
from vllm.v1.kv_offload.tiering.fs.manager import (
    FileSystemTierManager,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool
from vllm.v1.kv_offload.tiering.fs.worker import FileSystemWorkerTransferHandler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BLOCK_ELEMENTS = 128 * mmap.PAGESIZE  # 2MB per block for pagesize 4096.
_DTYPE: torch.dtype = torch.float32
_CTX = ReqContext(req_id="test")

_MOCK_VLLM_CONFIG = MagicMock()
_MOCK_VLLM_CONFIG.model_config.model = "test-model"
_MOCK_VLLM_CONFIG.cache_config.block_size = 16
_MOCK_VLLM_CONFIG.cache_config.cache_dtype = "torch.float32"
_MOCK_VLLM_CONFIG.parallel_config.tensor_parallel_size = 1
_MOCK_VLLM_CONFIG.parallel_config.pipeline_parallel_size = 1
_MOCK_VLLM_CONFIG.parallel_config.prefill_context_parallel_size = 1
_MOCK_VLLM_CONFIG.parallel_config.decode_context_parallel_size = 1
_MOCK_VLLM_CONFIG.parallel_config.rank = 0
_MOCK_VLLM_CONFIG.parallel_config.world_size = 1

_MOCK_KV_CACHE_CONFIG = MagicMock()
_MOCK_KV_CACHE_CONFIG.kv_cache_groups = []

_MOCK_OFFLOADING_SPEC = MagicMock()
_MOCK_OFFLOADING_SPEC.vllm_config = _MOCK_VLLM_CONFIG
_MOCK_OFFLOADING_SPEC.kv_cache_config = _MOCK_KV_CACHE_CONFIG
_MOCK_OFFLOADING_SPEC.block_size_factor = 1


def _make_offloading_spec(world_size: int = 1, rank: int = 0):
    vllm_config = MagicMock()
    vllm_config.instance_id = "test-instance"
    vllm_config.model_config.model = "test-model"
    vllm_config.cache_config.block_size = 16
    vllm_config.cache_config.cache_dtype = "torch.float32"
    vllm_config.parallel_config.tensor_parallel_size = world_size
    vllm_config.parallel_config.pipeline_parallel_size = 1
    vllm_config.parallel_config.prefill_context_parallel_size = 1
    vllm_config.parallel_config.decode_context_parallel_size = 1
    vllm_config.parallel_config.rank = rank
    vllm_config.parallel_config.world_size = world_size
    vllm_config.use_v2_model_runner = False

    offloading_spec = MagicMock()
    offloading_spec.vllm_config = vllm_config
    offloading_spec.kv_cache_config = _MOCK_KV_CACHE_CONFIG
    offloading_spec.block_size_factor = 1
    return offloading_spec


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


def drain(tier: FileSystemTierManager, max_rounds: int = 100) -> list:
    """
    Call get_finished_jobs() repeatedly until no new results arrive for 20
    consecutive rounds or max_rounds is reached.
    """
    results = []
    idle = 0
    for _ in range(max_rounds):
        time.sleep(0.01)
        new = list(tier.get_finished_jobs())
        results.extend(new)
        if new:
            idle = 0
        else:
            idle += 1
            if idle >= 20:
                break
    return results


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


def test_worker_store_commit_creates_visible_file(tmp_path):
    tensor = _page_aligned_zero_tensor(1, mmap.PAGESIZE)
    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        worker_transfers=True,
    )
    try:
        job = make_job(11, [key(1)], [0])
        src_spec, dst_spec = tier.build_worker_store_transfer(job)

        assert tier.uses_worker_transfers()
        assert isinstance(src_spec, CPULoadStoreSpec)
        assert isinstance(dst_spec, FileSystemLoadStoreSpec)
        assert dst_spec.temp_file_paths is not None
        final_path = dst_spec.file_paths[0]
        temp_path = dst_spec.temp_file_paths[0]

        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(bytes([7]) * dst_spec.block_size)

        assert not os.path.exists(final_path)
        tier.complete_worker_store(job, success=True)

        assert os.path.exists(final_path)
        assert not os.path.exists(temp_path)
        assert lookup_and_wait(tier, [key(1)]) == [LookupResult.HIT]
    finally:
        tier.shutdown()


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_worker_transfer_lookup_requires_all_rank_files(
    tmp_path, monkeypatch, use_c_ext
):
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    if use_c_ext and not mgr_mod._HAS_BATCH_LOOKUP_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(mgr_mod, "_HAS_BATCH_LOOKUP_C", use_c_ext)

    tensor = _page_aligned_zero_tensor(1, mmap.PAGESIZE)
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(world_size=2),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        worker_transfers=True,
    )
    try:
        missing_rank_key = key(1)
        complete_key = key(2)
        _, missing_rank_spec = tier.build_worker_store_transfer(
            make_job(21, [missing_rank_key], [0])
        )
        _, complete_spec = tier.build_worker_store_transfer(
            make_job(22, [complete_key], [0])
        )

        for path in [missing_rank_spec.file_paths[0], *complete_spec.file_paths]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(bytes([3]) * missing_rank_spec.block_size)

        assert lookup_and_wait(tier, [missing_rank_key, complete_key]) == [
            LookupResult.MISS,
            LookupResult.HIT,
        ]
    finally:
        tier.shutdown()


def test_worker_store_commit_missing_rank_temp_is_discarded(tmp_path):
    tensor = _page_aligned_zero_tensor(1, mmap.PAGESIZE)
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(world_size=2),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        worker_transfers=True,
    )
    try:
        stored_key = key(1)
        job = make_job(31, [stored_key], [0])
        _, dst_spec = tier.build_worker_store_transfer(job)
        assert dst_spec.temp_file_paths is not None

        rank0_temp = dst_spec.temp_file_paths[0]
        os.makedirs(os.path.dirname(rank0_temp), exist_ok=True)
        with open(rank0_temp, "wb") as f:
            f.write(bytes([7]) * dst_spec.block_size)

        tier.complete_worker_store(job, success=True)

        assert not os.path.exists(rank0_temp)
        assert all(not os.path.exists(path) for path in dst_spec.file_paths)
        assert lookup_and_wait(tier, [stored_key]) == [LookupResult.MISS]
    finally:
        tier.shutdown()


def test_filesystem_worker_handler_store_and_load_rank_slice(tmp_path):
    store_t0 = torch.arange(0, 8, dtype=torch.int8).view(2, 4)
    store_t1 = torch.arange(10, 18, dtype=torch.int8).view(2, 4)
    handler = FileSystemWorkerTransferHandler(
        [store_t0, store_t1], rank=0, n_read_threads=1, n_write_threads=1
    )
    final_path = str(tmp_path / "block.bin")
    temp_path = str(tmp_path / "block.tmp")
    try:
        assert handler.submit_store(
            1,
            CPULoadStoreSpec([1]),
            FileSystemLoadStoreSpec(
                file_paths=[final_path],
                temp_file_paths=[temp_path],
                block_size=8,
            ),
        )
        handler.wait()
        results = handler.get_finished()
        assert len(results) == 1
        assert results[0].job_id == 1
        assert results[0].success
        assert os.path.exists(temp_path)
        assert (tmp_path / "block.tmp").read_bytes() == bytes(
            store_t0[1].tolist() + store_t1[1].tolist()
        )
        os.replace(temp_path, final_path)

        load_t0 = torch.zeros((2, 4), dtype=torch.int8)
        load_t1 = torch.zeros((2, 4), dtype=torch.int8)
        load_handler = FileSystemWorkerTransferHandler(
            [load_t0, load_t1], rank=0, n_read_threads=1, n_write_threads=1
        )
        try:
            assert load_handler.submit_load(
                2,
                FileSystemLoadStoreSpec(file_paths=[final_path], block_size=8),
                CPULoadStoreSpec([0]),
            )
            load_handler.wait()
            results = load_handler.get_finished()
            assert len(results) == 1
            assert results[0].job_id == 2
            assert results[0].success
            assert torch.equal(load_t0[0], store_t0[1])
            assert torch.equal(load_t1[0], store_t1[1])
        finally:
            load_handler.shutdown()
    finally:
        handler.shutdown()


def test_filesystem_worker_handler_loads_global_rank_slice(tmp_path):
    rank_payload = bytes(range(20, 28))
    rank1_file = tmp_path / "rank1.bin"
    rank1_file.write_bytes(bytes(8) + rank_payload)

    load_t0 = torch.zeros((1, 4), dtype=torch.int8)
    load_t1 = torch.zeros((1, 4), dtype=torch.int8)
    handler = FileSystemWorkerTransferHandler(
        [load_t0, load_t1], rank=1, n_read_threads=1, n_write_threads=1
    )
    try:
        assert handler.submit_load(
            7,
            FileSystemLoadStoreSpec(
                file_paths=[str(tmp_path / "rank0.bin"), str(rank1_file)],
                block_size=16,
                num_ranks=2,
            ),
            CPULoadStoreSpec([0]),
        )
        handler.wait()
        results = handler.get_finished()
        assert len(results) == 1
        assert results[0].job_id == 7
        assert results[0].success
        assert bytes(load_t0[0].tolist() + load_t1[0].tolist()) == rank_payload
    finally:
        handler.shutdown()


def test_worker_transfer_store_then_load_all_rank_slices(tmp_path):
    manager_tensor = _page_aligned_zero_tensor(1, mmap.PAGESIZE)
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(world_size=2),
        primary_kv_view=memoryview(manager_tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        worker_transfers=True,
    )

    rank0_src0 = torch.arange(0, 8, dtype=torch.int8).view(2, 4)
    rank0_src1 = torch.arange(10, 18, dtype=torch.int8).view(2, 4)
    rank1_src0 = torch.arange(20, 28, dtype=torch.int8).view(2, 4)
    rank1_src1 = torch.arange(30, 38, dtype=torch.int8).view(2, 4)
    store_rank0 = FileSystemWorkerTransferHandler(
        [rank0_src0, rank0_src1], rank=0, n_read_threads=1, n_write_threads=1
    )
    store_rank1 = FileSystemWorkerTransferHandler(
        [rank1_src0, rank1_src1], rank=1, n_read_threads=1, n_write_threads=1
    )

    try:
        stored_key = key(71)
        store_job = make_job(71, [stored_key], [1])
        store_src, store_dst = tier.build_worker_store_transfer(store_job)

        assert store_rank0.submit_store(71, store_src, store_dst)
        assert store_rank1.submit_store(71, store_src, store_dst)
        store_rank0.wait()
        store_rank1.wait()
        assert [r.success for r in store_rank0.get_finished()] == [True]
        assert [r.success for r in store_rank1.get_finished()] == [True]

        tier.complete_worker_store(store_job, success=True)
        assert all(os.path.exists(path) for path in store_dst.file_paths)

        rank0_dst0 = torch.zeros((1, 4), dtype=torch.int8)
        rank0_dst1 = torch.zeros((1, 4), dtype=torch.int8)
        rank1_dst0 = torch.zeros((1, 4), dtype=torch.int8)
        rank1_dst1 = torch.zeros((1, 4), dtype=torch.int8)
        load_rank0 = FileSystemWorkerTransferHandler(
            [rank0_dst0, rank0_dst1], rank=0, n_read_threads=1, n_write_threads=1
        )
        load_rank1 = FileSystemWorkerTransferHandler(
            [rank1_dst0, rank1_dst1], rank=1, n_read_threads=1, n_write_threads=1
        )
        try:
            load_job = make_job(72, [stored_key], [0], is_promotion=True)
            load_src, load_dst = tier.build_worker_load_transfer(load_job)

            assert load_rank0.submit_load(72, load_src, load_dst)
            assert load_rank1.submit_load(72, load_src, load_dst)
            load_rank0.wait()
            load_rank1.wait()
            assert [r.success for r in load_rank0.get_finished()] == [True]
            assert [r.success for r in load_rank1.get_finished()] == [True]

            assert torch.equal(rank0_dst0[0], rank0_src0[1])
            assert torch.equal(rank0_dst1[0], rank0_src1[1])
            assert torch.equal(rank1_dst0[0], rank1_src0[1])
            assert torch.equal(rank1_dst1[0], rank1_src1[1])
        finally:
            load_rank0.shutdown()
            load_rank1.shutdown()
    finally:
        store_rank0.shutdown()
        store_rank1.shutdown()
        tier.shutdown()


def test_filesystem_worker_handler_wait_cleans_flushed_store_temp(tmp_path):
    tensor = torch.arange(0, 8, dtype=torch.int8).view(2, 4)
    handler = FileSystemWorkerTransferHandler(
        [tensor], rank=0, n_read_threads=1, n_write_threads=1
    )
    final_path = str(tmp_path / "block.bin")
    temp_path = str(tmp_path / "block.tmp")
    try:
        assert handler.submit_store(
            8,
            CPULoadStoreSpec([1]),
            FileSystemLoadStoreSpec(
                file_paths=[final_path],
                temp_file_paths=[temp_path],
                block_size=4,
            ),
        )
        handler.wait({8})
        assert not os.path.exists(temp_path)
        results = handler.get_finished()
        assert len(results) == 1
        assert results[0].job_id == 8
        assert results[0].success
    finally:
        handler.shutdown()


def test_filesystem_worker_handler_uses_global_rank_for_path_slice(tmp_path):
    store_t0 = torch.arange(0, 8, dtype=torch.int8).view(2, 4)
    store_t1 = torch.arange(10, 18, dtype=torch.int8).view(2, 4)
    handler = FileSystemWorkerTransferHandler(
        [store_t0, store_t1], rank=1, n_read_threads=1, n_write_threads=1
    )
    rank0_final = str(tmp_path / "rank0.bin")
    rank1_final = str(tmp_path / "rank1.bin")
    rank0_temp = str(tmp_path / "rank0.tmp")
    rank1_temp = str(tmp_path / "rank1.tmp")
    try:
        assert handler.submit_store(
            3,
            CPULoadStoreSpec([1]),
            FileSystemLoadStoreSpec(
                file_paths=[rank0_final, rank1_final],
                temp_file_paths=[rank0_temp, rank1_temp],
                block_size=16,
                num_ranks=2,
            ),
        )
        handler.wait()
        results = handler.get_finished()
        assert len(results) == 1
        assert results[0].job_id == 3
        assert results[0].success
        assert not os.path.exists(rank0_temp)
        assert os.path.exists(rank1_temp)
        assert (tmp_path / "rank1.tmp").read_bytes() == (
            bytes(8) + bytes(store_t0[1].tolist() + store_t1[1].tolist())
        )
    finally:
        handler.shutdown()


def test_filesystem_worker_handler_rejects_rank_out_of_range(tmp_path):
    tensor = torch.zeros((1, 4), dtype=torch.int8)
    handler = FileSystemWorkerTransferHandler(
        [tensor], rank=2, n_read_threads=1, n_write_threads=1
    )
    try:
        with pytest.raises(ValueError, match="outside num_ranks"):
            handler.submit_store(
                4,
                CPULoadStoreSpec([0]),
                FileSystemLoadStoreSpec(
                    file_paths=[
                        str(tmp_path / "rank0.bin"),
                        str(tmp_path / "rank1.bin"),
                    ],
                    temp_file_paths=[
                        str(tmp_path / "rank0.tmp"),
                        str(tmp_path / "rank1.tmp"),
                    ],
                    block_size=8,
                    num_ranks=2,
                ),
            )
    finally:
        handler.shutdown()


def test_filesystem_worker_handler_rejects_uneven_rank_paths(tmp_path):
    tensor = torch.zeros((1, 4), dtype=torch.int8)
    handler = FileSystemWorkerTransferHandler(
        [tensor], rank=1, n_read_threads=1, n_write_threads=1
    )
    try:
        with pytest.raises(ValueError, match="Cannot split"):
            handler.submit_load(
                5,
                FileSystemLoadStoreSpec(
                    file_paths=[
                        str(tmp_path / "rank0.bin"),
                        str(tmp_path / "rank1.bin"),
                        str(tmp_path / "extra.bin"),
                    ],
                    block_size=8,
                    num_ranks=2,
                ),
                CPULoadStoreSpec([0]),
            )
    finally:
        handler.shutdown()


def test_filesystem_worker_handler_rejects_invalid_num_ranks(tmp_path):
    tensor = torch.zeros((1, 4), dtype=torch.int8)
    handler = FileSystemWorkerTransferHandler(
        [tensor], rank=0, n_read_threads=1, n_write_threads=1
    )
    try:
        with pytest.raises(ValueError, match="Invalid num_ranks"):
            handler.submit_store(
                6,
                CPULoadStoreSpec([0]),
                FileSystemLoadStoreSpec(
                    file_paths=[str(tmp_path / "rank0.bin")],
                    temp_file_paths=[str(tmp_path / "rank0.tmp")],
                    block_size=4,
                    num_ranks=0,
                ),
            )
    finally:
        handler.shutdown()


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
    """Validates batch_lookup_C: empty, single, all-existing, all-missing,
    mixed ordering, and input type validation."""
    try:
        from vllm.fs_io_C import batch_lookup as batch_lookup_C
    except ImportError:
        pytest.skip("fs_io_C extension not built")

    # Setup
    all_exist = [str(tmp_path / f"e{i}.bin") for i in range(3)]
    for p in all_exist:
        open(p, "w").close()
    all_missing = [str(tmp_path / f"m{i}.bin") for i in range(3)]

    # Empty list
    assert batch_lookup_C([]) == []

    # Single existing / missing
    assert batch_lookup_C([all_exist[0]]) == [True]
    assert batch_lookup_C([all_missing[0]]) == [False]

    # All existing / all missing
    assert batch_lookup_C(all_exist) == [True, True, True]
    assert batch_lookup_C(all_missing) == [False, False, False]

    # Mixed — verifies index ordering is preserved
    paths = [val for pair in zip(all_exist, all_missing) for val in pair]
    assert batch_lookup_C(paths) == [True, False, True, False, True, False]

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
        batch_lookup_C([all_exist[0], None])  # valid first, invalid mid-list


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_batch_lookup_dispatch(fs_tier, monkeypatch, use_c_ext):
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    if use_c_ext and not mgr_mod._HAS_BATCH_LOOKUP_C:
        pytest.skip("fs_io_C extension not built")

    monkeypatch.setattr(mgr_mod, "_HAS_BATCH_LOOKUP_C", use_c_ext)

    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert all(r.success for r in drain(tier))

    results = lookup_and_wait(tier, [key(1), key(2)])
    assert results == [LookupResult.HIT, LookupResult.MISS]
