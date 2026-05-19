# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for FileSystemTierManagerPython.

These tests use real disk I/O to verify the Python filesystem tier implementation.
The tier manager writes KV cache blocks to disk and reads them back, verifying
data integrity throughout the process.
"""

import os
import time
from unittest.mock import MagicMock

import numpy as np

import pytest
import torch

from vllm.v1.kv_offload.base import OffloadKey, ReqContext, make_offload_key
from vllm.v1.kv_offload.tiering.base import JobMetadata
from vllm.v1.kv_offload.tiering.fs.manager import (
    FileSystemTierManager,
)
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BLOCK_ELEMENTS = 512 * 1024  # 2 MB per block (float32 × 512K = 2MB)
_DTYPE = torch.float32
_CTX = ReqContext(req_id="test")

# Create proper mocks for vLLM config
_MOCK_VLLM_CONFIG = MagicMock()
_MOCK_VLLM_CONFIG.model_config.model = "test-model"
_MOCK_VLLM_CONFIG.cache_config.block_size = 16
_MOCK_VLLM_CONFIG.cache_config.cache_dtype = "torch.float32"
_MOCK_VLLM_CONFIG.parallel_config.tensor_parallel_size = 1
_MOCK_VLLM_CONFIG.parallel_config.pipeline_parallel_size = 1
_MOCK_VLLM_CONFIG.parallel_config.prefill_context_parallel_size = 1
_MOCK_VLLM_CONFIG.parallel_config.decode_context_parallel_size = 1
_MOCK_VLLM_CONFIG.parallel_config.rank = 0

_MOCK_KV_CACHE_CONFIG = MagicMock()
_MOCK_KV_CACHE_CONFIG.kv_cache_groups = []

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


def drain(tier: FileSystemTierManager, max_rounds: int = 40) -> list:
    """
    Call get_finished() repeatedly until no new results arrive for 5
    consecutive rounds or max_rounds is reached.
    """
    results = []
    idle = 0
    for _ in range(max_rounds):
        time.sleep(0.01)
        new = list(tier.get_finished())
        results.extend(new)
        if new:
            idle = 0
        else:
            idle += 1
            if idle >= 5:
                break
    return results


# ---------------------------------------------------------------------------
# Basic functionality tests
# ---------------------------------------------------------------------------

class TestPythonFSTierBasic:
    """Tests for basic tier functionality"""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        tensor = torch.zeros((4, _BLOCK_ELEMENTS), dtype=_DTYPE)
        mock_view = memoryview(tensor.numpy())
        
        self.tier = FileSystemTierManager(
            vllm_config=_MOCK_VLLM_CONFIG,
            primary_kv_view=mock_view,
            root_dir=str(tmp_path),
            kv_cache_config=_MOCK_KV_CACHE_CONFIG,
            n_read_threads=4,
            n_write_threads=4,
        )
        yield

    def test_get_tier_type(self):
        assert FileSystemTierManager.get_tier_type() == "fs_python"

    def test_lookup_empty_tier(self):
        assert self.tier.lookup(key(1), _CTX) is False
        assert self.tier.lookup(key(2), _CTX) is False

    def test_store_creates_file_and_lookup_succeeds(self):
        job = make_job(1, [key(1)], [0])
        self.tier.submit_store(job)
        results = drain(self.tier)
        assert len(results) == 1
        assert results[0].success
        # Verify file exists and lookup returns True
        assert self.tier.lookup(key(1), _CTX) is True
        dest = self.tier.file_mapper.get_file_name(key(1))
        assert os.path.exists(dest), f"Expected file at {dest}"

    def test_store_then_load_roundtrip(self):
        job_s = make_job(1, [key(1), key(2)], [0, 1])
        self.tier.submit_store(job_s)
        store_results = drain(self.tier)
        assert all(r.success for r in store_results)

        for k in [key(1), key(2)]:
            path = self.tier.file_mapper.get_file_name(k)
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"File {path} exists with size {size}")
            else:
                print(f"File {path} does not exist")

        assert self.tier.lookup(key(1), _CTX) is True
        assert self.tier.lookup(key(2), _CTX) is True

        job_l = make_job(2, [key(1), key(2)], [2, 3], is_promotion=True)
        self.tier.submit_load(job_l)
        load_results = drain(self.tier)
        assert all(r.success for r in load_results)
        # Blocks stay on disk after load
        assert self.tier.lookup(key(1), _CTX) is True
        assert self.tier.lookup(key(2), _CTX) is True

    def test_invalid_path_raises_at_construction(self):
        """Construction must fail immediately when the config file cannot be written."""
        tensor = torch.zeros((32, _BLOCK_ELEMENTS), dtype=_DTYPE)
        mock_view = memoryview(tensor.numpy())

        with pytest.raises(OSError):
            FileSystemTierManager(
                vllm_config=_MOCK_VLLM_CONFIG,
                primary_kv_view=mock_view,
                root_dir="/dev/null/invalid_path",
                kv_cache_config=_MOCK_KV_CACHE_CONFIG,
            )

    def test_failed_load_missing_file(self):
        """Test that loading a block whose file does not exist results in a failed job."""
        job = make_job(1, [key(99)], [0], is_promotion=True)
        self.tier.submit_load(job)
        results = drain(self.tier)
        assert len(results) == 1
        assert not results[0].success

    def test_multiple_jobs_tracked_independently(self):
        job1 = make_job(1, [key(1)], [0])
        job2 = make_job(2, [key(2)], [1])
        self.tier.submit_store(job1)
        self.tier.submit_store(job2)
        results = drain(self.tier)
        job_ids = {r.job_id for r in results}
        assert job_ids == {1, 2}
        assert self.tier.lookup(key(1), _CTX) is True
        assert self.tier.lookup(key(2), _CTX) is True

    def test_multi_block_job_partial_failure(self):
        """A load job where one block file is missing yields a single failed JobResult."""
        # Store two of three keys
        self.tier.submit_store(make_job(1, [key(10), key(11)], [0, 1]))
        assert all(r.success for r in drain(self.tier))

        # Load all three — key(99) was never stored
        self.tier.submit_load(make_job(2, [key(10), key(11), key(99)], [0, 1, 2], is_promotion=True))
        results = drain(self.tier)

        assert len(results) == 1
        assert results[0].job_id == 2
        assert not results[0].success

    def test_shutdown_discards_pending_tasks(self, tmp_path):
        """Shutdown clears both queues and stops all worker threads without draining."""
        tensor = torch.zeros((10, _BLOCK_ELEMENTS), dtype=_DTYPE)
        mock_view = memoryview(tensor.numpy())
        
        tier = FileSystemTierManager(
            vllm_config=_MOCK_VLLM_CONFIG,
            primary_kv_view=mock_view,
            root_dir=str(tmp_path),
            kv_cache_config=_MOCK_KV_CACHE_CONFIG,
            n_read_threads=2,
            n_write_threads=2,
        )

        for i in range(10):
            tier.submit_store(make_job(i, [key(i)], [i % 10]))

        tier._pool.shutdown()

        assert len(tier._pool._load_q) == 0
        assert len(tier._pool._store_q) == 0
        assert all(not t.is_alive() for t in tier._pool._threads)

    def test_store_load_data_integrity(self, tmp_path):
        """Data written by store must be exactly recovered by load."""
        num_blocks = 4
        num_total = num_blocks*2
        tensor = torch.rand((num_total, _BLOCK_ELEMENTS), dtype=_DTYPE)
        mock_view = memoryview(tensor.numpy())
        
        # Create new tier with larger tensor
        tier = FileSystemTierManager(
            vllm_config=_MOCK_VLLM_CONFIG,
            primary_kv_view=mock_view,
            root_dir=str(tmp_path),
            kv_cache_config=_MOCK_KV_CACHE_CONFIG,
            n_read_threads=4,
            n_write_threads=4,
        )

        expected = tensor[:num_blocks].clone()

        block_ids = list(range(num_blocks))
        keys = [key(i) for i in range(num_blocks)]

        tier.submit_store(make_job(1, keys, block_ids))
        results = drain(tier)
        assert all(r.success for r in results)

        # Overwrite source blocks to prove data is read from disk
        tensor[:num_blocks] = 0.0

        load_ids = list(range(num_blocks, num_total))
        tier.submit_load(make_job(2, keys, load_ids, is_promotion=True))
        results = drain(tier)
        assert all(r.success for r in results)

        for i, bid in enumerate(load_ids):
            assert torch.allclose(
                tensor[bid], expected[i]
            ), f"Block {bid} data mismatch after store+load"

# ---------------------------------------------------------------------------
# End-to-end tests with primary tier integration
# ---------------------------------------------------------------------------

class TestPythonFileSystemTierE2EWithPrimary:
    """
    End-to-end tests integrating FileSystemTierManagerPython with
    CPUPrimaryTierOffloadingManager using real disk I/O.
    
    These tests verify full data integrity through cascade and promotion
    pipelines with actual file system operations.
    """

    @pytest.fixture
    def setup_manager(self, tmp_path):
        """Setup TieringOffloadingManager with real primary and Python filesystem tiers."""
        num_primary_blocks = 10

        # Create a mock mmap region
        from unittest.mock import MagicMock
        mock_region = MagicMock()
        cpu_tensor = torch.zeros((num_primary_blocks, _BLOCK_ELEMENTS), dtype=_DTYPE)
        mock_region.create_kv_memoryview.return_value = memoryview(cpu_tensor.numpy())
        mock_region._base = cpu_tensor
        mock_region.num_blocks = num_primary_blocks
        mock_region._row_stride = _BLOCK_ELEMENTS

        # Create primary tier
        primary_tier = CPUPrimaryTierOffloadingManager(
            num_blocks=num_primary_blocks,
            mmap_region=mock_region,
        )
        
        # Create Python filesystem tier with real I/O
        mock_view = primary_tier.get_kv_memoryview()
        fs_tier = FileSystemTierManager(
            vllm_config=_MOCK_VLLM_CONFIG,
            primary_kv_view=mock_view,
            root_dir=str(tmp_path / "kvcache"),
            kv_cache_config=_MOCK_KV_CACHE_CONFIG,
            n_read_threads=4,
            n_write_threads=4,
        )
        
        # Create tiering manager
        manager = TieringOffloadingManager(
            primary_tier=primary_tier,
            secondary_tiers=[fs_tier],
        )
        
        yield manager, primary_tier, fs_tier, cpu_tensor, _BLOCK_ELEMENTS
        
        # Cleanup
        manager.shutdown()

    def test_full_cascade_with_data_integrity(self, setup_manager):
        """
        Store blocks to primary tier with known data patterns, verify cascade
        to filesystem tier completes, and verify data integrity by reading
        files directly from disk.
        """
        manager, primary_tier, fs_tier, cpu_tensor, block_elements = setup_manager
        
        # Generate unique data patterns for each block
        num_blocks = 5
        keys = [key(100 + i) for i in range(num_blocks)]
        expected_data = {}
        
        # Prepare store to primary tier
        result = manager.prepare_store(keys, _CTX)
        assert result is not None
        assert len(result.keys_to_store) == num_blocks
        
        # Fill blocks with unique random data
        spec = result.store_spec
        for i, block_id in enumerate(spec.block_ids):
            data = torch.rand(block_elements, dtype=_DTYPE)
            cpu_tensor[int(block_id)] = data
            expected_data[keys[i]] = data.clone()
        
        # Complete store (triggers cascade to filesystem)
        manager.complete_store(keys, _CTX, success=True)
        
        # Wait for cascade to complete
        for _ in range(20):
            manager._process_finished_jobs()
            time.sleep(0.01)
        
        # Verify blocks are in both tiers
        for k in keys:
            assert primary_tier.lookup(k, _CTX) is True
            assert fs_tier.lookup(k) is True
        
        # Verify data integrity by reading from disk
        for k in keys:
            file_path = fs_tier.file_mapper.get_file_name(k)
            assert os.path.isfile(file_path), f"File not found: {file_path}"
            with open(file_path, "rb") as f:
                raw = f.read(block_elements * 4)
            actual = torch.frombuffer(bytearray(raw), dtype=_DTYPE)
            assert torch.allclose(actual, expected_data[k]), \
                f"Data mismatch for block {k}"

    def test_cascade_promotion_roundtrip(self, setup_manager):
        """
        Store blocks with random data to primary (triggers cascade),
        evict blocks from primary tier, lookup blocks to trigger promotion
        from filesystem, and verify data integrity after full roundtrip.
        """
        manager, primary_tier, fs_tier, cpu_tensor, block_elements = setup_manager
        
        # Store blocks with random data
        num_blocks = 3
        keys = [key(200 + i) for i in range(num_blocks)]
        expected_data = {}
        
        result = manager.prepare_store(keys, _CTX)
        assert result is not None

        spec = result.store_spec
        for i, block_id in enumerate(spec.block_ids):
            data = torch.rand(block_elements, dtype=_DTYPE)
            cpu_tensor[int(block_id)] = data
            expected_data[keys[i]] = data.clone()

        manager.complete_store(keys, _CTX, success=True)

        # Wait for cascade to complete before trying to evict
        for _ in range(20):
            manager._process_finished_jobs()
            all_done = all(
                primary_tier._policy.get(k).ref_cnt == 0
                for k in keys if primary_tier._policy.get(k) is not None
            )
            if all_done:
                break
            time.sleep(0.05)

        # Evict from primary by filling it (10 slots, 3 filled, store 10 more to trigger eviction)
        evict_keys = [key(300 + i) for i in range(10)]
        result = manager.prepare_store(evict_keys, _CTX)
        assert result is not None
        assert len(result.evicted_keys) >= num_blocks  # Original blocks should be evicted
        
        spec = result.store_spec
        for block_id in spec.block_ids:
            cpu_tensor[int(block_id)] = 0.0
        manager.complete_store(evict_keys, _CTX, success=True)
        
        # Wait for cascade of new blocks
        for _ in range(20):
            manager._process_finished_jobs()
            time.sleep(0.01)
        
        # Verify blocks are only in filesystem tier
        for k in keys:
            assert primary_tier.lookup(k, _CTX) is False
            assert fs_tier.lookup(k) is True
        
        # Trigger promotion (each lookup queues blocks in _pending_load_submissions)
        for k in keys:
            manager.lookup(k, _CTX)

        # Flush pending promotions — take_events() calls _flush_pending_promotions(),
        # which is the only path that actually calls submit_load().
        list(manager.take_events())

        # Wait for promotion I/O to complete
        for _ in range(20):
            manager._process_finished_jobs()
            time.sleep(0.05)
        
        # Verify blocks are now in primary tier
        assert all(manager.lookup(k, _CTX) is True for k in keys)
        
        # Verify data integrity after roundtrip
        load_spec = primary_tier.prepare_load(keys, _CTX)
        for i, block_id in enumerate(load_spec.block_ids):
            actual_data = cpu_tensor[int(block_id)]
            expected = expected_data[keys[i]]
            assert torch.allclose(actual_data, expected, rtol=1e-5, atol=1e-7), \
                f"Block {i} data mismatch after roundtrip"

    def test_prepare_load_after_promotion_via_manager(self, setup_manager):
        """
        Full scheduler-driven flow: store → cascade → evict → promote →
        manager.prepare_load → manager.complete_load.

        Unlike test_cascade_promotion_roundtrip, this test:
          - Drives every step boundary through take_events() (not
            _process_finished_jobs() directly), exercising the per-step gate.
          - Calls manager.prepare_load / complete_load through the tiering
            manager instead of the primary tier, covering the
            _maybe_process_finished_jobs() call inside prepare_load and the
            ref_cnt decrement inside complete_load.
        """
        manager, primary_tier, fs_tier, cpu_tensor, block_elements = setup_manager

        num_blocks = 3
        keys = [key(400 + i) for i in range(num_blocks)]
        expected_data = {}

        # Store blocks with known data.
        result = manager.prepare_store(keys, _CTX)
        assert result is not None
        for i, block_id in enumerate(result.store_spec.block_ids):
            data = torch.rand(block_elements, dtype=_DTYPE)
            cpu_tensor[int(block_id)] = data
            expected_data[keys[i]] = data.clone()
        manager.complete_store(keys, _CTX, success=True)

        # Wait for cascade to complete and ref_cnt to drop, driven by
        # take_events() so the per-step gate is exercised on every iteration.
        for _ in range(40):
            list(manager.take_events())
            if all(
                primary_tier._policy.get(k) is not None
                and primary_tier._policy.get(k).ref_cnt == 0
                for k in keys
            ):
                break
            time.sleep(0.05)
        assert all(
            primary_tier._policy.get(k).ref_cnt == 0 for k in keys
        ), "cascade never completed (ref_cnt still held)"
        assert all(fs_tier.lookup(k) is True for k in keys)

        # Evict original blocks by overfilling primary (10 slots).
        evict_keys = [key(500 + i) for i in range(10)]
        result = manager.prepare_store(evict_keys, _CTX)
        assert result is not None and len(result.evicted_keys) >= num_blocks
        for block_id in result.store_spec.block_ids:
            cpu_tensor[int(block_id)] = 0.0
        manager.complete_store(evict_keys, _CTX, success=True)

        # Wait for the evict_keys cascade to complete so their ref_cnt drops to
        # 0. Until that happens, all 10 primary slots are pinned and
        # prepare_write() cannot evict any of them to make room for the
        # promotion below.
        for _ in range(40):
            list(manager.take_events())
            if all(
                primary_tier._policy.get(k) is not None
                and primary_tier._policy.get(k).ref_cnt == 0
                for k in evict_keys
            ):
                break
            time.sleep(0.05)

        assert all(primary_tier.lookup(k, _CTX) is False for k in keys)

        # Trigger promotion via lookup; take_events() flushes submit_load.
        for k in keys:
            assert manager.lookup(k, _CTX) is None
        list(manager.take_events())

        # Wait for promotion I/O to complete, driven by take_events().
        for _ in range(40):
            list(manager.take_events())
            if all(primary_tier.lookup(k, _CTX) is True for k in keys):
                break
            time.sleep(0.05)
        assert all(primary_tier.lookup(k, _CTX) is True for k in keys), \
            "promotion never completed"

        # manager.prepare_load — the path under test.
        load_spec = manager.prepare_load(keys, _CTX)
        assert load_spec is not None
        assert len(load_spec.block_ids) == num_blocks
        for i, block_id in enumerate(load_spec.block_ids):
            assert torch.allclose(
                cpu_tensor[int(block_id)], expected_data[keys[i]],
                rtol=1e-5, atol=1e-7,
            ), f"Block {i} data mismatch after promotion + manager.prepare_load"

        # manager.complete_load must release ref_cnt.
        manager.complete_load(keys, _CTX)
        for k in keys:
            block = primary_tier._policy.get(k)
            assert block is not None and block.ref_cnt == 0