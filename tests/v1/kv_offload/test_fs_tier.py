# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for FileSystemTierManager.

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BLOCK_ELEMENTS = 512 * 1024  # 2 MB per block (float32 × 512K = 2MB)
_DTYPE = torch.float32
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

_MOCK_KV_CACHE_CONFIG = MagicMock()
_MOCK_KV_CACHE_CONFIG.kv_cache_groups = []

_MOCK_OFFLOADING_SPEC = MagicMock()
_MOCK_OFFLOADING_SPEC.vllm_config = _MOCK_VLLM_CONFIG
_MOCK_OFFLOADING_SPEC.kv_cache_config = _MOCK_KV_CACHE_CONFIG

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
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fs_tier(tmp_path):
    tensor = torch.zeros((4, _BLOCK_ELEMENTS), dtype=_DTYPE)
    mock_view = memoryview(tensor.numpy())
    return FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=mock_view,
        tier_type="fs_python",
        root_dir=str(tmp_path),
        n_read_threads=4,
        n_write_threads=4,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_get_tier_type():
    assert FileSystemTierManager.get_tier_type() == "fs_python"


def test_lookup_empty_tier(fs_tier):
    assert fs_tier.lookup(key(1), _CTX) is False
    assert fs_tier.lookup(key(2), _CTX) is False


def test_store_creates_file_and_lookup_succeeds(fs_tier):
    job = make_job(1, [key(1)], [0])
    fs_tier.submit_store(job)
    results = drain(fs_tier)
    assert len(results) == 1
    assert results[0].success
    assert fs_tier.lookup(key(1), _CTX) is True
    dest = fs_tier.file_mapper.get_file_name(key(1))
    assert os.path.exists(dest), f"Expected file at {dest}"


def test_store_then_load_roundtrip(fs_tier):
    job_s = make_job(1, [key(1), key(2)], [0, 1])
    fs_tier.submit_store(job_s)
    store_results = drain(fs_tier)
    assert all(r.success for r in store_results)

    assert fs_tier.lookup(key(1), _CTX) is True
    assert fs_tier.lookup(key(2), _CTX) is True

    job_l = make_job(2, [key(1), key(2)], [2, 3], is_promotion=True)
    fs_tier.submit_load(job_l)
    load_results = drain(fs_tier)
    assert all(r.success for r in load_results)
    # Blocks stay on disk after load
    assert fs_tier.lookup(key(1), _CTX) is True
    assert fs_tier.lookup(key(2), _CTX) is True


def test_invalid_path_raises_at_construction():
    """Construction must fail immediately when the config file cannot be written."""
    tensor = torch.zeros((32, _BLOCK_ELEMENTS), dtype=_DTYPE)
    mock_view = memoryview(tensor.numpy())

    with pytest.raises(OSError):
        FileSystemTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=mock_view,
            tier_type="fs_python",
            root_dir="/dev/null/invalid_path",
        )


def test_failed_load_missing_file(fs_tier):
    """Test that loading a block whose file does not exist results in a failed job."""
    job = make_job(1, [key(99)], [0], is_promotion=True)
    fs_tier.submit_load(job)
    results = drain(fs_tier)
    assert len(results) == 1
    assert not results[0].success


def test_multiple_jobs_tracked_independently(fs_tier):
    job1 = make_job(1, [key(1)], [0])
    job2 = make_job(2, [key(2)], [1])
    fs_tier.submit_store(job1)
    fs_tier.submit_store(job2)
    results = drain(fs_tier)
    job_ids = {r.job_id for r in results}
    assert job_ids == {1, 2}
    assert fs_tier.lookup(key(1), _CTX) is True
    assert fs_tier.lookup(key(2), _CTX) is True


def test_multi_block_job_partial_failure(fs_tier):
    """A load job where one block file is missing yields a single failed JobResult."""
    # Store two of three keys
    fs_tier.submit_store(make_job(1, [key(10), key(11)], [0, 1]))
    assert all(r.success for r in drain(fs_tier))

    # Load all three — key(99) was never stored
    fs_tier.submit_load(make_job(2, [key(10), key(11), key(99)], [0, 1, 2], is_promotion=True))
    results = drain(fs_tier)

    assert len(results) == 1
    assert results[0].job_id == 2
    assert not results[0].success


def test_shutdown_discards_pending_tasks(tmp_path):
    """Shutdown clears both queues and stops all worker threads without draining."""
    tensor = torch.zeros((10, _BLOCK_ELEMENTS), dtype=_DTYPE)
    mock_view = memoryview(tensor.numpy())

    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=mock_view,
        tier_type="fs_python",
        root_dir=str(tmp_path),
        n_read_threads=2,
        n_write_threads=2,
    )

    for i in range(10):
        tier.submit_store(make_job(i, [key(i)], [i % 10]))

    tier._pool.shutdown()

    assert len(tier._pool._load_q) == 0
    assert len(tier._pool._store_q) == 0
    assert all(not t.is_alive() for t in tier._pool._threads)


def test_store_load_data_integrity(tmp_path):
    """Data written by store must be exactly recovered by load."""
    num_blocks = 4
    num_total = num_blocks * 2
    tensor = torch.rand((num_total, _BLOCK_ELEMENTS), dtype=_DTYPE)
    mock_view = memoryview(tensor.numpy())

    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=mock_view,
        tier_type="fs_python",
        root_dir=str(tmp_path),
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
