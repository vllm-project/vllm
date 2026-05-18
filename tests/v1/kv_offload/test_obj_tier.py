# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for ObjectStoreSecondaryTierManager.

These tests use real NIXL OBJ (S3) I/O to verify the OBJ secondary tier
implementation. The tier writes KV cache blocks to an S3-compatible store via
NIXL and reads them back, verifying data integrity throughout the process.

Required environment variables (tests are skipped if any are absent):
    VLLM_TEST_S3_BUCKET           — S3 bucket name
    VLLM_TEST_S3_ENDPOINT         — S3 endpoint URL (e.g. http://minio:9000)
    VLLM_TEST_S3_ACCESS_KEY       — S3 access key
    VLLM_TEST_S3_SECRET_KEY       — S3 secret key
    VLLM_TEST_S3_SCHEME           — (optional) http or https, default http
    VLLM_TEST_S3_CA_BUNDLE        — (optional) path to CA bundle for TLS verification
"""

import os
import time
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.kv_offload.base import OffloadKey, ReqContext, make_offload_key
from vllm.v1.kv_offload.tiering.base import JobMetadata, JobResult
from vllm.v1.kv_offload.tiering.obj.manager import ObjectStoreSecondaryTierManager


def _make_vllm_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(model="test/model"),
        cache_config=SimpleNamespace(block_size=16, cache_dtype="float16"),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
    )


_VLLM_CONFIG = _make_vllm_config()

# ---------------------------------------------------------------------------
# S3 credentials — skip entire module if not configured
# ---------------------------------------------------------------------------

_S3_BUCKET = os.environ.get("VLLM_TEST_S3_BUCKET", "")
_S3_ENDPOINT = os.environ.get("VLLM_TEST_S3_ENDPOINT", "")
_S3_ACCESS_KEY = os.environ.get("VLLM_TEST_S3_ACCESS_KEY", "")
_S3_SECRET_KEY = os.environ.get("VLLM_TEST_S3_SECRET_KEY", "")
_S3_SCHEME = os.environ.get("VLLM_TEST_S3_SCHEME", "http")
_S3_CA_BUNDLE = os.environ.get("VLLM_TEST_S3_CA_BUNDLE", "")

if not all([_S3_BUCKET, _S3_ENDPOINT, _S3_ACCESS_KEY, _S3_SECRET_KEY]):
    pytest.skip(
        "S3 credentials not set — export VLLM_TEST_S3_BUCKET, "
        "VLLM_TEST_S3_ENDPOINT, VLLM_TEST_S3_ACCESS_KEY, VLLM_TEST_S3_SECRET_KEY",
        allow_module_level=True,
    )

_STORE_CONFIG = {
    "bucket": _S3_BUCKET,
    "endpoint_override": _S3_ENDPOINT,
    "access_key": _S3_ACCESS_KEY,
    "secret_key": _S3_SECRET_KEY,
    "scheme": _S3_SCHEME,
    "ca_bundle": _S3_CA_BUNDLE,
}

# Probe NIXL OBJ plugin availability
try:
    _probe_view = memoryview(torch.zeros(1, 1, dtype=torch.float32).numpy())
    _probe = ObjectStoreSecondaryTierManager(
        vllm_config=_VLLM_CONFIG,
        primary_kv_view=_probe_view,
        store_config=_STORE_CONFIG,
    )
    del _probe, _probe_view
except RuntimeError as _e:
    pytest.skip(f"NIXL OBJ plugin not available: {_e}", allow_module_level=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Small blocks keep S3 round-trips fast; 1 KB per block is enough for integrity
_BLOCK_ELEMENTS = 256  # float32 × 256 = 1 KB
_DTYPE = torch.float32

# Unique prefix per test session so parallel runs don't collide
_RUN_PREFIX = f"test/{uuid.uuid4().hex[:8]}"

_CTX = ReqContext(req_id="test-req")


def key(n: int) -> OffloadKey:
    return make_offload_key(n.to_bytes(8, "big"), 0)


def make_job(
    job_id: int,
    keys: list[OffloadKey],
    block_ids: list[int] | None = None,
) -> JobMetadata:
    if block_ids is None:
        block_ids = list(range(len(keys)))
    return JobMetadata(
        job_id=job_id,
        keys=keys,
        block_ids=np.array(block_ids, dtype=np.int64),
        is_promotion=False,
        req_context=_CTX,
    )


def make_tier(
    key_prefix: str = _RUN_PREFIX,
    num_blocks: int = 1,
    **kwargs,
) -> ObjectStoreSecondaryTierManager:
    view = memoryview(torch.zeros(num_blocks, _BLOCK_ELEMENTS, dtype=_DTYPE).numpy())
    return ObjectStoreSecondaryTierManager(
        vllm_config=_VLLM_CONFIG,
        primary_kv_view=view,
        store_config=_STORE_CONFIG,
        prefix=key_prefix,
        **kwargs,
    )


def make_tier_with_view(
    num_total_blocks: int = 8,
    key_prefix: str = _RUN_PREFIX,
    **kwargs,
) -> tuple[ObjectStoreSecondaryTierManager, torch.Tensor]:
    tensor = torch.zeros((num_total_blocks, _BLOCK_ELEMENTS), dtype=_DTYPE)
    tier = ObjectStoreSecondaryTierManager(
        vllm_config=_VLLM_CONFIG,
        primary_kv_view=memoryview(tensor.numpy()),
        store_config=_STORE_CONFIG,
        prefix=key_prefix,
        **kwargs,
    )
    return tier, tensor


def drain(tier: ObjectStoreSecondaryTierManager, max_rounds: int = 200) -> list[JobResult]:
    """Poll get_finished() until all pending jobs resolve or timeout."""
    results: list[JobResult] = []
    for _ in range(max_rounds):
        results.extend(tier.get_finished())
        if not tier._transfers:
            break
        time.sleep(0.1)
    return results


# ---------------------------------------------------------------------------
# Basic functionality tests
# ---------------------------------------------------------------------------


class TestObjTierBasic:
    """Tests for basic tier operations with real S3 I/O."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        prefix = f"{_RUN_PREFIX}/basic/{uuid.uuid4().hex[:6]}"
        self.tier, self.tensor = make_tier_with_view(
            num_total_blocks=4, key_prefix=prefix
        )
        yield
        self.tier.shutdown()

    def test_lookup_empty_tier(self):
        assert self.tier.lookup(key(1), _CTX) is False
        assert self.tier.lookup(key(2), _CTX) is False

    def test_store_and_lookup(self):
        job = make_job(1, [key(1)], [0])
        self.tier.submit_store(job)
        results = drain(self.tier)
        assert len(results) == 1
        assert results[0].success
        assert self.tier.lookup(key(1), _CTX) is True

    def test_store_unknown_key_returns_false(self):
        job = make_job(1, [key(1)], [0])
        self.tier.submit_store(job)
        drain(self.tier)
        assert self.tier.lookup(key(99), _CTX) is False

    def test_store_then_load_roundtrip(self):
        job_s = make_job(1, [key(1), key(2)], [0, 1])
        self.tier.submit_store(job_s)
        store_results = drain(self.tier)
        assert all(r.success for r in store_results)
        assert self.tier.lookup(key(1), _CTX) is True
        assert self.tier.lookup(key(2), _CTX) is True

        job_l = make_job(2, [key(1), key(2)], [2, 3])
        self.tier.submit_load(job_l)
        load_results = drain(self.tier)
        assert all(r.success for r in load_results)
        # Blocks remain in S3 after load
        assert self.tier.lookup(key(1), _CTX) is True
        assert self.tier.lookup(key(2), _CTX) is True

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



# ---------------------------------------------------------------------------
# Data integrity tests
# ---------------------------------------------------------------------------


class TestObjTierIO:
    """Data written by store must be exactly recovered by load."""

    def _make(self, num_total_blocks: int = 8) -> tuple[ObjectStoreSecondaryTierManager, torch.Tensor]:
        prefix = f"{_RUN_PREFIX}/io/{uuid.uuid4().hex[:6]}"
        return make_tier_with_view(num_total_blocks=num_total_blocks, key_prefix=prefix)

    def test_store_load_data_integrity(self):
        num_blocks = 4
        tier, tensor = self._make(num_total_blocks=num_blocks * 2)

        for bid in range(num_blocks):
            tensor[bid] = torch.rand((_BLOCK_ELEMENTS,), dtype=_DTYPE)
        expected = tensor[:num_blocks].clone()

        keys = [key(i) for i in range(num_blocks)]
        tier.submit_store(make_job(1, keys, list(range(num_blocks))))
        results = drain(tier)
        assert all(r.success for r in results)

        # Zero source slots to prove data comes from S3
        tensor[:num_blocks] = 0.0

        load_ids = list(range(num_blocks, num_blocks * 2))
        tier.submit_load(make_job(2, keys, load_ids))
        results = drain(tier)
        assert all(r.success for r in results)

        for i, bid in enumerate(load_ids):
            assert torch.equal(tensor[bid], expected[i]), (
                f"Block {bid} data mismatch after store+load"
            )

        tier.shutdown()

    def test_store_load_multiple_blocks(self):
        num_blocks = 8
        tier, tensor = self._make(num_total_blocks=num_blocks * 2)

        for bid in range(num_blocks):
            tensor[bid] = float(bid + 1)
        expected = tensor[:num_blocks].clone()

        keys = [key(i + 200) for i in range(num_blocks)]
        tier.submit_store(make_job(10, keys, list(range(num_blocks))))
        results = drain(tier)
        assert all(r.success for r in results)

        tensor[:num_blocks] = 0.0
        load_ids = list(range(num_blocks, num_blocks * 2))
        tier.submit_load(make_job(11, keys, load_ids))
        results = drain(tier)
        assert all(r.success for r in results)

        for i, bid in enumerate(load_ids):
            assert torch.equal(tensor[bid], expected[i])

        tier.shutdown()


# ---------------------------------------------------------------------------
# End-to-end tests with primary tier integration
# ---------------------------------------------------------------------------


class TestObjTierE2EWithPrimary:
    """
    End-to-end tests integrating ObjectStoreSecondaryTierManager with
    CPUPrimaryTierOffloadingManager using real S3 I/O via NIXL.

    Verifies full data integrity through cascade and promotion pipelines.
    """

    @pytest.fixture
    def setup_manager(self):
        from vllm.v1.kv_offload.tiering.manager import (
            CPUPrimaryTierOffloadingManager,
            TieringOffloadingManager,
        )

        num_primary_blocks = 10
        prefix = f"{_RUN_PREFIX}/e2e/{uuid.uuid4().hex[:6]}"

        cpu_tensor = torch.zeros((num_primary_blocks, _BLOCK_ELEMENTS), dtype=_DTYPE)
        mmap_region = SimpleNamespace(
            create_kv_memoryview=lambda: memoryview(cpu_tensor.numpy()),
            cleanup=lambda: None,
        )
        primary_tier = CPUPrimaryTierOffloadingManager(
            num_blocks=num_primary_blocks,
            mmap_region=mmap_region,
        )

        obj_tier = ObjectStoreSecondaryTierManager(
            vllm_config=_VLLM_CONFIG,
            primary_kv_view=memoryview(cpu_tensor.numpy()),
            store_config=_STORE_CONFIG,
            prefix=prefix,
        )

        manager = TieringOffloadingManager(
            primary_tier=primary_tier,
            secondary_tiers=[obj_tier],
        )

        yield manager, primary_tier, obj_tier, cpu_tensor
        manager.shutdown()

    def _wait_cascade(self, manager, primary_tier, keys, rounds=60):
        """Poll until all cascaded blocks have ref_cnt == 0."""
        for _ in range(rounds):
            manager._process_finished_jobs()
            all_done = all(
                primary_tier._policy.get(k) is None
                or primary_tier._policy.get(k).ref_cnt == 0
                for k in keys
            )
            if all_done:
                break
            time.sleep(0.1)

    def test_cascade_store_and_lookup(self, setup_manager):
        """Blocks stored to primary cascade to S3 and become findable."""
        manager, primary_tier, obj_tier, cpu_tensor = setup_manager

        num_blocks = 3
        keys = [key(100 + i) for i in range(num_blocks)]

        result = manager.prepare_store(keys, _CTX)
        assert result is not None
        manager.complete_store(keys, _CTX, success=True)

        self._wait_cascade(manager, primary_tier, keys)

        for k in keys:
            assert primary_tier.lookup(k, _CTX) is True
            assert obj_tier.lookup(k, _CTX) is True

    def test_full_cascade_data_integrity(self, setup_manager):
        """Data written to primary cascades to S3 with correct content."""
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

        manager, primary_tier, obj_tier, cpu_tensor = setup_manager

        num_blocks = 4
        keys = [key(200 + i) for i in range(num_blocks)]
        expected: dict[OffloadKey, torch.Tensor] = {}

        result = manager.prepare_store(keys, _CTX)
        assert result is not None
        spec = result.store_spec
        assert isinstance(spec, CPULoadStoreSpec)

        for i, bid in enumerate(spec.block_ids):
            data = torch.rand((_BLOCK_ELEMENTS,), dtype=_DTYPE)
            cpu_tensor[int(bid)] = data
            expected[keys[i]] = data.clone()

        manager.complete_store(keys, _CTX, success=True)
        self._wait_cascade(manager, primary_tier, keys)

        # Zero out primary slots; load back from S3 into fresh slots
        load_ids = list(range(5, 5 + num_blocks))
        for bid in spec.block_ids:
            cpu_tensor[int(bid)] = 0.0

        job = make_job(99, keys, load_ids)
        obj_tier.submit_load(job)
        drain(obj_tier)

        for i, bid in enumerate(load_ids):
            assert torch.equal(cpu_tensor[bid], expected[keys[i]]), (
                f"Block {i} data mismatch after cascade+load"
            )

    def test_promotion_from_obj_tier(self, setup_manager):
        """Blocks evicted from primary can be promoted back via lookup."""
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

        manager, primary_tier, obj_tier, cpu_tensor = setup_manager

        num_blocks = 3
        keys = [key(300 + i) for i in range(num_blocks)]
        expected: dict[OffloadKey, torch.Tensor] = {}

        # Store to primary (cascades to S3)
        result = manager.prepare_store(keys, _CTX)
        assert result is not None
        spec = result.store_spec
        assert isinstance(spec, CPULoadStoreSpec)
        for i, bid in enumerate(spec.block_ids):
            data = torch.rand((_BLOCK_ELEMENTS,), dtype=_DTYPE)
            cpu_tensor[int(bid)] = data
            expected[keys[i]] = data.clone()
        manager.complete_store(keys, _CTX, success=True)
        self._wait_cascade(manager, primary_tier, keys)

        # Evict from primary by filling it with new blocks
        evict_keys = [key(400 + i) for i in range(10)]
        result = manager.prepare_store(evict_keys, _CTX)
        assert result is not None
        for bid in result.store_spec.block_ids:
            cpu_tensor[int(bid)] = 0.0
        manager.complete_store(evict_keys, _CTX, success=True)
        self._wait_cascade(manager, primary_tier, evict_keys)

        # Original blocks should be gone from primary but present in S3
        for k in keys:
            assert primary_tier.lookup(k, _CTX) is False
            assert obj_tier.lookup(k, _CTX) is True

        # Trigger promotion via lookup
        for k in keys:
            manager.lookup(k, _CTX)

        # Wait for promotion to complete
        for _ in range(60):
            list(manager.take_events())
            time.sleep(0.1)

        # Blocks should now be back in primary
        assert all(manager.lookup(k, _CTX) is True for k in keys)

        # Verify data integrity after promotion
        load_spec = primary_tier.prepare_load(keys, _CTX)
        for i, bid in enumerate(load_spec.block_ids):
            assert torch.equal(
                cpu_tensor[int(bid)], expected[keys[i]]
            ), f"Block {i} data mismatch after promotion"
        primary_tier.complete_load(keys, _CTX)

    def test_cascade_promotion_roundtrip(self, setup_manager):
        """Full roundtrip: store -> cascade -> evict -> promote -> data intact."""
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

        manager, primary_tier, obj_tier, cpu_tensor = setup_manager

        num_blocks = 3
        keys = [key(500 + i) for i in range(num_blocks)]
        expected: dict[OffloadKey, torch.Tensor] = {}

        result = manager.prepare_store(keys, _CTX)
        assert result is not None
        spec = result.store_spec
        assert isinstance(spec, CPULoadStoreSpec)
        for i, bid in enumerate(spec.block_ids):
            data = torch.rand((_BLOCK_ELEMENTS,), dtype=_DTYPE)
            cpu_tensor[int(bid)] = data
            expected[keys[i]] = data.clone()
        manager.complete_store(keys, _CTX, success=True)
        self._wait_cascade(manager, primary_tier, keys)

        # Evict
        evict_keys = [key(600 + i) for i in range(10)]
        result = manager.prepare_store(evict_keys, _CTX)
        assert result is not None
        for bid in result.store_spec.block_ids:
            cpu_tensor[int(bid)] = 0.0
        manager.complete_store(evict_keys, _CTX, success=True)
        self._wait_cascade(manager, primary_tier, evict_keys)

        for k in keys:
            assert primary_tier.lookup(k, _CTX) is False

        # Promote
        for k in keys:
            manager.lookup(k, _CTX)
        for _ in range(60):
            list(manager.take_events())
            time.sleep(0.1)

        assert all(manager.lookup(k, _CTX) is True for k in keys)

        load_spec = primary_tier.prepare_load(keys, _CTX)
        for i, bid in enumerate(load_spec.block_ids):
            assert torch.equal(
                cpu_tensor[int(bid)], expected[keys[i]]
            ), f"Block {i} data mismatch after roundtrip"
        primary_tier.complete_load(keys, _CTX)

    def test_ref_cnt_released_after_cascade(self, setup_manager):
        """ref_cnt on primary blocks must reach 0 after S3 cascade completes."""
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

        manager, primary_tier, obj_tier, cpu_tensor = setup_manager

        keys = [key(700 + i) for i in range(3)]
        result = manager.prepare_store(keys, _CTX)
        assert result is not None
        spec = result.store_spec
        assert isinstance(spec, CPULoadStoreSpec)
        for bid in spec.block_ids:
            cpu_tensor[int(bid)] = torch.rand((_BLOCK_ELEMENTS,), dtype=_DTYPE)
        manager.complete_store(keys, _CTX, success=True)

        # Immediately after complete_store, ref_cnt must be 1 (cascade in flight)
        for k in keys:
            block = primary_tier._policy.get(k)
            assert block is not None
            assert block.ref_cnt == 1

        self._wait_cascade(manager, primary_tier, keys)

        for k in keys:
            block = primary_tier._policy.get(k)
            assert block is not None
            assert block.ref_cnt == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
