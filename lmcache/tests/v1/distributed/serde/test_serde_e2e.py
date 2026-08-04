# SPDX-License-Identifier: Apache-2.0
"""
End-to-end tests for the serde integration through StorageManager.

Drives the full L1 -> L2 store (with serialize) and L2 -> L1 prefetch
(with deserialize) paths using a real L1Manager + MockL2Adapter +
AsyncSerdeProcessor(fp8). Tests exercise the public StorageManager API
only (no private-member access), per AGENTS.md.

Coverage:
- Store + prefetch round-trip through fp8 serde
- Memory accounting (no leaks after full cycle)
- Partial prefix trimming with serde
- Mixed adapters (one with serde, one without)
- Serde disabled (None) preserves existing behavior exactly
- Serde failure propagation
"""

# Standard
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchRequestSpec,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.serde import SerdeConfig
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.platform import current_device_spec

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )

# =============================================================================
# Helpers
# =============================================================================


def make_object_key(chunk_id: int) -> ObjectKey:
    """Create a test ObjectKey with the given chunk ID."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def make_layout() -> MemoryLayoutDesc:
    """Create a small MemoryLayoutDesc for testing (bf16, ~200KB/chunk)."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


def wait_for_condition(
    predicate,
    timeout: float = 10.0,
    poll_interval: float = 0.05,
) -> bool:
    """Poll until a predicate returns True or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return False


def wait_for_prefetch_status(
    sm: StorageManager,
    handle,
    timeout: float = 10.0,
    poll_interval: float = 0.05,
) -> int | None:
    """Poll query_prefetch_status until it returns a non-None value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = sm.query_prefetch_status(handle)
        if result is not None:
            return result.count_leading_ones()
        time.sleep(poll_interval)
    return None


def make_mock_adapter_config(
    *,
    serde_type: str | None = "fp8",
) -> MockL2AdapterConfig:
    """Create a MockL2AdapterConfig, optionally with serde."""
    cfg = MockL2AdapterConfig(
        max_size_gb=0.1,
        mock_bandwidth_gb=10.0,
    )
    if serde_type is not None:
        cfg.serde_config = SerdeConfig(type=serde_type)
    return cfg


def make_storage_manager_config(
    adapter_configs: list[MockL2AdapterConfig],
    l1_size_mb: int = 256,
) -> StorageManagerConfig:
    """Build a StorageManagerConfig with the given L2 adapter configs."""
    return StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=l1_size_mb * 1024 * 1024,
                use_lazy=current_device_spec.is_pin_supported,
                init_size_in_bytes=min(l1_size_mb, 64) * 1024 * 1024,
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(adapters=list(adapter_configs)),
    )


def get_l2_stored_object_count(sm: StorageManager) -> int:
    """Return the total stored object count across all L2 adapters."""
    return sum(
        adapter["stored_object_count"] for adapter in sm.report_status()["l2_adapters"]
    )


def write_and_wait_for_l2(
    sm: StorageManager,
    keys: list[ObjectKey],
    layout: MemoryLayoutDesc,
    timeout: float = 10.0,
) -> None:
    """Write keys to L1 via StorageManager and wait for L2 store.

    Fills each chunk with deterministic data so round-trip can be verified.
    """
    stored_before = get_l2_stored_object_count(sm)

    ret = sm.reserve_write(keys, layout, mode="new")
    assert len(ret) == len(keys), f"reserve_write: {len(ret)}/{len(keys)} succeeded"

    # Fill with deterministic data per key
    for i, key in enumerate(keys):
        obj = ret[key]
        tensor = obj.tensor
        if tensor is not None:
            tensor.fill_(float(i + 1))

    sm.finish_write(list(ret.keys()))

    # Wait for StoreController to flush to L2. Polling the controller's queue
    # counters alone is racy: the background loop pops the pending keys
    # (pending_keys_count -> 0) before it submits the store tasks
    # (in_flight_task_count is still 0 in between), so a poll landing in that
    # window declares the store complete before anything reached L2. Anchor
    # the wait on the adapters' stored object counts, then use the queue
    # counters only to confirm the controller has settled.
    def flushed_to_l2() -> bool:
        status = sm.report_status()
        stored_total = sum(
            adapter["stored_object_count"] for adapter in status["l2_adapters"]
        )
        store_controller = status["store_controller"]
        return (
            stored_total >= stored_before + len(keys)
            and store_controller["in_flight_task_count"] == 0
            and store_controller["pending_keys_count"] == 0
        )

    ok = wait_for_condition(flushed_to_l2, timeout=timeout)
    assert ok, "Store to L2 did not complete within timeout"


def get_l1_memory_used(sm: StorageManager) -> int:
    """Return current L1 memory usage in bytes via public report_status."""
    return sm.report_status()["l1_manager"]["memory_used_bytes"]


def get_l1_object_count(sm: StorageManager) -> int:
    """Return current L1 object count via public report_status."""
    return sm.report_status()["l1_manager"]["total_object_count"]


def clear_and_wait_drained(sm: StorageManager, timeout: float = 10.0) -> None:
    """Clear L1 and poll until every object is evicted.

    After an L2 store the StoreController holds read locks on the stored objects
    for a short window, and ``StorageManager.clear`` keeps locked objects intact.
    A single clear right after the store therefore races the lock release and can
    leave objects behind. Retry clear() until the locks drop and L1 drains rather
    than relying on a fixed sleep.

    Raises:
        AssertionError: If L1 still holds objects after ``timeout`` seconds.
    """

    def drained() -> bool:
        sm.clear()
        return get_l1_object_count(sm) == 0

    if not wait_for_condition(drained, timeout=timeout):
        raise AssertionError(
            f"L1 did not drain after clear: {get_l1_object_count(sm)} objects remain"
        )


# =============================================================================
# Tests: Full round-trip through serde
# =============================================================================


class TestSerdeRoundTrip:
    """End-to-end store + prefetch through fp8 serde."""

    def test_store_and_prefetch_with_serde(self) -> None:
        """Write → L2 store (serialize) → clear L1 → prefetch (deserialize).

        Verifies all keys are recovered and L1 memory returns to clean state
        after all read locks are released.
        """
        cfg = make_storage_manager_config([make_mock_adapter_config(serde_type="fp8")])
        sm = StorageManager(cfg)
        layout = make_layout()
        keys = [make_object_key(i) for i in range(5)]

        write_and_wait_for_l2(sm, keys, layout)

        clear_and_wait_drained(sm)
        assert get_l1_object_count(sm) == 0

        # Prefetch from L2
        handle = sm.submit_prefetch_task(PrefetchRequestSpec(keys, layout))
        hits = wait_for_prefetch_status(sm, handle)
        assert hits == 5, f"Expected 5 L2 hits, got {hits}"

        # Read the data back
        with sm.read_prefetched_results(keys) as objs:
            assert objs is not None
            assert len(objs) == 5

        sm.finish_read_prefetched(keys)
        sm.close()

    def test_no_memory_leak_after_full_cycle(self) -> None:
        """After write → store → clear → prefetch → finish_read, L1 is clean."""
        cfg = make_storage_manager_config([make_mock_adapter_config(serde_type="fp8")])
        sm = StorageManager(cfg)
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]

        write_and_wait_for_l2(sm, keys, layout)
        clear_and_wait_drained(sm)

        # Prefetch
        handle = sm.submit_prefetch_task(PrefetchRequestSpec(keys, layout))
        hits = wait_for_prefetch_status(sm, handle)
        assert hits == 3

        # Release read locks
        sm.finish_read_prefetched(keys)

        # L1 should have objects (temporary prefetch results auto-delete on
        # finish_read), so memory should be back to 0.
        ok = wait_for_condition(
            lambda: get_l1_memory_used(sm) == 0,
            timeout=5.0,
        )
        assert ok, (
            f"L1 memory leak: {get_l1_memory_used(sm)} bytes still used "
            f"after releasing all read locks"
        )

        sm.close()


# =============================================================================
# Tests: Serde disabled (None) preserves existing behavior
# =============================================================================


class TestSerdeDisabled:
    """Verify that serde_config=None is equivalent to no-serde code path."""

    def test_store_and_prefetch_without_serde(self) -> None:
        """Same flow as the serde test, but without serde — must still work."""
        cfg = make_storage_manager_config([make_mock_adapter_config(serde_type=None)])
        sm = StorageManager(cfg)
        layout = make_layout()
        keys = [make_object_key(i) for i in range(5)]

        write_and_wait_for_l2(sm, keys, layout)
        clear_and_wait_drained(sm)

        handle = sm.submit_prefetch_task(PrefetchRequestSpec(keys, layout))
        hits = wait_for_prefetch_status(sm, handle)
        assert hits == 5

        with sm.read_prefetched_results(keys) as objs:
            assert objs is not None
            assert len(objs) == 5

        sm.finish_read_prefetched(keys)
        sm.close()

    def test_no_memory_leak_without_serde(self) -> None:
        """No-serde path should also leave L1 clean after a full cycle."""
        cfg = make_storage_manager_config([make_mock_adapter_config(serde_type=None)])
        sm = StorageManager(cfg)
        layout = make_layout()
        keys = [make_object_key(i) for i in range(3)]

        write_and_wait_for_l2(sm, keys, layout)
        clear_and_wait_drained(sm)

        handle = sm.submit_prefetch_task(PrefetchRequestSpec(keys, layout))
        hits = wait_for_prefetch_status(sm, handle)
        assert hits == 3
        sm.finish_read_prefetched(keys)

        ok = wait_for_condition(
            lambda: get_l1_memory_used(sm) == 0,
            timeout=5.0,
        )
        assert ok, f"L1 memory leak: {get_l1_memory_used(sm)} bytes still used"
        sm.close()


# =============================================================================
# Tests: Prefix trimming with serde
# =============================================================================


class TestSerdePartialPrefix:
    """Prefetch with serde when L2 has gaps in the key sequence."""

    def test_partial_prefix_with_serde(self) -> None:
        """L2 has keys {0,1,3,4} but not 2 → only prefix {0,1} returned."""
        cfg = make_storage_manager_config([make_mock_adapter_config(serde_type="fp8")])
        sm = StorageManager(cfg)
        layout = make_layout()

        # Write only keys 0, 1, 3, 4 (skip 2)
        keys_to_write = [make_object_key(i) for i in [0, 1, 3, 4]]
        write_and_wait_for_l2(sm, keys_to_write, layout)
        clear_and_wait_drained(sm)

        # Request all 5 keys — prefix should be 2 (gap at index 2)
        all_keys = [make_object_key(i) for i in range(5)]
        handle = sm.submit_prefetch_task(PrefetchRequestSpec(all_keys, layout))
        hits = wait_for_prefetch_status(sm, handle)

        assert hits is not None
        assert hits == 2, f"Expected prefix of 2, got {hits}"

        sm.finish_read_prefetched(all_keys[:hits])
        sm.close()


# =============================================================================
# Tests: Memory leak on repeated cycles
# =============================================================================


class TestSerdeMemoryStress:
    """Run multiple store-clear-prefetch cycles to catch temp buffer leaks."""

    def test_repeated_cycles_no_leak(self) -> None:
        """5 cycles of write → L2 → clear → prefetch → finish_read.

        After each cycle L1 should return to 0 bytes used. A temp buffer
        leak would accumulate across cycles.
        """
        cfg = make_storage_manager_config([make_mock_adapter_config(serde_type="fp8")])
        sm = StorageManager(cfg)
        layout = make_layout()

        for cycle in range(5):
            keys = [make_object_key(cycle * 10 + i) for i in range(3)]
            write_and_wait_for_l2(sm, keys, layout)
            clear_and_wait_drained(sm)

            handle = sm.submit_prefetch_task(PrefetchRequestSpec(keys, layout))
            hits = wait_for_prefetch_status(sm, handle)
            assert hits == 3, f"Cycle {cycle}: expected 3 hits, got {hits}"
            sm.finish_read_prefetched(keys)

            ok = wait_for_condition(
                lambda: get_l1_memory_used(sm) == 0,
                timeout=5.0,
            )
            assert ok, (
                f"Cycle {cycle}: L1 memory leak — {get_l1_memory_used(sm)} bytes used"
            )

        sm.close()


# =============================================================================
# Tests: Multiple keys with nothing in L2
# =============================================================================


class TestSerdeNoHits:
    """Prefetch with serde when L2 is empty — 0 hits, no crash."""

    def test_prefetch_no_hits_with_serde(self) -> None:
        """Empty L2 → prefetch returns 0 hits, no temp buffer leak."""
        cfg = make_storage_manager_config([make_mock_adapter_config(serde_type="fp8")])
        sm = StorageManager(cfg)
        layout = make_layout()

        keys = [make_object_key(i) for i in range(3)]
        handle = sm.submit_prefetch_task(PrefetchRequestSpec(keys, layout))
        hits = wait_for_prefetch_status(sm, handle)

        assert hits is not None
        assert hits == 0

        # No objects in L1 → memory should be 0
        ok = wait_for_condition(
            lambda: get_l1_memory_used(sm) == 0,
            timeout=5.0,
        )
        assert ok, (
            f"L1 memory leak after 0-hit prefetch: {get_l1_memory_used(sm)} bytes"
        )
        sm.close()


# =============================================================================
# Tests: Buffer sizing — no out-of-bound memory access
# =============================================================================


class TestSerdeBufferBounds:
    """Verify that the temp buffer sizing chain does not cause OOB access.

    The critical path:
      estimate_serialized_size(layout) -> buffer size (includes 1.5x margin)
      temp_buffer = estimate bytes as uint8
      serialize: writes num_elements bytes into temp (must fit)
      deserialize: reads num_elements bytes from temp (must fit)

    OOB would manifest as a crash (segfault / CUDA illegal access),
    wrong data, or a memory leak from corrupted L1 accounting.
    """

    def _run_roundtrip(
        self,
        layout: MemoryLayoutDesc,
        num_keys: int = 3,
        l1_size_mb: int = 512,
    ) -> None:
        """Helper: full store -> clear -> prefetch -> verify -> cleanup.

        Crashes here indicate OOB in serialize or deserialize.
        """
        cfg = make_storage_manager_config(
            [make_mock_adapter_config(serde_type="fp8")],
            l1_size_mb=l1_size_mb,
        )
        sm = StorageManager(cfg)
        keys = [make_object_key(i) for i in range(num_keys)]

        write_and_wait_for_l2(sm, keys, layout)
        clear_and_wait_drained(sm)
        assert get_l1_object_count(sm) == 0

        handle = sm.submit_prefetch_task(PrefetchRequestSpec(keys, layout))
        hits = wait_for_prefetch_status(sm, handle)
        assert hits == num_keys, f"Expected {num_keys} hits, got {hits}"

        with sm.read_prefetched_results(keys) as objs:
            assert objs is not None
            assert len(objs) == num_keys

        sm.finish_read_prefetched(keys)

        # Verify no memory leak (temp buffers fully cleaned up)
        ok = wait_for_condition(
            lambda: get_l1_memory_used(sm) == 0,
            timeout=5.0,
        )
        assert ok, f"L1 memory leak: {get_l1_memory_used(sm)} bytes after full cycle"
        sm.close()

    def test_bfloat16_layout(self) -> None:
        """bf16: 2 bytes/elem KV -> 1 byte/elem fp8. Buffer = 1.5 * numel."""
        layout = MemoryLayoutDesc(
            shapes=[torch.Size([100, 2, 512])],
            dtypes=[torch.bfloat16],
        )
        self._run_roundtrip(layout)

    def test_float16_layout(self) -> None:
        """fp16: 2 bytes/elem KV -> 1 byte/elem fp8. Same ratio as bf16."""
        layout = MemoryLayoutDesc(
            shapes=[torch.Size([100, 2, 512])],
            dtypes=[torch.float16],
        )
        self._run_roundtrip(layout)

    def test_float32_layout(self) -> None:
        """fp32: 4 bytes/elem KV -> 1 byte/elem fp8. 4x compression.

        The temp buffer (1.5 * numel bytes) is much smaller than the
        real KV buffer (4 * numel bytes). This is the highest compression
        ratio and the most likely to trigger sizing bugs.
        """
        layout = MemoryLayoutDesc(
            shapes=[torch.Size([50, 2, 256])],
            dtypes=[torch.float32],
        )
        self._run_roundtrip(layout)

    def test_large_tensor(self) -> None:
        """Large tensor (~4M elements, ~8MB bf16). Stress the buffer boundary."""
        layout = MemoryLayoutDesc(
            shapes=[torch.Size([256, 4, 2, 2048])],
            dtypes=[torch.bfloat16],
        )
        self._run_roundtrip(layout, num_keys=2, l1_size_mb=512)

    def test_small_tensor(self) -> None:
        """Tiny tensor (single element). Edge case for buffer sizing."""
        layout = MemoryLayoutDesc(
            shapes=[torch.Size([1])],
            dtypes=[torch.bfloat16],
        )
        self._run_roundtrip(layout)

    def test_odd_element_count(self) -> None:
        """Non-power-of-2 element count. Tests alignment edge cases.

        numel = 7 * 13 * 3 = 273, not divisible by any common alignment.
        """
        layout = MemoryLayoutDesc(
            shapes=[torch.Size([7, 13, 3])],
            dtypes=[torch.bfloat16],
        )
        self._run_roundtrip(layout)

    # NOTE: Multi-group layouts (multiple shapes/dtypes) are not tested here
    # because the fp8 serde accesses MemoryObj.tensor which only works for
    # single-group layouts. Multi-group would require per-group
    # serialize/deserialize via MemoryObj.get_tensor(index).
