# SPDX-License-Identifier: Apache-2.0
"""
End-to-end test for fp8 serde with a real filesystem L2 adapter.

Unlike test_serde_e2e.py (which uses MockL2Adapter), this test exercises
the full disk I/O path: L1 write -> fp8 serialize -> disk store -> L1
clear -> disk load -> fp8 deserialize -> L1 read. Verifies the data
round-trips within fp8 quantization error and that no temp buffers leak.
"""

# Standard
import os
import shutil
import tempfile
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
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import FSL2AdapterConfig
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


def _make_key(chunk_hash: bytes) -> ObjectKey:
    """Create an ObjectKey with the given raw hash bytes."""
    return ObjectKey(
        chunk_hash=chunk_hash,
        model_name="test-model",
        kv_rank=0,
    )


def wait_for_condition(
    predicate,
    timeout: float = 10.0,
    poll_interval: float = 0.1,
) -> bool:
    """Poll until predicate returns True or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return False


def wait_for_prefetch_status(
    sm: StorageManager,
    handle,
    timeout: float = 15.0,
    poll_interval: float = 0.1,
) -> int | None:
    """Poll query_prefetch_status until non-None or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = sm.query_prefetch_status(handle)
        if result is not None:
            return result.count_leading_ones()
        time.sleep(poll_interval)
    return None


# =============================================================================
# Test
# =============================================================================


class TestFp8SerdeFsRoundTrip:
    """Full disk-backed fp8 serde round-trip through StorageManager."""

    def test_write_serialize_clear_prefetch_deserialize(self) -> None:
        """Write KV → fp8 serialize → disk → clear L1 → prefetch → verify.

        Checks:
          - Serialized files appear on disk.
          - Prefetch returns all keys from L2.
          - Deserialized data correlates >0.95 with the original.
          - L1 memory returns to 0 after all locks released.
        """
        disk_path = tempfile.mkdtemp(prefix="lmcache_serde_fs_test_")
        try:
            self._run(disk_path)
        finally:
            shutil.rmtree(disk_path, ignore_errors=True)

    def _run(self, disk_path: str) -> None:
        # ---- Config ----
        fs_cfg = FSL2AdapterConfig(
            base_path=disk_path,
            relative_tmp_dir=None,
            read_ahead_size=None,
            use_odirect=False,
        )
        fs_cfg.serde_config = SerdeConfig(
            type="fp8", kwargs={"fp8_dtype": "float8_e4m3fn"}
        )

        sm_cfg = StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=4 << 30,
                    use_lazy=current_device_spec.is_pin_supported,
                    init_size_in_bytes=1 << 30,
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(adapters=[fs_cfg]),  # type: ignore[list-item]
        )

        sm = StorageManager(sm_cfg)

        kv_shape = torch.Size([2, 4, 256, 128])
        kv_dtype = torch.bfloat16
        layout = MemoryLayoutDesc(shapes=[kv_shape], dtypes=[kv_dtype])

        keys = [
            _make_key(b"\x00" * 31 + b"\x01"),
            _make_key(b"\x00" * 31 + b"\x02"),
        ]
        torch.manual_seed(0)
        originals = [torch.randn(kv_shape, dtype=kv_dtype) for _ in keys]

        # ---- Step 1: write to L1 ----
        reserved = sm.reserve_write(keys, layout, mode="new")
        assert len(reserved) == len(keys)
        for k, orig in zip(keys, originals, strict=True):
            mem_obj = reserved[k]
            assert mem_obj.tensor is not None
            mem_obj.tensor.view(kv_shape).view(kv_dtype).copy_(orig)
        sm.finish_write(keys)

        # ---- Step 2: wait for L2 store to disk ----
        # The FS adapter writes one file per key (staged as "*.tmp" in the
        # same directory), so wait for all final files rather than the
        # first: the controller's queue counters can both read zero in the
        # window between popping pending keys and submitting the store
        # tasks, so they only confirm settling after the files prove the
        # store actually happened.
        def count_stored_files() -> int:
            return sum(
                1
                for e in os.scandir(disk_path)
                if e.is_file() and not e.name.endswith(".tmp")
            )

        ok = wait_for_condition(
            lambda: count_stored_files() >= len(keys),
            timeout=10.0,
        )
        assert ok, f"Expected {len(keys)} files under {disk_path}"

        ok = wait_for_condition(
            lambda: (
                sm.report_status()["store_controller"]["in_flight_task_count"] == 0
                and sm.report_status()["store_controller"]["pending_keys_count"] == 0
            ),
            timeout=10.0,
        )
        assert ok, "Store controller did not finish in time"

        # ---- Step 3: clear L1 ----
        sm.clear(force=True)
        assert sm.report_status()["l1_manager"]["total_object_count"] == 0

        # ---- Step 4: prefetch (disk load + fp8 deserialize) ----
        handle = sm.submit_prefetch_task(PrefetchRequestSpec(keys, layout))
        prefix_hits = wait_for_prefetch_status(sm, handle)
        assert prefix_hits is not None, "Prefetch never completed"
        assert prefix_hits == len(keys), (
            f"Expected {len(keys)} prefix hits, got {prefix_hits}"
        )

        # ---- Step 5: verify fp8 round-trip ----
        with sm.read_prefetched_results(keys) as mem_objs:
            assert mem_objs is not None
            assert len(mem_objs) == len(keys)
            for orig, mem_obj in zip(originals, mem_objs, strict=True):
                assert mem_obj.tensor is not None
                got = mem_obj.tensor.view(kv_shape).view(kv_dtype)
                corr = torch.corrcoef(
                    torch.stack([got.float().flatten(), orig.float().flatten()])
                )[0, 1].item()
                assert corr > 0.95, f"fp8 round-trip correlation too low: {corr:.4f}"

        sm.finish_read_prefetched(keys)

        # ---- Step 6: verify no memory leak ----
        ok = wait_for_condition(
            lambda: sm.report_status()["l1_manager"]["memory_used_bytes"] == 0,
            timeout=5.0,
        )
        assert ok, (
            f"L1 memory leak: "
            f"{sm.report_status()['l1_manager']['memory_used_bytes']} bytes"
        )

        sm.close()
