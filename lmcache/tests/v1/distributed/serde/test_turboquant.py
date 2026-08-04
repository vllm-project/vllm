# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TurboQuant serde skeleton.
"""

# Standard
from pathlib import Path
from typing import cast
import shutil
import tempfile
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.native_storage_ops import Bitmap
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
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.serde import (
    SerdeConfig,
    create_serde_processor,
    get_registered_serde_types,
)
from lmcache.v1.distributed.serde.turboquant import (
    TurboQuantSerdeConfig,
    TurboQuantSerializer,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_management import MemoryObj


def test_turboquant_registered() -> None:
    assert "turboquant" in get_registered_serde_types()


def test_create_turboquant_serde_processor() -> None:
    processor = create_serde_processor(
        SerdeConfig(
            type="turboquant",
            kwargs={
                "preset": "turboquant_k8v4",
                "head_dim": 128,
                "block_size": 16,
                "max_workers": 1,
            },
        )
    )
    processor.close()


@pytest.mark.parametrize(
    (
        "preset",
        "key_fp8",
        "key_quant_bits",
        "key_mse_bits",
        "value_quant_bits",
        "norm_correction",
        "key_packed_size",
        "value_packed_size",
        "slot_size",
        "slot_size_aligned",
    ),
    [
        ("turboquant_k8v4", True, 8, 0, 4, False, 128, 68, 196, 196),
        ("turboquant_4bit_nc", False, 4, 4, 4, True, 66, 68, 134, 134),
        ("turboquant_k3v4_nc", False, 3, 3, 4, True, 50, 68, 118, 118),
        ("turboquant_3bit_nc", False, 3, 3, 3, True, 50, 52, 102, 102),
    ],
)
def test_turboquant_config_sizes_head_dim_128(
    preset: str,
    key_fp8: bool,
    key_quant_bits: int,
    key_mse_bits: int,
    value_quant_bits: int,
    norm_correction: bool,
    key_packed_size: int,
    value_packed_size: int,
    slot_size: int,
    slot_size_aligned: int,
) -> None:
    cfg = TurboQuantSerdeConfig(
        preset=preset,
        head_dim=128,
        block_size=16,
    )

    assert cfg.key_fp8 is key_fp8
    assert cfg.key_quant_bits == key_quant_bits
    assert cfg.key_mse_bits == key_mse_bits
    assert cfg.value_quant_bits == value_quant_bits
    assert cfg.norm_correction is norm_correction
    assert cfg.key_packed_size == key_packed_size
    assert cfg.value_packed_size == value_packed_size
    assert cfg.slot_size == slot_size
    assert cfg.slot_size_aligned == slot_size_aligned


def test_turboquant_config_rejects_invalid_preset() -> None:
    cfg = TurboQuantSerdeConfig(
        preset="turboquant_invalid",
        head_dim=128,
        block_size=16,
    )

    with pytest.raises(ValueError, match="Unsupported TurboQuant preset"):
        _ = cfg.key_quant_bits


def test_estimate_serialized_size_k8v4() -> None:
    cfg = TurboQuantSerdeConfig(
        preset="turboquant_k8v4",
        head_dim=128,
        block_size=16,
    )
    serializer = TurboQuantSerializer(cfg)

    # LMCache KV layout: [2, num_layers, num_tokens, hidden_dim]
    # hidden_dim = num_heads * head_dim = 4 * 128 = 512
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 3, 20, 512])],
        dtypes=[torch.bfloat16],
    )

    # num_layers = 3
    # default skip_first_layers=2 and skip_last_layers=2 leaves no middle
    # layers to compress, so all layers are stored as raw bfloat16 KV.
    expected = 2 * 3 * 20 * 512 * torch.bfloat16.itemsize

    assert serializer.estimate_serialized_size(layout) == expected


def test_estimate_serialized_size_rejects_invalid_kv_size() -> None:
    cfg = TurboQuantSerdeConfig(
        preset="turboquant_k8v4",
        head_dim=128,
        block_size=16,
    )
    serializer = TurboQuantSerializer(cfg)

    layout = MemoryLayoutDesc(
        shapes=[torch.Size([1, 3, 20, 512])],
        dtypes=[torch.bfloat16],
    )

    with pytest.raises(ValueError, match="kv_size=2"):
        serializer.estimate_serialized_size(layout)


def test_estimate_serialized_size_rejects_bad_head_dim() -> None:
    cfg = TurboQuantSerdeConfig(
        preset="turboquant_k8v4",
        head_dim=128,
        block_size=16,
    )
    serializer = TurboQuantSerializer(cfg)

    layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 3, 20, 500])],
        dtypes=[torch.bfloat16],
    )

    with pytest.raises(ValueError, match="must be divisible"):
        serializer.estimate_serialized_size(layout)


# =============================================================================
# GPU E2E test: StorageManager + SerdeL2AdapterWrapper + MockL2Adapter
# =============================================================================


def _make_turboquant_object_key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="turboquant_test_model",
        kv_rank=0,
    )


def _make_turboquant_layout() -> MemoryLayoutDesc:
    # TurboQuant serde currently expects [2, num_layers, num_tokens, hidden_dim].
    # hidden_dim = num_heads * head_dim = 4 * 128 = 512.
    return MemoryLayoutDesc(
        shapes=[torch.Size([2, 2, 32, 512])],
        dtypes=[torch.bfloat16],
    )


def _wait_for_condition(
    predicate,
    timeout: float = 20.0,
    poll_interval: float = 0.05,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return False


def _wait_for_prefetch_status(
    sm: StorageManager,
    handle,
    timeout: float = 20.0,
    poll_interval: float = 0.05,
) -> Bitmap | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = sm.query_prefetch_status(handle)
        if result is not None:
            return result
        time.sleep(poll_interval)
    return None


def _finish_read_prefetched_until_clean(
    sm: StorageManager,
    keys: list[ObjectKey],
    timeout: float = 30.0,
) -> None:
    """Release prefetched temporary objects and wait for L1 cleanup.

    StorageManager / serde wrapper paths may hold more than one read lock
    on temporary prefetched objects. Release repeatedly until L1 is clean.
    """
    for _ in range(4):
        sm.finish_read_prefetched(keys)
        ok = _wait_for_condition(
            lambda: (
                sm.report_status()["l1_manager"]["memory_used_bytes"] == 0
                and sm.report_status()["l1_manager"]["total_object_count"] == 0
                and sm.report_status()["l1_manager"]["read_locked_count"] == 0
                and sm.report_status()["l1_manager"]["write_locked_count"] == 0
                and sm.report_status()["l1_manager"]["temporary_count"] == 0
            ),
            timeout=timeout,
        )
        if ok:
            return

    raise AssertionError(f"L1 memory not released: {sm.report_status()['l1_manager']}")


def _make_turboquant_storage_manager(preset: str) -> StorageManager:
    adapter_cfg = MockL2AdapterConfig(
        max_size_gb=0.1,
        mock_bandwidth_gb=10.0,
    )
    adapter_cfg.serde_config = SerdeConfig(
        type="turboquant",
        kwargs={
            "preset": preset,
            "head_dim": 128,
            "block_size": 16,
            "max_workers": 1,
        },
    )

    cfg = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=256 * 1024 * 1024,
                use_lazy=torch.cuda.is_available(),
                init_size_in_bytes=64 * 1024 * 1024,
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(adapters=[adapter_cfg]),
    )
    return StorageManager(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    ("preset", "corr_lower_bound"),
    [
        ("turboquant_k8v4", 0.95),
        ("turboquant_4bit_nc", 0.90),
        ("turboquant_k3v4_nc", 0.85),
        ("turboquant_3bit_nc", 0.80),
    ],
)
def test_turboquant_storage_manager_roundtrip(
    preset: str,
    corr_lower_bound: float,
) -> None:
    """Store/load through StorageManager with TurboQuant serde.

    This verifies:
    reserve_write -> finish_write -> StoreController
    -> SerdeL2AdapterWrapper serialize -> MockL2Adapter store
    -> clear L1 -> prefetch -> MockL2Adapter load
    -> SerdeL2AdapterWrapper deserialize -> read_prefetched_results.
    """
    sm = _make_turboquant_storage_manager(preset)
    layout = _make_turboquant_layout()
    keys = [_make_turboquant_object_key(i) for i in range(3)]

    try:
        ret = sm.reserve_write(keys, layout, mode="new")
        assert len(ret) == len(keys), f"reserve_write got {len(ret)} / {len(keys)}"

        original_by_key = {}
        for i, key in enumerate(keys):
            obj = ret[key]
            assert obj.tensor is not None

            torch.manual_seed(1234 + i)
            data = torch.randn(
                obj.tensor.shape,
                dtype=obj.tensor.dtype,
                device=obj.tensor.device,
            )
            data = data + float(i)

            obj.tensor.copy_(data)
            original_by_key[key] = data.detach().clone()

        sm.finish_write(list(ret.keys()))

        # Wait until both the store controller and the serde wrapper cleanup
        # have released temporary objects and locks.
        ok = _wait_for_condition(
            lambda: (
                sm.report_status()["store_controller"]["in_flight_task_count"] == 0
                and sm.report_status()["store_controller"]["pending_keys_count"] == 0
                and sm.report_status()["l1_manager"]["write_locked_count"] == 0
                and sm.report_status()["l1_manager"]["read_locked_count"] == 0
                and sm.report_status()["l1_manager"]["temporary_count"] == 0
            ),
            timeout=120.0,
        )
        assert ok, "Store to L2 did not fully complete"

        sm.clear()
        ok = _wait_for_condition(
            lambda: (
                sm.report_status()["l1_manager"]["total_object_count"] == 0
                and sm.report_status()["l1_manager"]["memory_used_bytes"] == 0
                and sm.report_status()["l1_manager"]["read_locked_count"] == 0
                and sm.report_status()["l1_manager"]["write_locked_count"] == 0
                and sm.report_status()["l1_manager"]["temporary_count"] == 0
            ),
            timeout=120.0,
        )
        assert ok, f"L1 not cleared: {sm.report_status()['l1_manager']}"

        handle = sm.submit_prefetch_task(PrefetchRequestSpec(keys, layout))
        hit_bitmap = _wait_for_prefetch_status(sm, handle, timeout=120.0)
        assert hit_bitmap is not None
        hits = hit_bitmap.count_leading_ones()
        assert hits == len(keys), f"Expected {len(keys)} hits, got {hits}"

        with sm.read_prefetched_results(keys) as objs:
            assert objs is not None
            assert len(objs) == len(keys)

            for key, obj in zip(keys, objs, strict=True):
                assert obj.tensor is not None

                recovered = obj.tensor
                original = original_by_key[key]

                orig_f = original.float().flatten()
                rec_f = recovered.float().flatten()

                corr = torch.corrcoef(torch.stack([orig_f, rec_f]))[0, 1].item()
                mae = torch.mean(torch.abs(orig_f - rec_f)).item()
                mse = torch.mean((orig_f - rec_f) ** 2).item()

                assert corr > corr_lower_bound, (
                    f"low corr for preset={preset}, key={key}: "
                    f"corr={corr}, mae={mae}, mse={mse}"
                )

        _finish_read_prefetched_until_clean(sm, keys)
    finally:
        sm.close()


# =============================================================================
# GPU direct test: TurboQuantSerializer + TurboQuantDeserializer
# =============================================================================


class _FakeMemoryObj:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    ("preset", "expected_ratio_lower_bound", "corr_lower_bound"),
    [
        ("turboquant_k8v4", 2.60, 0.95),
        ("turboquant_4bit_nc", 3.75, 0.90),
        ("turboquant_k3v4_nc", 4.20, 0.85),
        ("turboquant_3bit_nc", 4.85, 0.80),
    ],
)
def test_turboquant_direct_roundtrip_cuda(
    preset: str,
    expected_ratio_lower_bound: float,
    corr_lower_bound: float,
) -> None:
    """Direct GPU round-trip through TurboQuant serializer/deserializer."""
    # First Party
    from lmcache.v1.distributed.serde.turboquant import TurboQuantDeserializer

    device = torch.device("cuda:0")
    dtype = torch.float16

    # LMCache KV layout: [2, num_layers, num_tokens, hidden_dim]
    num_layers = 4
    num_tokens = 128
    num_heads = 8
    head_dim = 128
    hidden_dim = num_heads * head_dim

    cfg = TurboQuantSerdeConfig(
        preset=preset,
        head_dim=head_dim,
        block_size=16,
        skip_first_layers=0,
        skip_last_layers=0,
    )

    torch.manual_seed(2026)
    shape = torch.Size([2, num_layers, num_tokens, hidden_dim])
    original = torch.randn(shape, dtype=dtype, device=device)

    serializer = TurboQuantSerializer(cfg)
    deserializer = TurboQuantDeserializer(cfg)

    layout = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])
    n_bytes = serializer.estimate_serialized_size(layout)

    compressed = torch.empty(n_bytes, dtype=torch.uint8, device=device)
    recovered = torch.empty_like(original)

    written = serializer.serialize(
        cast(MemoryObj, _FakeMemoryObj(original)),
        cast(MemoryObj, _FakeMemoryObj(compressed)),
        _make_turboquant_object_key(0),
    )
    assert written == n_bytes

    deserializer.deserialize(
        cast(MemoryObj, _FakeMemoryObj(compressed)),
        cast(MemoryObj, _FakeMemoryObj(recovered)),
        _make_turboquant_object_key(0),
    )

    orig_f = original.float().flatten()
    rec_f = recovered.float().flatten()

    corr = torch.corrcoef(torch.stack([orig_f, rec_f]))[0, 1].item()
    mae = torch.mean(torch.abs(orig_f - rec_f)).item()
    mse = torch.mean((orig_f - rec_f) ** 2).item()

    original_bytes = original.numel() * original.element_size()
    ratio = original_bytes / n_bytes

    assert ratio >= expected_ratio_lower_bound
    assert corr > corr_lower_bound, (
        f"low corr for preset={preset}: corr={corr}, mae={mae}, mse={mse}"
    )


# =============================================================================
# GPU E2E test: StorageManager + SerdeL2AdapterWrapper + FSL2Adapter
# =============================================================================


def _make_turboquant_fs_storage_manager(
    base_path: str,
    preset: str,
) -> StorageManager:
    adapter_cfg = FSL2AdapterConfig(
        base_path=base_path,
        relative_tmp_dir="tmp",
        use_odirect=False,
    )
    adapter_cfg.serde_config = SerdeConfig(
        type="turboquant",
        kwargs={
            "preset": preset,
            "head_dim": 128,
            "block_size": 16,
            "max_workers": 1,
        },
    )

    cfg = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=256 * 1024 * 1024,
                use_lazy=torch.cuda.is_available(),
                init_size_in_bytes=64 * 1024 * 1024,
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(adapters=[adapter_cfg]),
    )
    return StorageManager(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    ("preset", "corr_lower_bound"),
    [
        ("turboquant_k8v4", 0.95),
        ("turboquant_4bit_nc", 0.90),
        ("turboquant_k3v4_nc", 0.85),
        ("turboquant_3bit_nc", 0.80),
    ],
)
def test_turboquant_fs_storage_manager_roundtrip(
    preset: str,
    corr_lower_bound: float,
) -> None:
    """Store/load through FSL2Adapter with TurboQuant serde."""
    base_dir = tempfile.mkdtemp(prefix="lmcache_turboquant_fs_")
    sm = None

    try:
        sm = _make_turboquant_fs_storage_manager(base_dir, preset)
        layout = _make_turboquant_layout()
        keys = [_make_turboquant_object_key(i) for i in range(3)]

        ret = sm.reserve_write(keys, layout, mode="new")
        assert len(ret) == len(keys), f"reserve_write got {len(ret)} / {len(keys)}"

        original_by_key = {}
        for i, key in enumerate(keys):
            obj = ret[key]
            assert obj.tensor is not None

            torch.manual_seed(5678 + i)
            data = torch.randn(
                obj.tensor.shape,
                dtype=obj.tensor.dtype,
                device=obj.tensor.device,
            )
            data = data + float(i)

            obj.tensor.copy_(data)
            original_by_key[key] = data.detach().clone()

        sm.finish_write(list(ret.keys()))

        ok = _wait_for_condition(
            lambda: (
                sm.report_status()["store_controller"]["in_flight_task_count"] == 0
                and sm.report_status()["store_controller"]["pending_keys_count"] == 0
                and sm.report_status()["l1_manager"]["write_locked_count"] == 0
                and sm.report_status()["l1_manager"]["read_locked_count"] == 0
                and sm.report_status()["l1_manager"]["temporary_count"] == 0
            ),
            timeout=120.0,
        )
        assert ok, "Store to FS L2 did not fully complete"

        stored_files = [p for p in Path(base_dir).rglob("*") if p.is_file()]
        assert len(stored_files) >= len(keys)

        sm.clear()
        ok = _wait_for_condition(
            lambda: (
                sm.report_status()["l1_manager"]["total_object_count"] == 0
                and sm.report_status()["l1_manager"]["memory_used_bytes"] == 0
                and sm.report_status()["l1_manager"]["read_locked_count"] == 0
                and sm.report_status()["l1_manager"]["write_locked_count"] == 0
                and sm.report_status()["l1_manager"]["temporary_count"] == 0
            ),
            timeout=120.0,
        )
        assert ok, f"L1 not cleared: {sm.report_status()['l1_manager']}"

        handle = sm.submit_prefetch_task(PrefetchRequestSpec(keys, layout))
        hit_bitmap = _wait_for_prefetch_status(sm, handle, timeout=120.0)
        assert hit_bitmap is not None
        hits = hit_bitmap.count_leading_ones()
        assert hits == len(keys), f"Expected {len(keys)} hits, got {hits}"

        with sm.read_prefetched_results(keys) as objs:
            assert objs is not None
            assert len(objs) == len(keys)

            for key, obj in zip(keys, objs, strict=True):
                assert obj.tensor is not None

                recovered = obj.tensor
                original = original_by_key[key]

                orig_f = original.float().flatten()
                rec_f = recovered.float().flatten()

                corr = torch.corrcoef(torch.stack([orig_f, rec_f]))[0, 1].item()
                mae = torch.mean(torch.abs(orig_f - rec_f)).item()
                mse = torch.mean((orig_f - rec_f) ** 2).item()

                assert corr > corr_lower_bound, (
                    f"low corr for preset={preset}, key={key}: "
                    f"corr={corr}, mae={mae}, mse={mse}"
                )

        _finish_read_prefetched_until_clean(sm, keys)

    finally:
        if sm is not None:
            sm.close()
        shutil.rmtree(base_dir, ignore_errors=True)
