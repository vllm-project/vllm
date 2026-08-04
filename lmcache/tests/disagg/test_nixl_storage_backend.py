# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Tuple
import argparse
import asyncio
import os
import tempfile
import threading
import time

# Third Party
import pytest
import torch

pytest.importorskip("nixl", reason="nixl package is required for nixl tests")

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.nixl_storage_backend import (
    NixlStorageBackend,
    NixlStorageConfig,
)
from lmcache.v1.transfer_channel.transfer_utils import get_correct_device

logger = init_logger(__name__)


def generate_test_data(
    num_objs: int, shape: torch.Size, dtype: torch.dtype = torch.bfloat16
) -> Tuple[List[CacheEngineKey], List[MemoryObj]]:
    keys = []
    objs = []
    allocator = AdHocMemoryAllocator(
        device=torch_device_type if torch_dev.is_available() else "cpu",
    )
    for i in range(num_objs):
        keys.append(
            CacheEngineKey(
                model_name="test_model",
                world_size=1,
                worker_id=0,
                chunk_hash=i,
                dtype=dtype,
            )
        )
        obj = allocator.allocate(shape, dtype, fmt=MemoryFormat.KV_2LTD)
        obj.tensor.fill_((i + 1) / num_objs)  # Fill with some test data
        objs.append(obj)
    return keys, objs


def calculate_throughput(total_bytes: int, elapsed_time: float) -> float:
    """Calculate throughput in GB/s"""
    if elapsed_time == 0:
        return float("inf")
    gb = total_bytes / (1024 * 1024 * 1024)
    return gb / elapsed_time


def create_test_config(
    buffer_device: str = torch_device_type if torch_dev.is_available() else "cpu",
    backend: str = "GDS_MT" if torch_dev.is_available() else "POSIX",
) -> LMCacheEngineConfig:
    """Create a test configuration for NixlStorageBackend"""
    config = LMCacheEngineConfig()
    config.nixl_buffer_size = 2**32  # 4GB
    config.nixl_buffer_device = buffer_device
    config.extra_config = {
        "enable_nixl_storage": True,
        "nixl_backend": backend,
        "nixl_pool_size": 10,
        "nixl_path": tempfile.mkdtemp(),  # Create a temporary directory for testing
    }
    return config


def create_test_metadata() -> LMCacheMetadata:
    """Create test metadata for NixlStorageBackend"""
    return LMCacheMetadata(
        model_name="test_model",
        worker_id=0,
        local_world_size=1,
        local_worker_id=0,
        world_size=1,
        kv_dtype=torch.bfloat16,
        kv_shape=(
            32,
            2,
            256,
            1024,
            128,
        ),  # (num_layer, 2, chunk_size, num_kv_head, head_size)
    )


@pytest.mark.no_shared_allocator
@pytest.mark.cuda
def test_nixl_storage_config():
    """Test NixlStorageConfig creation and validation"""
    config = create_test_config()
    metadata = create_test_metadata()

    nixl_config = NixlStorageConfig.from_cache_engine_config(config, metadata)
    if config.nixl_buffer_device == "cpu":
        # CPU mode shares LocalCPUBackend's pool (sized by max_local_cpu_size),
        # so the NIXL buffer_size is left at 0.
        assert nixl_config.buffer_size == 0
    else:
        assert nixl_config.buffer_size == config.nixl_buffer_size
    # buffer_device is normalized to include the worker's device id
    # (e.g. "cuda" -> f"{torch_device_type}:0"); "cpu" stays "cpu".
    assert nixl_config.buffer_device == get_correct_device(
        config.nixl_buffer_device, metadata.worker_id
    )
    assert nixl_config.backend == config.extra_config["nixl_backend"]
    assert nixl_config.pool_size == config.extra_config["nixl_pool_size"]
    assert nixl_config.path == config.extra_config["nixl_path"]

    # Test validation
    assert NixlStorageConfig.validate_nixl_backend("GDS", "cuda")
    assert NixlStorageConfig.validate_nixl_backend("GDS", "cpu")
    assert NixlStorageConfig.validate_nixl_backend("GDS_MT", "cuda")
    assert NixlStorageConfig.validate_nixl_backend("GDS_MT", "cpu")
    assert NixlStorageConfig.validate_nixl_backend("POSIX", "cpu")
    assert not NixlStorageConfig.validate_nixl_backend("POSIX", "cuda")
    assert not NixlStorageConfig.validate_nixl_backend("INVALID", "cpu")


def _make_obj_config(
    extra_overrides: dict | None = None,
) -> LMCacheEngineConfig:
    """Create a minimal OBJ-backend config for endpoint-list tests."""
    config = LMCacheEngineConfig()
    # CPU mode: nixl_buffer_size is rejected; the pool comes from LocalCPUBackend.
    config.nixl_buffer_device = "cpu"
    config.extra_config = {
        "enable_nixl_storage": True,
        "nixl_backend": "OBJ",
        "nixl_pool_size": 0,
        "nixl_path": tempfile.mkdtemp(),
    }
    if extra_overrides:
        config.extra_config.update(extra_overrides)
    return config


def _make_metadata(local_worker_id: int = 0) -> LMCacheMetadata:
    """Create test metadata with a configurable local_worker_id."""
    return LMCacheMetadata(
        model_name="test_model",
        worker_id=local_worker_id,
        local_world_size=1,
        local_worker_id=local_worker_id,
        world_size=1,
        kv_dtype=torch.bfloat16,
        kv_shape=(32, 2, 256, 1024, 128),
    )


@pytest.mark.no_shared_allocator
def test_endpoint_list_round_robin():
    """nixl_endpoint_list should assign endpoints to workers round-robin."""
    endpoints = [
        "https://node-0:9021",
        "https://node-1:9021",
        "https://node-2:9021",
    ]
    config = _make_obj_config({"nixl_endpoint_list": endpoints})

    for local_worker_id in range(6):
        metadata = _make_metadata(local_worker_id=local_worker_id)
        nixl_config = NixlStorageConfig.from_cache_engine_config(config, metadata)
        expected = endpoints[local_worker_id % len(endpoints)]
        assert nixl_config.backend_params["endpoint_override"] == expected


@pytest.mark.no_shared_allocator
def test_endpoint_list_overrides_endpoint_override():
    """nixl_endpoint_list takes precedence over backend_params endpoint_override."""
    endpoints = ["https://node-0:9021"]
    config = _make_obj_config(
        {
            "nixl_endpoint_list": endpoints,
            "nixl_backend_params": {
                "endpoint_override": "https://should-be-ignored:9021",
                "access_key": "key",
            },
        }
    )
    metadata = _make_metadata(local_worker_id=0)

    nixl_config = NixlStorageConfig.from_cache_engine_config(config, metadata)

    assert nixl_config.backend_params["endpoint_override"] == endpoints[0]
    # other params must be preserved
    assert nixl_config.backend_params["access_key"] == "key"


@pytest.mark.no_shared_allocator
def test_endpoint_list_does_not_mutate_original_config():
    """Setting nixl_endpoint_list must not mutate the original backend_params dict."""
    original_params = {"access_key": "key", "secret_key": "secret"}
    config = _make_obj_config(
        {
            "nixl_endpoint_list": ["https://node-0:9021"],
            "nixl_backend_params": original_params,
        }
    )
    metadata = _make_metadata(local_worker_id=0)

    NixlStorageConfig.from_cache_engine_config(config, metadata)

    assert "endpoint_override" not in original_params


@pytest.mark.no_shared_allocator
def test_endpoint_list_empty_raises():
    """nixl_endpoint_list=[] should raise ValueError before any nixl ops."""
    config = _make_obj_config({"nixl_endpoint_list": []})
    metadata = _make_metadata(local_worker_id=0)

    with pytest.raises(ValueError, match="nixl_endpoint_list is set but empty"):
        NixlStorageConfig.from_cache_engine_config(config, metadata)


@pytest.mark.no_shared_allocator
def test_no_endpoint_list_leaves_backend_params_unchanged():
    """When nixl_endpoint_list is absent, endpoint_override must not be injected."""
    config = _make_obj_config()  # no nixl_endpoint_list key
    metadata = _make_metadata(local_worker_id=0)

    nixl_config = NixlStorageConfig.from_cache_engine_config(config, metadata)

    assert "endpoint_override" not in nixl_config.backend_params


@pytest.mark.no_shared_allocator
def test_endpoint_list_malformed_url_raises():
    """A non-http(s) entry in nixl_endpoint_list should raise ValueError."""
    config = _make_obj_config({"nixl_endpoint_list": ["htps://typo.example.com"]})
    metadata = _make_metadata(local_worker_id=0)

    with pytest.raises(ValueError, match="is not a valid URL"):
        NixlStorageConfig.from_cache_engine_config(config, metadata)


@pytest.mark.no_shared_allocator
@pytest.mark.cuda
@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason="Requires available {torch_device_type} runtime",
)
def test_nixl_storage_backend_basic():
    """Test basic NixlStorageBackend operations"""
    config = create_test_config()
    metadata = create_test_metadata()

    thread_loop = None
    thread = None
    backend = None
    try:
        thread_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=thread_loop.run_forever)
        thread.start()

        backend = NixlStorageBackend.CreateNixlStorageBackend(
            config=config,
            loop=thread_loop,
            metadata=metadata,
        )

        # Test allocation
        shape = torch.Size([32, 2, 256, 1024])
        dtype = torch.bfloat16
        obj = backend.allocate(shape, dtype)
        assert obj is not None
        assert obj.tensor is not None
        assert obj.tensor.shape == shape
        assert obj.tensor.dtype == dtype

        # Test batched allocation
        batch_size = 5
        objs = backend.batched_allocate(shape, dtype, batch_size)
        assert objs is not None
        assert len(objs) == batch_size
        for obj in objs:
            assert obj.tensor is not None
            assert obj.tensor.shape == shape
            assert obj.tensor.dtype == dtype

    except Exception:
        raise
    finally:
        if backend:
            backend.close()
        if thread_loop and thread_loop.is_running():
            thread_loop.call_soon_threadsafe(thread_loop.stop)
        if thread and thread.is_alive():
            thread.join()
        # Cleanup temporary directory
        if os.path.exists(config.extra_config["nixl_path"]):
            os.rmdir(config.extra_config["nixl_path"])


@pytest.mark.no_shared_allocator
@pytest.mark.cuda
@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason="Requires available {torch_device_type} runtime",
)
def test_nixl_storage_backend_put_get():
    """Test put and get operations in NixlStorageBackend"""
    config = create_test_config()
    metadata = create_test_metadata()

    thread_loop = None
    thread = None
    backend = None
    try:
        thread_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=thread_loop.run_forever)
        thread.start()

        backend = NixlStorageBackend.CreateNixlStorageBackend(
            config=config,
            loop=thread_loop,
            metadata=metadata,
        )

        # Generate test data
        keys, objs = generate_test_data(10, torch.Size([32, 2, 256, 1024]))

        # Test contains before put
        for key in keys:
            assert not backend.contains(key)
            assert not backend.exists_in_put_tasks(key)

        # Test put
        backend.batched_submit_put_task(keys, objs)

        # Test get
        for key, original_obj in zip(keys, objs, strict=False):
            assert backend.contains(key)
            retrieved_obj = backend.get_blocking(key)
            assert retrieved_obj is not None
            assert retrieved_obj.tensor is not None
            assert torch.equal(retrieved_obj.tensor, original_obj.tensor)

        # Test batched get
        retrieved_objs = asyncio.run(
            backend.batched_get_non_blocking(lookup_id="test", keys=keys)
        )
        assert len(retrieved_objs) == len(objs)
        for retrieved_obj, original_obj in zip(retrieved_objs, objs, strict=False):
            assert retrieved_obj is not None
            assert retrieved_obj.tensor is not None
            assert torch.equal(retrieved_obj.tensor, original_obj.tensor)

        # Test remove
        for key in keys:
            backend.remove(key)
            assert not backend.contains(key)

    except Exception:
        raise
    finally:
        if backend:
            backend.close()
        if thread_loop and thread_loop.is_running():
            thread_loop.call_soon_threadsafe(thread_loop.stop)
        if thread and thread.is_alive():
            thread.join()
        # Cleanup temporary directory
        if os.path.exists(config.extra_config["nixl_path"]):
            os.rmdir(config.extra_config["nixl_path"])


@pytest.mark.no_shared_allocator
@pytest.mark.cuda
@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason="Requires available {torch_device_type} runtime",
)
def test_nixl_storage_backend_different_backends():
    """Test NixlStorageBackend with different backend types"""
    backends = (
        [
            ("GDS_MT", "cuda"),
            ("GDS", "cuda"),
            ("GDS_MT", "cpu"),
            ("GDS", "cpu"),
            ("POSIX", "cpu"),
        ]
        if torch_dev.is_available()
        else [
            ("GDS_MT", "cpu"),
            ("GDS", "cpu"),
            ("POSIX", "cpu"),
        ]
    )

    for backend_type, device in backends:
        config = create_test_config(buffer_device=device, backend=backend_type)
        metadata = create_test_metadata()

        thread_loop = None
        thread = None
        backend = None
        try:
            thread_loop = asyncio.new_event_loop()
            thread = threading.Thread(target=thread_loop.run_forever)
            thread.start()

            backend = NixlStorageBackend.CreateNixlStorageBackend(
                config=config,
                loop=thread_loop,
                metadata=metadata,
            )

            # Basic allocation test
            obj = backend.allocate(torch.Size([32, 2, 256, 1024]), torch.bfloat16)
            assert obj is not None
            assert obj.tensor is not None

        except Exception:
            raise
        finally:
            if backend:
                backend.close()
            if thread_loop and thread_loop.is_running():
                thread_loop.call_soon_threadsafe(thread_loop.stop)
            if thread and thread.is_alive():
                thread.join()
            # Cleanup temporary directory
            if os.path.exists(config.extra_config["nixl_path"]):
                os.rmdir(config.extra_config["nixl_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test NixlStorageBackend with different configurations"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="GDS_MT",
        choices=["GDS_MT", "GDS", "POSIX"],
        help="NIXL backend type to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=torch_device_type,
        choices=[torch_device_type, "cpu"],
        help="Device to use for buffer",
    )
    parser.add_argument(
        "--num-objs",
        type=int,
        default=100,
        help="Number of objects to test with",
    )
    args = parser.parse_args()

    # Create config and metadata
    config = create_test_config(buffer_device=args.device, backend=args.backend)
    metadata = create_test_metadata()

    thread_loop = None
    thread = None
    backend = None
    try:
        thread_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=thread_loop.run_forever)
        thread.start()

        # Create backend
        backend = NixlStorageBackend.CreateNixlStorageBackend(
            config=config,
            loop=thread_loop,
            metadata=metadata,
        )

        # Generate and test with data
        keys, objs = generate_test_data(args.num_objs, torch.Size([32, 2, 256, 1024]))
        total_size = sum(obj.get_size() for obj in objs)
        logger.info(
            "Generated %d objects with total size %.2f MB",
            len(objs),
            total_size / (1024 * 1024),
        )

        # Test put performance
        start_time = time.time()
        backend.batched_submit_put_task(keys, objs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        throughput = calculate_throughput(total_size, elapsed_time)
        logger.info("Put throughput: %.2f GB/s", throughput)

        # Test get performance
        start_time = time.time()
        retrieved_objs = asyncio.run(
            backend.batched_get_non_blocking(lookup_id="test", keys=keys)
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        throughput = calculate_throughput(total_size, elapsed_time)
        logger.info("Get throughput: %.2f GB/s", throughput)

        # Verify data
        for retrieved_obj, original_obj in zip(retrieved_objs, objs, strict=False):
            assert torch.equal(retrieved_obj.tensor, original_obj.tensor)

        logger.info("All tests passed successfully!")

    except Exception:
        raise
    finally:
        if backend:
            backend.close()
        if thread_loop and thread_loop.is_running():
            thread_loop.call_soon_threadsafe(thread_loop.stop)
        if thread and thread.is_alive():
            thread.join()
        # Cleanup temporary directory
        if os.path.exists(config.extra_config["nixl_path"]):
            os.rmdir(config.extra_config["nixl_path"])
