# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from unittest.mock import patch
import asyncio
import tempfile

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.pin_memory_allocator import PinMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend import LocalCPUBackend
from lmcache.v1.storage_backend.connector import CreateConnector

# Local
from .utils import (
    check_mem_obj_equal,
    close_asyncio_loop,
    dumb_cache_engine_key,
    init_asyncio_loop,
)


@pytest.mark.parametrize("lmserver_v1_process", ["cpu"], indirect=True)
@pytest.mark.parametrize(
    "url",
    [
        "lm://localhost:65000",
    ],
)
def test_lm_connector(url, autorelease_v1, lmserver_v1_process):
    if url.startswith("lm"):
        url = lmserver_v1_process.server_url

    async_loop, async_thread = init_asyncio_loop()
    memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    local_cpu_backend = _create_local_cpu_backend(memory_allocator, False)
    connector = autorelease_v1(CreateConnector(url, async_loop, local_cpu_backend))

    random_key = dumb_cache_engine_key()
    future = asyncio.run_coroutine_threadsafe(connector.exists(random_key), async_loop)
    assert not future.result()

    num_tokens = 1000
    mem_obj_shape = torch.Size([2, 32, num_tokens, 1024])
    dtype = torch.bfloat16
    memory_obj = local_cpu_backend.allocate(mem_obj_shape, dtype)
    memory_obj.ref_count_up()

    torch.manual_seed(42)
    test_tensor = torch.randint(0, 100, memory_obj.raw_data.shape, dtype=torch.int64)
    memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

    future = asyncio.run_coroutine_threadsafe(
        connector.put(random_key, memory_obj), async_loop
    )
    future.result()

    future = asyncio.run_coroutine_threadsafe(connector.exists(random_key), async_loop)
    assert future.result()
    assert memory_obj.get_ref_count() == 1

    future = asyncio.run_coroutine_threadsafe(connector.get(random_key), async_loop)
    retrieved_memory_obj = future.result()

    check_mem_obj_equal(
        [retrieved_memory_obj],
        [memory_obj],
    )

    close_asyncio_loop(async_loop, async_thread)
    local_cpu_backend.close()


@pytest.mark.parametrize("full_chunk", [True, False])
@pytest.mark.parametrize("save_chunk_meta", [True, False])
@pytest.mark.parametrize("use_mla", [True, False])
def test_fs_connector(autorelease_v1, full_chunk, save_chunk_meta, use_mla):
    """
    Test FSConnector: exists, put, get, list, and file store
    with the following conditions:
    full_chunk: is the block full
    save_chunk_meta: save the metadata of the chunk or not
    use_mla: is mla enabled
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup
        url = f"fs://host:0/{temp_dir}/"
        async_loop, async_thread = init_asyncio_loop()
        memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
        # full chunk's kv_shape (num_layer, 2, chunk_size, num_kv_head, head_size)
        kv_shape = (32, 1 if use_mla else 2, 256, 1 if use_mla else 8, 128)
        dtype = torch.bfloat16
        config = LMCacheEngineConfig.from_defaults(
            extra_config={"save_chunk_meta": save_chunk_meta}
        )
        local_cpu_backend = _create_local_cpu_backend(memory_allocator, use_mla, config)
        connector = autorelease_v1(
            CreateConnector(url, async_loop, local_cpu_backend, config)
        )
        random_key = dumb_cache_engine_key()

        # Test 1: Verify key doesn't exist initially
        future = asyncio.run_coroutine_threadsafe(
            connector.exists(random_key), async_loop
        )
        assert not future.result()

        # Test 2: Create and store test data
        # The size of the full chunk is 256.
        # If test unfull chunk, use 100 (<256) to allocate memory_obj.
        memory_obj_shape = torch.Size(
            [
                kv_shape[1],
                kv_shape[0],
                kv_shape[2] if full_chunk else 100,
                kv_shape[3] * kv_shape[4],
            ]
        )
        memory_obj = local_cpu_backend.allocate(memory_obj_shape, dtype)
        memory_obj.ref_count_up()
        # Fill with deterministic test data
        torch.manual_seed(42)
        test_tensor = torch.randint(
            0, 100, memory_obj.raw_data.shape, dtype=torch.int64
        )
        memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

        future = asyncio.run_coroutine_threadsafe(
            connector.put(random_key, memory_obj), async_loop
        )
        future.result()

        # Test 3: Verify key exists after putting data
        future = asyncio.run_coroutine_threadsafe(
            connector.exists(random_key), async_loop
        )
        assert future.result()
        assert memory_obj.get_ref_count() == 1

        # Test 4: Retrieve and verify data
        future = asyncio.run_coroutine_threadsafe(connector.get(random_key), async_loop)
        check_mem_obj_equal([future.result()], [memory_obj], use_mla)

        # Test 5: List the keys
        future = asyncio.run_coroutine_threadsafe(connector.list(), async_loop)
        assert future.result() == [random_key.to_string()]

        # Test 6: Verify file existence and other attributes
        # file name
        files = list(Path(temp_dir).glob("*.data"))
        assert len(files) == 1
        assert files[0].name == f"{random_key.to_string()}.data"

        # file size
        dtype_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = 1
        for dim in memory_obj_shape:
            num_elements *= dim
        expected_file_size = dtype_size * num_elements + (28 if save_chunk_meta else 0)
        assert files[0].stat().st_size == expected_file_size

        close_asyncio_loop(async_loop, async_thread)
        local_cpu_backend.close()


@pytest.mark.parametrize(
    "url",
    [
        "redis://localhost:6379",
        "redis://user:password@localhost:6379/0",
        "redis://:password@localhost:6379/1",
        "rediss://user:password@localhost:6380?ssl_cert_reqs=CERT_REQUIRED",
        "unix:///tmp/redis.sock",
        "plugin://redis",
    ],
)
def test_redis_connector(url, autorelease_v1):
    """Test Redis connector: exists, put, get operations.

    This test uses the MockRedis from conftest.py to simulate
    Redis behavior without requiring an actual Redis server.
    """

    async_loop, async_thread = init_asyncio_loop()
    memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    local_cpu_backend = _create_local_cpu_backend(memory_allocator, False)
    connector = autorelease_v1(CreateConnector(url, async_loop, local_cpu_backend))

    random_key = dumb_cache_engine_key()

    # Test 1: Verify key doesn't exist initially
    future = asyncio.run_coroutine_threadsafe(connector.exists(random_key), async_loop)
    assert not future.result()

    # Test 2: Create and store test data
    num_tokens = 1000
    mem_obj_shape = torch.Size([2, 32, num_tokens, 1024])
    dtype = torch.bfloat16
    memory_obj = local_cpu_backend.allocate(mem_obj_shape, dtype)
    memory_obj.ref_count_up()

    torch.manual_seed(42)
    test_tensor = torch.randint(0, 100, memory_obj.raw_data.shape, dtype=torch.int64)
    memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

    # Test 3: Put data
    future = asyncio.run_coroutine_threadsafe(
        connector.put(random_key, memory_obj), async_loop
    )
    future.result()

    # Test 4: Verify key exists after putting data
    future = asyncio.run_coroutine_threadsafe(connector.exists(random_key), async_loop)
    assert future.result()
    assert memory_obj.get_ref_count() == 1

    # Test 5: Retrieve and verify data
    future = asyncio.run_coroutine_threadsafe(connector.get(random_key), async_loop)
    retrieved_memory_obj = future.result()

    check_mem_obj_equal(
        [retrieved_memory_obj],
        [memory_obj],
    )

    close_asyncio_loop(async_loop, async_thread)
    local_cpu_backend.close()


@pytest.mark.parametrize(
    "url",
    [
        "redis-sentinel://localhost:26379,localhost:26380,localhost:26381",
        "redis-sentinel://user:password@localhost:26379,localhost:26380",
        "redis-sentinel://localhost:26379",
    ],
)
def test_redis_sentinel_connector(url, autorelease_v1):
    """Test Redis Sentinel connector: exists, put, get operations.

    This test uses the MockRedisSentinel from conftest.py to simulate
    Redis Sentinel behavior without requiring an actual Redis Sentinel setup.
    """
    # Standard
    import os

    # Set required environment variables for Redis Sentinel
    os.environ["REDIS_SERVICE_NAME"] = "mymaster"
    os.environ["REDIS_TIMEOUT"] = "5"

    async_loop, async_thread = init_asyncio_loop()
    memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    local_cpu_backend = _create_local_cpu_backend(memory_allocator, False)
    connector = autorelease_v1(CreateConnector(url, async_loop, local_cpu_backend))

    random_key = dumb_cache_engine_key()

    # Test 1: Verify key doesn't exist initially
    future = asyncio.run_coroutine_threadsafe(connector.exists(random_key), async_loop)
    assert not future.result()

    # Test 2: Create and store test data
    num_tokens = 1000
    mem_obj_shape = torch.Size([2, 32, num_tokens, 1024])
    dtype = torch.bfloat16
    memory_obj = local_cpu_backend.allocate(mem_obj_shape, dtype)
    memory_obj.ref_count_up()

    # Fill with deterministic test data for Redis Sentinel test
    torch.manual_seed(123)
    test_tensor = torch.randint(0, 100, memory_obj.raw_data.shape, dtype=torch.int64)
    memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

    # Test 3: Put data
    future = asyncio.run_coroutine_threadsafe(
        connector.put(random_key, memory_obj), async_loop
    )
    future.result()

    # Test 4: Verify key exists after putting data
    future = asyncio.run_coroutine_threadsafe(connector.exists(random_key), async_loop)
    assert future.result()

    # Test 5: Retrieve and verify data
    future = asyncio.run_coroutine_threadsafe(connector.get(random_key), async_loop)
    future.result()

    close_asyncio_loop(async_loop, async_thread)
    local_cpu_backend.close()


REDIS_CLUSTER_URLS = [
    "redis-cluster://host1:7000,host2:7000,host3:7000",
    "redis-cluster://clustercfg.cluster-name.id.region.cache.amazonaws.com:6379",
    "redis-cluster://user:password@host1:7000,host2:7000,host3:7000",
]


@pytest.mark.parametrize("url", REDIS_CLUSTER_URLS)
def test_redis_cluster_connector(url, autorelease_v1):
    """Test Redis Cluster connector: exists, put, get operations.

    This test uses the MockRedisCluster from conftest.py to simulate
    Redis Cluster behavior without requiring an actual Redis Cluster setup.
    """

    # Standard
    import os

    os.environ["REDIS_TIMEOUT"] = "3.5"

    async_loop, async_thread = init_asyncio_loop()
    memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    local_cpu_backend = _create_local_cpu_backend(memory_allocator, False)
    connector = autorelease_v1(CreateConnector(url, async_loop, local_cpu_backend))

    random_key = dumb_cache_engine_key()

    # Test 1: Verify key doesn't exist initially, test contains key not exist
    future = asyncio.run_coroutine_threadsafe(connector.exists(random_key), async_loop)
    assert not future.result()

    # Test 2: Create and store test data
    num_tokens = 1000
    mem_obj_shape = torch.Size([2, 32, num_tokens, 1024])
    dtype = torch.bfloat16
    memory_obj = local_cpu_backend.allocate(mem_obj_shape, dtype)
    memory_obj.ref_count_up()

    # Fill with deterministic test data
    torch.manual_seed(42)
    test_tensor = torch.randint(0, 100, memory_obj.raw_data.shape, dtype=torch.int64)
    memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

    # Test 3: Put data
    future = asyncio.run_coroutine_threadsafe(
        connector.put(random_key, memory_obj), async_loop
    )
    future.result()

    # Test 4: Verify key exists after putting data, test contains key exists
    future = asyncio.run_coroutine_threadsafe(connector.exists(random_key), async_loop)
    assert future.result()
    assert memory_obj.get_ref_count() == 1

    # Test 5: Retrieve and verify data
    future = asyncio.run_coroutine_threadsafe(connector.get(random_key), async_loop)
    retrieved_memory_obj = future.result()

    check_mem_obj_equal([retrieved_memory_obj], [memory_obj])

    close_asyncio_loop(async_loop, async_thread)
    local_cpu_backend.close()


@pytest.mark.parametrize("url", REDIS_CLUSTER_URLS)
def test_cluster_metadata_without_kv_bytes(url, autorelease_v1):
    async_loop, async_thread = init_asyncio_loop()
    memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    local_cpu_backend = _create_local_cpu_backend(memory_allocator, False)
    connector = autorelease_v1(CreateConnector(url, async_loop, local_cpu_backend))

    random_key = dumb_cache_engine_key()
    # build a small mem obj to get correct metadata bytes
    memory_obj = local_cpu_backend.allocate(torch.Size([2, 32, 8, 64]), torch.bfloat16)
    kv_bytes = memory_obj.byte_array
    meta = RemoteMetadata(
        len(kv_bytes),
        memory_obj.get_shapes(),
        memory_obj.get_dtypes(),
        memory_obj.get_memory_format(),
    )
    metadata_bytes = meta.serialize()

    # clean up memory object after getting metadata
    memory_obj.ref_count_down()

    # inject only metadata, no kv_bytes
    meta_key = random_key.to_string() + "metadata"
    connector._connector.cluster.set(meta_key, metadata_bytes)

    # get() should return None and remove the metadata without kv_bytes pair
    future = asyncio.run_coroutine_threadsafe(connector.get(random_key), async_loop)
    assert future.result() is None

    future = asyncio.run_coroutine_threadsafe(connector.exists(random_key), async_loop)
    assert not future.result()

    close_asyncio_loop(async_loop, async_thread)
    local_cpu_backend.close()


def _get_metadata(use_mla: bool):
    kv_shape = (32, 1 if use_mla else 2, 256, 1 if use_mla else 8, 128)
    dtype = torch.bfloat16
    metadata = LMCacheMetadata(
        model_name="deepseek/DeepSeek-R1",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=dtype,
        kv_shape=kv_shape,
        use_mla=use_mla,
    )
    return metadata


def _create_local_cpu_backend(memory_allocator, use_mla, config=None):
    if config is None:
        config = LMCacheEngineConfig.from_defaults()
    metadata = _get_metadata(use_mla)
    return LocalCPUBackend(
        config=config, metadata=metadata, memory_allocator=memory_allocator
    )


@patch("lmcache.v1.storage_backend.connector.redis_connector.RedisConnector")
def test_redis_plugin_custom_url(mock_redis_connector, autorelease_v1) -> None:
    """Verify that RedisConnectorAdapter extracts custom Redis URL
    from extra_config when loaded dynamically as a plugin.
    """
    async_loop, async_thread = init_asyncio_loop()
    memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    local_cpu_backend = _create_local_cpu_backend(memory_allocator, False)

    # Define custom Redis URL inside extra_config
    custom_url = "redis://my-custom-redis-host:6379"
    config = LMCacheEngineConfig.from_defaults(
        extra_config={"remote_storage_plugin.redis.redis_url": custom_url}
    )

    # Create connector using dynamic plugin schema URL
    autorelease_v1(
        CreateConnector("plugin://redis", async_loop, local_cpu_backend, config)
    )

    mock_redis_connector.assert_called_once_with(
        url=custom_url,
        loop=async_loop,
        local_cpu_backend=local_cpu_backend,
    )

    close_asyncio_loop(async_loop, async_thread)
    local_cpu_backend.close()
