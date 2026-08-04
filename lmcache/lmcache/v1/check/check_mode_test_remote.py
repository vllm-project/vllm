# SPDX-License-Identifier: Apache-2.0
"""Test mode implementation for basic checks"""

# Standard
import asyncio

# First Party
from lmcache.integration.vllm.utils import lmcache_get_or_create_config
from lmcache.v1.check import check_mode

# Import shared utilities
from lmcache.v1.check.utils import (
    DEFAULT_KV_DTYPE_STR,
    EventLoopManager,
    _get_default_metadata,
    create_test_key,
    parse_kv_dtype,
    run_common_test_framework,
    validate_get_results,
)
from lmcache.v1.memory_management import MemoryObj

# Import from lmcache with absolute paths
from lmcache.v1.storage_backend import RemoteBackend
from lmcache.v1.storage_backend.connector import InstrumentedRemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


async def async_contains_backend(backend, key):
    """Async wrapper for backend contains method"""
    return backend.contains(key)


async def async_get_backend(backend, key):
    """Async wrapper for backend get_blocking method"""
    return backend.get_blocking(key)


async def async_submit_put_backend(backend, key, memory_obj):
    """Async wrapper for backend submit_put_task"""
    future = backend.submit_put_task(key, memory_obj)
    # Wait for the future to complete with timeout
    try:
        await asyncio.wait_for(asyncio.wrap_future(future), timeout=10.0)
        return True
    except asyncio.TimeoutError:
        print(f"Put task timed out for key: {key}")
        return False


def create_test_memory_obj(
    backend: RemoteBackend, local_cpu_backend: LocalCPUBackend
) -> MemoryObj:
    """Create a test MemoryObj for testing."""
    if backend.connection is None:
        raise ValueError("Backend connection is None")

    if isinstance(backend.connection, InstrumentedRemoteConnector):
        connector = backend.connection.getWrappedConnector()
    else:
        connector = backend.connection

    return local_cpu_backend.allocate(
        connector.meta_shapes, connector.meta_dtypes, connector.meta_fmt
    )


def create_test_data_for_backend(
    backend,
    local_cpu_backend,
    model,
    num_tests,
    kv_dtype=None,
):
    """Create test data for backend based tests"""
    # Group 1: Non-existing keys
    kw = {} if kv_dtype is None else {"kv_dtype": kv_dtype}
    non_exist_keys = [
        create_test_key(model, f"non_exist_{i}", **kw) for i in range(num_tests)
    ]

    # Group 2: Existing keys
    exist_keys = [create_test_key(model, f"exist_{i}", **kw) for i in range(num_tests)]
    exist_memories = [
        create_test_memory_obj(backend, local_cpu_backend) for _ in range(num_tests)
    ]

    return non_exist_keys, exist_keys, exist_memories, num_tests


@check_mode("test_remote")
async def run_test_mode(model: str, **kwargs):
    """Run connector test mode"""
    kv_dtype_str = kwargs.get("kv_dtype") or DEFAULT_KV_DTYPE_STR
    kv_dtype = parse_kv_dtype(kv_dtype_str)
    if kv_dtype is None:
        print("Error: unsupported --kv-dtype '%s'" % kv_dtype_str)
        return

    obj_size = kwargs.get("obj_size")

    config = lmcache_get_or_create_config()
    metadata = _get_default_metadata(model, kv_dtype=kv_dtype, obj_size=obj_size)

    # Create and start event loop manager
    loop_manager = EventLoopManager()
    loop_manager.start()

    local_cpu_backend = LocalCPUBackend(
        config=config, metadata=metadata, dst_device="cpu"
    )

    backend = RemoteBackend(
        config=config,
        metadata=metadata,
        loop=loop_manager.get_loop(),
        local_cpu_backend=local_cpu_backend,
        dst_device="cpu",
    )

    try:
        # Create test context for the common framework
        test_context = {
            "create_test_data_func": create_test_data_for_backend,
            "async_contains_func": async_contains_backend,
            "async_put_func": async_submit_put_backend,
            "async_get_func": async_get_backend,
            "validate_get_func": validate_get_results,
            "test_object": backend,
            "extra_args": [
                local_cpu_backend,
            ],
            "kv_dtype": kv_dtype,
            "obj_size": obj_size,
        }

        # Run the common test framework
        num_tests = kwargs.get("num_keys", 5)
        settle_time = kwargs.get("settle_time", 0.0)
        await run_common_test_framework(
            test_context, model, num_tests=num_tests, settle_time=settle_time
        )

    except Exception as e:
        print(f"Test Failed - Error: {e}")
    finally:
        # Clean up
        try:
            if backend:
                backend.close()
        except Exception as e:
            print(f"Error closing backend: {e}")

        # Stop the event loop
        loop_manager.stop()
