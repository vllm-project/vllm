# SPDX-License-Identifier: Apache-2.0
"""Test mode implementation for basic checks"""

# Standard
import asyncio

# First Party
from lmcache.v1.check import check_mode

# Import shared utilities
from lmcache.v1.check.utils import (
    DEFAULT_KV_DTYPE_STR,
    create_storage_manager_with_config,
    create_test_key,
    create_test_memory_obj_for_storage_manager,
    find_remote_backend,
    parse_kv_dtype,
    run_common_test_framework,
    validate_get_results,
    wait_put_tasks_complete,
)


async def async_contains_storage_manager(storage_manager, key):
    """Async wrapper for storage manager contains method"""
    # Use asyncio.to_thread to make the synchronous call truly async
    # This allows for proper timeout handling and non-blocking execution
    result = await asyncio.to_thread(storage_manager.contains, key)
    return result is not None


async def async_get_storage_manager(storage_manager, key):
    """Async wrapper for storage manager get method"""
    # Use asyncio.to_thread to make the synchronous call truly async
    # This allows for proper timeout handling and non-blocking execution
    return await asyncio.to_thread(storage_manager.get, key)


async def async_submit_put_storage_manager(storage_manager, key, memory_obj):
    """Async wrapper for storage manager batched_put"""
    try:
        # Use asyncio.to_thread to make the synchronous calls truly async
        # This allows for proper timeout handling and non-blocking execution
        await asyncio.to_thread(storage_manager.batched_put, [key], [memory_obj])
        await asyncio.to_thread(
            wait_put_tasks_complete, find_remote_backend(storage_manager)
        )
        return True
    except Exception as e:
        print(f"Put task failed for key: {key}, error: {e}")
        return False


def create_test_data_for_storage_manager(
    storage_manager,
    metadata,
    model,
    num_tests,
    kv_dtype=None,
):
    """Create test data for storage manager based tests"""
    # Group 1: Non-existing keys
    kw = {} if kv_dtype is None else {"kv_dtype": kv_dtype}
    non_exist_keys = [
        create_test_key(model, f"non_exist_{i}", **kw) for i in range(num_tests)
    ]

    # Group 2: Existing keys
    exist_keys = [create_test_key(model, f"exist_{i}", **kw) for i in range(num_tests)]
    exist_memories = []
    for i in range(num_tests):
        memory_obj = create_test_memory_obj_for_storage_manager(
            storage_manager, metadata
        )
        if memory_obj is not None:
            # Fill with unique test data for each memory object
            if memory_obj.tensor is not None:
                # Fill with a pattern based on the index to make each object unique
                memory_obj.tensor.fill_(float(i + 1))
            memory_obj.ref_count_up()
            exist_memories.append(memory_obj)

    if len(exist_memories) != num_tests:
        print(
            f"Warning: Could only allocate {len(exist_memories)}/{num_tests} "
            f"memory objects"
        )
        num_tests = len(exist_memories)
        exist_keys = exist_keys[:num_tests]

    return non_exist_keys, exist_keys, exist_memories, num_tests


@check_mode("test_storage_manager")
async def run_test_mode(model: str, **kwargs):
    """Run connector test mode"""
    kv_dtype_str = kwargs.get("kv_dtype") or DEFAULT_KV_DTYPE_STR
    kv_dtype = parse_kv_dtype(kv_dtype_str)
    if kv_dtype is None:
        print("Error: unsupported --kv-dtype '%s'" % kv_dtype_str)
        return

    obj_size = kwargs.get("obj_size")

    # Create storage manager using common function
    storage_manager = create_storage_manager_with_config(
        model,
        kv_dtype=kv_dtype,
        obj_size=obj_size,
    )

    try:
        print("Test: Passed - Created storage manager with valid config")

        # Create test context for the common framework
        test_context = {
            "create_test_data_func": create_test_data_for_storage_manager,
            "async_contains_func": async_contains_storage_manager,
            "async_put_func": async_submit_put_storage_manager,
            "async_get_func": async_get_storage_manager,
            "validate_get_func": validate_get_results,
            "test_object": storage_manager,
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
            if storage_manager:
                storage_manager.close()
        except Exception as e:
            print(f"Error closing storage manager: {e}")
