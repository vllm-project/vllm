# SPDX-License-Identifier: Apache-2.0
"""Generate mode implementation for key generation"""

# Third Party
import tqdm

# First Party
from lmcache.v1.check import check_mode
from lmcache.v1.check.utils import (
    _get_default_metadata,
    create_memory_objects_batch,
    create_storage_manager_with_config,
    create_test_key,
    find_remote_backend,
    flow_control_check,
    wait_put_tasks_complete,
)


@check_mode("gen")
async def run_gen_mode(
    model: str, num_keys: int, concurrency: int, offset: int = 0, **kwargs
):
    """Run key generation mode"""
    # Create storage manager using common function
    storage_manager = create_storage_manager_with_config(model)
    metadata = _get_default_metadata(model)

    try:
        print("Generate: Passed - Created storage manager with valid config")

        # Find remote backend for flow control
        remote_backend = find_remote_backend(storage_manager)

        # Create limited number of memory objects for reuse (memory efficiency)
        batch_size = min(concurrency, 100)  # Limit to 100 for memory efficiency
        memory_objs = create_memory_objects_batch(storage_manager, metadata, batch_size)

        if not memory_objs:
            print("Generate: Failed - Could not allocate any memory objects")
            return

        # Create progress bar
        progress_bar = tqdm.tqdm(
            total=num_keys, desc="Generating keys", unit="key", unit_scale=True
        )
        sleep_count = 1.0
        # Process keys in batches of concurrency size
        for batch_start in range(0, num_keys, concurrency):
            batch_end = min(batch_start + concurrency, num_keys)
            batch_keys = []
            batch_memory_objs = []

            # Create keys and reuse memory objects for this batch
            for i in range(batch_start, batch_end):
                key = create_test_key(model, f"gen_{offset + i}")
                # Reuse memory objects in round-robin fashion
                memory_obj = memory_objs[i % len(memory_objs)]
                batch_keys.append(key)
                batch_memory_objs.append(memory_obj)
                memory_obj.ref_count_up()

            # Flow control: check if remote backend has too many pending tasks
            sleep_count = await flow_control_check(
                remote_backend, concurrency, sleep_count
            )

            # Use batched_put to store the batch of memory objects
            storage_manager.batched_put(batch_keys, batch_memory_objs)

            # Update progress bar
            progress_bar.update(len(batch_keys))

        progress_bar.close()
        print(f"Generate: Successfully generated {num_keys} keys")

        # Wait for remote backend put_tasks to complete
        wait_put_tasks_complete(find_remote_backend(storage_manager))

    except Exception as e:
        print(
            f"Generate: Failed - Error creating storage manager with valid config: {e}"
        )
    finally:
        if storage_manager:
            storage_manager.close()
