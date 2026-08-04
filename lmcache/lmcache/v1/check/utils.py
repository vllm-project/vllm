# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for check modes"""

# Standard
from typing import Optional
import asyncio
import hashlib
import threading
import time

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.kv_layer_groups import DTYPE_MAP

# Import from lmcache with absolute paths
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.remote_backend import RemoteBackend
from lmcache.v1.storage_backend.storage_manager import StorageManager

DEFAULT_KV_DTYPE_STR = "bfloat16"
DEFAULT_OBJ_SIZE = 1024


def _compute_kv_shape(
    obj_size: int,
) -> tuple:
    """Compute a kv_shape that yields the given obj_size.

    The returned shape is in vllm format:
    ``(num_layers, 2, num_tokens, num_heads, head_size)``.

    The final KV_2LTD tensor has
    ``2 * num_layers * num_tokens * (num_heads * head_size)``
    elements, which equals *obj_size*.

    We fix ``num_layers=1, num_heads=1`` and split the
    remaining factor between ``num_tokens`` and ``head_size``
    so that ``num_tokens * head_size = obj_size // 2``.
    """
    if obj_size % 2 != 0:
        raise ValueError("obj_size must be even (got %d)" % obj_size)
    half = obj_size // 2
    # (num_layers, kv_dim, num_tokens, num_heads, head_size)
    return (1, 2, half, 1, 1)


def parse_kv_dtype(kv_dtype_str: str) -> Optional[torch.dtype]:
    """Parse a kv_dtype string to a torch.dtype.

    Returns None if the string is not recognized.
    """
    return DTYPE_MAP.get(kv_dtype_str)


def _get_default_metadata(
    model: str,
    kv_dtype: torch.dtype = torch.bfloat16,
    obj_size: Optional[int] = None,
) -> LMCacheMetadata:
    """Get default metadata for testing.

    When *obj_size* is given the ``kv_shape`` is computed so
    that the resulting KV_2LTD tensor has exactly *obj_size*
    elements.  Otherwise a small default shape is used.
    """
    if obj_size is not None:
        kv_shape = _compute_kv_shape(obj_size)
    else:
        kv_shape = _compute_kv_shape(DEFAULT_OBJ_SIZE)
    return LMCacheMetadata(
        model_name=model,
        world_size=8,
        local_world_size=8,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=kv_dtype,
        kv_shape=kv_shape,
    )


def create_test_key(
    model: str,
    key_id: str = "test_key",
    kv_dtype: torch.dtype = torch.bfloat16,
) -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey(
        model_name=model,
        world_size=8,
        worker_id=0,
        chunk_hash=int(hashlib.sha256(key_id.encode()).hexdigest(), 16),
        dtype=kv_dtype,
    )


def create_test_memory_obj_for_storage_manager(
    storage_manager: StorageManager, metadata: LMCacheMetadata
) -> Optional[MemoryObj]:
    """Create a test MemoryObj for testing with StorageManager."""
    # The metadata.kv_shape is in vllm format:
    # [num_layers, 2, num_tokens, num_heads, head_size]
    # For KV_2LTD format, we need shape: [2, num_layers, num_tokens, hidden_dim]
    # where hidden_dim = num_heads * head_size

    vllm_shape = metadata.kv_shape  # [num_layers, 2, num_tokens, num_heads, head_size]
    num_layers = vllm_shape[0]  # 8
    kv_dim = vllm_shape[1]  # 2 (K and V)
    num_tokens = vllm_shape[2]  # 16
    num_heads = vllm_shape[3]  # 8
    head_size = vllm_shape[4]  # 16

    # Convert to KV_2LTD format shape: [2, num_layers, num_tokens, hidden_dim]
    hidden_dim = num_heads * head_size
    kv_2ltd_shape = torch.Size([kv_dim, num_layers, num_tokens, hidden_dim])

    memory_obj = storage_manager.allocate(
        kv_2ltd_shape,
        metadata.kv_dtype,
        fmt=MemoryFormat.KV_2LTD,
        eviction=True,
        busy_loop=False,
    )
    return memory_obj


def create_storage_manager_with_config(
    model: str,
    kv_dtype: torch.dtype = torch.bfloat16,
    obj_size: Optional[int] = None,
):
    """Create storage manager with default configuration"""
    # First Party
    from lmcache.integration.vllm.utils import lmcache_get_or_create_config
    from lmcache.v1.event_manager import EventManager

    config = lmcache_get_or_create_config()
    metadata = _get_default_metadata(model, kv_dtype=kv_dtype, obj_size=obj_size)

    # Create event manager
    event_manager = EventManager()

    # Create storage manager
    storage_manager = StorageManager(
        config=config,
        metadata=metadata,
        event_manager=event_manager,
    )

    return storage_manager


def find_remote_backend(storage_manager: StorageManager) -> Optional[RemoteBackend]:
    """Find remote backend from storage manager"""
    for backend_name, backend in storage_manager.storage_backends.items():
        if isinstance(backend, RemoteBackend):
            return backend
    return None


def wait_put_tasks_complete(
    remote_backend: Optional[RemoteBackend], max_wait_time: float = 5.0
):
    """Wait for remote backend put tasks to complete"""
    if remote_backend is None:
        return

    check_interval = 0.001
    elapsed_time = 0.0

    while elapsed_time < max_wait_time:
        if not remote_backend.put_tasks:
            break
        time.sleep(check_interval)
        elapsed_time += check_interval

    # Log warning if timeout
    remaining_tasks = len(remote_backend.put_tasks)
    if remaining_tasks > 0:
        print(
            f"Warning: {remaining_tasks} remote put tasks still "
            f"pending after {max_wait_time}s timeout"
        )


def create_memory_objects_batch(
    storage_manager: StorageManager, metadata: LMCacheMetadata, batch_size: int
) -> list[MemoryObj]:
    """Create a batch of memory objects for reuse"""
    memory_objs = []
    for i in range(batch_size):
        memory_obj = create_test_memory_obj_for_storage_manager(
            storage_manager, metadata
        )
        if memory_obj is not None:
            memory_obj.ref_count_up()
            memory_objs.append(memory_obj)
    return memory_objs


async def flow_control_check(
    remote_backend: Optional[RemoteBackend], concurrency: int, sleep_count: float = 1.0
) -> float:
    """Check flow control and wait if necessary"""
    if remote_backend is None:
        return sleep_count

    high_watermark = 100 * concurrency
    low_watermark = 10 * concurrency
    current_tasks = len(remote_backend.put_tasks)

    while current_tasks > high_watermark:
        current_tasks = len(remote_backend.put_tasks)
        if current_tasks > high_watermark:
            # Too many pending tasks, wait before proceeding
            sleep_sec = 0.1 * sleep_count
            current_tasks = len(remote_backend.put_tasks)
            await asyncio.sleep(sleep_sec)
            current_tasks_after_sleep = len(remote_backend.put_tasks)
            if current_tasks_after_sleep > low_watermark:
                sleep_count *= 2.0
            elif current_tasks_after_sleep == 0:
                sleep_count /= 2.0
            continue
        if current_tasks <= low_watermark:
            break

    return sleep_count


async def run_perf_test_with_timeout(func, args_list, timeout=30.0):
    """Common performance test framework with timeout handling"""
    times = []
    results = []  # Collect results for each operation
    for i, args in enumerate(args_list):
        try:
            start = time.perf_counter()
            result = await asyncio.wait_for(func(*args), timeout=timeout)
            end = time.perf_counter()
            times.append((end - start) * 1000)
            results.append(result)
            print(
                f"  Test {i + 1}/{len(args_list)} completed in "
                f"{(end - start) * 1000:.2f}ms"
            )
        except asyncio.TimeoutError:
            print(f"  Test {i + 1}/{len(args_list)} timed out after {timeout}s")
            times.append(timeout * 1000)
            results.append(None)
        except Exception as e:
            print(f"  Test {i + 1}/{len(args_list)} failed: {e}")
            times.append(0)
            results.append(None)

    if times:
        return {
            "time_stats": {
                "avg": sum(times) / len(times),
                "max": max(times),
                "min": min(times),
            },
            "results": results,
        }
    else:
        return {"time_stats": {"avg": 0, "max": 0, "min": 0}, "results": []}


def _format_throughput(avg_ms: float, obj_bytes: int) -> str:
    """Format throughput as a human-readable string."""
    if avg_ms <= 0 or obj_bytes <= 0:
        return "N/A"
    bps = obj_bytes / (avg_ms / 1000.0)
    if bps >= 1 << 30:
        return "%.2f GB/s" % (bps / (1 << 30))
    if bps >= 1 << 20:
        return "%.2f MB/s" % (bps / (1 << 20))
    return "%.2f KB/s" % (bps / (1 << 10))


def print_performance_results(
    stats_data,
    obj_bytes: int = 0,
    throughput_ops: Optional[set] = None,
):
    """Print performance results in a formatted table.

    Args:
        stats_data: list of (op, stats, results, pass_count).
        obj_bytes: size of one object in bytes.  When > 0 a
            throughput column is shown for operations listed
            in *throughput_ops*.
        throughput_ops: set of operation name prefixes that
            should show throughput (e.g. {"STORE", "LOAD"}).
            Defaults to common data-transfer operations.
    """
    if throughput_ops is None:
        throughput_ops = {
            "STORE",
            "LOAD",
            "PUT",
            "GET",
        }
    show_tp = obj_bytes > 0

    sep_len = 118 if show_tp else 100
    tp_hdr = " | %s" % "Throughput".center(14) if show_tp else ""
    print("\nPerformance Results:")
    print("-" * sep_len)
    print(
        f"| {'Operation':<20} | {'Avg (ms)':>12} "
        f"| {'Max (ms)':>12} "
        f"| {'Min (ms)':>12} "
        f"| {'Pass/All':>10} "
        f"| {'Pass Rate':>10} |" + tp_hdr
    )
    print("-" * sep_len)
    for op, stats, results, pass_count in stats_data:
        total = len(results)
        pass_all = f"{pass_count}/{total}"
        pass_rate = pass_count / total * 100 if total > 0 else 0
        tp_col = ""
        if show_tp:
            is_data_op = any(op.startswith(p) for p in throughput_ops)
            if is_data_op:
                tp_col = " | %s" % _format_throughput(
                    stats["avg"],
                    obj_bytes,
                ).center(14)
            else:
                tp_col = " | %s" % "-".center(14)

        print(
            f"| {op:<20} "
            f"| {stats['avg']:>12.6f} "
            f"| {stats['max']:>12.6f} "
            f"| {stats['min']:>12.6f} "
            f"| {pass_all:>10} "
            f"| {pass_rate:>9.1f}% |" + tp_col
        )
    print("-" * sep_len)


def validate_get_results(get_results, exist_keys, exist_memories, num_tests):
    """Validate GET operation results and return statistics"""
    content_valid_count = 0
    for i, result in enumerate(get_results["results"]):
        if result is None:
            print(f"  GET for key {exist_keys[i]} returned None result")
            continue
        try:
            if result.tensor is None:
                print(f"  GET for key {exist_keys[i]} returned None tensor")
                continue

            if exist_memories[i].tensor is None:
                print(f"  Original memory object {i} has None tensor")
                continue

            # Compare data content
            data_match = torch.equal(result.tensor, exist_memories[i].tensor)

            if data_match:
                content_valid_count += 1
            else:
                print(f"  GET for key {exist_keys[i]} returned incorrect memory object")
                print("    Data content mismatch detected")

        except Exception as e:
            print(f"  Data comparison failed for key {exist_keys[i]}: {e}")
            # Standard
            import traceback

            traceback.print_exc()

    # Calculate pass rates
    not_none_count = sum(1 for r in get_results["results"] if r is not None)
    content_pass_rate = content_valid_count / num_tests * 100
    print(f"  Validation (not None): {not_none_count}/{num_tests} passed")
    print(
        f"  Validation (content correct): {content_valid_count}/{num_tests}"
        f" passed ({content_pass_rate:.1f}%)"
    )
    return content_valid_count, not_none_count


async def run_common_test_framework(
    test_context,
    model: str,
    num_tests: int = 5,
    settle_time: float = 0.0,
):
    """
    Common test framework for both storage manager and remote backend tests.

    Args:
        test_context: A dictionary containing test-specific functions and objects:
            - 'create_test_data_func': Function to create test data
            - 'async_contains_func': Async function for contains operations
            - 'async_put_func': Async function for put operations
            - 'async_get_func': Async function for get operations
            - 'validate_get_func': Function to validate get results
            - 'test_object': The main test object (storage_manager or backend)
            - 'extra_args': Extra arguments for test data creation (optional)
        model: Model name for testing
        num_tests: Number of tests to run
    """
    print("Testing basic operations...")

    # Create test data using the provided function
    extra_args = test_context.get("extra_args", [])
    extra_kwargs = {}
    if "kv_dtype" in test_context:
        extra_kwargs["kv_dtype"] = test_context["kv_dtype"]
    if extra_args:
        non_exist_keys, exist_keys, exist_memories, num_tests = test_context[
            "create_test_data_func"
        ](test_context["test_object"], *extra_args, model, num_tests, **extra_kwargs)
    else:
        kv_dtype = test_context.get("kv_dtype")
        obj_size = test_context.get("obj_size")
        meta_kw = {}
        if kv_dtype:
            meta_kw["kv_dtype"] = kv_dtype
        if obj_size is not None:
            meta_kw["obj_size"] = obj_size
        metadata = _get_default_metadata(model, **meta_kw)
        non_exist_keys, exist_keys, exist_memories, num_tests = test_context[
            "create_test_data_func"
        ](test_context["test_object"], metadata, model, num_tests, **extra_kwargs)

    # Phase 1: exists test (key does not exist)
    print("Phase 1: Testing exists for non-existing keys...")

    exists_non_exist_res = await run_perf_test_with_timeout(
        test_context["async_contains_func"],
        [(test_context["test_object"], key) for key in non_exist_keys],
    )
    exists_non_exist_stats = exists_non_exist_res["time_stats"]
    # Validation: All non-existing keys should return False
    exists_non_exist_pass_count = sum(
        1 for r in exists_non_exist_res["results"] if r is False
    )
    pass_rate = exists_non_exist_pass_count / len(non_exist_keys) * 100
    print(
        f"  Validation: {exists_non_exist_pass_count}/{len(non_exist_keys)} "
        f"passed ({pass_rate:.1f}%)"
    )

    # Phase 2: put test (create new key)
    print("Phase 2: Testing put operations...")

    put_res = await run_perf_test_with_timeout(
        test_context["async_put_func"],
        [
            (test_context["test_object"], exist_keys[i], exist_memories[i])
            for i in range(num_tests)
        ],
    )
    put_stats = put_res["time_stats"]
    # Validation: All PUT operations should return True
    put_pass_count = sum(1 for r in put_res["results"] if r is True)
    pass_rate = put_pass_count / num_tests * 100
    print(f"  Validation: {put_pass_count}/{num_tests} passed ({pass_rate:.1f}%)")

    if settle_time > 0:
        print("  Waiting %.1fs for data to settle..." % settle_time)
        await asyncio.sleep(settle_time)

    # Phase 3: exists test (key exists)
    print("Phase 3: Testing exists for existing keys...")

    exists_exist_res = await run_perf_test_with_timeout(
        test_context["async_contains_func"],
        [(test_context["test_object"], key) for key in exist_keys],
    )
    exists_exist_stats = exists_exist_res["time_stats"]
    # Validation: All existing keys should return True
    exists_exist_pass_count = sum(1 for r in exists_exist_res["results"] if r is True)
    pass_rate = exists_exist_pass_count / num_tests * 100
    print(
        f"  Validation: {exists_exist_pass_count}/{num_tests} passed ({pass_rate:.1f}%)"
    )

    # Phase 4: get test (key exists)
    print("Phase 4: Testing get operations...")

    get_res = await run_perf_test_with_timeout(
        test_context["async_get_func"],
        [(test_context["test_object"], key) for key in exist_keys],
    )
    get_stats = get_res["time_stats"]
    # Validation: Check for non-None results and content correctness
    content_valid_count, not_none_count = test_context["validate_get_func"](
        get_res, exist_keys, exist_memories, num_tests
    )
    # Use content_valid_count as the pass_count for GET operations
    get_pass_count = content_valid_count

    stats_data = [
        (
            "EXISTS (non-exist)",
            exists_non_exist_stats,
            exists_non_exist_res["results"],
            exists_non_exist_pass_count,
        ),
        ("PUT", put_stats, put_res["results"], put_pass_count),
        (
            "EXISTS (exist)",
            exists_exist_stats,
            exists_exist_res["results"],
            exists_exist_pass_count,
        ),
        ("GET", get_stats, get_res["results"], get_pass_count),
    ]

    # Compute per-object byte size for throughput display
    tp_obj_size = test_context.get("obj_size") or DEFAULT_OBJ_SIZE
    tp_kv_dtype = test_context.get("kv_dtype") or torch.bfloat16
    obj_bytes = tp_obj_size * torch.tensor([], dtype=tp_kv_dtype).element_size()

    # Use common performance results printing
    print_performance_results(stats_data, obj_bytes=obj_bytes)


class EventLoopManager:
    """Manages a dedicated event loop in a separate thread"""

    def __init__(self):
        self.loop = None
        self.thread = None
        self._loop_started = threading.Event()

    def start(self):
        """Start the event loop in a separate thread"""
        if self.thread is not None and self.thread.is_alive():
            return

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self._loop_started.wait()

    def _run_loop(self):
        """Run the event loop"""
        asyncio.set_event_loop(self.loop)
        self._loop_started.set()
        try:
            self.loop.run_forever()
        except Exception as e:
            print(f"Event loop error: {e}")
        finally:
            self.loop.close()

    def stop(self):
        """Stop the event loop and thread"""
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)

    def get_loop(self):
        """Get the event loop"""
        return self.loop
