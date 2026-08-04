# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from collections import defaultdict
from typing import TYPE_CHECKING, Optional
import asyncio
import copy
import threading
import time

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.memory_management import MemoryObj, TensorMemoryObj
from lmcache.v1.platform import create_event_notifier

logger = init_logger(__name__)

# Helper function


def clone_tensor_memory_obj(obj: MemoryObj) -> TensorMemoryObj:
    assert isinstance(obj, TensorMemoryObj), (
        "Only TensorMemoryObj is supported in this mock adapter"
    )
    raw_tensor = obj.raw_tensor
    assert raw_tensor is not None, (
        "The tensor data of the object cannot be None for cloning"
    )

    new_obj = TensorMemoryObj(
        raw_data=raw_tensor.detach().clone(),
        metadata=copy.deepcopy(obj.metadata),
        parent_allocator=None,
    )

    return new_obj


# Config class


class MockL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for a mock L2 adapter (for testing).

    Fields:
    - max_size_gb: maximum size in GB.
    - mock_bandwidth_gb: simulated bandwidth in GB/sec.
    """

    def __init__(
        self,
        max_size_gb: float,
        mock_bandwidth_gb: float,
    ):
        self.max_size_gb = max_size_gb
        self.mock_bandwidth_gb = mock_bandwidth_gb

    @classmethod
    def from_dict(cls, d: dict) -> "MockL2AdapterConfig":
        max_size_gb = d.get("max_size_gb")
        if not isinstance(max_size_gb, (int, float)) or max_size_gb <= 0:
            raise ValueError("max_size_gb must be a positive number")

        mock_bandwidth_gb = d.get("mock_bandwidth_gb")
        if not isinstance(mock_bandwidth_gb, (int, float)) or mock_bandwidth_gb <= 0:
            raise ValueError("mock_bandwidth_gb must be a positive number")

        return cls(
            max_size_gb=max_size_gb,
            mock_bandwidth_gb=mock_bandwidth_gb,
        )

    @classmethod
    def help(cls) -> str:
        return (
            "Mock L2 adapter config fields:\n"
            "- max_size_gb (float): maximum size of "
            "the adapter in GB (required, >0)\n"
            "- mock_bandwidth_gb (float): simulated "
            "bandwidth in GB/sec (required, >0)"
        )


# Main class


class MockL2Adapter(L2AdapterInterface):
    """
    A mock-up L2 adapter with a specific RAM size and mocked bandwidth
    """

    def __init__(self, config: MockL2AdapterConfig):
        max_capacity_bytes = int(config.max_size_gb * (1024**3))
        super().__init__(max_capacity_bytes=max_capacity_bytes)
        self._config = config
        self._bandwidth_byte_ps = int(config.mock_bandwidth_gb * (1024**3))

        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        self._memory_objects: dict[ObjectKey, MemoryObj] = {}
        self._locked_keys: dict[ObjectKey, int] = defaultdict(int)
        self._current_size_bytes: int = 0

        # Task ID management
        self._next_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}
        self._lock = threading.Lock()  # lock for all shared state

        # Asyncio event loop running in a background thread
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()

    # --------------------
    # Event Fd Interface
    # --------------------

    def get_store_event_fd(self) -> int:
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_efd.fileno()

    # --------------------
    # Store Interface
    # --------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """
        Submit a store task to store a batch of memory objects associated with
        a batch of keys.

        For the mock adapter, the store operation simulates bandwidth-limited
        transfer by delaying completion based on object size and configured bandwidth.

        Args:
            keys (list[ObjectKey]): the list of keys to be stored.
            objects (list[MemoryObj]): the list of memory objects to be stored.
                The length of the objects list should be the same as the length of
                the keys list.

        Returns:
            L2TaskId: the task id of the submitted store task.
        """
        with self._lock:
            task_id = self._get_next_task_id()

        asyncio.run_coroutine_threadsafe(
            self._execute_store_in_the_loop(keys, objects, task_id), self._loop
        )

        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        """
        Pop all the completed store tasks with their results.

        Returns:
            dict[L2TaskId, L2StoreResult]: a dictionary mapping the task id to
            an L2StoreResult encoding success/failure and bytes transferred.
        """
        with self._lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
        return completed

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], layout_desc: MemoryLayoutDesc
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        # Schedule the lookup operation in the event loop thread
        self._loop.call_soon_threadsafe(self._execute_lookup_in_the_loop, keys, task_id)
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        def _unlock_keys(keys: list[ObjectKey]) -> None:
            """
            Coroutine to unlock keys in the event loop thread.
            This is a helper function to avoid blocking the main thread.
            """
            for key in keys:
                if key not in self._locked_keys:
                    continue
                if self._locked_keys[key] <= 1:
                    del self._locked_keys[key]
                else:
                    self._locked_keys[key] -= 1

        # Schedule the unlock operation in the event loop thread
        self._loop.call_soon_threadsafe(_unlock_keys, keys)

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        # Schedule the load operation in the event loop thread
        asyncio.run_coroutine_threadsafe(
            self._execute_load_in_loop(keys, objects, task_id), self._loop
        )

        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_load_tasks.pop(task_id, None)

    def close(self):
        # Stop the event loop and wait for the thread to finish
        async def _stop_tasks():
            tasks = [
                t
                for t in asyncio.all_tasks(self._loop)
                if t is not asyncio.current_task()
            ]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        if self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(_stop_tasks(), self._loop)
            try:
                future.result(timeout=5)  # Wait for tasks to be cancelled
            except Exception:
                pass  # Ignore exceptions during shutdown
            self._loop.call_soon_threadsafe(self._loop.stop)

        self._loop_thread.join()
        self._loop.close()

        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()

    ##################
    # Debug / test-only functions
    ##################

    def report_status(self) -> dict:
        """Return a status dict for the mock L2 adapter."""
        with self._lock:
            return {
                "is_healthy": True,
                "type": "MockL2Adapter",
                "stored_object_count": len(self._memory_objects),
                "locked_key_count": len(self._locked_keys),
                "current_size_bytes": self._current_size_bytes,
                "max_capacity_bytes": self._max_capacity_bytes,
            }

    def debug_get_stored_object_count(self) -> int:
        """
        Return the number of objects currently stored in the mock adapter.

        This method is intended for testing and debugging only.

        Returns:
            int: Number of stored objects.
        """
        with self._lock:
            return len(self._memory_objects)

    def debug_get_locked_key_count(self) -> int:
        """
        Return the number of currently locked keys.

        This method is intended for testing and debugging only.

        Returns:
            int: Number of locked keys.
        """
        with self._lock:
            return len(self._locked_keys)

    def debug_has_key(self, key: ObjectKey) -> bool:
        """
        Check whether a specific key is stored in the mock adapter.

        This method is intended for testing and debugging only.

        Args:
            key: The object key to check.

        Returns:
            bool: True if the key is stored.
        """
        with self._lock:
            return key in self._memory_objects

    ##################
    # Helper functions
    ##################

    def _run_event_loop(self) -> None:
        """Run the asyncio event loop in a background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _get_next_task_id(self) -> L2TaskId:
        """Get the next task ID and increment the counter."""
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    #####################
    # Eviction Interface
    #####################

    def delete(self, keys: list[ObjectKey]) -> None:
        """Delete a batch of objects from the mock adapter."""
        deleted_keys: list[ObjectKey] = []
        deleted_sizes: list[int] = []
        with self._lock:
            for key in keys:
                if key not in self._memory_objects:
                    continue
                obj = self._memory_objects.pop(key)
                size = obj.get_size()
                self._current_size_bytes -= size
                deleted_keys.append(key)
                deleted_sizes.append(size)
        if deleted_keys:
            self._notify_keys_deleted(deleted_keys, deleted_sizes)

    # ``get_usage()`` is inherited from ``L2AdapterInterface``, which derives
    # the report from the byte counters maintained by ``_notify_keys_*``.
    # ``_current_size_bytes`` above is a local within-batch accumulator
    # used for the synchronous capacity check inside the store loop and is
    # NOT the source of truth for external reporting.

    def _signal_store_event(self) -> None:
        """Signal the store event fd to notify completion."""
        self._store_efd.notify()

    async def _execute_store_in_the_loop(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        """
        Execute the store operation in the event loop thread.
        This is a helper function to avoid blocking the main thread.
        """
        total_bytes = 0
        success = True
        start = time.perf_counter()

        stored_keys: list[ObjectKey] = []
        stored_sizes: list[int] = []
        try:
            for key, obj in zip(keys, objects, strict=True):
                obj_size = obj.get_size()

                # If the object is larger than max capacity, skip it
                if obj_size > self._max_capacity_bytes:
                    continue

                # If key already exists, simply skip
                if key in self._memory_objects:
                    continue

                # Skip if there is not enough capacity
                if self._current_size_bytes + obj_size > self._max_capacity_bytes:
                    logger.warning(
                        "MockL2Adapter: not enough capacity to store key %s "
                        "(used=%d, needed=%d, max=%d); skipping.",
                        key,
                        self._current_size_bytes,
                        obj_size,
                        self._max_capacity_bytes,
                    )
                    continue

                # Store the object
                new_obj = clone_tensor_memory_obj(obj)
                self._memory_objects[key] = new_obj
                self._current_size_bytes += obj_size
                total_bytes += obj_size
                stored_keys.append(key)
                stored_sizes.append(obj_size)
        except Exception:
            success = False

        # Calculate delay based on bandwidth simulation
        end = time.perf_counter()
        delay_seconds = (
            total_bytes / self._bandwidth_byte_ps if self._bandwidth_byte_ps > 0 else 0
        )
        delay_seconds -= end - start
        delay_seconds = max(delay_seconds, 0)  # Ensure non-negative delay

        # Schedule completion coroutine on the event loop
        await asyncio.sleep(delay_seconds)
        with self._lock:
            # ``total_bytes`` counts only objects actually written into
            # ``self._memory_objects`` — duplicates and capacity-skipped
            # keys are excluded.  Reporting this lets the L2 throughput
            # subscriber distinguish real I/O from fast-pathed no-ops.
            self._completed_store_tasks[task_id] = L2StoreResult(success, total_bytes)

        if stored_keys:
            self._notify_keys_stored(stored_keys, stored_sizes)
        self._signal_store_event()

    def _signal_lookup_event(self) -> None:
        """Signal the lookup event fd to notify completion."""
        self._lookup_efd.notify()

    def _execute_lookup_in_the_loop(
        self, keys: list[ObjectKey], task_id: L2TaskId
    ) -> None:
        bitmap = Bitmap(len(keys))
        for i, key in enumerate(keys):
            if key not in self._memory_objects:
                continue
            bitmap.set(i)
            self._locked_keys[key] += 1
        with self._lock:
            self._completed_lookup_tasks[task_id] = bitmap
        self._signal_lookup_event()

    def _signal_load_event(self) -> None:
        """Signal the load event fd to notify completion."""
        self._load_efd.notify()

    async def _execute_load_in_loop(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        """
        Execute the load operation in the event loop thread.
        This is a helper function to avoid blocking the main thread.
        """
        bitmap = Bitmap(len(keys))
        total_bytes = 0
        accessed_keys: list[ObjectKey] = []
        start = time.perf_counter()

        for i, key in enumerate(keys):
            if key not in self._memory_objects:
                continue
            # load data into the provided memory object
            obj = self._memory_objects[key]
            src_tensor = obj.tensor
            dst_tensor = objects[i].tensor
            assert src_tensor is not None
            assert dst_tensor is not None
            dst_tensor.copy_(src_tensor)
            bitmap.set(i)
            total_bytes += obj.get_size()
            accessed_keys.append(key)

        end = time.perf_counter()
        delay_seconds = (
            total_bytes / self._bandwidth_byte_ps if self._bandwidth_byte_ps > 0 else 0
        )
        delay_seconds -= end - start
        delay_seconds = max(delay_seconds, 0)  # Ensure non-negative delay

        await asyncio.sleep(delay_seconds)
        if accessed_keys:
            self._notify_keys_accessed(accessed_keys)
        with self._lock:
            self._completed_load_tasks[task_id] = bitmap
        self._signal_load_event()


# Self-register config type and adapter factory
register_l2_adapter_type("mock", MockL2AdapterConfig)


def _create_mock_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create a MockL2Adapter from config."""
    return MockL2Adapter(config)  # type: ignore[arg-type]


register_l2_adapter_factory("mock", _create_mock_adapter)
