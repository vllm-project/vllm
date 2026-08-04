# SPDX-License-Identifier: Apache-2.0
"""
AsyncSerdeProcessor: wraps sync Serializer/Deserializer into the async
SerdeProcessor interface expected by the controllers.

Runs serialization/deserialization tasks in a thread pool and signals
event notifiers on completion, matching the L2 adapter async pattern.
The notifiers come from :mod:`lmcache.v1.platform` so the same code
runs on Linux (eventfd) and other POSIX platforms (pipe fallback).
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
import enum
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.base import (
    Deserializer,
    SerdeProcessor,
    SerdeTaskId,
    Serializer,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import create_event_notifier

logger = init_logger(__name__)


class _TaskType(enum.Enum):
    SERIALIZE = enum.auto()
    DESERIALIZE = enum.auto()


class AsyncSerdeProcessor(SerdeProcessor):
    """Wraps sync Serializer/Deserializer into async SerdeProcessor.

    Runs each submitted task in a thread pool. On completion, stores
    the result and signals the appropriate event notifier so the
    controller's poll loop wakes up.

    Args:
        serializer: Sync serializer implementation.
        deserializer: Sync deserializer implementation.
        max_workers: Thread pool size. Default 1 (serialization is
            typically CPU-bound, more threads may help if the transform
            releases the GIL).
    """

    def __init__(
        self,
        serializer: Serializer,
        deserializer: Deserializer,
        max_workers: int = 1,
    ) -> None:
        self._serializer = serializer
        self._deserializer = deserializer

        self._serialize_efd = create_event_notifier()
        self._deserialize_efd = create_event_notifier()

        self._lock = threading.Lock()
        self._next_task_id: SerdeTaskId = 0
        # task_id -> (success: bool) for completed tasks, partitioned by type
        self._completed_serialize: dict[SerdeTaskId, bool] = {}
        self._completed_deserialize: dict[SerdeTaskId, bool] = {}

        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    # ----- Event fds -----

    def get_serialize_event_fd(self) -> int:
        """Return the fd signaled on serialize completion."""
        return self._serialize_efd.fileno()

    def get_deserialize_event_fd(self) -> int:
        """Return the fd signaled on deserialize completion."""
        return self._deserialize_efd.fileno()

    # ----- Serialize -----

    def submit_serialize(
        self,
        src_objs: list[MemoryObj],
        dst_objs: list[MemoryObj],
        keys: list[ObjectKey],
    ) -> SerdeTaskId:
        """Submit a batch serialize task to the thread pool."""
        task_id = self._alloc_task_id()
        logger.debug(
            "Serde: submitted serialize task %d (%d objects)",
            task_id,
            len(src_objs),
        )
        self._pool.submit(
            self._run_task,
            task_id,
            _TaskType.SERIALIZE,
            src_objs,
            dst_objs,
            keys,
        )
        return task_id

    def query_serialize_result(self, task_id: SerdeTaskId) -> bool | None:
        """Pop and return the serialize task result, or None if pending."""
        with self._lock:
            return self._completed_serialize.pop(task_id, None)

    # ----- Deserialize -----

    def submit_deserialize(
        self,
        src_objs: list[MemoryObj],
        dst_objs: list[MemoryObj],
        keys: list[ObjectKey],
    ) -> SerdeTaskId:
        """Submit a batch deserialize task to the thread pool."""
        task_id = self._alloc_task_id()
        logger.debug(
            "Serde: submitted deserialize task %d (%d objects)",
            task_id,
            len(src_objs),
        )
        self._pool.submit(
            self._run_task,
            task_id,
            _TaskType.DESERIALIZE,
            src_objs,
            dst_objs,
            keys,
        )
        return task_id

    def query_deserialize_result(self, task_id: SerdeTaskId) -> bool | None:
        """Pop and return the deserialize task result, or None if pending."""
        with self._lock:
            return self._completed_deserialize.pop(task_id, None)

    # ----- Size estimation (delegates to sync serializer) -----

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        """Delegate to the sync serializer's estimate (includes margin)."""
        return self._serializer.estimate_serialized_size(layout_desc)

    # ----- Lifecycle -----

    def close(self) -> None:
        """Shut down the thread pool and close event notifiers."""
        self._pool.shutdown(wait=True)
        self._serialize_efd.close()
        self._deserialize_efd.close()

    # ----- Internal -----

    def _alloc_task_id(self) -> SerdeTaskId:
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
        return task_id

    def _run_task(
        self,
        task_id: SerdeTaskId,
        task_type: _TaskType,
        src_objs: list[MemoryObj],
        dst_objs: list[MemoryObj],
        keys: list[ObjectKey],
    ) -> None:
        """Execute a serialize/deserialize task in the thread pool.

        On completion (success or failure), stores the result and
        signals the event notifier.
        """
        success = True
        try:
            if task_type == _TaskType.SERIALIZE:
                for src, dst, key in zip(src_objs, dst_objs, keys, strict=True):
                    # ``serialize`` returns the actual number of bytes
                    # written, which may be smaller than the destination
                    # buffer (since the destination is sized from
                    # ``estimate_serialized_size``, an upper bound).
                    # Narrow the destination's logical size to ``n`` so
                    # downstream L2 adapters that read the size via
                    # ``obj.get_size()`` / ``obj.byte_array`` store
                    # exactly ``n`` bytes -- not the over-allocated
                    # estimate.  Guarded so duck-typed test fakes that
                    # don't implement the interface still work, and so a
                    # serializer that does not honor the ``-> int``
                    # contract (returns ``None``) is skipped rather than
                    # tripping set_used_size's range check.
                    n = self._serializer.serialize(src, dst, key)
                    if isinstance(n, int) and hasattr(dst, "set_used_size"):
                        dst.set_used_size(n)
            else:
                for src, dst, key in zip(src_objs, dst_objs, keys, strict=True):
                    self._deserializer.deserialize(src, dst, key)
        except Exception:
            logger.exception(
                "Serde task %d (%s) failed",
                task_id,
                task_type.name,
            )
            success = False

        if success:
            logger.debug(
                "Serde: %s task %d completed successfully (%d objects)",
                task_type.name.lower(),
                task_id,
                len(src_objs),
            )
        else:
            logger.warning(
                "Serde: %s task %d failed (%d objects)",
                task_type.name.lower(),
                task_id,
                len(src_objs),
            )

        with self._lock:
            if task_type == _TaskType.SERIALIZE:
                self._completed_serialize[task_id] = success
            else:
                self._completed_deserialize[task_id] = success

        # Signal the appropriate notifier to wake the controller's poll loop
        notifier = (
            self._serialize_efd
            if task_type == _TaskType.SERIALIZE
            else self._deserialize_efd
        )
        try:
            notifier.notify()
        except OSError:
            logger.exception("Failed to signal notifier for serde task %d", task_id)
