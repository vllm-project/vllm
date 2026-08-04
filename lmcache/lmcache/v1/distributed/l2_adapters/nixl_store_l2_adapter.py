# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from typing import Optional
import asyncio
import os
import threading
import uuid

# Third Party
from nixl._api import nixl_agent as NixlAgent
from nixl._api import nixl_agent_config as NixlAgentConfig
from nixl._api import nixl_prepped_dlist_handle as NixlDlistHandle
from nixl._api import nixl_xfer_handle as NixlXferHandle
from nixl._api import (
    nixlBind,
)

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L1MemoryDesc, L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import create_event_notifier

logger = init_logger(__name__)

# Main class


@dataclass
class NixlStoreObj:
    """
    The object stored in Nixl L2 cache.
    Can be used for both file and object.
    """

    page_indices: list[int]

    size: int  # in bytes

    layout: Optional[MemoryLayoutDesc] = None

    pin_count: int = 0
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def increase_pin_count(self):
        with self._lock:
            self.pin_count += 1

    def decrease_pin_count(self):
        with self._lock:
            if self.pin_count > 0:
                self.pin_count -= 1
            else:
                logger.warning(
                    "Trying to decrease pin count of object at page indices %s below 0",
                    self.page_indices,
                )


class NixlObjPool:
    """Thread-safe pool of integer indices representing pre-allocated storage slots."""

    def __init__(self, num_total_objs: int):
        """
        Args:
            num_total_objs: Total number of storage slots to manage.
        """
        self.indices = list(range(num_total_objs))
        self._total = num_total_objs
        self._lock = threading.Lock()

    @property
    def total_objs(self) -> int:
        """Total number of storage slots this pool manages (allocated + free)."""
        return self._total

    def batched_allocate(self, num_objs: int) -> list[int]:
        """
        Allocate a batch of storage slot indices.

        Args:
            num_objs: Number of indices to allocate.

        Returns:
            list[int]: The allocated indices.

        Raises:
            RuntimeError: If fewer than ``num_objs`` slots remain in the pool.
        """
        with self._lock:
            if num_objs > len(self.indices):
                logger.debug("NixlObjPool allocation failure.")
                return []
            allocated = self.indices[:num_objs]
            self.indices = self.indices[num_objs:]
            return allocated

    def batched_free(self, obj_indices: list[int]) -> None:
        """
        Return a batch of storage slot indices back to the pool.

        Args:
            obj_indices: Indices previously obtained from ``batched_allocate``.
        """
        with self._lock:
            self.indices.extend(obj_indices)

    def get_slot_usage(self) -> tuple[float, float]:
        """
        Return (current_usage, usage_after_ongoing_eviction) in [0, 1] for
        the slot pool. Renamed from ``get_usage`` to avoid colliding with
        the byte-based ``L2AdapterInterface.get_usage`` shape — this is
        an internal pool helper, not the adapter-level usage report.

        Both values are identical because slot frees are synchronous.
        """
        with self._lock:
            if self._total == 0:
                return (0.0, 0.0)
            usage = (self._total - len(self.indices)) / self._total
            return (usage, usage)


class NixlStorageAgent:
    agent_name: str
    nixl_agent: NixlAgent
    mem_reg_descs: nixlBind.nixlRegDList
    mem_xfer_handler: NixlDlistHandle

    def __init__(
        self,
        device: str,
        backend: str,
        backend_params: dict[str, str],
        pool_size: int,
        l1_memory_desc: L1MemoryDesc,
    ):
        """
        Initialize the NixlStorageAgent.

        Args:
            device: Device type of the L1 memory buffer (e.g. "cpu", "cuda").
            backend: Nixl storage backend to use. One of: GDS, GDS_MT, POSIX,
                HF3FS (file-based) or OBJ, AZURE_BLOB (object-based).
            backend_params: Backend-specific parameters. File-based backends
                require "file_path" and "use_direct_io" keys.
            pool_size: Number of storage descriptor slots to pre-allocate.
            l1_memory_desc: Descriptor of the L1 memory buffer to register with Nixl
                for data transfers.
        """
        self.backend = backend
        self.pool_size = pool_size
        self.device = device
        self.backend_params = backend_params

        self.l1_align_bytes = l1_memory_desc.align_bytes

        self.agent_name = "NixlAgent_" + str(uuid.uuid4())
        nixl_conf = NixlAgentConfig(backends=[])
        self.nixl_agent = NixlAgent(self.agent_name, nixl_conf)
        self.nixl_agent.create_backend(backend, backend_params)

        self.init_mem_handlers(
            self.device,
            l1_memory_desc.ptr,
            l1_memory_desc.size,
            l1_memory_desc.align_bytes,
            device_id=0,  # 0 indicates cpu
        )

        if self.backend in ["GDS", "GDS_MT", "POSIX", "HF3FS"]:
            file_size = int(
                self.backend_params.get("file_size", l1_memory_desc.align_bytes)
            )
            pages_per_file = file_size // l1_memory_desc.align_bytes
            self.pool = NixlObjPool(num_total_objs=self.pool_size * pages_per_file)
            self.init_storage_handlers_file(
                num_files=self.pool_size,
                page_size=l1_memory_desc.align_bytes,
                file_size=file_size,
                file_path=self.backend_params["file_path"],
                # TODO(Jiayi): Need to make argument parsing more elegant
                use_direct_io=str(self.backend_params["use_direct_io"]).lower()
                == "true",
            )
        elif self.backend in ["OBJ", "AZURE_BLOB"]:
            self.pool = NixlObjPool(num_total_objs=self.pool_size)
            self.init_storage_handlers_object(
                page_size=l1_memory_desc.align_bytes,
                num_pages=self.pool_size,
            )
        else:
            raise TypeError(f"Unsupported backend type: {self.backend}")

    def init_mem_handlers(self, device, buffer_ptr, buffer_size, page_size, device_id):
        """
        Initialize memory handlers for the given device and buffer.
        """
        reg_list = [(buffer_ptr, buffer_size, device_id, "")]
        xfer_desc = [
            (base_addr, page_size, device_id)
            for base_addr in range(buffer_ptr, buffer_ptr + buffer_size, page_size)
        ]

        if device == "cpu":
            mem_type = "DRAM"
        else:
            mem_type = "VRAM"

        reg_descs = self.nixl_agent.register_memory(reg_list, mem_type=mem_type)
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, mem_type=mem_type)
        xfer_handler = self.nixl_agent.prep_xfer_dlist(
            "", xfer_descs, mem_type=mem_type
        )

        self.mem_reg_descs = reg_descs
        self.mem_xfer_handler = xfer_handler

    def init_storage_handlers_file(
        self,
        num_files: int,
        page_size: int,
        file_size: int,
        file_path: str,
        use_direct_io: bool,
    ):
        """Initialize storage handlers for file-based backends.

        Each file holds ``file_size // page_size`` pages at successive offsets.
        ``file_size`` must be a multiple of ``page_size``.

        Args:
            num_files: Number of storage files to create.
            page_size: Granularity of L1 memory pages (transfer unit size).
            file_size: Size in bytes of each storage file. Must be a multiple
                of ``page_size``.
            file_path: Directory where storage files are created.
            use_direct_io: Whether to open files with O_DIRECT.
        """
        os.makedirs(file_path, exist_ok=True)
        if file_size % page_size != 0:
            raise ValueError(
                f"file_size ({file_size}) must be a multiple of page_size ({page_size})"
            )

        pages_per_file = file_size // page_size
        num_pages = num_files * pages_per_file

        # Create file descriptors for Nixl to register
        fds: list[int] = []
        flags = os.O_CREAT | os.O_RDWR
        if use_direct_io:
            if hasattr(os, "O_DIRECT"):
                flags |= os.O_DIRECT
            else:
                logger.warning(
                    "use_direct_io is True, but O_DIRECT is not available on "
                    "this system. Falling back to buffered I/O."
                )
        for i in range(num_files):
            filename = f"obj_{i}_{uuid.uuid4().hex[0:4]}.bin"
            tmp_path = os.path.join(file_path, filename)
            fd = os.open(tmp_path, flags)
            fds.append(fd)

        # Register each file covering the full file_size.
        # Build one xfer_desc entry per page slot (page index i maps to
        # offset (i % pages_per_file) * page_size inside fd[i // pages_per_file]).
        reg_list = []
        xfer_desc = []
        for fd in fds:
            reg_list.append((0, file_size, fd, ""))
        for page_idx in range(num_pages):
            fd = fds[page_idx // pages_per_file]
            offset = (page_idx % pages_per_file) * page_size
            xfer_desc.append((offset, page_size, fd))
        reg_descs = self.nixl_agent.register_memory(reg_list, mem_type="FILE")
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, mem_type="FILE")
        xfer_handler = self.nixl_agent.prep_xfer_dlist(
            self.agent_name, xfer_descs, mem_type="FILE"
        )

        self.storage_fds = fds
        self.storage_reg_descs = reg_descs
        self.storage_xfer_descs = xfer_descs
        self.storage_xfer_handler = xfer_handler

    def init_storage_handlers_object(
        self,
        page_size: int,
        num_pages: int,
    ):
        """Initialize storage handlers for object-based backends."""

        # Create object keys for Nixl to register
        keys = []

        for i in range(num_pages):
            key = f"obj_{i}_{uuid.uuid4().hex[0:4]}"
            keys.append(key)

        # Register and prepare xfer handler
        reg_list = []
        xfer_desc = []
        for i, key in enumerate(keys):
            reg_list.append((0, page_size, i, key))
            xfer_desc.append((0, page_size, i))
        reg_descs = self.nixl_agent.register_memory(reg_list, mem_type="OBJ")
        xfer_descs = self.nixl_agent.get_xfer_descs(xfer_desc, mem_type="OBJ")
        xfer_handler = self.nixl_agent.prep_xfer_dlist(
            self.agent_name, xfer_descs, mem_type="OBJ"
        )

        self.storage_reg_descs = reg_descs
        self.storage_xfer_descs = xfer_descs
        self.storage_xfer_handler = xfer_handler

    def get_mem_to_storage_handle(self, mem_indices, storage_indices) -> NixlXferHandle:
        """Get a Nixl transfer handle for transferring data from memory to storage."""

        return self.nixl_agent.make_prepped_xfer(
            "WRITE",
            self.mem_xfer_handler,
            mem_indices,
            self.storage_xfer_handler,
            storage_indices,
        )

    def get_storage_to_mem_handle(self, mem_indices, storage_indices) -> NixlXferHandle:
        """Get a Nixl transfer handle for transferring data from storage to memory."""
        return self.nixl_agent.make_prepped_xfer(
            "READ",
            self.mem_xfer_handler,
            mem_indices,
            self.storage_xfer_handler,
            storage_indices,
        )

    def post_blocking(self, handle: NixlXferHandle):
        """Post a Nixl transfer handle and block until the transfer is done."""

        state = self.nixl_agent.transfer(handle)

        while state != "DONE" and state != "ERR":
            try:
                state = self.nixl_agent.check_xfer_state(handle)
            except nixlBind.nixlBackendError:
                raise

        if state == "ERR":
            raise RuntimeError("NIXL transfer failed")

    async def post_non_blocking(self, handle: NixlXferHandle):
        """Post a Nixl transfer handle and await until the transfer is done."""

        state = self.nixl_agent.transfer(handle)

        while state != "DONE" and state != "ERR":
            try:
                state = self.nixl_agent.check_xfer_state(handle)
            except nixlBind.nixlBackendError:
                raise

            # TODO(Jiayi): Tune this for better perf
            await asyncio.sleep(0.01)

        if state == "ERR":
            raise RuntimeError("NIXL transfer failed")

    def get_storage_indices(self, num_objs: int) -> list[int]:
        # TODO(Jiayi): Support eviction
        return self.pool.batched_allocate(num_objs)

    def get_memory_indices(self, raw_addr: int, mem_size: int) -> list[int]:
        """Get memory indices for the given raw address and size."""
        # TODO(Jiayi): Now we assume the memory is contiguous and page-aligned. We may
        # want to support more flexible memory layout in the future.
        if raw_addr % self.l1_align_bytes != 0:
            raise ValueError(
                f"Raw address {raw_addr} is not aligned to "
                f"page size {self.l1_align_bytes}"
            )
        if mem_size % self.l1_align_bytes != 0:
            raise ValueError(
                f"Memory size {mem_size} is not a multiple of "
                f"page size {self.l1_align_bytes}"
            )
        num_pages = mem_size // self.l1_align_bytes
        return [(raw_addr // self.l1_align_bytes + i) for i in range(num_pages)]

    def release_handle(self, handle):
        self.nixl_agent.release_xfer_handle(handle)

    def close(self):
        self.nixl_agent.release_dlist_handle(self.storage_xfer_handler)
        self.nixl_agent.release_dlist_handle(self.mem_xfer_handler)
        self.nixl_agent.deregister_memory(self.storage_reg_descs)
        self.nixl_agent.deregister_memory(self.mem_reg_descs)
        for fd in getattr(self, "storage_fds", []):
            os.close(fd)


class NixlStoreL2Adapter(L2AdapterInterface):
    """
    A Nixl-based L2 adapter
    """

    def __init__(self, config: NixlStoreL2AdapterConfig, l1_memory_desc: L1MemoryDesc):
        """
        Initialize the NixlStoreL2Adapter.

        Args:
            config: Nixl-specific adapter configuration including backend type,
                backend parameters, and storage pool size.
            l1_memory_desc: Descriptor of the L1 memory buffer to register with the
                Nixl backend for DMA transfers.
        """
        # Initialize Nixl agent first so we know the actual page count the
        # backend allocated; we then forward the byte capacity to the base
        # class so ``get_usage()`` / ``supports_global_eviction`` reflect the real
        # storage size.
        self.nixl_agent = NixlStorageAgent(
            device="cpu",
            backend=config.backend,
            backend_params=config.backend_params,
            pool_size=config.pool_size,
            l1_memory_desc=l1_memory_desc,
        )
        max_capacity_bytes = (
            self.nixl_agent.pool.total_objs * l1_memory_desc.align_bytes
        )
        super().__init__(max_capacity_bytes=max_capacity_bytes)
        self._config = config

        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        # Cache data structures
        self._memory_objects: dict[ObjectKey, NixlStoreObj] = {}

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

    #####################
    # Store Interface
    #####################

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """
        Submit a store task to store a batch of memory objects associated with
        a batch of keys.

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
        """Pop all completed store tasks.

        Returns:
            dict[L2TaskId, L2StoreResult]: a dictionary mapping the task
            id to an ``L2StoreResult`` that encodes both the success flag
            and the bytes actually transferred.
        """
        with self._lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
        return completed

    #####################
    # Lookup and Lock Interface
    #####################

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
            Unlock keys in the event loop thread.
            """
            for key in keys:
                if (obj := self._memory_objects.get(key)) is not None:
                    obj.decrease_pin_count()

        # Schedule the unlock operation in the event loop thread
        self._loop.call_soon_threadsafe(_unlock_keys, keys)

    #####################
    # Load Interface
    ######################

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

    def close(self) -> None:
        """Close the adapter and release its event-loop and NIXL resources."""

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

        # Gate on is_closed() rather than is_running(): the loop thread may not
        # have reached run_forever() yet, and skipping the stop below would leave
        # join() blocking forever. Both threadsafe calls are valid on a loop that
        # has not started; their callbacks run once it does.
        if not self._loop.is_closed():
            try:
                future = asyncio.run_coroutine_threadsafe(_stop_tasks(), self._loop)

                # Wait for tasks to be cancelled, with a timeout
                future.result(timeout=5)
            finally:
                self._loop.call_soon_threadsafe(self._loop.stop)

        self._loop_thread.join()
        try:
            self.nixl_agent.close()
        finally:
            self._loop.close()
            self._store_efd.close()
            self._lookup_efd.close()
            self._load_efd.close()

    #####################
    # Eviction Interface
    #####################

    def delete(self, keys: list[ObjectKey]) -> None:
        """
        Delete a batch of objects from Nixl storage, freeing their page slots.

        Pinned objects (pin_count > 0) are skipped to avoid racing with an
        in-flight load; the eviction controller will retry them on the next
        cycle once they are unpinned.
        """
        # TODO(Jiayi): Optimize lock usage here
        deleted_keys: list[ObjectKey] = []
        deleted_sizes: list[int] = []
        with self._lock:
            for key in keys:
                obj = self._memory_objects.get(key)
                if obj is None:
                    continue
                if obj.pin_count > 0:
                    logger.debug(
                        "Skipping eviction of pinned key %s (pin_count=%d)",
                        key,
                        obj.pin_count,
                    )
                    continue
                del self._memory_objects[key]
                self.nixl_agent.pool.batched_free(obj.page_indices)
                deleted_keys.append(key)
                deleted_sizes.append(obj.size)
        if deleted_keys:
            self._notify_keys_deleted(deleted_keys, deleted_sizes)

    # ``get_usage()`` is inherited from ``L2AdapterInterface``. The base
    # class derives the report from ``_notify_keys_*`` totals which we
    # update with the byte sizes from each store/delete. The Nixl pool's
    # own slot-based ``get_usage()`` is still used internally for
    # capacity-check style decisions but is no longer exposed via the
    # adapter interface.

    #####################
    # Status Interface
    #####################

    def report_status(self) -> dict:
        """Return a status dict for the Nixl L2 adapter."""
        # NOTE(Jiayi): This function looks pretty slow.
        with self._lock:
            stored_object_count = len(self._memory_objects)
            pinned_object_count = sum(
                1 for obj in self._memory_objects.values() if obj.pin_count > 0
            )
        pool = self.nixl_agent.pool
        with pool._lock:
            pool_free_slots = len(pool.indices)
        return {
            "is_healthy": self._loop_thread.is_alive(),
            "type": "NixlStoreL2Adapter",
            "backend": self._config.backend,
            "stored_object_count": stored_object_count,
            "pinned_object_count": pinned_object_count,
            "pool_size": self._config.pool_size,
            "pool_free_slots": pool_free_slots,
            "event_loop_alive": self._loop_thread.is_alive(),
        }

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

    def _evict_if_needed(
        self,
    ) -> None:
        """
        Evict objects from the cache using desired caching policy.
        """

        # TODO(Jiayi): Support eviction

        pass

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
        Coroutine that performs a batched store to Nixl storage.

        For each key-object pair, memory page indices are mapped to storage
        slot indices and a single batched DMA write is issued. On success the
        key-to-storage mapping is recorded in ``_memory_objects``. On transfer
        failure, all allocated storage slots are freed and the task is marked
        as failed.

        Args:
            keys: Keys identifying each object to store.
            objects: Memory objects whose contents will be written to storage.
                Must be the same length as ``keys``.
            task_id: Identifier used to report completion via
                ``_completed_store_tasks``.
        """
        success = True
        try:
            # Get memory page indices and storage slot indices
            mem_indices_flat = []
            storage_indices_flat = []
            stored_keys = []
            storage_objs = []
            for key, obj in zip(keys, objects, strict=False):
                # Skip if key already exists to avoid leaking pool slots
                with self._lock:
                    if key in self._memory_objects:
                        continue

                mem_addr = obj.meta.address
                mem_size = obj.meta.phy_size
                mem_indices = self.nixl_agent.get_memory_indices(mem_addr, mem_size)
                storage_indices = self.nixl_agent.get_storage_indices(
                    num_objs=len(mem_indices)
                )

                if storage_indices == []:
                    break

                mem_indices_flat.extend(mem_indices)
                storage_indices_flat.extend(storage_indices)

                stored_keys.append(key)
                storage_objs.append(
                    NixlStoreObj(
                        page_indices=storage_indices,
                        size=obj.meta.phy_size,
                        layout=MemoryLayoutDesc(
                            [obj.meta.shape],
                            [obj.meta.dtype],
                        ),
                        pin_count=1,
                    )
                )

            if not mem_indices_flat:
                # Nothing to store (all keys already existed or pool empty)
                with self._lock:
                    self._completed_store_tasks[task_id] = L2StoreResult(True, 0)
                self._signal_store_event()
                return

            handle = self.nixl_agent.get_mem_to_storage_handle(
                mem_indices_flat,
                storage_indices_flat,
            )

            await self.nixl_agent.post_non_blocking(handle)
            self.nixl_agent.release_handle(handle)

            with self._lock:
                for key, storage_obj in zip(stored_keys, storage_objs, strict=False):
                    self._memory_objects[key] = storage_obj
                    storage_obj.decrease_pin_count()
            # ``stored_keys`` and ``storage_objs`` are built together in the
            # pre-alloc loop above, so the size lists stay aligned even
            # when the pool ran out of slots mid-batch.
            if stored_keys:
                stored_sizes = [obj.size for obj in storage_objs]
                self._notify_keys_stored(stored_keys, stored_sizes)
            bytes_transferred = sum(obj.size for obj in storage_objs)

        # success is only set to false for transfer failures
        except Exception:
            logger.exception("NIXL store task %d failed", task_id)
            success = False
            bytes_transferred = 0

            # free storage indices if transfer fails
            self.nixl_agent.pool.batched_free(storage_indices_flat)

        with self._lock:
            self._completed_store_tasks[task_id] = L2StoreResult(
                success, bytes_transferred
            )

        self._signal_store_event()

    def _signal_lookup_event(self) -> None:
        """Signal the lookup event fd to notify completion."""
        self._lookup_efd.notify()

    def _execute_lookup_in_the_loop(
        self, keys: list[ObjectKey], task_id: L2TaskId
    ) -> None:
        """
        Performs a batched lookup and pin in the event loop.

        For each key present in ``_memory_objects``, its bit is set in the
        result bitmap and its pin count is incremented to prevent eviction
        while the caller holds the lock. Keys not found are left unset.

        Args:
            keys: Keys to look up.
            task_id: Identifier used to report completion via
                ``_completed_lookup_tasks``.
        """
        bitmap = Bitmap(len(keys))
        with self._lock:
            for i, key in enumerate(keys):
                if (obj := self._memory_objects.get(key)) is None:
                    continue
                bitmap.set(i)
                obj.increase_pin_count()
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
        Coroutine that performs a batched load from Nixl storage into L1 memory.

        For each key that exists in ``_memory_objects``, the corresponding
        storage page indices are gathered and a single batched DMA read is
        issued into the provided memory objects. Keys that are not found are
        silently skipped and their bit in the result bitmap is left unset.

        Args:
            keys: Keys identifying each object to load.
            objects: Pre-allocated memory objects that will receive the loaded
                data. Must be the same length as ``keys``.
            task_id: Identifier used to report completion via
                ``_completed_load_tasks``.
        """
        bitmap = Bitmap(len(keys))
        accessed_keys: list[ObjectKey] = []
        try:
            mem_indices_flat = []
            storage_indices_flat = []

            with self._lock:
                for i, key in enumerate(keys):
                    if (storage_obj := self._memory_objects.get(key)) is None:
                        continue
                    mem_addr = objects[i].meta.address
                    mem_size = objects[i].meta.phy_size
                    mem_indices = self.nixl_agent.get_memory_indices(mem_addr, mem_size)

                    mem_indices_flat.extend(mem_indices)
                    storage_indices_flat.extend(storage_obj.page_indices)

                    bitmap.set(i)
                    accessed_keys.append(key)

            if mem_indices_flat:
                handle = self.nixl_agent.get_storage_to_mem_handle(
                    mem_indices_flat,
                    storage_indices_flat,
                )
                await self.nixl_agent.post_non_blocking(handle)
                self.nixl_agent.release_handle(handle)
        except Exception:
            logger.exception("NIXL load task %d failed", task_id)

        if accessed_keys:
            self._notify_keys_accessed(accessed_keys)
        with self._lock:
            self._completed_load_tasks[task_id] = bitmap
        self._signal_load_event()


# ---------------------------------------------------------------------
# Config and self-registration
# ---------------------------------------------------------------------

_VALID_NIXL_BACKENDS = (
    "GDS",
    "GDS_MT",
    "POSIX",
    "HF3FS",
    "OBJ",
    "AZURE_BLOB",
)
_FILE_BACKENDS = ("GDS", "GDS_MT", "POSIX", "HF3FS")


class NixlStoreL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for a Nixl-store-based L2 adapter.

    Fields:
    - backend: Nixl storage backend
      (GDS, GDS_MT, POSIX, HF3FS, OBJ, AZURE_BLOB).
    - backend_params: Backend-specific parameters as a
      dict of string key-value pairs. For file-based
      backends (GDS, GDS_MT, POSIX, HF3FS), must include
      ``file_path``. May also include ``use_direct_io``
      (default ``"false"``) and other backend-specific
      keys.
    - pool_size: Number of storage descriptors to
      pre-allocate (must be > 0).
    """

    def __init__(
        self,
        backend: str,
        backend_params: dict[str, str],
        pool_size: int,
    ):
        if backend in _FILE_BACKENDS:
            if "file_path" not in backend_params:
                raise ValueError(
                    "backend_params must include "
                    "'file_path' for file-based "
                    "backend %r" % backend
                )
            if "use_direct_io" not in backend_params:
                raise ValueError(
                    "backend_params must include "
                    "'use_direct_io' for file-based "
                    "backend %r" % backend
                )
        self.backend = backend
        self.backend_params = backend_params
        self.pool_size = pool_size

    @classmethod
    def from_dict(cls, d: dict) -> "NixlStoreL2AdapterConfig":
        backend = d.get("backend")
        if backend not in _VALID_NIXL_BACKENDS:
            raise ValueError(
                "backend must be one of %s, got %r" % (_VALID_NIXL_BACKENDS, backend)
            )

        backend_params = d.get("backend_params", {})
        if not isinstance(backend_params, dict):
            raise ValueError("backend_params must be a dict of string key-value pairs")

        pool_size = d.get("pool_size")
        if not isinstance(pool_size, int) or pool_size <= 0:
            raise ValueError("pool_size must be a positive integer")

        return cls(
            backend=backend,
            backend_params=backend_params,
            pool_size=pool_size,
        )

    @classmethod
    def help(cls) -> str:
        return (
            "Nixl store L2 adapter config fields:\n"
            "- backend (str): Nixl storage backend, "
            "one of %s (required)\n"
            "- backend_params (dict): backend-specific "
            "string key-value pairs (optional, "
            "default empty). File-based backends "
            "require file_path. Optional keys include "
            "'use_direct_io' (default 'false') and "
            "'file_size' (int, size in bytes of each "
            "storage file slot; defaults to the L1 "
            "page size if not set).\n"
            "- pool_size (int): number of storage "
            "descriptors to pre-allocate (required, "
            ">0)" % (_VALID_NIXL_BACKENDS,)
        )


# Self-register config type and adapter factory
register_l2_adapter_type("nixl_store", NixlStoreL2AdapterConfig)


def _create_nixl_store_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: Optional[L1MemoryDesc] = None,
) -> L2AdapterInterface:
    """Create a NixlStoreL2Adapter from config."""
    if l1_memory_desc is None:
        raise ValueError("l1_memory_desc is required to create a NixlStoreL2Adapter.")
    return NixlStoreL2Adapter(config, l1_memory_desc)  # type: ignore[arg-type]


register_l2_adapter_factory("nixl_store", _create_nixl_store_adapter)
