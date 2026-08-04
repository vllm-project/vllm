# SPDX-License-Identifier: Apache-2.0
"""
In-memory L2 adapter plugin for LMCache.

A minimal but fully functional L2 adapter that stores KV
cache objects in plain Python dicts.  Intended as a
reference implementation for third-party plugin authors.
"""

# Standard
from collections import defaultdict
from typing import Any, Union
import asyncio
import copy
import threading
import time

# Third Party
import torch  # noqa: F401  # must precede native_storage_ops

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
)
from lmcache.v1.memory_management import (
    MemoryObj,
    TensorMemoryObj,
)
from lmcache.v1.platform import create_event_notifier

logger = init_logger(__name__)


def _clone_tensor_memory_obj(
    obj: MemoryObj,
) -> TensorMemoryObj:
    """Deep-clone a TensorMemoryObj."""
    assert isinstance(obj, TensorMemoryObj), "Only TensorMemoryObj is supported"
    raw = obj.raw_tensor
    assert raw is not None, "tensor data cannot be None"
    return TensorMemoryObj(
        raw_data=raw.detach().clone(),
        metadata=copy.deepcopy(obj.metadata),
        parent_allocator=None,
    )


class InMemoryL2AdapterConfig(L2AdapterConfigBase):
    """Config for the in-memory L2 adapter.

    Fields:
    - max_size_gb: Maximum cache size in GiB.
    - mock_bandwidth_gb: Simulated bandwidth in GiB/s.
    """

    def __init__(
        self,
        max_size_gb: float = 0.5,
        mock_bandwidth_gb: float = 10.0,
    ):
        self.max_size_gb = max_size_gb
        self.mock_bandwidth_gb = mock_bandwidth_gb

    @classmethod
    def from_dict(cls, d: dict) -> "InMemoryL2AdapterConfig":
        return cls(
            max_size_gb=float(d.get("max_size_gb", 0.5)),
            mock_bandwidth_gb=float(d.get("mock_bandwidth_gb", 10.0)),
        )

    @classmethod
    def help(cls) -> str:
        return (
            "InMemoryL2Adapter config fields:\n"
            "- max_size_gb (float): max cache size "
            "in GiB (default 0.5)\n"
            "- mock_bandwidth_gb (float): simulated "
            "bandwidth in GiB/s (default 10.0)\n"
        )


class InMemoryL2Adapter(L2AdapterInterface):
    """In-memory L2 adapter loaded as an external plugin.

    Constructor accepts either an
    ``InMemoryL2AdapterConfig`` instance (when the
    framework auto-discovers the config class) **or**
    a plain dict (raw-dict mode, when no config class
    is registered).

    Any extra ``**kwargs`` forwarded by the framework
    (e.g. ``l1_memory_desc``) are accepted but not used
    by this simple example adapter.
    """

    def __init__(
        self,
        config: Union[
            InMemoryL2AdapterConfig,
            dict[str, Any],
        ],
        **_kwargs: object,
    ):
        if isinstance(config, dict):
            config = InMemoryL2AdapterConfig.from_dict(config)
        max_size_gb = config.max_size_gb
        mock_bandwidth_gb = config.mock_bandwidth_gb
        cap = int(max_size_gb * (1024**3))
        bw = int(mock_bandwidth_gb * (1024**3))

        self._max_cap = cap
        self._bw_bps = bw
        self._cur_size = 0

        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        self._objects: dict[ObjectKey, MemoryObj] = {}
        self._key_queue: list[ObjectKey] = []
        self._locked: dict[ObjectKey, int] = defaultdict(int)

        self._next_id: L2TaskId = 0
        self._done_store: dict[L2TaskId, L2StoreResult] = {}
        self._done_lookup: dict[L2TaskId, Bitmap] = {}
        self._done_load: dict[L2TaskId, Bitmap] = {}
        self._lock = threading.Lock()

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        logger.info(
            "InMemoryL2Adapter created: max_size_gb=%s, mock_bandwidth_gb=%s",
            max_size_gb,
            mock_bandwidth_gb,
        )

    # ---- event fd ------------------------------------------

    def get_store_event_fd(self) -> int:
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_efd.fileno()

    # ---- store ---------------------------------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._lock:
            tid = self._alloc_id()
        asyncio.run_coroutine_threadsafe(
            self._do_store(keys, objects, tid),
            self._loop,
        )
        return tid

    def pop_completed_store_tasks(
        self,
    ) -> dict[L2TaskId, L2StoreResult]:
        with self._lock:
            done = self._done_store
            self._done_store = {}
        return done

    # ---- lookup & lock -------------------------------------

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], layout_desc: MemoryLayoutDesc
    ) -> L2TaskId:
        # TODO: this example ignores layout_desc; a real adapter may need it (the hint is forwarded by the P2P adapter).
        with self._lock:
            tid = self._alloc_id()
        self._loop.call_soon_threadsafe(self._do_lookup, keys, tid)
        return tid

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._done_lookup.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        def _unlock(ks: list[ObjectKey]) -> None:
            for k in ks:
                if k not in self._locked:
                    continue
                if self._locked[k] <= 1:
                    del self._locked[k]
                else:
                    self._locked[k] -= 1

        self._loop.call_soon_threadsafe(_unlock, keys)

    # ---- load ----------------------------------------------

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._lock:
            tid = self._alloc_id()
        asyncio.run_coroutine_threadsafe(
            self._do_load(keys, objects, tid),
            self._loop,
        )
        return tid

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._done_load.pop(task_id, None)

    # ---- close ---------------------------------------------

    def close(self) -> None:
        async def _cancel() -> None:
            tasks = [
                t
                for t in asyncio.all_tasks(self._loop)
                if t is not asyncio.current_task()
            ]
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        if self._loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(_cancel(), self._loop)
            try:
                fut.result(timeout=5)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)

        self._thread.join()
        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()

    # ---- helpers -------------------------------------------

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _alloc_id(self) -> L2TaskId:
        tid = self._next_id
        self._next_id += 1
        return tid

    def _evict(self, needed: int) -> None:
        tries = len(self._key_queue)
        while self._cur_size + needed > self._max_cap and tries > 0:
            tries -= 1
            k = self._key_queue.pop(0)
            if self._locked.get(k, 0) > 0:
                self._key_queue.append(k)
                continue
            if k in self._objects:
                self._cur_size -= self._objects.pop(k).get_size()

    async def _do_store(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        tid: L2TaskId,
    ) -> None:
        total = 0
        ok = True
        t0 = time.perf_counter()
        try:
            for key, obj in zip(keys, objects, strict=False):
                sz = obj.get_size()
                if sz > self._max_cap:
                    continue
                if key in self._objects:
                    continue
                self._evict(sz)
                clone = _clone_tensor_memory_obj(obj)
                self._objects[key] = clone
                self._key_queue.append(key)
                self._cur_size += sz
                total += sz
        except Exception:
            ok = False

        delay = total / self._bw_bps if self._bw_bps > 0 else 0
        delay -= time.perf_counter() - t0
        if delay > 0:
            await asyncio.sleep(delay)

        with self._lock:
            self._done_store[tid] = L2StoreResult(ok, total)
        self._store_efd.notify()

    def _do_lookup(
        self,
        keys: list[ObjectKey],
        tid: L2TaskId,
    ) -> None:
        bm = Bitmap(len(keys))
        for i, k in enumerate(keys):
            if k not in self._objects:
                continue
            bm.set(i)
            self._locked[k] += 1
        with self._lock:
            self._done_lookup[tid] = bm
        self._lookup_efd.notify()

    async def _do_load(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        tid: L2TaskId,
    ) -> None:
        bm = Bitmap(len(keys))
        total = 0
        t0 = time.perf_counter()
        for i, k in enumerate(keys):
            if k not in self._objects:
                continue
            src = self._objects[k].tensor
            dst = objects[i].tensor
            assert src is not None and dst is not None
            dst.copy_(src)
            bm.set(i)
            total += self._objects[k].get_size()

        delay = total / self._bw_bps if self._bw_bps > 0 else 0
        delay -= time.perf_counter() - t0
        if delay > 0:
            await asyncio.sleep(delay)

        with self._lock:
            self._done_load[tid] = bm
        self._load_efd.notify()
