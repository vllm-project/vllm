# SPDX-License-Identifier: Apache-2.0
"""
Single source of truth for the RESPClient class.

This module provides a sync and asyncio wrapper around the LMCacheRedisClient,
which is a C++ extension built with PyTorch. The pybinding interface allows
working with both sync and async code seamlessly.

Used by:
- lmcache/v1/storage_backend/connector/redis_connector.py (RESPConnector)
- examples/kv_cache_reuse/remote_backends/resp/example_resp_client.py (benchmarks)
"""

# Standard
from typing import Dict, Optional, Tuple, Union
import asyncio
import concurrent.futures

try:
    # First Party
    from lmcache.lmcache_redis import LMCacheRedisClient

    REDIS_AVAILABLE = True
except ImportError:
    # C++ extension not built (e.g., in non-CUDA CI environments)
    # This is fine - RESP tests will be skipped
    REDIS_AVAILABLE = False
    LMCacheRedisClient = None  # type: ignore


class RESPClient:
    """
    Sync and asyncio wrapper around the native LMCacheRedisClient.

    This class bridges the C++ RESP client with Python's asyncio event loop,
    allowing for efficient async I/O operations with Redis using the RESP protocol.
    """

    def __init__(
        self,
        host: str,
        port: int,
        num_workers: int,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        username: str = "",
        password: str = "",
    ):
        if not REDIS_AVAILABLE:
            raise RuntimeError(
                "RESPClient requires the C++ Redis extension. "
                "Build with: pip install -e ."
            )
        self.loop = loop or asyncio.get_running_loop()
        self._client = LMCacheRedisClient(host, port, num_workers, username, password)
        self._fd = int(self._client.event_fd())
        self._closed = False

        # future_id -> (Future, op_name)
        # we support both types of futures since we only their basic interface
        self._pending: Dict[
            int, Tuple[Union[asyncio.Future, concurrent.futures.Future], str]
        ] = {}

        self.loop.add_reader(self._fd, self._on_ready)

    def _on_ready(self) -> None:
        if self._closed:
            return

        try:
            # drain until empty; completions can race in while processing
            while True:
                items = self._client.drain_completions()
                if not items:
                    return

                for future_id, ok, error, result_bools in items:
                    fid = int(future_id)
                    entry = self._pending.pop(fid, None)
                    if entry is None:
                        continue

                    fut, op = entry
                    # fut can be asyncio.Future OR concurrent.future.Future
                    # the .done() and .set_result() and .set_exception()
                    # interface is the same
                    if fut.done():
                        continue

                    if ok:
                        if op == "exists":
                            # result_bools is a list with 1 element for single
                            # exists
                            if result_bools is not None and len(result_bools) > 0:
                                fut.set_result(bool(result_bools[0]))
                            else:
                                # should not happen but handle gracefully
                                fut.set_result(False)
                        elif op == "batch_exists":
                            # result_bools is a list of booleans (or None if empty)
                            if result_bools is not None:
                                fut.set_result(list(result_bools))
                            else:
                                fut.set_result([])
                        else:
                            fut.set_result(None)
                    else:
                        fut.set_exception(RuntimeError(str(error)))

        except Exception as e:
            # Native layer is likely broken; fail everything and tear down.
            self._fail_all(RuntimeError(f"native drain_completions failed: {e}"))
            self._shutdown_native(best_effort=True)

    def _fail_all(self, exc: Exception) -> None:
        """Fail all pending futures with the given exception."""
        for fid, (fut, _) in list(self._pending.items()):
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()

    def _shutdown_native(self, best_effort: bool = False) -> None:
        """Shutdown the native client and cleanup resources."""
        try:
            self._closed = True
            self.loop.remove_reader(self._fd)
        except Exception:
            if not best_effort:
                raise

    def _register_future_async(self, op: str, future_id: int) -> asyncio.Future:
        fut = self.loop.create_future()
        self._pending[int(future_id)] = (fut, op)
        return fut

    def _register_future_sync(
        self, op: str, future_id: int
    ) -> concurrent.futures.Future:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._pending[int(future_id)] = (fut, op)
        return fut

    async def get(self, key: str, buf: memoryview) -> None:
        future_id = int(self._client.submit_get(key, buf))
        fut = self._register_future_async("get", future_id)
        return await fut

    def get_sync(self, key: str, buf: memoryview) -> None:
        future_id = int(self._client.submit_get(key, buf))
        fut = self._register_future_sync("get", future_id)
        return fut.result()

    async def set(self, key: str, buf: memoryview) -> None:
        future_id = int(self._client.submit_set(key, buf))
        fut = self._register_future_async("set", future_id)
        return await fut

    def set_sync(self, key: str, buf: memoryview) -> None:
        future_id = int(self._client.submit_set(key, buf))
        fut = self._register_future_sync("set", future_id)
        return fut.result()

    async def exists(self, key: str) -> bool:
        future_id = int(self._client.submit_exists(key))
        fut = self._register_future_async("exists", future_id)
        return await fut

    def exists_sync(self, key: str) -> bool:
        future_id = int(self._client.submit_exists(key))
        fut = self._register_future_sync("exists", future_id)
        return fut.result()

    async def batch_get(self, keys: list[str], bufs: list[memoryview]) -> None:
        if len(keys) != len(bufs):
            raise ValueError("keys and bufs length mismatch")
        future_id = int(self._client.submit_batch_get(keys, bufs))
        fut = self._register_future_async("batch_get", future_id)
        return await fut

    def batch_get_sync(self, keys: list[str], bufs: list[memoryview]) -> None:
        if len(keys) != len(bufs):
            raise ValueError("keys and bufs length mismatch")
        future_id = int(self._client.submit_batch_get(keys, bufs))
        fut = self._register_future_sync("batch_get", future_id)
        return fut.result()

    async def batch_set(self, keys: list[str], bufs: list[memoryview]) -> None:
        if len(keys) != len(bufs):
            raise ValueError("keys and bufs length mismatch")
        future_id = int(self._client.submit_batch_set(keys, bufs))
        fut = self._register_future_async("batch_set", future_id)
        return await fut

    def batch_set_sync(self, keys: list[str], bufs: list[memoryview]) -> None:
        if len(keys) != len(bufs):
            raise ValueError("keys and bufs length mismatch")
        future_id = int(self._client.submit_batch_set(keys, bufs))
        fut = self._register_future_sync("batch_set", future_id)
        return fut.result()

    async def batch_exists(self, keys: list[str]) -> list[bool]:
        """Check existence of multiple keys in a single batch operation."""
        future_id = int(self._client.submit_batch_exists(keys))
        fut = self._register_future_async("batch_exists", future_id)
        return await fut

    def batch_exists_sync(self, keys: list[str]) -> list[bool]:
        """Check existence of multiple keys in a single
        batch operation (sync version)."""
        future_id = int(self._client.submit_batch_exists(keys))
        fut = self._register_future_sync("batch_exists", future_id)
        return fut.result()

    async def batched_exists(self, keys: list[str]) -> list[bool]:
        """Alias for batch_exists."""
        return await self.batch_exists(keys)

    def batched_exists_sync(self, keys: list[str]) -> list[bool]:
        """Alias for batch_exists_sync."""
        return self.batch_exists_sync(keys)

    def close(self) -> None:
        """Close the client and cleanup resources."""
        if not self._closed:
            self._shutdown_native(best_effort=True)
            self._fail_all(RuntimeError("Client closed"))
            self._client.close()
