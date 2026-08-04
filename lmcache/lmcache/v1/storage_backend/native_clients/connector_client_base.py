# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union
import asyncio
import concurrent.futures

NativeClientT = TypeVar("NativeClientT")


class ConnectorClientBase(Generic[NativeClientT]):
    def __init__(
        self,
        native_client: NativeClientT,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.loop = loop or asyncio.get_running_loop()
        self._client: NativeClientT = native_client
        self._fd = int(self._client.event_fd())  # type: ignore[attr-defined]
        self._closed = False
        # Keepalive refs prevent buffers passed to native code from being
        # garbage-collected while C++ worker threads still hold raw pointers.
        self._pending: Dict[
            int,
            Tuple[
                Union[asyncio.Future, concurrent.futures.Future], str, Tuple[Any, ...]
            ],
        ] = {}
        self.loop.add_reader(self._fd, self._on_ready)

    def _on_ready(self) -> None:
        if self._closed:
            return

        try:
            while True:
                items = self._client.drain_completions()  # type: ignore[attr-defined]
                if not items:
                    return

                for future_id, ok, error, result_bools in items:
                    fid = int(future_id)
                    entry = self._pending.pop(fid, None)
                    if entry is None:
                        continue

                    fut, op, _keepalive = entry
                    if fut.done():
                        continue

                    if ok:
                        if op == "exists":
                            if result_bools is not None and len(result_bools) > 0:
                                fut.set_result(bool(result_bools[0]))
                            else:
                                fut.set_result(False)
                        elif op == "batch_exists":
                            if result_bools is not None:
                                fut.set_result(list(result_bools))
                            else:
                                fut.set_result([])
                        else:
                            fut.set_result(None)
                    else:
                        fut.set_exception(RuntimeError(str(error)))
        except Exception as e:
            self._fail_all(RuntimeError(f"native drain_completions failed: {e}"))
            self._shutdown_native(best_effort=True)

    def _fail_all(self, exc: Exception) -> None:
        for fid, (fut, _, _keepalive) in list(self._pending.items()):
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()

    def _shutdown_native(self, best_effort: bool = False) -> None:
        try:
            self._closed = True
            self.loop.remove_reader(self._fd)
        except Exception:
            if not best_effort:
                raise

    def _register_future_async(
        self, op: str, future_id: int, keepalive: Tuple[Any, ...] = ()
    ) -> asyncio.Future:
        fut = self.loop.create_future()
        self._pending[int(future_id)] = (fut, op, keepalive)
        return fut

    def _register_future_sync(
        self, op: str, future_id: int, keepalive: Tuple[Any, ...] = ()
    ) -> concurrent.futures.Future:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._pending[int(future_id)] = (fut, op, keepalive)
        return fut

    async def get(self, key: str, buf: memoryview) -> None:
        return await self.batch_get([key], [buf])

    async def set(self, key: str, buf: memoryview) -> None:
        return await self.batch_set([key], [buf])

    async def exists(self, key: str) -> bool:
        results = await self.batch_exists([key])
        return results[0]

    async def batch_get(self, keys: list[str], bufs: list[memoryview]) -> None:
        if len(keys) != len(bufs):
            raise ValueError("keys and bufs length mismatch")
        future_id = int(self._client.submit_batch_get(keys, bufs))  # type: ignore[attr-defined]
        fut = self._register_future_async("batch_get", future_id, (keys, tuple(bufs)))
        return await fut

    async def batch_set(self, keys: list[str], bufs: list[memoryview]) -> None:
        if len(keys) != len(bufs):
            raise ValueError("keys and bufs length mismatch")
        future_id = int(self._client.submit_batch_set(keys, bufs))  # type: ignore[attr-defined]
        fut = self._register_future_async("batch_set", future_id, (keys, tuple(bufs)))
        return await fut

    async def batch_exists(self, keys: list[str]) -> list[bool]:
        future_id = int(self._client.submit_batch_exists(keys))  # type: ignore[attr-defined]
        fut = self._register_future_async("batch_exists", future_id)
        return await fut

    async def batched_exists(self, keys: list[str]) -> list[bool]:
        return await self.batch_exists(keys)

    def get_sync(self, key: str, buf: memoryview) -> None:
        return self.batch_get_sync([key], [buf])

    def set_sync(self, key: str, buf: memoryview) -> None:
        return self.batch_set_sync([key], [buf])

    def exists_sync(self, key: str) -> bool:
        results = self.batch_exists_sync([key])
        return results[0]

    def batch_get_sync(self, keys: list[str], bufs: list[memoryview]) -> None:
        if len(keys) != len(bufs):
            raise ValueError("keys and bufs length mismatch")
        future_id = int(self._client.submit_batch_get(keys, bufs))  # type: ignore[attr-defined]
        fut = self._register_future_sync("batch_get", future_id, (keys, tuple(bufs)))
        return fut.result()

    def batch_set_sync(self, keys: list[str], bufs: list[memoryview]) -> None:
        if len(keys) != len(bufs):
            raise ValueError("keys and bufs length mismatch")
        future_id = int(self._client.submit_batch_set(keys, bufs))  # type: ignore[attr-defined]
        fut = self._register_future_sync("batch_set", future_id, (keys, tuple(bufs)))
        return fut.result()

    def batch_exists_sync(self, keys: list[str]) -> list[bool]:
        future_id = int(self._client.submit_batch_exists(keys))  # type: ignore[attr-defined]
        fut = self._register_future_sync("batch_exists", future_id)
        return fut.result()

    def batched_exists_sync(self, keys: list[str]) -> list[bool]:
        return self.batch_exists_sync(keys)

    def close(self) -> None:
        if not self._closed:
            self._shutdown_native(best_effort=True)
            self._fail_all(RuntimeError("Client closed"))
            self._client.close()  # type: ignore[attr-defined]
