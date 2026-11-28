# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading
from collections.abc import Callable
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.utils import CircuitBreaker
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import ReqMeta
from vllm.logger import init_logger

logger = init_logger(__name__)


class CacheCircuitOpen(RuntimeError):
    pass


class _StoreCircuitBreaker(CircuitBreaker):
    """Keep native probes active and ignore results from before an outage."""

    def __init__(self, name: str, probe_window_s: float) -> None:
        super().__init__(name, 30.0, 3, probe_window_s)
        self._lock = threading.Lock()
        self._probe_active = False
        self._generation = 0

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._probe_active or super().is_open

    def begin(self) -> int | None:
        with self._lock:
            if self._probe_active or not self.allow():
                return None
            self._probe_active = super().is_open
            return self._generation

    def finish(
        self, generation: int, success: bool | None, *, release_probe: bool = False
    ) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._probe_active = False
            if release_probe:
                self._open_until = 0.0
            if success is True:
                self.record_success()
            elif success is False:
                self.record_failure()
                if super().is_open:
                    self._generation += 1


class MooncakeStoreService:
    """Own cache I/O policy; transfer threads retain CUDA buffer ownership."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self.save_queue: queue.Queue[ReqMeta | None] = queue.Queue()
        self.load_queue: queue.Queue[ReqMeta] = queue.Queue()
        self.save_breaker = _StoreCircuitBreaker("Mooncake save", 5.0)
        self._save_generation: int | None = None
        # None: no PUT; True: successful PUT; False: any SDK failure.
        self._save_outcome: bool | None = None
        self.read_breaker = _StoreCircuitBreaker("Mooncake read", 1.0)
        self._read_lock = threading.Lock()
        self._read_failures = 0

    def submit_save(self, request: ReqMeta, finish: Callable[[ReqMeta], None]) -> None:
        if self.save_breaker.is_open:
            finish(request)
            return
        self.save_queue.put_nowait(request)

    @staticmethod
    def _validate_results(results: list[int], count: int) -> bool:
        if len(results) == count:
            return True
        logger.warning("Mooncake returned %d results for %d keys", len(results), count)
        return False

    def start_save(self) -> None:
        self._save_generation = None
        self._save_outcome = None

    def check_save_exists(self, keys: list[str]) -> list[int]:
        if self._save_generation is None:
            self._save_generation = self.save_breaker.begin()
            if self._save_generation is None:
                raise CacheCircuitOpen("Mooncake save circuit breaker is open")
        try:
            results = self.store.batch_is_exist(keys)
            if not self._validate_results(results, len(keys)):
                results = [-1] * len(keys)
        except Exception:
            self._save_outcome = False
            raise
        if any(result < 0 for result in results):
            self._save_outcome = False
        return results

    def put(
        self,
        keys: list[str],
        addrs: list[list[int]],
        sizes: list[list[int]],
        replicate_config: Any,
    ) -> list[int]:
        try:
            results = self.store.batch_put_from_multi_buffers(
                keys, addrs, sizes, replicate_config
            )
            if not self._validate_results(results, len(keys)):
                results = [-1] * len(keys)
        except Exception:
            self._save_outcome = False
            raise
        if any(result < 0 for result in results):
            self._save_outcome = False
        elif self._save_outcome is not False:
            self._save_outcome = True
        return results

    def finish_save(self) -> None:
        if self._save_generation is not None:
            self.save_breaker.finish(self._save_generation, self._save_outcome)

    def get(
        self,
        keys: list[str],
        addrs: list[list[int]],
        sizes: list[list[int]],
    ) -> list[int]:
        return self._read(
            lambda: self.store.batch_get_into_multi_buffers(keys, addrs, sizes),
            len(keys),
            lookup=False,
        )

    def lookup_exists(
        self,
        keys: list[str],
        exists_fn: Callable[[list[str]], list[int]],
    ) -> list[int]:
        return self._read(
            lambda: exists_fn(keys),
            len(keys),
            lookup=True,
        )

    def _read(
        self, call: Callable[[], list[int]], count: int, *, lookup: bool
    ) -> list[int]:
        if count == 0:
            return []
        generation = self.read_breaker.begin()
        if generation is None:
            raise CacheCircuitOpen("Mooncake read circuit breaker is open")
        failed = True
        try:
            results = call()
            if not self._validate_results(results, count):
                results = [-1] * count
            # Cache eviction/missing keys are expected; they require recompute,
            # not an outage. Existence checks use 0/1 rather than these codes.
            failed = any(r < 0 and (lookup or r not in (-400, -704)) for r in results)
            return results
        finally:
            if failed and not lookup:
                with self._read_lock:
                    self._read_failures += 1
            success = not failed and not lookup and all(r >= 0 for r in results)
            # An existence query cannot prove that owner transfers recovered.
            self.read_breaker.finish(
                generation,
                False if failed else (True if success else None),
                release_probe=not failed,
            )

    def take_read_failures(self) -> int:
        with self._read_lock:
            failures, self._read_failures = self._read_failures, 0
            return failures
