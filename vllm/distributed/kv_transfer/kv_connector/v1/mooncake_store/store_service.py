# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
from collections.abc import Callable
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import ReqMeta
from vllm.logger import init_logger

logger = init_logger(__name__)


class MooncakeStoreService:
    """Own cache I/O queues; transfer threads retain CUDA buffer ownership."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self.save_queue: queue.Queue[ReqMeta | None] = queue.Queue()
        self.load_queue: queue.Queue[ReqMeta] = queue.Queue()

    @staticmethod
    def _validate_results(results: list[int], count: int) -> bool:
        if len(results) == count:
            return True
        logger.warning("Mooncake returned %d results for %d keys", len(results), count)
        return False

    def check_save_exists(self, keys: list[str]) -> list[int]:
        results = self.store.batch_is_exist(keys)
        return (
            results if self._validate_results(results, len(keys)) else [-1] * len(keys)
        )

    def put(
        self,
        keys: list[str],
        addrs: list[list[int]],
        sizes: list[list[int]],
        replicate_config: Any,
    ) -> list[int]:
        results = self.store.batch_put_from_multi_buffers(
            keys, addrs, sizes, replicate_config
        )
        return (
            results if self._validate_results(results, len(keys)) else [-1] * len(keys)
        )

    def get(
        self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]
    ) -> list[int]:
        results = self.store.batch_get_into_multi_buffers(keys, addrs, sizes)
        return (
            results if self._validate_results(results, len(keys)) else [-1] * len(keys)
        )

    def lookup_exists(
        self, keys: list[str], exists_fn: Callable[[list[str]], list[int]]
    ) -> list[int]:
        results = exists_fn(keys)
        if not self._validate_results(results, len(keys)):
            results = [-1] * len(keys)
        return results
