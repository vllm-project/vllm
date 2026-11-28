# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import math
import queue
import threading
import time
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import regex as re

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import ReqMeta
from vllm.logger import init_logger

logger = init_logger(__name__)

# mooncake_master serves Prometheus metrics on --http_metrics_port, 9003 by default.
MOONCAKE_MASTER_METRICS_PORT = 9003
_SEGMENT_CAPACITY = re.compile(
    r'^segment_total_capacity_bytes\{segment="([^"]+)"\} ([0-9.eE+]+)$', re.MULTILINE
)


class StagePlacement:
    """Chooses one Mooncake segment per chunk for pipeline-parallel puts.

    Each pipeline stage stores its layers of a chunk as its own object, and a
    hit needs every stage's object. Placed independently, the objects land on
    different clients, which offload them to different SSDs and evict them
    with separate LRUs, so a chunk can lose one stage and keep the others.
    Every stage picks the same segment from the chunk hash (rendezvous hashing
    weighted by segment capacity), so a chunk's objects share one client like
    the single object of a PP=1 worker. The segment list comes from the
    master's metrics and is refreshed in the background; Mooncake falls back
    to other segments when the chosen one is full.
    """

    def __init__(self, metrics_url: str, refresh_s: float = 60.0) -> None:
        self._url = metrics_url
        self._refresh_s = refresh_s
        self._segments: tuple[tuple[str, float], ...] = ()
        self._readable = True
        self._refresh()
        threading.Thread(
            target=self._refresh_loop, name="MooncakeStagePlacement", daemon=True
        ).start()

    def _refresh(self) -> None:
        try:
            with urllib.request.urlopen(self._url, timeout=5) as r:
                text = r.read().decode()
        except Exception as e:
            if self._readable:
                logger.warning(
                    "Mooncake stage placement: %s unreadable (%s); keeping "
                    "segments %s and retrying every %.0f s",
                    self._url,
                    e,
                    self._segments,
                    self._refresh_s,
                )
            self._readable = False
            return
        self._readable = True
        segments = tuple(
            sorted(
                (m.group(1), float(m.group(2)))
                for m in _SEGMENT_CAPACITY.finditer(text)
            )
        )
        if segments != self._segments:
            logger.info("Mooncake stage placement over segments %s", segments)
            self._segments = segments

    def _refresh_loop(self) -> None:
        while True:
            time.sleep(self._refresh_s)
            self._refresh()

    def pick(self, chunk_hash: str) -> str | None:
        best, best_score = None, -math.inf
        for name, capacity in self._segments:
            digest = hashlib.blake2b(f"{chunk_hash}|{name}".encode(), digest_size=8)
            u = (int.from_bytes(digest.digest(), "big") + 0.5) / 2**64
            score = -capacity / math.log(u)
            if score > best_score:
                best, best_score = name, score
        return best


class MooncakeStoreService:
    """Own cache I/O queues; transfer threads retain CUDA buffer ownership."""

    def __init__(self, store: Any, placement: StagePlacement | None = None) -> None:
        self.store = store
        self.placement = placement
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
        if self.placement is None:
            return self._put(keys, addrs, sizes, replicate_config)
        # One put per target segment; keys end with their chunk hash.
        by_segment: dict[str | None, list[int]] = defaultdict(list)
        for i, key in enumerate(keys):
            by_segment[self.placement.pick(key.rsplit("@", 1)[-1])].append(i)
        group_ids = getattr(replicate_config, "group_ids", None)
        results = [0] * len(keys)
        try:
            for segment, idx in by_segment.items():
                replicate_config.preferred_segments = [segment] if segment else []
                if group_ids is not None:
                    replicate_config.group_ids = [group_ids[i] for i in idx]
                part = self._put(
                    [keys[i] for i in idx],
                    [addrs[i] for i in idx],
                    [sizes[i] for i in idx],
                    replicate_config,
                )
                for i, result in zip(idx, part, strict=True):
                    results[i] = result
        finally:
            replicate_config.preferred_segments = []
            if group_ids is not None:
                replicate_config.group_ids = group_ids
        return results

    def _put(
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
