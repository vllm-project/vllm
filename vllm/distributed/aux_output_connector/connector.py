# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side control plane for execution auxiliary outputs."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING

import numpy as np

from vllm.distributed.kv_events import MEDIUM_CPU, BlockRemoved, BlockStored
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash

if TYPE_CHECKING:
    from vllm.distributed.kv_events import ExternalBlockHash, KVCacheEvent
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
    from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


@dataclass
class PackedBlockHashes:
    """Contiguous, self-contained block hashes for scheduler-worker IPC."""

    data: bytes
    item_size: int

    def __iter__(self) -> Iterator[bytes]:
        for start in range(0, len(self.data), self.item_size):
            yield self.data[start : start + self.item_size]


@dataclass
class AuxOutputConnectorMetadata:
    generation: int
    requests: dict[str, int]
    block_hashes: dict[str, PackedBlockHashes]
    finished_requests: tuple[str, ...]
    # Hashes of blocks whose objects the KV cache tiers start or stop holding.
    retained_block_hashes: PackedBlockHashes | None = None
    released_block_hashes: PackedBlockHashes | None = None


@dataclass
class AuxRequestOutput:
    token_start: int
    rows: np.ndarray


class AuxOutputSchedulerConnector:
    """Build worker metadata without owning auxiliary output payloads or stores.

    The worker retains a finished request's GPU-cached blocks while the GPU or
    CPU KV cache holds them, so every GPU-cached full block has its outputs.
    """

    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        kv_connector: KVConnectorBase_V1 | None = None,
    ) -> None:
        # Number of hashes already sent to the worker for each active request.
        self._sent_hash_counts: dict[str, int] = {}
        # Terminal events are delivered with the next connector metadata.
        self._finished_requests: dict[str, PackedBlockHashes | None] = {}
        self._generation = 0
        # Full attention caches the whole prefix that a hit can reuse.
        coordinator = kv_cache_manager.coordinator
        self._group_id = getattr(coordinator, "full_attention_group_id", None) or 0
        self._kv_cache_manager = kv_cache_manager
        self._block_pool = kv_cache_manager.block_pool
        # HiSparse moves this group's cache to a host pool after construction.
        self._manager = coordinator.single_type_managers[self._group_id]
        self._block_size = self._manager.block_size
        # A trimmed external hit ends where the KV connector can stop loading;
        # None if it loads only whole hits, or a Mamba state ends the load.
        alignment = kv_connector.external_load_alignment if kv_connector else None
        self._external_alignment = (
            math.lcm(coordinator.scheduler_block_size, alignment)
            if alignment and not kv_cache_manager.kv_cache_config.has_mamba_layers
            else None
        )
        # Retained block -> its hash-block hashes; the last is the block's hash.
        self._kv_cached: dict[ExternalBlockHash, Sequence[BlockHash]] = {}
        # CPU chunks holding each block; overlapping chunks announce it again.
        self._cpu_cached: Counter[ExternalBlockHash] = Counter()
        # Blocks of pending async loads, retained until their request runs.
        self._loading: dict[str, set[ExternalBlockHash]] = {}
        # Retained blocks no tier is known to hold, oldest first. A KV connector
        # without residency events may still hold them, so the oldest are only
        # released once there are more than KV cache blocks.
        self._unheld: dict[ExternalBlockHash, None] = {}
        self._max_unheld = kv_cache_manager.kv_cache_config.num_blocks
        self._retained_hashes: list[bytes] = []
        self._released_hashes: list[bytes] = []

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
        requests: dict[str, Request],
    ) -> AuxOutputConnectorMetadata:
        """Build one step's incremental worker metadata."""
        # Release this step's evictions before the worker stores new outputs;
        # update_kv_cache_events counts these events later.
        for event in self._block_pool.kv_event_queue:
            for block in getattr(event, "block_hashes", ()):
                self._release_uncached(block)
        scheduled_requests: dict[str, int] = {}
        block_hashes_by_request: dict[str, PackedBlockHashes] = {}
        for request_id in scheduler_output.num_scheduled_tokens:
            num_sent = self._sent_hash_counts.setdefault(request_id, 0)
            request = requests[request_id]
            packed = self._pack_new_hashes(request.block_hashes, num_sent)
            if packed is not None:
                block_hashes_by_request[request_id] = packed
            self._sent_hash_counts[request_id] = len(request.block_hashes)
            self._end_loading(request_id)
            assert request.sampling_params is not None
            # A P/D prefiller may cut the prompt short of the requested start.
            scheduled_requests[request_id] = max(
                min(
                    request.sampling_params.routed_experts_prompt_start,
                    request.num_prompt_tokens,
                ),
                0 if request.num_output_tokens == 0 else request.num_tokens - 1,
            )
        # A settled token can complete a hash block after the next async schedule
        # was built. Send a hash-only update if the request was not rescheduled.
        if len(self._sent_hash_counts) != len(scheduled_requests):
            for request_id, num_sent in self._sent_hash_counts.items():
                if request_id in scheduled_requests:
                    continue
                request = requests[request_id]
                packed = self._pack_new_hashes(request.block_hashes, num_sent)
                if packed is None:
                    continue
                block_hashes_by_request[request_id] = packed
                self._sent_hash_counts[request_id] = len(request.block_hashes)
        # Sending transfers ownership of these one-shot events.
        finished_requests = tuple(self._finished_requests)
        block_hashes_by_request.update(
            (request_id, block_hashes)
            for request_id, block_hashes in self._finished_requests.items()
            if block_hashes is not None
        )
        self._finished_requests = {}
        metadata = AuxOutputConnectorMetadata(
            self._generation,
            scheduled_requests,
            block_hashes_by_request,
            finished_requests,
            self._pack_new_hashes(self._retained_hashes, 0),
            self._pack_new_hashes(self._released_hashes, 0),
        )
        self._retained_hashes, self._released_hashes = [], []
        return metadata

    def max_cache_hit(self, request: Request) -> int:
        """Return the longest hit on full blocks that are GPU-cached or retained."""
        num_blocks = 0
        for block, keys in self._kv_blocks(request.block_hashes):
            if block not in self._kv_cached and not self._gpu_cached(keys[-1]):
                break
            num_blocks += 1
        return min(num_blocks * self._block_size, request.num_tokens - 1)

    def cap_local_hit(
        self, request: Request, blocks: KVCacheBlocks, num_tokens: int
    ) -> tuple[KVCacheBlocks, int]:
        """Drop a local hit past the retained outputs; it cannot be cut shorter.

        With a full-attention group every GPU-cached block is retained or held
        by a running request, so only partial-block and sparse hits are dropped.
        """
        if num_tokens <= self.max_cache_hit(request):
            return blocks, num_tokens
        return self._kv_cache_manager.empty_kv_cache_blocks, 0

    def cap_external_hit(
        self, request: Request, num_local: int, num_external: int
    ) -> int:
        """Trim an external hit to the retained outputs.

        Capture resumes on a block boundary, and a trimmed hit on a boundary
        the KV connector can load to.
        """
        num_hit = min(num_local + num_external, self.max_cache_hit(request))
        num_hit -= num_hit % self._kv_cache_manager.coordinator.scheduler_block_size
        if num_external <= num_hit - num_local:
            return num_external
        if self._external_alignment is None:
            return 0
        return max(num_hit - num_hit % self._external_alignment - num_local, 0)

    def request_loading(self, request: Request, num_tokens: int) -> None:
        """Keep an async load's blocks retained until its request is scheduled."""
        num_blocks = num_tokens // self._block_size
        blocks = islice(self._kv_blocks(request.block_hashes), num_blocks)
        previous = self._loading.get(request.request_id, set())
        self._loading[request.request_id] = loading = {block for block, _ in blocks}
        # A retried load may cover fewer blocks than the failed one.
        for block in previous | loading:
            self._release_uncached(block)

    def _end_loading(self, request_id: str) -> None:
        for block in self._loading.pop(request_id, ()):
            self._release_uncached(block)

    def update_kv_cache_events(self, events: Iterable[KVCacheEvent]) -> None:
        """Release retained blocks that no KV cache tier holds any more."""
        for event in events:
            if not isinstance(event, (BlockStored, BlockRemoved)):
                continue
            on_cpu = event.medium == MEDIUM_CPU and event.group_idx == self._group_id
            for block in event.block_hashes:
                if on_cpu:
                    self._cpu_cached[block] += (
                        1 if isinstance(event, BlockStored) else -1
                    )
                    if self._cpu_cached[block] <= 0:
                        del self._cpu_cached[block]
                        # The CPU tier evicted the block: release it now.
                        self._release_uncached(block, now=True)
                self._release_uncached(block)

    def reset_connector_cache(self) -> None:
        """Forget the CPU tier after the KV connector cleared it."""
        self._cpu_cached.clear()
        for block in list(self._kv_cached):
            self._release_uncached(block, now=True)

    def _release_uncached(self, block: ExternalBlockHash, now: bool = False) -> None:
        keys = self._kv_cached.get(block)
        if not keys:
            return
        if (
            block in self._cpu_cached
            or self._gpu_cached(keys[-1])
            or any(block in blocks for blocks in self._loading.values())
        ):
            self._unheld.pop(block, None)
            return
        if not now:
            self._unheld.setdefault(block)
            if len(self._unheld) <= self._max_unheld:
                return
            block = next(iter(self._unheld))
        self._unheld.pop(block, None)
        self._released_hashes.extend(self._kv_cached.pop(block))

    def _gpu_cached(self, block_hash: BlockHash) -> list[KVCacheBlock] | None:
        return self._manager.block_pool.get_cached_block(block_hash, [self._group_id])

    def _kv_blocks(
        self, hashes: Sequence[BlockHash]
    ) -> Iterator[tuple[ExternalBlockHash, Sequence[BlockHash]]]:
        step = self._block_size // self._block_pool.hash_block_size
        for end in range(step, len(hashes) + 1, step):
            yield maybe_convert_block_hash(hashes[end - 1]), hashes[end - step : end]

    def take_output(
        self,
        request: Request,
        output: dict[str, AuxRequestOutput] | None,
    ) -> np.ndarray | None:
        """Return the accepted R3 rows for one scheduled request."""
        request_id = request.request_id
        assert output is not None and request_id in output, (
            f"auxiliary output worker output is missing {request_id}"
        )
        request_output = output[request_id]
        token_end = request.num_tokens - 1
        local_end = token_end - request_output.token_start
        if local_end < 0:
            assert not request.is_finished(), (
                "finished auxiliary output output has no accepted token range: "
                f"request={request_id}, token_end={token_end}, "
                f"output_start={request_output.token_start}, "
                "output_end="
                f"{request_output.token_start + len(request_output.rows)}"
            )
            return None
        assert local_end <= len(request_output.rows), (
            "auxiliary output worker output has an invalid token range: "
            f"request={request_id}, token_end={token_end}, "
            f"output_start={request_output.token_start}, "
            f"output_end={request_output.token_start + len(request_output.rows)}"
        )
        return request_output.rows[:local_end]

    def request_finished(self, request: Request) -> None:
        """Queue a request's terminal event and final block hashes."""
        request_id = request.request_id
        num_sent = self._sent_hash_counts.pop(request_id, None)
        self._end_loading(request_id)
        # Retain the cached blocks the request held, before it releases them.
        num_blocks = request.num_computed_tokens // self._block_size
        for block, keys in islice(self._kv_blocks(request.block_hashes), num_blocks):
            if block not in self._kv_cached and self._gpu_cached(keys[-1]):
                self._kv_cached[block] = keys
                self._retained_hashes.extend(keys)
        if num_sent is None:
            return
        # The next metadata delivers this terminal event to the worker.
        self._finished_requests[request_id] = self._pack_new_hashes(
            request.block_hashes, num_sent
        )

    @staticmethod
    def _pack_new_hashes(
        block_hashes: Sequence[bytes], num_sent: int
    ) -> PackedBlockHashes | None:
        assert num_sent <= len(block_hashes), "KV block-hash history shrank"
        new_hashes = block_hashes[num_sent:]
        if not new_hashes:
            return None
        return PackedBlockHashes(b"".join(new_hashes), len(new_hashes[0]))

    def reset(self) -> None:
        """Start a new auxiliary output namespace after a prefix-cache reset."""
        # The worker drops temporary state on generation changes; resend hashes.
        self._sent_hash_counts.clear()
        self._finished_requests.clear()
        self._generation += 1
        # The worker releases these before it switches generations.
        self._released_hashes.extend(
            k for keys in self._kv_cached.values() for k in keys
        )
        self._kv_cached.clear()
        self._unheld.clear()
