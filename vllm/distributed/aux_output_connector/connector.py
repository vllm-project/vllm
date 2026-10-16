# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side control plane for execution auxiliary outputs."""

from __future__ import annotations

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
    from vllm.v1.core.kv_cache_manager import KVCacheManager
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
    # Objects to keep, and to stop keeping, for KV-cached blocks.
    pinned_hashes: PackedBlockHashes | None = None
    unpinned_hashes: PackedBlockHashes | None = None


@dataclass
class AuxRequestOutput:
    token_start: int
    rows: np.ndarray


class KVCacheRetention:
    """Keep the R3 objects of blocks a running request held while a KV tier
    caches them. Hits are capped at kept or held blocks, whose objects exist."""

    def __init__(self, kv_cache_manager: KVCacheManager) -> None:
        # Full attention caches the whole prefix that a hit can reuse.
        coordinator = kv_cache_manager.coordinator
        self._group_id = getattr(coordinator, "full_attention_group_id", None) or 0
        self._block_pool = kv_cache_manager.block_pool
        self._block_size = coordinator.single_type_managers[self._group_id].block_size
        self.hash_block_size = self._block_pool.hash_block_size
        self._hashes_per_block = self._block_size // self.hash_block_size
        self.reset()

    def reset(self) -> None:
        # Kept block -> its hash-block hashes; the last one is the block's hash.
        self._kept: dict[ExternalBlockHash, Sequence[BlockHash]] = {}
        # Kept blocks cached only on CPU, oldest first; residency per CPU chunk.
        self._cpu_only: dict[ExternalBlockHash, None] = {}
        self._cpu_counts: Counter[ExternalBlockHash] = Counter()
        # Blocks of accepted external hits that the worker does not retain yet.
        self.loading: dict[str, set[ExternalBlockHash]] = {}
        self.pins: list[bytes] = []
        self.unpins: list[bytes] = []

    def _blocks(
        self, hashes: Sequence[BlockHash]
    ) -> Iterator[tuple[ExternalBlockHash, Sequence[BlockHash]]]:
        step = self._hashes_per_block
        for end in range(step, len(hashes) + 1, step):
            yield maybe_convert_block_hash(hashes[end - 1]), hashes[end - step : end]

    def _cached(self, block_hash: BlockHash) -> list[KVCacheBlock] | None:
        return self._block_pool.get_cached_block(block_hash, [self._group_id])

    def max_hit_tokens(self, request: Request) -> int:
        """Return the longest hit whose blocks are all kept or held."""
        num_blocks = 0
        for block, keys in self._blocks(request.block_hashes):
            if block not in self._kept:
                cached = self._cached(keys[-1])
                if not cached or cached[0].ref_cnt == 0:
                    break
            num_blocks += 1
        return min(num_blocks * self._block_size, request.num_tokens - 1)

    def request_loading(self, request: Request, num_tokens: int) -> None:
        """Protect an async load's hit until the request's hashes reach the worker."""
        blocks = islice(
            self._blocks(request.block_hashes), num_tokens // self._block_size
        )
        self.loading[request.request_id] = {block for block, _ in blocks}

    def request_finished(self, request: Request) -> None:
        """Keep the cached blocks that a finished or preempted request held."""
        self.loading.pop(request.request_id, None)
        num_blocks = request.num_computed_tokens // self._block_size
        for block, keys in islice(self._blocks(request.block_hashes), num_blocks):
            if block not in self._kept and self._cached(keys[-1]):
                self._kept[block] = keys
                self.pins.extend(keys)

    def observe(self, events: Iterable[KVCacheEvent]) -> None:
        """Re-check kept blocks named by KV events; repeated events are no-ops."""
        for event in events:
            if not isinstance(event, (BlockStored, BlockRemoved)):
                continue
            for block in event.block_hashes:
                if event.medium == MEDIUM_CPU and event.group_idx == self._group_id:
                    self._cpu_counts[block] += (
                        1 if isinstance(event, BlockStored) else -1
                    )
                    if self._cpu_counts[block] <= 0:
                        del self._cpu_counts[block]
                self._recheck(block)
        # The worker store holds a second KV cache's worth of CPU-only blocks;
        # drop the oldest beyond that.
        excess = len(self._cpu_only) - self._block_pool.num_gpu_blocks
        if excess > 0:
            loading = set().union(*self.loading.values())
            oldest = (block for block in self._cpu_only if block not in loading)
            for block in list(islice(oldest, excess)):
                self._unpin(block)

    def cpu_cleared(self) -> None:
        self._cpu_counts.clear()
        for block in list(self._cpu_only):
            self._recheck(block)

    def _recheck(self, block: ExternalBlockHash) -> None:
        keys = self._kept.get(block)
        if keys is None or self._cached(keys[-1]):
            self._cpu_only.pop(block, None)
        elif block in self._cpu_counts:
            self._cpu_only.setdefault(block)
        else:
            self._unpin(block)

    def _unpin(self, block: ExternalBlockHash) -> None:
        self._cpu_only.pop(block, None)
        self.unpins.extend(self._kept.pop(block))


class AuxOutputSchedulerConnector:
    """Build worker metadata without owning auxiliary output payloads or stores."""

    def __init__(self, retention: KVCacheRetention | None = None) -> None:
        self._retention = retention
        # Number of hashes already sent to the worker for each active request.
        self._sent_hash_counts: dict[str, int] = {}
        # Terminal events are delivered with the next connector metadata.
        self._finished_requests: dict[str, PackedBlockHashes | None] = {}
        self._generation = 0

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
        requests: dict[str, Request],
    ) -> AuxOutputConnectorMetadata:
        """Build one step's incremental worker metadata."""
        scheduled_requests: dict[str, int] = {}
        block_hashes_by_request: dict[str, PackedBlockHashes] = {}
        for request_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            num_sent = self._sent_hash_counts.setdefault(request_id, 0)
            request = requests[request_id]
            num_hashes = self._num_hashes(request, num_tokens, num_sent)
            packed = self._pack_new_hashes(request.block_hashes, num_sent, num_hashes)
            if packed is not None:
                block_hashes_by_request[request_id] = packed
            self._sent_hash_counts[request_id] = num_hashes
            if self._retention is not None:
                self._retention.loading.pop(request_id, None)
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
                num_hashes = self._num_hashes(request, 0, num_sent)
                packed = self._pack_new_hashes(
                    request.block_hashes, num_sent, num_hashes
                )
                if packed is None:
                    continue
                block_hashes_by_request[request_id] = packed
                self._sent_hash_counts[request_id] = num_hashes
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
        )
        if (retention := self._retention) is not None:
            metadata.pinned_hashes = self._pack_new_hashes(retention.pins, 0)
            metadata.unpinned_hashes = self._pack_new_hashes(retention.unpins, 0)
            retention.pins, retention.unpins = [], []
        return metadata

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
        if self._retention is not None:
            self._retention.request_finished(request)
        if num_sent is None:
            return
        # The next metadata delivers this terminal event to the worker.
        self._finished_requests[request_id] = self._pack_new_hashes(
            request.block_hashes, num_sent, self._num_hashes(request, 0, num_sent)
        )

    def _num_hashes(self, request: Request, num_scheduled: int, num_sent: int) -> int:
        if (retention := self._retention) is None:
            return len(request.block_hashes)
        # Retaining a later chunk's objects would hold slots no KV block backs.
        num_tokens = request.num_computed_tokens + num_scheduled
        num_hashes = min(
            len(request.block_hashes), num_tokens // retention.hash_block_size
        )
        return max(num_hashes, num_sent)

    @staticmethod
    def _pack_new_hashes(
        block_hashes: Sequence[bytes], num_sent: int, num_hashes: int | None = None
    ) -> PackedBlockHashes | None:
        assert num_sent <= len(block_hashes), "KV block-hash history shrank"
        new_hashes = block_hashes[num_sent:num_hashes]
        if not new_hashes:
            return None
        return PackedBlockHashes(b"".join(new_hashes), len(new_hashes[0]))

    def reset(self) -> None:
        """Start a new auxiliary output namespace after a prefix-cache reset."""
        # The worker drops temporary state on generation changes; resend hashes.
        self._sent_hash_counts.clear()
        self._finished_requests.clear()
        self._generation += 1
        if self._retention is not None:
            self._retention.reset()
