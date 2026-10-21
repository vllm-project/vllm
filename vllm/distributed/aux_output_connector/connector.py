# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side control plane for execution auxiliary outputs."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import chain, islice
from typing import TYPE_CHECKING

import numpy as np

from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash

if TYPE_CHECKING:
    from vllm.distributed.kv_events import ExternalBlockHash
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
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
    retained_block_hashes: PackedBlockHashes | None = None
    released_block_hashes: PackedBlockHashes | None = None


@dataclass
class AuxRequestOutput:
    token_start: int
    rows: np.ndarray


class AuxOutputSchedulerConnector:
    """Build worker metadata without owning auxiliary output payloads or stores."""

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
        coordinator = kv_cache_manager.coordinator
        self._group_id = getattr(coordinator, "full_attention_group_id", None) or 0
        self._block_pool = kv_cache_manager.block_pool
        self._manager = coordinator.single_type_managers[self._group_id]
        self._block_size = self._manager.block_size
        self._hash_block_size = self._block_pool.hash_block_size
        self._kv_cached: OrderedDict[ExternalBlockHash, Sequence[BlockHash]] = (
            OrderedDict()
        )
        self._loading: dict[str, set[ExternalBlockHash]] = {}
        self._capacity = kv_cache_manager.kv_cache_config.num_blocks * (
            2 if kv_connector else 1
        )
        self._retained_hashes: list[bytes] = []
        self._released_hashes: list[bytes] = []

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
        requests: dict[str, Request],
    ) -> AuxOutputConnectorMetadata:
        """Build one step's incremental worker metadata."""
        scheduled_requests: dict[str, int] = {}
        block_hashes_by_request: dict[str, PackedBlockHashes] = {}
        for request_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            request = requests[request_id]
            if self._skips_routing(request):
                continue
            self._sent_hash_counts.setdefault(request_id, 0)
            packed = self._take_new_hashes(
                request, request.num_computed_tokens + num_tokens
            )
            if packed is not None:
                block_hashes_by_request[request_id] = packed
            self._loading.pop(request_id, None)
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
            for request_id in self._sent_hash_counts:
                if request_id in scheduled_requests:
                    continue
                request = requests[request_id]
                packed = self._take_new_hashes(request, request.num_computed_tokens)
                if packed is not None:
                    block_hashes_by_request[request_id] = packed
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
            self._pack_hashes(self._retained_hashes),
            self._pack_hashes(self._released_hashes),
        )
        self._retained_hashes, self._released_hashes = [], []
        return metadata

    def max_cache_hit(self, request: Request) -> int:
        if self._skips_routing(request):
            # No routing to serve, so the KV cache alone bounds the hit.
            return request.num_tokens - 1
        num_blocks = 0
        for block, keys in self._kv_blocks(request.block_hashes):
            if block in self._kv_cached:
                self._kv_cached.move_to_end(block)
            elif not self._gpu_cached(keys[-1]):
                break
            num_blocks += 1
        return min(num_blocks * self._block_size, request.num_tokens - 1)

    def cap_external_hit(
        self, request: Request, num_local: int, num_external: int
    ) -> int:
        if self._skips_routing(request):
            return num_external
        num_hit = num_local + num_external
        if num_hit % self._block_size == 0 and num_hit <= self.max_cache_hit(request):
            return num_external
        return 0

    def request_loading(self, request: Request, num_tokens: int) -> None:
        if self._skips_routing(request):
            return
        blocks = islice(
            self._kv_blocks(request.block_hashes), num_tokens // self._block_size
        )
        self._loading[request.request_id] = {block for block, _ in blocks}

    def _release_unheld(self) -> None:
        excess = len(self._kv_cached) - self._capacity
        if excess <= 0:
            return
        loading = set().union(*self._loading.values())
        victims = []
        for block, keys in self._kv_cached.items():
            if block not in loading and not self._gpu_cached(keys[-1]):
                victims.append(block)
                if len(victims) == excess:
                    break
        for block in victims:
            self._released_hashes.extend(self._kv_cached.pop(block))

    def _gpu_cached(self, block_hash: BlockHash) -> list[KVCacheBlock] | None:
        return self._manager.block_pool.get_cached_block(block_hash, [self._group_id])

    def _kv_blocks(
        self, hashes: Sequence[BlockHash]
    ) -> Iterator[tuple[ExternalBlockHash, Sequence[BlockHash]]]:
        step = self._block_size // self._hash_block_size
        for end in range(step, len(hashes) + 1, step):
            yield maybe_convert_block_hash(hashes[end - 1]), hashes[end - step : end]

    def take_output(
        self,
        request: Request,
        output: dict[str, AuxRequestOutput] | None,
    ) -> np.ndarray | None:
        """Return the accepted R3 rows for one scheduled request."""
        if self._skips_routing(request):
            return None
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
        self._loading.pop(request_id, None)
        if self._skips_routing(request):
            # Its blocks have no stored routing to retain.
            return
        num_blocks = request.num_computed_tokens // self._block_size
        for block, keys in islice(self._kv_blocks(request.block_hashes), num_blocks):
            if block not in self._kv_cached and self._gpu_cached(keys[-1]):
                self._kv_cached[block] = keys
                self._retained_hashes.extend(keys)
        self._release_unheld()
        if request_id not in self._sent_hash_counts:
            return
        # The next metadata delivers this terminal event to the worker.
        self._finished_requests[request_id] = self._take_new_hashes(
            request, request.num_computed_tokens
        )
        del self._sent_hash_counts[request_id]

    @staticmethod
    def _skips_routing(request: Request) -> bool:
        params = request.sampling_params
        return params is not None and not params.return_routed_experts

    @staticmethod
    def _pack_hashes(hashes: Sequence[bytes]) -> PackedBlockHashes | None:
        if not hashes:
            return None
        return PackedBlockHashes(b"".join(hashes), len(hashes[0]))

    def _take_new_hashes(
        self, request: Request, num_computed_tokens: int
    ) -> PackedBlockHashes | None:
        block_hashes = request.block_hashes
        num_sent = self._sent_hash_counts[request.request_id]
        assert num_sent <= len(block_hashes), "KV block-hash history shrank"
        num_keyed = min(len(block_hashes), num_computed_tokens // self._hash_block_size)
        if num_keyed <= num_sent:
            return None
        self._sent_hash_counts[request.request_id] = num_keyed
        return self._pack_hashes(block_hashes[num_sent:num_keyed])

    def reset(self) -> None:
        """Start a new auxiliary output namespace after a prefix-cache reset."""
        # The worker drops temporary state on generation changes; resend hashes.
        self._sent_hash_counts.clear()
        self._finished_requests.clear()
        self._generation += 1
        self._released_hashes.extend(chain.from_iterable(self._kv_cached.values()))
        self._kv_cached.clear()
