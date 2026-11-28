# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side control plane for execution auxiliary outputs."""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import chain, islice
from typing import TYPE_CHECKING

import numpy as np

from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    get_routed_experts_bytes_per_token,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import (
    maybe_convert_block_hash,
    resolve_kv_cache_block_sizes,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_events import ExternalBlockHash
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


def retained_objects(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> tuple[int, int]:
    """Number of hash-block objects whose routed experts the scheduler retains
    and the worker stores, and the tokens per object."""
    block_size, hash_block_size = resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    max_bytes = vllm_config.aux_output_config.max_bytes
    if max_bytes is not None:
        object_nbytes = hash_block_size * get_routed_experts_bytes_per_token(
            vllm_config
        )
        return max(max_bytes // object_nbytes, 1), hash_block_size
    num_blocks = kv_cache_config.num_blocks
    if vllm_config.kv_transfer_config is not None:
        num_blocks *= 2
    return num_blocks * (block_size // hash_block_size), hash_block_size


@dataclass(slots=True)
class PackedBlockHashes:
    """Contiguous, self-contained block hashes for scheduler-worker IPC."""

    data: bytes
    item_size: int

    def __iter__(self) -> Iterator[bytes]:
        for start in range(0, len(self.data), self.item_size):
            yield self.data[start : start + self.item_size]


@dataclass(slots=True)
class AuxOutputConnectorMetadata:
    generation: int
    requests: dict[str, int]
    block_hashes: dict[str, PackedBlockHashes]
    finished_requests: tuple[str, ...]
    retained_block_hashes: PackedBlockHashes | None = None
    released_block_hashes: PackedBlockHashes | None = None


@dataclass(slots=True)
class AuxRequestOutput:
    token_start: int
    rows: np.ndarray


class AuxOutputSchedulerConnector:
    """Build worker metadata without owning auxiliary output payloads or stores."""

    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        num_objects: int,
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
        # KV blocks whose rows are not all here: loaded without rows, computed
        # from mid-block on, or computed by an opted-out request. Any other
        # cached GPU block has rows.
        self._rowless: dict[ExternalBlockHash, BlockHash] = {}
        # First token each active request computes here.
        self._computed_from: dict[str, int] = {}
        # Per active opted-out request: KV blocks checked, the last one's hash,
        # whether it was scheduled since its lookup, and the blocks it computes,
        # held so their marks and rows outlive pruning until it finishes.
        self._skip_marked: dict[str, tuple[int, BlockHash | None, bool]] = {}
        self._skip_rowless: dict[str, list[ExternalBlockHash]] = {}
        self._skip_held: Counter[ExternalBlockHash] = Counter()
        self._capacity = num_objects // (self._block_size // self._hash_block_size)
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
            if self._skips(request):
                self._mark_rowless(request)
                continue
            self._sent_hash_counts.setdefault(request_id, 0)
            self._computed_from.setdefault(request_id, request.num_computed_tokens)
            packed = self._take_new_hashes(
                request, request.num_computed_tokens + num_tokens
            )
            if packed is not None:
                block_hashes_by_request[request_id] = packed
            self._loading.pop(request_id, None)
            scheduled_requests[request_id] = self._emit_start(request)
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

    @staticmethod
    def _emit_start(request: Request) -> int:
        if request.num_output_tokens:
            return request.num_tokens - 1
        assert request.sampling_params is not None
        # A P/D prefiller may cut the prompt short of the requested start.
        return min(
            request.sampling_params.routed_experts_prompt_start,
            request.num_prompt_tokens,
        )

    def max_cache_hit(self, request: Request) -> int:
        if self._skips(request):
            self._mark_rowless(request, lookup=True)
            return request.num_tokens - 1
        num_blocks = 0
        for block, keys in self._kv_blocks(request.block_hashes):
            if block in self._kv_cached:
                self._kv_cached.move_to_end(block)
            elif block in self._rowless or not self._gpu_cached(keys[-1]):
                break
            num_blocks += 1
        # A hit needs rows only from the first returned position on.
        num_hit = max(num_blocks * self._block_size, self._emit_start(request))
        return min(num_hit, request.num_tokens - 1)

    def cap_external_hit(
        self, request: Request, num_local: int, num_external: int
    ) -> int:
        if self._skips(request):
            return num_external
        num_hit = num_local + num_external
        if num_hit <= self._emit_start(request):
            for block, keys in islice(
                self._kv_blocks(request.block_hashes),
                num_local // self._block_size,
                cdiv(num_hit, self._block_size),
            ):
                if block not in self._kv_cached:
                    self._rowless[block] = keys[-1]
        elif num_hit % self._block_size or num_hit > self.max_cache_hit(request):
            return 0
        return num_external

    def request_loading(self, request: Request, num_tokens: int) -> None:
        if self._skips(request):
            return
        blocks = islice(
            self._kv_blocks(request.block_hashes), cdiv(num_tokens, self._block_size)
        )
        self._loading[request.request_id] = {block for block, _ in blocks}

    def _release_unheld(self) -> None:
        loading = set().union(*self._loading.values())
        if len(self._rowless) > self._capacity:
            # An opted-out request may cache its marked blocks only later.
            self._rowless = {
                block: block_hash
                for block, block_hash in self._rowless.items()
                if block in loading
                or block in self._skip_held
                or self._gpu_cached(block_hash)
            }
        excess = len(self._kv_cached) - self._capacity
        if excess <= 0:
            return
        victims = []
        for block, keys in self._kv_cached.items():
            if (
                block not in loading
                and block not in self._skip_held
                and not self._gpu_cached(keys[-1])
            ):
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
        if self._skips(request):
            # Its new tokens can complete a block before the next schedule caches it.
            self._mark_rowless(request)
            return None
        request_id = request.request_id
        # The block this request started mid-block has no rows here. Its hash
        # exists only once its last token is known; record the block before
        # the scheduler can cache it.
        computed_from = self._computed_from.get(request_id, 0)
        index, offset = divmod(computed_from, self._block_size)
        last = (index + 1) * (self._block_size // self._hash_block_size) - 1
        if offset and self._sent_hash_counts[request_id] <= last < len(
            request.block_hashes
        ):
            block_hash = request.block_hashes[last]
            block = maybe_convert_block_hash(block_hash)
            if block not in self._kv_cached:
                self._rowless[block] = block_hash
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
        # A session can change return_aux_outputs between chunks: drop all state.
        self._skip_marked.pop(request_id, None)
        self._drop_skip_holds(request_id)
        self._loading.pop(request_id, None)
        if request_id in self._sent_hash_counts:
            # The next metadata delivers this terminal event to the worker.
            self._finished_requests[request_id] = self._take_new_hashes(
                request, request.num_computed_tokens
            )
            del self._sent_hash_counts[request_id]
        computed_from = self._computed_from.pop(request_id, request.num_tokens)
        if self._skips(request):
            return
        num_blocks = request.num_computed_tokens // self._block_size
        blocks = islice(self._kv_blocks(request.block_hashes), num_blocks)
        for index, (block, keys) in enumerate(blocks):
            if index * self._block_size >= computed_from:
                self._rowless.pop(block, None)
            if (
                block not in self._kv_cached
                and block not in self._rowless
                and self._gpu_cached(keys[-1])
            ):
                self._kv_cached[block] = keys
                self._retained_hashes.extend(keys)
        self._release_unheld()

    def _mark_rowless(self, request: Request, lookup: bool = False) -> None:
        """Mark the KV blocks an opted-out request computes as rowless and hold
        them; blocks whose rows are kept here are held without a mark."""
        request_id = request.request_id
        step = self._block_size // self._hash_block_size
        hashes = request.block_hashes
        num_checked, last, scheduled = self._skip_marked.get(
            request_id, (0, None, True)
        )
        start = num_checked
        if lookup:
            # A lookup is not an admission: recheck from the start.
            self._drop_skip_holds(request_id)
            start = 0
        elif not scheduled:
            # The first scheduled step knows the real hit.
            start = request.num_computed_tokens // self._block_size
        elif num_checked and (
            len(hashes) < num_checked * step or hashes[num_checked * step - 1] != last
        ):
            # A session turn rewrote hashes: recheck from the start.
            self._drop_skip_holds(request_id)
            start = 0
        held = self._skip_rowless.setdefault(request_id, [])
        hit = lookup
        for block, keys in self._kv_blocks(hashes[start * step :]):
            hit = hit and self._gpu_cached(keys[-1]) is not None
            if hit:
                continue
            if block not in self._kv_cached:
                self._rowless[block] = keys[-1]
            held.append(block)
            self._skip_held[block] += 1
        num_blocks = len(hashes) // step
        self._skip_marked[request_id] = (
            num_blocks,
            hashes[num_blocks * step - 1] if num_blocks else None,
            not lookup,
        )

    def _drop_skip_holds(self, request_id: str) -> None:
        for block in self._skip_rowless.pop(request_id, ()):
            self._skip_held[block] -= 1
            if not self._skip_held[block]:
                del self._skip_held[block]

    @staticmethod
    def _skips(request: Request) -> bool:
        params = request.sampling_params
        return params is not None and not params.return_aux_outputs

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
        self._computed_from.clear()
        self._finished_requests.clear()
        self._generation += 1
        self._released_hashes.extend(chain.from_iterable(self._kv_cached.values()))
        self._kv_cached.clear()
        self._rowless.clear()
        self._skip_marked.clear()
        self._skip_rowless.clear()
        self._skip_held.clear()
