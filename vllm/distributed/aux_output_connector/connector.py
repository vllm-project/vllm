# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side control plane for execution auxiliary outputs."""

from __future__ import annotations

import json
import uuid
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import chain, islice
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    get_routed_experts_bytes_per_token,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import (
    maybe_convert_block_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_events import ExternalBlockHash
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.engine import EngineCoreRequest
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


_AUX_SALT_PREFIX = "vllm-aux:"


def aux_output_cache_salt(
    cache_salt: str | None,
    sampling_params: SamplingParams | None,
    object_key_prefix: str | None = None,
) -> str | None:
    """Salt a request by the auxiliary outputs its KV blocks carry. A request
    that skips them has blocks without recorded routing, and a request with
    `aux_namespace` has blocks whose routing is in objects named by its key
    prefix and namespace, so each gets a salt in the reserved form, which no
    other request can produce."""
    salt: dict[str, Any] = {}
    if sampling_params is not None and not sampling_params.return_aux_outputs:
        salt["aux_outputs"] = False
    if sampling_params is not None and sampling_params.aux_namespace is not None:
        salt["aux_namespace"] = sampling_params.aux_namespace
        salt["object_key_prefix"] = object_key_prefix
    if not salt and not (cache_salt or "").startswith(_AUX_SALT_PREFIX):
        return cache_salt
    salt["cache_salt"] = cache_salt
    return _AUX_SALT_PREFIX + json.dumps(salt, sort_keys=True)


def salt_aux_outputs(
    request: EngineCoreRequest, object_key_prefix: str | None = None
) -> None:
    """Apply aux_output_cache_salt to a request entering an engine that records
    auxiliary outputs. Frontend prompt hashes do not include a changed salt, so
    they are dropped and the engine hashes the prompt."""
    cache_salt = aux_output_cache_salt(
        request.cache_salt, request.sampling_params, object_key_prefix
    )
    if cache_salt != request.cache_salt:
        request.cache_salt = cache_salt
        request.block_hashes = None


AuxObject = tuple[str, int, int]
"""`(key, start, end)`: positions `[start, end)` are the first `end - start`
rows of the object `key`, whose first position is `start`."""


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
    namespace_runs: tuple[str, ...] = ()
    object_writes: dict[str, list[AuxObject]] = field(default_factory=dict)


@dataclass(slots=True)
class AuxRequestOutput:
    token_start: int
    rows: np.ndarray


@dataclass(slots=True)
class _NamespaceRequest:
    request: Request
    namespace: str
    prompt_start: int
    listing: list[AuxObject] = field(default_factory=list)
    written: int = 0
    running: bool = False
    in_worker: bool = False
    preempted: int | None = None
    valid: int | None = None


class AuxOutputSchedulerConnector:
    """Build worker metadata without owning auxiliary output payloads or stores."""

    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        num_objects: int,
        key_prefix: str | None = None,
    ) -> None:
        # Number of hashes already sent to the worker for each active request.
        self._sent_hash_counts: dict[str, int] = {}
        # Terminal events are delivered with the next connector metadata.
        self._finished_requests: dict[str, PackedBlockHashes | None] = {}
        self._generation = 0
        self._kv_cache_manager = kv_cache_manager
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
        # KV blocks whose rows are not all here: loaded without rows, or
        # computed from mid-block on. Any other cached GPU block has rows.
        self._rowless: dict[ExternalBlockHash, BlockHash] = {}
        # First token each active request computes here.
        self._computed_from: dict[str, int] = {}
        self._capacity = num_objects // (self._block_size // self._hash_block_size)
        self._retained_hashes: list[bytes] = []
        self._released_hashes: list[bytes] = []
        if key_prefix is not None and (
            coordinator.enable_partial_hash_hits
            or self._block_pool.num_hash_lookahead_tokens > 1
            or type(self._manager) is not FullAttentionManager
            or any(
                group.prefix_replay_tokens
                for group in kv_cache_manager.kv_cache_config.kv_cache_groups
            )
        ):
            raise ValueError(
                "AuxOutputConfig.object_key_prefix needs a plain full-attention "
                "KV cache group, prefix-cache hits on KV block boundaries, at "
                "most one hash look-ahead token and no prefix replay."
            )
        self._key_prefix = key_prefix
        self._namespaces: dict[str, _NamespaceRequest] = {}
        self._runs: list[str] = []
        self._writes: dict[str, list[AuxObject]] = {}
        self._listings: dict[str, list[AuxObject]] = {}

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
            if (namespace := self._namespace(request)) is not None:
                self._run_namespace(request, namespace)
                continue
            if self._skips(request):
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
        return self._take_metadata(scheduled_requests, block_hashes_by_request)

    def drain_metadata(self) -> AuxOutputConnectorMetadata | None:
        """Metadata for the write barrier of a pause: it ends the runs of
        preempted namespace requests and writes every settled block; `None`
        without an object key prefix."""
        if self._key_prefix is None:
            return None
        for state in self._namespaces.values():
            self._end_preempted(state)
        return self._take_metadata({}, {})

    def _take_metadata(
        self,
        scheduled_requests: dict[str, int],
        block_hashes_by_request: dict[str, PackedBlockHashes],
    ) -> AuxOutputConnectorMetadata:
        # Sending transfers ownership of these one-shot events.
        finished_requests = tuple(self._finished_requests)
        block_hashes_by_request.update(
            (request_id, block_hashes)
            for request_id, block_hashes in self._finished_requests.items()
            if block_hashes is not None
        )
        self._finished_requests = {}
        for state in self._namespaces.values():
            if state.running:
                request = state.request
                self._write_blocks(
                    state, request.num_computed_tokens - request.num_in_flight_tokens
                )
        metadata = AuxOutputConnectorMetadata(
            self._generation,
            scheduled_requests,
            block_hashes_by_request,
            finished_requests,
            self._pack_hashes(self._retained_hashes),
            self._pack_hashes(self._released_hashes),
            tuple(self._runs),
            self._writes,
        )
        self._retained_hashes, self._released_hashes = [], []
        self._runs, self._writes, self._listings = [], {}, {}
        return metadata

    def _namespace(self, request: Request) -> str | None:
        params = request.sampling_params
        if self._key_prefix is None or params is None:
            return None
        return params.aux_namespace

    def _run_namespace(self, request: Request, namespace: str) -> None:
        """Start a run of a namespace request that is scheduled: the worker
        keeps the rows it computes from here on. A preempted run ends first,
        and its objects are written from the rows its worker kept."""
        state = self._namespaces.get(request.request_id)
        if state is None:
            state = _NamespaceRequest(request, namespace, self._emit_start(request))
            self._namespaces[request.request_id] = state
        if state.running:
            return
        self._end_preempted(state)
        state.running = state.in_worker = True
        state.written = request.num_computed_tokens
        self._runs.append(request.request_id)

    def _end_preempted(self, state: _NamespaceRequest) -> None:
        """End the run of a preempted namespace request at the positions it
        computed before the last token it holds: its stale output has been
        delivered or dropped by now, and a dropped step computed no position
        before that token but prompt positions."""
        if state.preempted is not None:
            self._end_run(state, state.preempted, state.preempted)

    def _object_key(self, state: _NamespaceRequest, start: int) -> str:
        block_hash = state.request.block_hashes[
            (start + self._block_size) // self._hash_block_size - 1
        ]
        return f"{self._key_prefix}/{state.namespace}/{block_hash.hex()}"

    def _write_blocks(self, state: _NamespaceRequest, limit: int) -> None:
        """Write the blocks of the current run that end at `limit` or before,
        and before the last token: the row of that token's position is the
        one the next token is sampled from, and another run may sample it."""
        end = min(limit, state.request.num_tokens - 1)
        if state.valid is not None:
            end = min(end, state.valid)
        end -= end % self._block_size
        writes = [
            (self._object_key(state, start), start, start + self._block_size)
            for start in range(state.written, end, self._block_size)
        ]
        if writes:
            self._writes.setdefault(state.request.request_id, []).extend(writes)
            state.written = end

    def _end_run(self, state: _NamespaceRequest, end: int, limit: int) -> None:
        """End the current run: write its blocks up to `limit`, and list its
        positions up to `end` after those the listing covers, as blocks and a
        tail object for the positions after the last full block."""
        request = state.request
        end = min(end, request.num_tokens - 1)
        if state.valid is not None:
            end = min(end, state.valid)
        self._write_blocks(state, limit)
        listing = state.listing
        covered = listing[-1][2] if listing else state.prompt_start
        boundary = end - end % self._block_size
        listing.extend(
            (self._object_key(state, start), start, start + self._block_size)
            for start in range(
                covered - covered % self._block_size, boundary, self._block_size
            )
        )
        if end > max(boundary, covered):
            key = f"{self._key_prefix}/{state.namespace}/tail/{uuid.uuid4().hex}"
            listing.append((key, boundary, end))
            self._writes.setdefault(request.request_id, []).append((key, boundary, end))
        state.running = False
        state.preempted = state.valid = None

    def pop_aux_objects(self, request_id: str) -> list[AuxObject] | None:
        """The listing of a namespace request that finished since the last
        step, which its final output carries."""
        return self._listings.pop(request_id, None)

    @staticmethod
    def _emit_start(request: Request) -> int:
        if request.num_output_tokens:
            return request.num_tokens - 1
        assert request.sampling_params is not None
        # A P/D prefiller may cut the prompt short of the requested start.
        return min(request.sampling_params.aux_prompt_start, request.num_prompt_tokens)

    def max_cache_hit(self, request: Request) -> int:
        if self._skips(request):
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
        if self._namespace(request) is not None:
            aligned = (num_local + num_external) % self._block_size == 0
            return num_external if aligned else 0
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
            self._rowless = {
                block: block_hash
                for block, block_hash in self._rowless.items()
                if block in loading or self._gpu_cached(block_hash)
            }
        excess = len(self._kv_cached) - self._capacity
        if excess <= 0:
            return
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
        if self._skips(request):
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
        if self._namespace(request) is not None:
            self._finish_namespace(request)
            return
        if self._skips(request):
            return
        request_id = request.request_id
        self._loading.pop(request_id, None)
        if request_id in self._sent_hash_counts:
            # The next metadata delivers this terminal event to the worker.
            self._finished_requests[request_id] = self._take_new_hashes(
                request, request.num_computed_tokens
            )
            del self._sent_hash_counts[request_id]
        computed_from = self._computed_from.pop(request_id, request.num_tokens)
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

    def _finish_namespace(self, request: Request) -> None:
        """Record where a preempted namespace request stopped computing; its
        run ends when it is scheduled again, once its stale output is
        delivered or dropped. End the run of a finished one and keep its
        listing for its final output: the positions whose tokens it returned.
        """
        state = self._namespaces.get(request.request_id)
        if state is None:
            return
        num_computed = request.num_computed_tokens
        if not request.is_finished():
            state.running = False
            state.preempted = num_computed
            return
        del self._namespaces[request.request_id]
        if state.running:
            self._end_run(
                state, num_computed - request.num_in_flight_tokens, num_computed
            )
        self._end_preempted(state)
        self._listings[request.request_id] = state.listing
        if state.in_worker:
            self._finished_requests[request.request_id] = None

    def invalidate_kv(
        self, requests: Iterable[Request], invalid_block_ids: set[int]
    ) -> list[Request]:
        """Evict the blocks of the requests with rows a failed synchronous KV
        load affected from their first invalid block on, and keep the objects
        of a namespace request among them to the positions before it. Return
        those requests: they must compute again from there in a new run, as
        their rows only move forward and the store keeps the first copy."""
        affected = []
        for request in requests:
            params = request.sampling_params
            if params is not None and not params.return_aux_outputs:
                continue
            (block_ids,) = self._kv_cache_manager.get_block_ids(request.request_id)
            first = next(
                index
                for index, block_id in enumerate(block_ids)
                if block_id in invalid_block_ids
            )
            self._kv_cache_manager.evict_blocks(set(block_ids[first:]))
            if (state := self._namespaces.get(request.request_id)) is not None:
                state.valid = first * self._block_size
            affected.append(request)
        return affected

    def _skips(self, request: Request) -> bool:
        """Whether the request has no rows here: it skips its auxiliary
        outputs, or delivers them as objects."""
        params = request.sampling_params
        return params is not None and (
            not params.return_aux_outputs or self._namespace(request) is not None
        )

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
        """Start a new auxiliary output namespace after a prefix-cache reset.

        The worker drops its runs, so the runs of preempted namespace requests
        end now; their listings are kept.
        """
        for state in self._namespaces.values():
            self._end_preempted(state)
            state.in_worker = False
        # The worker drops temporary state on generation changes; resend hashes.
        self._sent_hash_counts.clear()
        self._computed_from.clear()
        self._finished_requests.clear()
        self._generation += 1
        self._released_hashes.extend(chain.from_iterable(self._kv_cached.values()))
        self._kv_cached.clear()
        self._rowless.clear()
