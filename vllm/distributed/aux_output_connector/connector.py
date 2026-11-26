# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side control plane for execution auxiliary outputs."""

from __future__ import annotations

import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import chain, islice
from typing import TYPE_CHECKING, Any

import numpy as np

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    get_routed_experts_bytes_per_token,
)
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.core.kv_cache_utils import (
    maybe_convert_block_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_events import ExternalBlockHash
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


def flush_aux_output(scheduler: object, collective_rpc: Callable[..., Any]) -> None:
    """Queue the object writes due when the engine pauses; the pause does not
    wait for them."""
    connector = getattr(scheduler, "aux_output_connector", None)
    if connector is not None and (metadata := connector.flush_metadata()) is not None:
        collective_rpc("flush_aux_output", args=(metadata,))


def finished_aux_objects(
    requests: Iterable[Request],
) -> dict[str, list[tuple[str, int, int]]] | None:
    """Object listings of the finished requests that have one, for their final
    outputs."""
    listing = {
        request.request_id: request.aux_objects
        for request in requests
        if request.aux_objects is not None
    }
    return listing or None


def object_put_timeout_s() -> float:
    """Seconds an object write may take from being queued: 80% of the
    execute-model timeout, so that a step waiting on a full queue returns in
    time."""
    return 0.8 * envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS


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
    # Namespace requests whose worker keeps the rows they compute from now
    # on, in a new run; and the objects to write from them.
    namespace_runs: list[str] | None = None
    object_writes: dict[str, list[AuxObject]] | None = None
    # Positions per object.
    object_block_size: int = 0


@dataclass(slots=True)
class AuxRequestOutput:
    token_start: int
    rows: np.ndarray


@dataclass(slots=True)
class _NamespaceRequest:
    request: Request
    namespace: str
    prompt_start: int
    # Start of its current run, end of the spans its runs wrote, and
    # positions computed when it was last preempted.
    start: int = 0
    written: int = 0
    computed: int = 0
    running: bool = False
    in_worker: bool = False


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
        coordinator = kv_cache_manager.coordinator
        self._group_id = getattr(coordinator, "full_attention_group_id", None) or 0
        self._manager = coordinator.single_type_managers[self._group_id]
        self._block_size = self._manager.block_size
        self._hash_block_size = kv_cache_manager.block_pool.hash_block_size
        # Positions per object: every prefix-cache hit ends on an object
        # boundary.
        self._object_size = (
            self._hash_block_size
            if key_prefix is not None and coordinator.enable_partial_hash_hits
            else self._block_size
        )
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
        if key_prefix is not None and not kv_cache_manager.enable_caching:
            raise ValueError("AuxOutputConfig.object_key_prefix needs prefix caching.")
        if key_prefix is not None and any(
            group.prefix_replay_tokens
            for group in kv_cache_manager.kv_cache_config.kv_cache_groups
        ):
            raise ValueError(
                "AuxOutputConfig.object_key_prefix does not support prefix "
                "replay: pass --no-swa-bounded-replay."
            )
        self._key_prefix = key_prefix
        self._namespaces: dict[str, _NamespaceRequest] = {}
        self._runs: list[str] = []
        self._writes: dict[str, list[AuxObject]] = {}

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

    def flush_metadata(self) -> AuxOutputConnectorMetadata | None:
        """Metadata that hands the worker the writes due when the engine
        pauses, which no step would send before it resumes. Nothing waits
        for them. `None` without an object key prefix."""
        if self._key_prefix is None:
            return None
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
        size = self._object_size
        for state in self._namespaces.values():
            # Only settled steps publish; most steps complete no span.
            if (
                state.running
                and (settled := self._settled(state.request)) >= state.written + size
            ):
                self._write_blocks(state, settled)
        metadata = AuxOutputConnectorMetadata(
            self._generation,
            scheduled_requests,
            block_hashes_by_request,
            finished_requests,
            self._pack_hashes(self._retained_hashes),
            self._pack_hashes(self._released_hashes),
            self._runs or None,
            self._writes or None,
            self._object_size,
        )
        self._retained_hashes, self._released_hashes = [], []
        self._runs, self._writes = [], {}
        return metadata

    def _namespace(self, request: Request) -> str | None:
        """The namespace of a request that delivers its rows as objects."""
        params = request.sampling_params
        if self._key_prefix is None or params is None:
            return None
        return params.aux_namespace

    def _run_namespace(self, request: Request, namespace: str) -> None:
        """Start a run of a scheduled namespace request, in which the worker
        keeps the rows it computes. A preempted request runs again from its
        prefix-cache hit, a span boundary: the positions after it are
        computed again, and their rows replace the earlier ones, as for inline
        rows. The blocks before it were written at the preemption. A hit of
        the whole prompt computes its last position again; the run starts at
        the end of the hit, so that row is neither written nor listed."""
        state = self._namespaces.get(request.request_id)
        if state is None:
            state = _NamespaceRequest(request, namespace, self._emit_start(request))
            self._namespaces[request.request_id] = state
        if state.running:
            return
        state.running = state.in_worker = True
        state.start = state.written = round_up(
            request.num_computed_tokens, self._object_size
        )
        self._runs.append(request.request_id)

    def _object_key(self, state: _NamespaceRequest, start: int) -> str:
        block_hash = state.request.block_hashes[
            (start + self._object_size) // self._hash_block_size - 1
        ]
        return f"{self._key_prefix}/{state.namespace}/{block_hash.hex()}"

    def _write_blocks(self, state: _NamespaceRequest, limit: int) -> None:
        """Write the full spans of the current run that end at `limit` or
        before and are hashed, whatever any KV cache group keeps: every span
        another request can hit was computed by a run that writes it."""
        request, size = state.request, self._object_size
        end = min(limit, len(request.block_hashes) * self._hash_block_size)
        writes = [
            (self._object_key(state, start), start, start + size)
            for start in range(state.written, end - end % size, size)
        ]
        if writes:
            self._writes.setdefault(request.request_id, []).extend(writes)
            state.written = writes[-1][2]

    def _end_run(
        self, state: _NamespaceRequest, end: int, limit: int, tail: bool = True
    ) -> list[AuxObject]:
        """End the run of a finished request: write its blocks up to `limit`,
        and list its positions from the block holding its prompt start up to
        `end`. A listing names only objects that are written: the blocks the
        run wrote, one object per span, and one tail object, written from the
        run's rows, for the positions after them. A run that holds no rows,
        or none before `end`, lists its written blocks up to `end` and no
        tail."""
        request = state.request
        end = min(end, request.num_tokens - 1)
        self._write_blocks(state, limit)
        size = self._object_size
        first = state.prompt_start - state.prompt_start % size
        written = min(state.written, end)
        boundary = max(written - written % size, first)
        listing = [
            (self._object_key(state, start), start, start + size)
            for start in range(first, boundary, size)
        ]
        if tail and end > max(boundary, state.prompt_start) and boundary >= state.start:
            key = f"{self._key_prefix}/{state.namespace}/tail-{uuid.uuid4().hex}"
            listing.append((key, boundary, end))
            self._writes.setdefault(request.request_id, []).append((key, boundary, end))
        return listing

    @staticmethod
    def _settled(request: Request) -> int:
        """Positions the current run computed whose step outputs were
        processed. Stale output still in flight belongs to the run a
        preemption ended, not to this one."""
        in_flight = request.num_in_flight_tokens - request.num_stale_output_tokens
        return request.num_computed_tokens - in_flight

    @staticmethod
    def _emit_start(request: Request) -> int:
        if request.num_output_tokens:
            return request.num_tokens - 1
        params = request.sampling_params
        assert params is not None
        start = params.aux_prompt_start or params.routed_experts_prompt_start
        # A P/D prefiller may cut the prompt short of the requested start.
        return min(start, request.num_prompt_tokens)

    def max_cache_hit(self, request: Request) -> int:
        if self._namespace(request) is not None:
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
        num_hit = num_local + num_external
        if self._namespace(request) is not None:
            hashed = len(request.block_hashes) * self._hash_block_size
            usable = num_hit % self._object_size == 0 and num_hit <= hashed
            return num_external if usable else 0
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
        if self._namespace(request) is not None:
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
        if self._namespace(request) is not None:
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
        if self._namespace(request) is not None:
            self._finish_namespace(request)
            return
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

    def loads_failed(self, request_ids: set[str]) -> None:
        """Requests about to end with an error because KV they hold failed to
        load: what their steps in flight compute is not written."""
        for request_id in request_ids:
            if (state := self._namespaces.get(request_id)) is not None:
                state.running = False

    def _finish_namespace(self, request: Request) -> None:
        """Write the full blocks a preempted namespace request computed, which
        the prefix cache keeps; it runs again from its prefix-cache hit.
        End the run of a finished one and keep its listing for its final
        output: the positions whose tokens it returned. One that ended with an
        error lists nothing and writes what it computed, which other requests
        may hit, unless KV it held failed to load (`loads_failed`).
        """
        request_id = request.request_id
        state = self._namespaces.get(request_id)
        if state is None:
            return
        num_computed = request.num_computed_tokens
        if not request.is_finished():
            self._write_blocks(state, num_computed)
            state.running = False
            state.computed = num_computed
            return
        del self._namespaces[request_id]
        if state.in_worker:
            self._finished_requests[request_id] = None
        if request.status == RequestStatus.FINISHED_ERROR:
            if state.running:
                self._write_blocks(state, num_computed)
            return
        tail = True
        if state.running and self._settled(request) >= state.written:
            end, limit = self._settled(request), num_computed
        elif state.running or not state.in_worker:
            # Its run holds no rows to list: a recompute of a failed load
            # took it back below the blocks it wrote, or a prefix-cache reset
            # dropped the run. It lists its written blocks.
            end = limit = state.written
            tail = False
        else:
            # Preempted: its run still holds the rows it computed, and rows
            # computed over a draft it did not keep end before its last token.
            end = state.computed
            limit = min(state.computed, request.num_tokens - 1)
        listing = self._end_run(state, end, limit, tail)
        aborted = request.status == RequestStatus.FINISHED_ABORTED
        request.aux_objects = None if aborted and not listing else listing

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

        The worker drops its runs: a preempted namespace request starts a new
        one when it runs again.
        """
        for state in self._namespaces.values():
            state.in_worker = False
        # The worker drops temporary state on generation changes; resend hashes.
        self._sent_hash_counts.clear()
        self._computed_from.clear()
        self._finished_requests.clear()
        self._generation += 1
        self._released_hashes.extend(chain.from_iterable(self._kv_cached.values()))
        self._kv_cached.clear()
        self._rowless.clear()
