# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side control plane for execution auxiliary outputs."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from itertools import chain, islice
from typing import TYPE_CHECKING

import numpy as np

from vllm.distributed.aux_output_connector.session_objects import (
    aux_object_key,
    aux_segment_id,
    aux_tail_key,
    session_version_segments,
)
from vllm.exceptions import VLLMValidationError
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    get_routed_experts_bytes_per_token,
)
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import (
    generate_block_hash_extra_keys,
    hash_block_tokens,
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
    sessions: dict[str, str] = field(default_factory=dict)
    session_tails: dict[str, tuple[str, int, int]] = field(default_factory=dict)
    session_object_size: int = 0
    flushed_block_hashes: dict[str, PackedBlockHashes] = field(default_factory=dict)
    session_hash_starts: dict[str, int] = field(default_factory=dict)
    session_orphans: dict[str, list[str]] = field(default_factory=dict)


@dataclass(slots=True)
class AuxRequestOutput:
    token_start: int
    """Logical token offset of the first row in both payloads."""
    routed_rows: np.ndarray | None = None
    """Routed expert IDs per token, or None when disabled."""
    indexer_rows: np.ndarray | None = None
    """Raw TopK indices per source layer, or None when disabled."""

    def get_num_rows(self) -> int:
        """Return the common token-axis length of the present artifacts."""
        row_counts = {
            len(rows)
            for rows in (self.routed_rows, self.indexer_rows)
            if rows is not None
        }
        assert row_counts, "auxiliary output has no artifact rows"
        assert len(row_counts) == 1, "auxiliary output artifact row counts differ"
        return row_counts.pop()


@dataclass(slots=True)
class _SessionRequest:
    request: Request
    session: str


class AuxOutputSchedulerConnector:
    """Build worker metadata without owning auxiliary output stores."""

    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        num_objects: int,
        *,
        enable_routed_experts: bool = True,
        enable_indexer: bool = False,
        key_prefix: str | None = None,
        hash_algo: str = "sha256",
    ) -> None:
        self._output_fields = tuple(
            (rows, prompt_start)
            for enabled, rows, prompt_start in (
                (enable_routed_experts, "routed_rows", "routed_experts_prompt_start"),
                (enable_indexer, "indexer_rows", "indexer_topk_prompt_start"),
            )
            if enabled
        )
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
        # KV blocks whose rows are not all here: loaded without rows, or
        # computed from mid-block on. Any other cached GPU block has rows.
        self._rowless: dict[ExternalBlockHash, BlockHash] = {}
        # First token each active request computes here.
        self._computed_from: dict[str, int] = {}
        self._capacity = num_objects // (self._block_size // self._hash_block_size)
        self._retained_hashes: list[bytes] = []
        self._released_hashes: list[bytes] = []
        if key_prefix is not None:
            from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

            if (
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
        self._hash_fn = get_hash_fn_by_name(hash_algo)
        self._weight_version = "default"
        self._session_requests: dict[str, _SessionRequest] = {}
        self._session_tails: dict[str, tuple[str, int, int]] = {}
        self._flushed_hashes: dict[str, PackedBlockHashes] = {}
        self._written: dict[str, list[BlockHash]] = {}
        self._listed: dict[str, set[str]] = {}
        self._orphans: dict[str, list[str]] = {}

    def _segment_id(self, request: Request) -> str:
        return aux_segment_id(request.request_id, self._weight_version)

    def _session(self, request: Request) -> str | None:
        params = request.sampling_params
        if self._key_prefix is None or params is None:
            return None
        return params.aux_session

    def _untracked(self, request: Request) -> bool:
        """Whether a request has no rows: no inline output and no session."""
        return not self._output_fields and self._session(request) is None

    def add_request(self, request: Request) -> None:
        """Salt a session request's positions before any KV connector sees its
        block hashes: with its `aux_version_segments` up to the last full hash
        block of the prompt they cover, then with its own segment id."""
        session = self._session(request)
        if session is None:
            return
        self._session_requests[request.request_id] = _SessionRequest(request, session)
        assert request.sampling_params is not None
        end, segments = session_version_segments(
            request.sampling_params.aux_version_segments,
            request.num_prompt_tokens,
            self._block_size,
            self._segment_id(request),
        )
        if segments != request.cache_salt_segments:
            request.cache_salt_segments = segments
            del request.block_hashes[end // self._hash_block_size :]
            request.update_block_hashes()

    def set_weight_version(self, weight_version: str) -> None:
        """Set the weight version of later steps; call while no step runs.
        A running session request names the positions it has not computed yet
        with it; a waiting one from its first unhashed position, and from its
        hit when it is scheduled."""
        if self._key_prefix is not None and not (
            isinstance(weight_version, str)
            and weight_version
            and "@" not in weight_version
        ):
            raise VLLMValidationError(
                "weight version must be a non-empty string without '@'",
                parameter="weight_version",
            )
        self._weight_version = weight_version
        for state in self._session_requests.values():
            request = state.request
            computed = request.num_computed_tokens
            running = request.status == RequestStatus.RUNNING
            if running:
                del request.block_hashes[computed // self._hash_block_size :]
            request.add_cache_salt_segment(
                computed, self._segment_id(request), self._hash_block_size
            )
            if running:
                request.update_block_hashes()

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
        requests: dict[str, Request],
    ) -> AuxOutputConnectorMetadata:
        """Build one step's incremental worker metadata."""
        scheduled_requests: dict[str, int] = {}
        block_hashes_by_request: dict[str, PackedBlockHashes] = {}
        sessions: dict[str, str] = {}
        hash_starts: dict[str, int] = {}
        for request_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            request = requests[request_id]
            if self._untracked(request):
                continue
            state = self._session_requests.get(request_id)
            if request_id not in self._sent_hash_counts:
                self._sent_hash_counts[request_id] = 0
                if state is not None:
                    sessions[request_id] = state.session
                    first = request.num_computed_tokens // self._hash_block_size
                    self._sent_hash_counts[request_id] = hash_starts[request_id] = first
            self._computed_from.setdefault(request_id, request.num_computed_tokens)
            packed = self._take_new_hashes(
                request, request.num_computed_tokens + num_tokens
            )
            if packed is not None:
                block_hashes_by_request[request_id] = packed
            self._loading.pop(request_id, None)
            scheduled_requests[request_id] = (
                0 if state is not None else self._emit_start(request)
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
        metadata = self._take_events(
            scheduled_requests, block_hashes_by_request, sessions, hash_starts
        )
        for request_id in scheduler_output.finished_req_ids or ():
            self._session_requests.pop(request_id, None)
        return metadata

    def drain_metadata(self) -> AuxOutputConnectorMetadata | None:
        if self._key_prefix is None:
            return None
        return self._take_events({}, {}, {}, {})

    def finishing(self, request_ids: set[str]) -> bool:
        return not self._finished_requests.keys().isdisjoint(request_ids)

    def _take_events(
        self,
        scheduled_requests: dict[str, int],
        block_hashes_by_request: dict[str, PackedBlockHashes],
        sessions: dict[str, str],
        hash_starts: dict[str, int],
    ) -> AuxOutputConnectorMetadata:
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
            sessions,
            self._session_tails,
            self._block_size,
            self._flushed_hashes,
            hash_starts,
        )
        self._flushed_hashes = {}
        metadata.session_orphans, self._orphans = self._orphans, {}
        self._retained_hashes, self._released_hashes = [], []
        self._session_tails = {}
        return metadata

    def _emit_start(self, request: Request) -> int:
        if request.num_output_tokens:
            return request.num_tokens - 1
        assert request.sampling_params is not None
        prompt_start: int = min(
            getattr(request.sampling_params, field) for _, field in self._output_fields
        )
        # A P/D prefiller may cut the prompt short of the requested start.
        return min(prompt_start, request.num_prompt_tokens)

    def max_cache_hit(self, request: Request) -> int:
        if self._session(request) is not None:
            return (request.num_tokens - 1) // self._block_size * self._block_size
        if self._untracked(request):
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

    def salt_from_hit(self, request: Request, num_hit: int) -> None:
        if self._session(request) is None:
            return
        segment_id = self._segment_id(request)
        segments = request.cache_salt_segments
        ends = [start for start, _ in segments[1:]]
        start = next(
            (
                max(begin, num_hit)
                for (begin, salt), end in zip(segments, [*ends, request.num_tokens])
                if salt != segment_id and end > num_hit
            ),
            None,
        )
        if start is not None:
            hash_size = self._hash_block_size
            del request.block_hashes[start // hash_size :]
            request.add_cache_salt_segment(start, segment_id, hash_size)
            request.update_block_hashes()

    def cap_external_hit(
        self, request: Request, num_local: int, num_external: int
    ) -> int:
        if self._untracked(request):
            return num_external
        num_hit = num_local + num_external
        if self._session(request) is None and num_hit <= self._emit_start(request):
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
        if self._untracked(request):
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
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return accepted artifacts with independent prompt-prefix offsets."""
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
        num_rows = request_output.get_num_rows()
        token_end = request.num_tokens - 1
        local_end = token_end - request_output.token_start
        if local_end < 0:
            assert not request.is_finished(), (
                "finished auxiliary output output has no accepted token range: "
                f"request={request_id}, token_end={token_end}, "
                f"output_start={request_output.token_start}, "
                f"output_end={request_output.token_start + num_rows}"
            )
            return None, None
        assert local_end <= num_rows, (
            "auxiliary output worker output has an invalid token range: "
            f"request={request_id}, token_end={token_end}, "
            f"output_start={request_output.token_start}, "
            f"output_end={request_output.token_start + num_rows}"
        )
        assert request.sampling_params is not None
        accepted: dict[str, np.ndarray] = {}
        for row_field, prompt_start_field in self._output_fields:
            rows: np.ndarray | None = getattr(request_output, row_field)
            assert rows is not None, f"auxiliary output is missing {row_field}"
            start = max(
                0,
                min(
                    getattr(request.sampling_params, prompt_start_field),
                    request.num_prompt_tokens,
                )
                - request_output.token_start,
            )
            accepted[row_field] = rows[start:local_end]
        return accepted.get("routed_rows"), accepted.get("indexer_rows")

    def take_rows_and_keys(
        self,
        request: Request,
        output: dict[str, AuxRequestOutput] | None,
        stopped: bool,
    ) -> tuple[
        np.ndarray | None,
        np.ndarray | None,
        list[str] | None,
        list[tuple[int, str | None]] | None,
    ]:
        """Rows of one output, or for a stopped session request the keys of
        the objects that hold its rows and the version segments of their
        names."""
        if self._untracked(request):
            return None, None, None, None
        state = self._session_requests.get(request.request_id)
        if state is None:
            return *self.take_output(request, output), None, None
        if not stopped or request.request_id not in self._sent_hash_counts:
            return None, None, None, None
        size = self._block_size
        hash_size = self._hash_block_size
        token_end = request.num_tokens - 1
        block_hashes = request.block_hashes
        computed = request.num_computed_tokens - request.num_in_flight_tokens
        num_sent = self._sent_hash_counts[request.request_id] * hash_size
        end = min(num_sent, computed) // size * size
        boundary = min(token_end, end)
        step = size // hash_size
        keys = [
            aux_object_key(self._key_prefix, state.session, block_hashes[last].hex())
            for last in range(step - 1, boundary // hash_size, step)
        ]
        if boundary < token_end:
            parent = block_hashes[boundary // hash_size - 1] if boundary else None
            extra_keys, _ = generate_block_hash_extra_keys(
                request, boundary, token_end, 0
            )
            tail_hash = hash_block_tokens(
                self._hash_fn,
                parent,
                request.all_token_ids[boundary:token_end],
                ("aux_tail", extra_keys),
            )
            key = aux_tail_key(self._key_prefix, state.session, tail_hash.hex())
            self._session_tails[request.request_id] = (key, boundary, token_end)
            keys.append(key)
        segments: list[tuple[int, str | None]] = [
            segment for segment in request.cache_salt_segments if segment[0] < end
        ]
        segments.append((end, None))
        self._listed[request.request_id] = set(keys)
        return None, None, keys, segments

    def request_finished(self, request: Request) -> None:
        """Queue a request's terminal event and final block hashes."""
        if self._untracked(request):
            return
        request_id = request.request_id
        self._loading.pop(request_id, None)
        session = request_id in self._session_requests
        num_computed = request.num_computed_tokens
        if session:
            num_computed -= request.num_in_flight_tokens
        if request_id in self._sent_hash_counts:
            # The next metadata delivers this terminal event to the worker.
            self._finished_requests[request_id] = self._take_new_hashes(
                request, num_computed
            )
            del self._sent_hash_counts[request_id]
        if session and request.is_finished():
            self._release_unlisted(request)
        computed_from = self._computed_from.pop(request_id, request.num_tokens)
        num_blocks = request.num_computed_tokens // self._block_size
        if self._session(request) is not None:
            num_blocks = 0
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
        if request.request_id in self._session_requests:
            self._record_written(request, num_sent, num_keyed)
        return self._pack_hashes(block_hashes[num_sent:num_keyed])

    def _record_written(self, request: Request, first: int, end: int) -> None:
        step = self._block_size // self._hash_block_size
        self._written.setdefault(request.request_id, []).extend(
            request.block_hashes[index]
            for index in range(first, end)
            if (index + 1) % step == 0
        )

    def _release_unlisted(self, request: Request) -> None:
        listed = self._listed.pop(request.request_id, set())
        session = self._session_requests[request.request_id].session
        keys = (
            aux_object_key(self._key_prefix, session, block_hash.hex())
            for block_hash in self._written.pop(request.request_id, ())
        )
        orphans = [key for key in dict.fromkeys(keys) if key not in listed]
        if orphans:
            self._orphans[request.request_id] = orphans

    def reset(self) -> None:
        """Start a new auxiliary output namespace after a prefix-cache reset."""
        # The worker drops temporary state on generation changes; resend hashes.
        self._sent_hash_counts.clear()
        self._computed_from.clear()
        finished = {}
        for request_id, block_hashes in self._finished_requests.items():
            state = self._session_requests.get(request_id)
            if state is None:
                continue
            if state.request.is_finished():
                finished[request_id] = block_hashes
            elif block_hashes is not None:
                self._flushed_hashes[request_id] = block_hashes
        self._finished_requests = finished
        self._generation += 1
        self._released_hashes.extend(chain.from_iterable(self._kv_cached.values()))
        self._kv_cached.clear()
        self._rowless.clear()
