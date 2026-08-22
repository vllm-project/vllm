# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-describing KV cache events for the offloading connector.

The OffloadingManager identifies an offloaded chunk only by its OffloadKey,
so its raw events carry no token ids, parent hash, or block size.
:class:`OffloadingEventsTracker` keeps a request-scoped key locator on
``ReqContext`` and builds the ``BlockStored`` payload only when a store event
arrives. Store jobs keep that context through completion; only a minimal
detached hash record survives for the legacy removal contract. A chunk event
may carry multiple constituent per-block hashes, and evictions fan out to the
same hashes. Chunks overlapping a non-chunk-aligned shared prefix re-announce
the shared hashes once per chunk; consumers are expected to deduplicate
(reference-count) repeated store/remove announcements of the same hash.
Opt-in via ``kv_connector_extra_config["self_describing_kv_events"]``;
inert unless KV cache events are enabled. See the PR description for the
full design.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

from vllm.distributed.kv_events import (
    MEDIUM_CPU,
    MEDIUM_STORAGE,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_block_hash_extra_keys,
    maybe_convert_block_hash,
    resolve_block_hashes,
)
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    get_kv_cache_spec_kind,
    get_kv_cache_spec_sliding_window,
)
from vllm.v1.kv_offload.base import (
    Medium,
    OffloadingEvent,
    OffloadingKVEventsConfig,
    OffloadKey,
    ReqContext,
    get_offload_block_hash,
    get_offload_group_idx,
    make_offload_key,
)
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        GroupOffloadConfig,
    )

logger = init_logger(__name__)

_MEDIUM_TO_EVENT_STR: dict[Medium, str] = {
    Medium.CPU: MEDIUM_CPU,
    Medium.STORAGE: MEDIUM_STORAGE,
}


class OffloadingEventGroupSpec(NamedTuple):
    kv_cache_spec_kind: str | None
    kv_cache_spec_sliding_window: int | None


def get_offloading_event_group_spec(
    kv_cache_group: KVCacheGroupSpec,
) -> OffloadingEventGroupSpec:
    kv_cache_spec = kv_cache_group.kv_cache_spec
    return OffloadingEventGroupSpec(
        kv_cache_spec_kind=get_kv_cache_spec_kind(kv_cache_spec).value,
        kv_cache_spec_sliding_window=get_kv_cache_spec_sliding_window(kv_cache_spec),
    )


@dataclass(frozen=True, slots=True)
class _OffloadEventMetadata:
    """Immutable ``BlockStored`` payload snapshot for one offload key."""

    # The chunk's constituent block hashes; the last one is the OffloadKey.
    block_hashes: tuple[BlockHash, ...]
    parent_block_hash: BlockHash | None
    token_ids: tuple[int, ...]
    block_size: int
    lora_id: int | None
    lora_name: str | None
    extra_keys: tuple[tuple[Any, ...] | None, ...] | None
    group_idx: int
    kv_cache_spec: OffloadingEventGroupSpec


@dataclass(slots=True)
class _RequestEventContext:
    """Lazy event locators owned by one request."""

    request: Request
    group_configs: dict[int, "GroupOffloadConfig"]
    supports_partial_tail: bool
    indexed_hash_count: int = 0
    locators: dict[OffloadKey, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _RemovalMetadata:
    """Minimal detached record retained for the legacy removal contract."""

    block_hashes: tuple[BlockHash, ...]
    group_idx: int


class OffloadingEventsTracker:
    """Translates offload events using request-scoped key locators."""

    def __init__(self, config: OffloadingKVEventsConfig):
        self.config = config
        self.self_describing_enabled = (
            config.enable_kv_cache_events and config.self_describing_kv_events
        )

        # PR 1 preserves the existing self-describing removal behavior. Keep
        # only the hashes needed for that compatibility path, detached from
        # the request-scoped stored-event payload.
        self._removal_metadata: dict[tuple[Medium, OffloadKey], _RemovalMetadata] = {}

    def on_new_request(
        self,
        req_context: ReqContext,
        request: Request,
        group_configs: tuple["GroupOffloadConfig", ...],
        supports_partial_tail: bool = False,
    ) -> None:
        """Attach a lazy event context to one request."""
        if not self.self_describing_enabled:
            return
        req_context.set_state(
            _RequestEventContext(
                request=request,
                group_configs={config.group_idx: config for config in group_configs},
                supports_partial_tail=supports_partial_tail,
            )
        )

    @staticmethod
    def _request_event_context(
        req_context: ReqContext | None,
    ) -> _RequestEventContext | None:
        if req_context is None:
            return None
        return req_context.get_state(_RequestEventContext)

    @staticmethod
    def _extend_locators(state: _RequestEventContext) -> None:
        request = state.request
        if len(request.block_hashes) < state.indexed_hash_count:
            state.indexed_hash_count = 0
            state.locators.clear()
        for hash_idx in range(state.indexed_hash_count, len(request.block_hashes)):
            block_hash = request.block_hashes[hash_idx]
            for group_config in state.group_configs.values():
                if group_config.sliding_window_size_in_chunks is not None:
                    continue
                hash_boundary = hash_idx + 1
                if (
                    hash_boundary % group_config.hashes_per_chunk != 0
                    and not state.supports_partial_tail
                ):
                    continue
                tokens_per_hash = group_config.tokens_per_chunk // (
                    group_config.hashes_per_chunk
                )
                key = make_offload_key(block_hash, group_config.group_idx)
                state.locators.setdefault(key, hash_boundary * tokens_per_hash)
        state.indexed_hash_count = len(request.block_hashes)

    def _locator_for(
        self,
        state: _RequestEventContext,
        offload_key: OffloadKey,
    ) -> tuple["GroupOffloadConfig", int] | None:
        self._extend_locators(state)
        boundary_tokens = state.locators.get(offload_key)
        group_config = state.group_configs.get(get_offload_group_idx(offload_key))
        if boundary_tokens is None or group_config is None:
            return None
        return group_config, boundary_tokens

    def _metadata_for(
        self,
        state: _RequestEventContext,
        offload_key: OffloadKey,
    ) -> _OffloadEventMetadata | None:
        if not self._request_is_event_safe(state.request):
            return None
        locator = self._locator_for(state, offload_key)
        if locator is None:
            return None
        group_config, boundary_tokens = locator

        if boundary_tokens % group_config.tokens_per_chunk == 0:
            chunk_idx = boundary_tokens // group_config.tokens_per_chunk - 1
            return self._build_event_metadata(state.request, group_config, chunk_idx)
        return self._build_partial_event_metadata(
            state.request, group_config, boundary_tokens
        )

    @staticmethod
    def _request_is_event_safe(request: Request) -> bool:
        # Resumable sessions can replace a sampled token after its block hash
        # was appended. NIXL/Mooncake Mamba prefill can similarly truncate a
        # request when combined with this connector. Until those paths roll
        # back the hashes and offload keys, do not pair stale hashes with the
        # mutated token list in a self-describing event.
        if getattr(request, "resumable", False) is True:
            return False
        params = request.kv_transfer_params
        return not (isinstance(params, dict) and params.get("_p_side_truncated"))

    def record_hit(self, req_context: ReqContext, offload_key: OffloadKey) -> None:
        """Backfill the detached removal record for a primary-tier hit."""
        if not self.self_describing_enabled:
            return
        removal_key = (Medium.CPU, offload_key)
        if removal_key in self._removal_metadata:
            return
        state = self._request_event_context(req_context)
        if state is not None and not self._request_is_event_safe(state.request):
            return
        locator = self._locator_for(state, offload_key) if state is not None else None
        if state is None or locator is None:
            return
        group_config, boundary_tokens = locator
        block_hashes = self._block_hashes_for(
            state.request,
            group_config,
            boundary_tokens,
        )
        self._removal_metadata[removal_key] = _RemovalMetadata(
            block_hashes=block_hashes,
            group_idx=group_config.group_idx,
        )

    @staticmethod
    def _block_hashes_for(
        req: Request,
        group_config: "GroupOffloadConfig",
        boundary_tokens: int,
    ) -> tuple[BlockHash, ...]:
        tokens_per_hash = group_config.tokens_per_chunk // (
            group_config.hashes_per_chunk
        )
        last_hash_idx = boundary_tokens // tokens_per_hash
        if boundary_tokens % group_config.tokens_per_chunk == 0:
            first_hash_idx = last_hash_idx - group_config.hashes_per_chunk
            block_hashes = resolve_block_hashes(
                req.block_hashes[first_hash_idx:last_hash_idx],
                tokens_per_hash,
                group_config.tokens_per_block,
            )
        else:
            chunk_start = (
                (boundary_tokens - 1) // group_config.tokens_per_chunk
            ) * group_config.tokens_per_chunk
            first_hash_idx = chunk_start // tokens_per_hash
            block_hashes = req.block_hashes[first_hash_idx:last_hash_idx]
        assert first_hash_idx >= 0
        assert last_hash_idx <= len(req.block_hashes)
        resolved_hashes = tuple(
            block_hash for block_hash in block_hashes if block_hash is not None
        )
        assert resolved_hashes and len(resolved_hashes) == len(block_hashes)
        return resolved_hashes

    def _build_partial_event_metadata(
        self,
        req: Request,
        group_config: "GroupOffloadConfig",
        boundary_tokens: int,
    ) -> _OffloadEventMetadata:
        """Build metadata for the valid prefix of one physical cache block.

        A partial recurrent tail ends on a hash boundary but before its
        physical cache block is full. The event describes only the valid
        hashes and tokens, not the unused remainder of that physical block.
        """
        tokens_per_hash = group_config.tokens_per_chunk // group_config.hashes_per_chunk
        # Subtract one so the boundary token itself cannot select the next
        chunk_start = (
            (boundary_tokens - 1) // group_config.tokens_per_chunk
        ) * group_config.tokens_per_chunk
        first_hash_idx = chunk_start // tokens_per_hash
        last_hash_idx = boundary_tokens // tokens_per_hash
        assert chunk_start < boundary_tokens
        assert boundary_tokens % tokens_per_hash == 0
        assert last_hash_idx <= len(req.block_hashes)

        # Unlike a complete chunk, a partial tail contains only the hashes
        # between its physical block start and its valid token boundary.
        maybe_block_hashes = req.block_hashes[first_hash_idx:last_hash_idx]
        block_hashes = tuple(
            block_hash for block_hash in maybe_block_hashes if block_hash is not None
        )
        assert block_hashes and len(block_hashes) == len(maybe_block_hashes)
        parent_block_hash = (
            req.block_hashes[first_hash_idx - 1] if first_hash_idx > 0 else None
        )
        assert first_hash_idx == 0 or parent_block_hash is not None

        lora_id = req.lora_request.adapter_id if req.lora_request is not None else None
        lora_name = req.lora_request.name if req.lora_request is not None else None
        extra_keys = self._build_extra_keys(
            req,
            chunk_start,
            boundary_tokens,
            tokens_per_hash,
        )
        return _OffloadEventMetadata(
            block_hashes=block_hashes,
            parent_block_hash=parent_block_hash,
            token_ids=tuple(req.all_token_ids[chunk_start:boundary_tokens]),
            block_size=tokens_per_hash,
            lora_id=lora_id,
            lora_name=lora_name,
            extra_keys=extra_keys,
            group_idx=group_config.group_idx,
            kv_cache_spec=group_config.kv_event_group_spec,
        )

    def take_events(self, events: Iterable[OffloadingEvent]) -> Iterable[KVCacheEvent]:
        """Translate raw OffloadingEvents into self-describing KV events.

        Complete metadata is available only for full-attention groups when
        the tracker is enabled. Other shapes retain the legacy placeholder
        payload so consumers can ignore them.

        Yields:
            ``BlockStored`` or ``BlockRemoved`` events corresponding to
            the underlying :class:`OffloadingEvent` stream.
        """
        for event in events:
            if event.removed:
                yield from self._take_removed_event(event)
            else:
                yield from self._take_stored_event(event)

    def reset(self) -> None:
        """Drop detached CPU removal records after a cache reset."""
        self._removal_metadata.clear()

    @staticmethod
    def _build_extra_keys(
        req: Request,
        start_token_idx: int,
        end_token_idx: int,
        block_size: int,
    ) -> tuple[tuple[Any, ...] | None, ...]:
        """Match GPU event extra-key generation at event block granularity."""
        assert start_token_idx % block_size == 0
        assert end_token_idx % block_size == 0
        curr_mm_idx = 0
        extra_keys: list[tuple[Any, ...] | None] = []
        for block_start in range(start_token_idx, end_token_idx, block_size):
            block_extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                req,
                block_start,
                block_start + block_size,
                curr_mm_idx,
            )
            extra_keys.append(block_extra_keys)
        return tuple(extra_keys)

    def _build_event_metadata(
        self,
        req: Request,
        group_config: "GroupOffloadConfig",
        chunk_idx: int,
    ) -> _OffloadEventMetadata:
        """Build the payload snapshot for one offloaded chunk: its
        constituent per-block hashes, the whole chunk's tokens, and the
        per-block ``block_size``."""
        hashes_per_chunk = group_config.hashes_per_chunk
        assert hashes_per_chunk > 0
        assert chunk_idx >= 0
        tokens_per_hash = group_config.tokens_per_chunk // hashes_per_chunk
        # Each chunk's final raw hash is its OffloadKey. Resolve the raw hash
        # chain to the owning KV group's block granularity, matching GPU events.
        first_hash_idx = chunk_idx * hashes_per_chunk
        last_hash_idx = first_hash_idx + hashes_per_chunk
        assert first_hash_idx >= 0
        assert last_hash_idx <= len(req.block_hashes)
        raw_chunk_hashes = req.block_hashes[first_hash_idx:last_hash_idx]
        chunk_hashes = resolve_block_hashes(
            raw_chunk_hashes,
            tokens_per_hash,
            group_config.tokens_per_block,
        )
        for block_hash in chunk_hashes:
            assert block_hash is not None
        assert len(chunk_hashes) == (
            group_config.tokens_per_chunk // group_config.tokens_per_block
        )

        if group_config.sliding_window_size_in_chunks is not None:
            # The recording methods filter these out before calling this helper.
            raise AssertionError("self-describing events only support full attention")

        parent_block_hash: BlockHash | None
        if first_hash_idx == 0:
            parent_block_hash = None
        else:
            parent_block_hash = req.block_hashes[first_hash_idx - 1]
            assert parent_block_hash is not None

        tok_start = chunk_idx * group_config.tokens_per_chunk
        tok_end = tok_start + group_config.tokens_per_chunk
        assert tok_end <= len(req.all_token_ids)
        token_ids = tuple(req.all_token_ids[tok_start:tok_end])
        extra_keys = self._build_extra_keys(
            req,
            tok_start,
            tok_end,
            group_config.tokens_per_block,
        )

        lora_id: int | None = None
        lora_name: str | None = None
        if req.lora_request is not None:
            lora_id = req.lora_request.adapter_id
            lora_name = req.lora_request.name

        return _OffloadEventMetadata(
            block_hashes=tuple(chunk_hashes),
            parent_block_hash=parent_block_hash,
            token_ids=token_ids,
            block_size=group_config.tokens_per_block,
            lora_id=lora_id,
            lora_name=lora_name,
            extra_keys=extra_keys,
            group_idx=group_config.group_idx,
            kv_cache_spec=group_config.kv_event_group_spec,
        )

    def _placeholder_stored(
        self,
        key: OffloadKey,
        medium: Medium,
        locality: str | None,
    ) -> BlockStored:
        return BlockStored(
            block_hashes=[
                maybe_convert_block_hash(BlockHash(get_offload_block_hash(key)))
            ],
            parent_block_hash=None,
            token_ids=[],
            lora_id=None,
            block_size=0,
            medium=_MEDIUM_TO_EVENT_STR[medium],
            lora_name=None,
            group_idx=get_offload_group_idx(key),
            locality=locality,
        )

    def _take_stored_event(self, event: OffloadingEvent) -> Iterable[KVCacheEvent]:
        locality = event.locality.value if event.locality is not None else None
        state = (
            self._request_event_context(event.req_context)
            if self.self_describing_enabled
            else None
        )
        for key in event.keys:
            meta = self._metadata_for(state, key) if state is not None else None
            if meta is None:
                if self.self_describing_enabled:
                    logger.warning_once(
                        "OffloadingEventsTracker: no event metadata for "
                        "offload key during BlockStored emission; emitting a "
                        "placeholder payload. Expected for external jobs and "
                        "unsupported cache shapes."
                    )
                yield self._placeholder_stored(key, event.medium, locality)
                continue

            if event.medium is Medium.CPU:
                self._record_removal_metadata(event.medium, key, meta)

            yield BlockStored(
                block_hashes=list(
                    maybe_convert_block_hash(h) for h in meta.block_hashes
                ),
                parent_block_hash=(
                    maybe_convert_block_hash(meta.parent_block_hash)
                    if meta.parent_block_hash is not None
                    else None
                ),
                token_ids=list(meta.token_ids),
                block_size=meta.block_size,
                lora_id=meta.lora_id,
                medium=_MEDIUM_TO_EVENT_STR[event.medium],
                lora_name=meta.lora_name,
                extra_keys=(
                    list(meta.extra_keys) if meta.extra_keys is not None else None
                ),
                group_idx=meta.group_idx,
                kv_cache_spec_kind=meta.kv_cache_spec.kv_cache_spec_kind,
                kv_cache_spec_sliding_window=(
                    meta.kv_cache_spec.kv_cache_spec_sliding_window
                ),
                locality=locality,
            )

    def _take_removed_event(self, event: OffloadingEvent) -> Iterable[KVCacheEvent]:
        # Keep group_idx unambiguous if a manager batch spans groups.
        locality = event.locality.value if event.locality is not None else None
        by_group: dict[int, list] = {}
        for key in event.keys:
            meta = self._removal_metadata.pop((event.medium, key), None)
            if meta is not None:
                group_idx = meta.group_idx
                by_group.setdefault(group_idx, []).extend(
                    maybe_convert_block_hash(h) for h in meta.block_hashes
                )
            else:
                if self.self_describing_enabled:
                    logger.warning_once(
                        "OffloadingEventsTracker: no event metadata for "
                        "offload key during BlockRemoved emission; emitting a "
                        "placeholder removal. Expected if the matching store "
                        "used the legacy placeholder payload; otherwise "
                        "indicates missing store metadata."
                    )
                group_idx = get_offload_group_idx(key)
                by_group.setdefault(group_idx, []).append(
                    maybe_convert_block_hash(BlockHash(get_offload_block_hash(key)))
                )

        for group_idx, hashes in by_group.items():
            yield BlockRemoved(
                block_hashes=hashes,
                medium=_MEDIUM_TO_EVENT_STR[event.medium],
                group_idx=group_idx,
                locality=locality,
            )

    def _record_removal_metadata(
        self,
        medium: Medium,
        key: OffloadKey,
        metadata: _OffloadEventMetadata,
    ) -> None:
        self._removal_metadata.setdefault(
            (medium, key),
            _RemovalMetadata(
                block_hashes=metadata.block_hashes,
                group_idx=metadata.group_idx,
            ),
        )
