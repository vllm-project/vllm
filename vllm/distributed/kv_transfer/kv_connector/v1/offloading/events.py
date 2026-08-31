# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-describing KV cache events for the offloading connector.

The OffloadingManager identifies an offloaded chunk only by its OffloadKey,
so its raw events carry no token ids, parent hash, or block size.
:class:`OffloadingEventsTracker` snapshots each chunk's full ``BlockStored``
payload while the ``Request`` is alive and publishes stores as block-granular
payloads: a chunk event may carry multiple constituent per-block hashes, and
evictions fan out to the same hashes. Chunks overlapping a non-chunk-aligned
shared prefix re-announce the shared hashes once per chunk; consumers are
expected to deduplicate (reference-count) repeated store/remove announcements
of the same hash. Opt-in via
``kv_connector_extra_config["self_describing_kv_events"]``; inert unless
KV cache events are enabled. See the PR description for the full design.
"""

from collections.abc import Iterable
from dataclasses import dataclass
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
    get_offload_block_hash,
    get_offload_group_idx,
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


@dataclass(slots=True)
class _OffloadEventMetadata:
    """BlockStored payload snapshot for one OffloadKey, captured while the
    Request is available and kept until the final matching removal event.
    ``medium`` and ``ownership`` are forwarded from the OffloadingEvent."""

    # The chunk's constituent block hashes; the last one is the OffloadKey.
    block_hashes: tuple[BlockHash, ...]
    parent_block_hash: BlockHash | None
    token_ids: tuple[int, ...]
    block_size: int
    lora_id: int | None
    lora_name: str | None
    # Deferred: needs the same incremental curr_mm_idx handling as GPU events.
    extra_keys: tuple[tuple[Any, ...] | None, ...] | None
    group_idx: int
    kv_cache_spec: OffloadingEventGroupSpec
    active_residencies: set[tuple[Medium, str | None]]


class OffloadingEventsTracker:
    """Tracks offloaded chunks' KV event payloads from store to eviction.

    The scheduler calls :meth:`record_store` from ``_build_store_jobs`` and
    :meth:`record_lookup` for ready primary-tier hits while the ``Request`` is
    available. Deferred and missing lookups add no state. Under the connector's
    supported success-only transfer model, entries remain until the final
    observed residency removal or :meth:`reset`.
    """

    def __init__(self, config: OffloadingKVEventsConfig):
        self.config = config
        self.self_describing_enabled = (
            config.enable_kv_cache_events and config.self_describing_kv_events
        )

        # OffloadKey -> payload snapshot, kept until final removal or reset.
        self._pending_event_metadata: dict[OffloadKey, _OffloadEventMetadata] = {}

    def record_store(
        self,
        req: Request,
        group_config: "GroupOffloadConfig",
        chunk_idx: int,
        offload_key: OffloadKey,
    ) -> None:
        """Snapshot the KV cache event payload for one offloaded chunk.

        No-op when self-describing event capture is disabled or for
        sliding-window / SSM groups, which keep the legacy placeholder payload.
        """
        if not self.self_describing_enabled:
            return
        if group_config.sliding_window_size_in_chunks is not None:
            return
        meta = self._build_event_metadata(req, group_config, chunk_idx)
        if existing := self._pending_event_metadata.get(offload_key):
            meta.active_residencies.update(existing.active_residencies)
        self._pending_event_metadata[offload_key] = meta

    def record_lookup(
        self,
        req: Request,
        group_config: "GroupOffloadConfig",
        chunk_idx: int,
        offload_key: OffloadKey,
    ) -> None:
        """Snapshot metadata for a ready primary-tier lookup hit."""
        if not self.self_describing_enabled:
            return
        if group_config.sliding_window_size_in_chunks is not None:
            return
        if offload_key not in self._pending_event_metadata:
            self._pending_event_metadata[offload_key] = self._build_event_metadata(
                req, group_config, chunk_idx
            )

    def record_partial_store(
        self,
        req: Request,
        group_config: "GroupOffloadConfig",
        boundary_tokens: int,
        offload_key: OffloadKey,
    ) -> None:
        """Snapshot metadata for a newly stored partial recurrent tail."""
        if group_config.sliding_window_size_in_chunks is not None:
            return
        self._record_partial_tail(req, group_config, boundary_tokens, offload_key)

    def record_partial_lookup(
        self,
        req: Request,
        group_config: "GroupOffloadConfig",
        boundary_tokens: int,
        offload_key: OffloadKey,
    ) -> None:
        """Backfill metadata for a partial recurrent tail lookup hit."""
        if group_config.sliding_window_size_in_chunks is not None:
            return
        if offload_key not in self._pending_event_metadata:
            self._record_partial_tail(req, group_config, boundary_tokens, offload_key)

    def _record_partial_tail(
        self,
        req: Request,
        group_config: "GroupOffloadConfig",
        boundary_tokens: int,
        offload_key: OffloadKey,
    ) -> None:
        """Build metadata for the valid prefix of one physical cache block.

        A partial recurrent tail ends on a hash boundary but before its
        physical cache block is full. The event describes only the valid
        hashes and tokens, not the unused remainder of that physical block.
        """
        if not self.self_describing_enabled:
            return

        tokens_per_hash = group_config.tokens_per_chunk // group_config.hashes_per_chunk
        # Subtract one so the boundary token itself cannot select the next
        # physical block when the boundary lies exactly on a block edge.
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
        meta = _OffloadEventMetadata(
            block_hashes=block_hashes,
            parent_block_hash=parent_block_hash,
            token_ids=tuple(req.all_token_ids[chunk_start:boundary_tokens]),
            block_size=tokens_per_hash,
            lora_id=lora_id,
            lora_name=lora_name,
            extra_keys=None,
            group_idx=group_config.group_idx,
            kv_cache_spec=group_config.kv_event_group_spec,
            active_residencies={(Medium.CPU, None)},
        )
        if existing := self._pending_event_metadata.get(offload_key):
            meta.active_residencies.update(existing.active_residencies)
        self._pending_event_metadata[offload_key] = meta

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
        """Drop all tracked state; pending payloads are stale after a
        manager cache reset."""
        self._pending_event_metadata.clear()

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
        # Each chunk's final raw hash is its OffloadKey.
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
            extra_keys=None,
            group_idx=group_config.group_idx,
            kv_cache_spec=group_config.kv_event_group_spec,
            active_residencies={(Medium.CPU, None)},
        )

    def _placeholder_stored(
        self,
        key: OffloadKey,
        medium: Medium,
        locality: str | None,
        ownership: str | None,
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
            ownership=ownership,
        )

    def _take_stored_event(self, event: OffloadingEvent) -> Iterable[KVCacheEvent]:
        # Metadata is read, NOT popped: the entry must survive until the
        # eviction event so BlockRemoved can fan out to the same hashes.
        # Events are self-contained (own parent), so key order is free.
        locality = event.locality.value if event.locality is not None else None
        for key in event.keys:
            meta = self._pending_event_metadata.get(key)
            if meta is None:
                if self.self_describing_enabled:
                    # Expected for unsupported shapes; warn once only.
                    logger.warning_once(
                        "OffloadingEventsTracker: no event metadata for "
                        "offload key during BlockStored emission; emitting a "
                        "placeholder payload. Expected for non-full-attention "
                        "groups and promotions not observed as a primary-tier "
                        "hit before translation."
                    )
                yield self._placeholder_stored(
                    key, event.medium, locality, event.ownership
                )
                continue

            if event.removal_expected:
                meta.active_residencies.add((event.medium, event.ownership))
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
                ownership=event.ownership,
            )

    def _take_removed_event(self, event: OffloadingEvent) -> Iterable[KVCacheEvent]:
        # Keep group_idx unambiguous if a manager batch spans groups.
        locality = event.locality.value if event.locality is not None else None
        by_group: dict[int, list] = {}
        for key in event.keys:
            meta = self._pending_event_metadata.get(key)
            if meta is not None:
                group_idx = meta.group_idx
                by_group.setdefault(group_idx, []).extend(
                    maybe_convert_block_hash(h) for h in meta.block_hashes
                )
                meta.active_residencies.discard((event.medium, event.ownership))
                if not meta.active_residencies:
                    self._pending_event_metadata.pop(key)
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
                ownership=event.ownership,
            )
