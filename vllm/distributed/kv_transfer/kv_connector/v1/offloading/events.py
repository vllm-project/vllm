# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-describing KV cache events for the offloading connector.

The OffloadingManager identifies an offloaded chunk only by its OffloadKey,
so its raw events carry no token ids, parent hash, or block size.
:class:`OffloadingEventsTracker` snapshots each chunk's full ``BlockStored``
payload while the ``Request`` is alive and publishes stores as ordinary
per-block events; evictions fan out to the same hashes. Chunks overlapping
a non-chunk-aligned shared prefix re-announce the shared hashes once per
chunk; consumers are expected to deduplicate (reference-count) repeated
store/remove announcements of the same hash. Opt-in via
``kv_connector_extra_config["self_describing_kv_events"]``; inert unless
KV cache events are enabled. See the PR description for the full design.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
from vllm.v1.kv_offload.base import (
    OffloadingEvent,
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


@dataclass(slots=True)
class _OffloadEventMetadata:
    """BlockStored payload snapshot for one OffloadKey, captured at store
    time and kept until the matching eviction event. ``medium`` is forwarded
    from the OffloadingEvent."""

    # The chunk's constituent block hashes; the last one is the OffloadKey.
    block_hashes: list[BlockHash]
    parent_block_hash: BlockHash | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None
    lora_name: str | None
    # Deferred: needs the same incremental curr_mm_idx handling as GPU events.
    extra_keys: list[tuple[Any, ...] | None] | None
    group_idx: int
    kv_cache_spec_kind: str | None
    kv_cache_spec_sliding_window: int | None


class OffloadingEventsTracker:
    """Tracks offloaded chunks' KV event payloads from store to eviction.

    The scheduler calls :meth:`record_store` from ``_build_store_jobs``
    while the ``Request`` is available, and routes the manager's raw
    :class:`OffloadingEvent` stream through :meth:`take_events`. All state
    is bounded by the CPU pool capacity and cleared by :meth:`reset`.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled

        # OffloadKey -> payload snapshot, kept until the eviction event so
        # BlockRemoved can fan out. Bounded: one entry per offloaded chunk.
        self._pending_event_metadata: dict[OffloadKey, _OffloadEventMetadata] = {}

    def record_store(
        self,
        req: Request,
        group_config: "GroupOffloadConfig",
        offload_block_idx: int,
        offload_key: OffloadKey,
    ) -> None:
        """Snapshot the KV cache event payload for one offloaded chunk.

        No-op when the tracker is disabled or for sliding-window / SSM
        groups, which keep the legacy placeholder payload.
        """
        if not self.enabled:
            return
        if group_config.sliding_window_size_in_blocks is not None:
            return
        meta = self._build_event_metadata(req, group_config, offload_block_idx)
        if meta is not None:
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
        offload_block_idx: int,
    ) -> _OffloadEventMetadata | None:
        """Build the payload snapshot for one offloaded chunk: its
        constituent per-block hashes, the whole chunk's tokens, and the
        per-block ``block_size``. Returns None when metadata is incomplete
        so take_events falls back to the placeholder payload."""
        hbf = group_config.hash_block_size_factor
        # per-block token count (= the GPU/hash block size)
        sub_block_size = group_config.offloaded_block_size // hbf
        # chunk c covers hash-blocks [c*hbf, (c+1)*hbf); its tail block's hash
        # is the chunk's OffloadKey.
        first_hash_idx = offload_block_idx * hbf
        last_hash_idx = first_hash_idx + hbf
        if first_hash_idx < 0 or last_hash_idx > len(req.block_hashes):
            return None
        chunk_hashes = list(req.block_hashes[first_hash_idx:last_hash_idx])
        if any(h is None for h in chunk_hashes):
            return None

        parent_block_hash: BlockHash | None
        if first_hash_idx == 0:
            parent_block_hash = None
        else:
            parent_block_hash = req.block_hashes[first_hash_idx - 1]
            if parent_block_hash is None:
                return None

        tok_start = offload_block_idx * group_config.offloaded_block_size
        tok_end = tok_start + group_config.offloaded_block_size
        if tok_end > len(req.all_token_ids):
            return None
        token_ids = list(req.all_token_ids[tok_start:tok_end])

        lora_id: int | None = None
        lora_name: str | None = None
        if req.lora_request is not None:
            lora_id = req.lora_request.adapter_id
            lora_name = req.lora_request.name

        return _OffloadEventMetadata(
            block_hashes=chunk_hashes,
            parent_block_hash=parent_block_hash,
            token_ids=token_ids,
            block_size=sub_block_size,
            lora_id=lora_id,
            lora_name=lora_name,
            extra_keys=None,
            group_idx=group_config.group_idx,
            kv_cache_spec_kind=group_config.kv_cache_spec_kind,
            kv_cache_spec_sliding_window=(group_config.kv_cache_spec_sliding_window),
        )

    def _placeholder_stored(self, key: OffloadKey, medium: str) -> BlockStored:
        return BlockStored(
            block_hashes=[
                maybe_convert_block_hash(BlockHash(get_offload_block_hash(key)))
            ],
            parent_block_hash=None,
            token_ids=[],
            lora_id=None,
            block_size=0,
            medium=medium,
            lora_name=None,
            group_idx=get_offload_group_idx(key),
        )

    def _take_stored_event(self, event: OffloadingEvent) -> Iterable[KVCacheEvent]:
        # Metadata is read, NOT popped: the entry must survive until the
        # eviction event so BlockRemoved can fan out to the same hashes.
        # Events are self-contained (own parent), so key order is free.
        for key in event.keys:
            meta = self._pending_event_metadata.get(key)
            if meta is None:
                if self.enabled:
                    # Expected for unsupported shapes; warn once only.
                    logger.warning_once(
                        "OffloadingEventsTracker: no event metadata for "
                        "offload key during BlockStored emission; emitting a "
                        "placeholder payload. Expected for non-full-attention "
                        "groups or when block hashes/tokens were unavailable "
                        "at store time; otherwise indicates a missing "
                        "populate path."
                    )
                yield self._placeholder_stored(key, event.medium)
                continue

            yield BlockStored(
                block_hashes=[maybe_convert_block_hash(h) for h in meta.block_hashes],
                parent_block_hash=(
                    maybe_convert_block_hash(meta.parent_block_hash)
                    if meta.parent_block_hash is not None
                    else None
                ),
                token_ids=meta.token_ids,
                block_size=meta.block_size,
                lora_id=meta.lora_id,
                medium=event.medium,
                lora_name=meta.lora_name,
                extra_keys=meta.extra_keys,
                group_idx=meta.group_idx,
                kv_cache_spec_kind=meta.kv_cache_spec_kind,
                kv_cache_spec_sliding_window=meta.kv_cache_spec_sliding_window,
            )

    def _take_removed_event(self, event: OffloadingEvent) -> Iterable[KVCacheEvent]:
        # Keep group_idx unambiguous if a manager batch spans groups.
        by_group: dict[int, list] = {}
        for key in event.keys:
            meta = self._pending_event_metadata.pop(key, None)
            if meta is not None:
                group_idx = meta.group_idx
                hashes = [maybe_convert_block_hash(h) for h in meta.block_hashes]
            else:
                group_idx = get_offload_group_idx(key)
                hashes = [
                    maybe_convert_block_hash(BlockHash(get_offload_block_hash(key)))
                ]
            by_group.setdefault(group_idx, []).extend(hashes)

        for group_idx, hashes in by_group.items():
            yield BlockRemoved(
                block_hashes=hashes,
                medium=event.medium,
                group_idx=group_idx,
            )
