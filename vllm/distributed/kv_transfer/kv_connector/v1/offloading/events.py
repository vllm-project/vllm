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

from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    get_kv_cache_spec_kind,
    get_kv_cache_spec_sliding_window,
)
from vllm.v1.kv_offload.base import (
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
    Request is available and kept until the matching eviction event. ``medium``
    is forwarded from the OffloadingEvent."""

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


class OffloadingEventsTracker:
    """Tracks offloaded chunks' KV event payloads from store to eviction.

    The scheduler calls :meth:`record_store` from ``_build_store_jobs`` and
    :meth:`record_lookup` for ready primary-tier hits while the ``Request`` is
    available. Deferred and missing lookups add no state. Under the connector's
    supported success-only transfer model, entries follow primary allocations
    until CPU removal translation or :meth:`reset`.
    """

    def __init__(self, config: OffloadingKVEventsConfig):
        self.config = config
        self.self_describing_enabled = (
            config.enable_kv_cache_events and config.self_describing_kv_events
        )

        # OffloadKey -> payload snapshot, kept until CPU removal or reset.
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
        hbf = group_config.hashes_per_chunk
        assert hbf > 0
        assert chunk_idx >= 0
        # per-block token count (= the GPU/hash block size)
        tokens_per_hash = group_config.tokens_per_chunk // hbf
        # chunk c covers hash-blocks [c*hbf, (c+1)*hbf); its tail block's hash
        # is the chunk's OffloadKey.
        first_hash_idx = chunk_idx * hbf
        last_hash_idx = first_hash_idx + hbf
        assert first_hash_idx >= 0
        assert last_hash_idx <= len(req.block_hashes)
        chunk_hashes: list[BlockHash] = []
        for block_hash in req.block_hashes[first_hash_idx:last_hash_idx]:
            assert block_hash is not None
            chunk_hashes.append(block_hash)
        assert len(chunk_hashes) == hbf

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
            block_size=tokens_per_hash,
            lora_id=lora_id,
            lora_name=lora_name,
            extra_keys=None,
            group_idx=group_config.group_idx,
            kv_cache_spec=group_config.kv_event_group_spec,
        )

    def _placeholder_stored(
        self,
        key: OffloadKey,
        medium: str,
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
            medium=medium,
            lora_name=None,
            group_idx=get_offload_group_idx(key),
            locality=locality,
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
                yield self._placeholder_stored(key, event.medium, locality)
                continue

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
                medium=event.medium,
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
            meta = self._pending_event_metadata.pop(key, None)
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
                medium=event.medium,
                group_idx=group_idx,
                locality=locality,
            )
