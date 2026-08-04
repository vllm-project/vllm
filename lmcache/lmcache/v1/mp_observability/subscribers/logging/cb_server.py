# SPDX-License-Identifier: Apache-2.0

"""Blend logging subscriber — debug logs for cache blending events."""

# Future
from __future__ import annotations

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

logger = init_logger(__name__)


class BlendLoggingSubscriber(EventSubscriber):
    """Logs cache blending (CB) events at debug level."""

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        """Return the mapping of event types to handler callbacks."""
        return {
            EventType.CB_STORE_PRE_COMPUTED_START: self._on_store_pre_start,
            EventType.CB_STORE_PRE_COMPUTED_END: self._on_store_pre_end,
            EventType.CB_LOOKUP_START: self._on_lookup_start,
            EventType.CB_LOOKUP_END: self._on_lookup_end,
            EventType.CB_RETRIEVE_START: self._on_retrieve_start,
            EventType.CB_RETRIEVE_END: self._on_retrieve_end,
            EventType.CB_STORE_FINAL_START: self._on_store_final_start,
            EventType.CB_STORE_FINAL_END: self._on_store_final_end,
            EventType.CB_FINGERPRINTS_REGISTERED: self._on_fingerprints_registered,
            EventType.CB_CHUNKS_EVICTED: self._on_chunks_evicted,
        }

    def _on_store_pre_start(self, event: Event) -> None:
        logger.debug(
            "CB store_pre_computed start: session=%s instance_id=%s num_tokens=%s",
            event.session_id,
            event.metadata.get("instance_id"),
            event.metadata.get("num_tokens"),
        )

    def _on_store_pre_end(self, event: Event) -> None:
        logger.debug(
            "CB store_pre_computed end: session=%s instance_id=%s"
            " num_tokens=%s stored_chunks=%s success=%s",
            event.session_id,
            event.metadata.get("instance_id"),
            event.metadata.get("num_tokens"),
            event.metadata.get("stored_chunks"),
            event.metadata.get("success"),
        )

    def _on_lookup_start(self, event: Event) -> None:
        logger.debug(
            "CB lookup start: session=%s num_tokens=%s",
            event.session_id,
            event.metadata.get("num_tokens"),
        )

    def _on_lookup_end(self, event: Event) -> None:
        logger.debug(
            "CB lookup end: session=%s num_tokens=%s"
            " fingerprint_hits=%s storage_hits=%s stale_chunks=%s no_gpu_context=%s",
            event.session_id,
            event.metadata.get("num_tokens"),
            event.metadata.get("fingerprint_hits"),
            event.metadata.get("storage_hits"),
            event.metadata.get("stale_chunks"),
            event.metadata.get("no_gpu_context"),
        )

    def _on_retrieve_start(self, event: Event) -> None:
        logger.debug(
            "CB retrieve start: session=%s instance_id=%s num_chunks=%s",
            event.session_id,
            event.metadata.get("instance_id"),
            event.metadata.get("num_chunks"),
        )

    def _on_retrieve_end(self, event: Event) -> None:
        logger.debug(
            "CB retrieve end: session=%s instance_id=%s num_chunks=%s success=%s",
            event.session_id,
            event.metadata.get("instance_id"),
            event.metadata.get("num_chunks"),
            event.metadata.get("success"),
        )

    def _on_store_final_start(self, event: Event) -> None:
        logger.debug(
            "CB store_final start: session=%s instance_id=%s num_tokens=%s",
            event.session_id,
            event.metadata.get("instance_id"),
            event.metadata.get("num_tokens"),
        )

    def _on_store_final_end(self, event: Event) -> None:
        logger.debug(
            "CB store_final end: session=%s instance_id=%s"
            " num_tokens=%s stored_chunks=%s success=%s",
            event.session_id,
            event.metadata.get("instance_id"),
            event.metadata.get("num_tokens"),
            event.metadata.get("stored_chunks"),
            event.metadata.get("success"),
        )

    def _on_fingerprints_registered(self, event: Event) -> None:
        logger.debug(
            "CB fingerprint table: +%s chunks (%s tokens)",
            event.metadata.get("num_chunks"),
            event.metadata.get("num_tokens"),
        )

    def _on_chunks_evicted(self, event: Event) -> None:
        logger.debug(
            "CB fingerprint table: evicted %s stale chunks",
            event.metadata.get("num_chunks"),
        )
