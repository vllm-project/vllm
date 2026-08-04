# SPDX-License-Identifier: Apache-2.0

"""Blend metrics subscriber — OTel counters for cache blending events."""

# Future
from __future__ import annotations

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


class BlendMetricsSubscriber(EventSubscriber):
    """Maintains OTel counters for cache blending (CB) operations.

    Metrics:
    - ``lmcache_blend.lookup_requests``              — total CB lookup calls
    - ``lmcache_blend.lookup_requested_tokens``      — chunk-aligned tokens
      submitted for CB lookup (denominator of the blend token-level hit
      rate).  Sub-chunk trailing tokens are excluded because they cannot
      hit by design.
    - ``lmcache_blend.lookup_hit_tokens``            — tokens served by
      blend during the lookup (numerator of the blend token-level hit
      rate).  Equal to ``storage_hits * chunk_size``.
    - ``lmcache_blend.lookup_fingerprint_hits``      — fingerprint table hits
    - ``lmcache_blend.lookup_storage_hits``          — chunks confirmed in storage
    - ``lmcache_blend.lookup_stale_chunks``          — fingerprint hits evicted as stale
    - ``lmcache_blend.lookup_no_gpu_context_errors`` — lookup failures: no GPU context
    - ``lmcache_blend.retrieve_requests``            — total CB retrieve calls
    - ``lmcache_blend.retrieve_chunks``              — chunks requested for retrieval
    - ``lmcache_blend.retrieve_failures``            — retrieves with success=False
    - ``lmcache_blend.store_pre_computed_requests``  — total CB store_pre_computed calls
    - ``lmcache_blend.store_pre_computed_chunks``    — chunks via store_pre_computed
    - ``lmcache_blend.store_pre_computed_failures``  — store_pre_computed failures
    - ``lmcache_blend.store_final_requests``         — total CB store_final calls
    - ``lmcache_blend.store_final_chunks``           — chunks stored via store_final
    - ``lmcache_blend.store_final_failures``         — store_final failures
    - ``lmcache_blend.fingerprints_registered``      — chunks in fingerprint table
    - ``lmcache_blend.chunks_evicted``               — evicted from fingerprint table
    """

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache.blend")

        self._lookup_requests = meter.create_counter(
            "lmcache_blend.lookup_requests",
            description="Total CB lookup requests",
        )
        self._lookup_requested_tokens = meter.create_counter(
            "lmcache_blend.lookup_requested_tokens",
            description=(
                "Total tokens submitted for CB lookup (denominator of the "
                "blend token-level hit rate). Only chunk-aligned tokens "
                "are counted."
            ),
            unit="tokens",
        )
        self._lookup_hit_tokens = meter.create_counter(
            "lmcache_blend.lookup_hit_tokens",
            description=(
                "Total tokens served by blend during lookup (numerator of "
                "the blend token-level hit rate). Equal to "
                "storage_hits * chunk_size."
            ),
            unit="tokens",
        )
        self._lookup_prefix_hit_tokens = meter.create_counter(
            "lmcache_blend.lookup_prefix_hit_tokens",
            description="Tokens served by blend from the prefix (L1+L2).",
            unit="tokens",
        )
        self._lookup_non_prefix_hit_tokens = meter.create_counter(
            "lmcache_blend.lookup_non_prefix_hit_tokens",
            description="Tokens served by blend from non-prefix (shifted) chunks.",
            unit="tokens",
        )
        self._lookup_segmented_prefix_hit_tokens = meter.create_counter(
            "lmcache_blend.lookup_segmented_prefix_hit_tokens",
            description=(
                "Tokens served by blend from the segmented-prefix tail "
                "(post-gap same-position chunks reused via the prefix leg)."
            ),
            unit="tokens",
        )
        self._lookup_fingerprint_hits = meter.create_counter(
            "lmcache_blend.lookup_fingerprint_hits",
            description="Chunks matched by local fingerprint table",
        )
        self._lookup_storage_hits = meter.create_counter(
            "lmcache_blend.lookup_storage_hits",
            description="Chunks confirmed present in storage after prefetch",
        )
        self._lookup_stale_chunks = meter.create_counter(
            "lmcache_blend.lookup_stale_chunks",
            description="Fingerprint hits evicted as stale (not in storage)",
        )
        self._lookup_no_gpu_ctx_errors = meter.create_counter(
            "lmcache_blend.lookup_no_gpu_context_errors",
            description="Lookup failures due to missing GPU context",
        )
        self._retrieve_requests = meter.create_counter(
            "lmcache_blend.retrieve_requests",
            description="Total CB retrieve requests",
        )
        self._retrieve_chunks = meter.create_counter(
            "lmcache_blend.retrieve_chunks",
            description="Total chunks requested for CB retrieval",
        )
        self._retrieve_failures = meter.create_counter(
            "lmcache_blend.retrieve_failures",
            description="CB retrieve operations that returned success=False",
        )
        self._store_pre_computed_requests = meter.create_counter(
            "lmcache_blend.store_pre_computed_requests",
            description="Total CB store_pre_computed requests",
        )
        self._store_pre_computed_chunks = meter.create_counter(
            "lmcache_blend.store_pre_computed_chunks",
            description="Chunks stored via CB store_pre_computed",
        )
        self._store_pre_computed_failures = meter.create_counter(
            "lmcache_blend.store_pre_computed_failures",
            description="CB store_pre_computed failures",
        )
        self._store_final_requests = meter.create_counter(
            "lmcache_blend.store_final_requests",
            description="Total CB store_final requests",
        )
        self._store_final_chunks = meter.create_counter(
            "lmcache_blend.store_final_chunks",
            description="Chunks stored via CB store_final",
        )
        self._store_final_failures = meter.create_counter(
            "lmcache_blend.store_final_failures",
            description="CB store_final failures",
        )
        self._fingerprints_registered = meter.create_counter(
            "lmcache_blend.fingerprints_registered",
            description="Chunks indexed into the fingerprint table",
        )
        self._chunks_evicted = meter.create_counter(
            "lmcache_blend.chunks_evicted",
            description="Stale chunks evicted from the fingerprint table",
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        """Return the mapping of event types to handler callbacks."""
        return {
            EventType.CB_LOOKUP_START: self._on_lookup_start,
            EventType.CB_LOOKUP_END: self._on_lookup_end,
            EventType.CB_RETRIEVE_START: self._on_retrieve_start,
            EventType.CB_RETRIEVE_END: self._on_retrieve_end,
            EventType.CB_STORE_PRE_COMPUTED_START: self._on_store_pre_start,
            EventType.CB_STORE_PRE_COMPUTED_END: self._on_store_pre_end,
            EventType.CB_STORE_FINAL_START: self._on_store_final_start,
            EventType.CB_STORE_FINAL_END: self._on_store_final_end,
            EventType.CB_FINGERPRINTS_REGISTERED: self._on_fingerprints_registered,
            EventType.CB_CHUNKS_EVICTED: self._on_chunks_evicted,
        }

    def _on_lookup_start(self, event: Event) -> None:
        self._lookup_requests.add(1)

    def _on_lookup_end(self, event: Event) -> None:
        self._lookup_requested_tokens.add(event.metadata["requested_tokens"])
        self._lookup_hit_tokens.add(event.metadata["hit_tokens"])
        self._lookup_prefix_hit_tokens.add(event.metadata.get("prefix_hit_tokens", 0))
        self._lookup_non_prefix_hit_tokens.add(
            event.metadata.get("non_prefix_hit_tokens", 0)
        )
        self._lookup_segmented_prefix_hit_tokens.add(
            event.metadata.get("segmented_prefix_hit_tokens", 0)
        )
        self._lookup_fingerprint_hits.add(event.metadata["fingerprint_hits"])
        self._lookup_storage_hits.add(event.metadata["storage_hits"])
        self._lookup_stale_chunks.add(event.metadata["stale_chunks"])
        if event.metadata.get("no_gpu_context"):
            self._lookup_no_gpu_ctx_errors.add(1)

    def _on_retrieve_start(self, event: Event) -> None:
        self._retrieve_requests.add(1)
        self._retrieve_chunks.add(event.metadata["num_chunks"])

    def _on_retrieve_end(self, event: Event) -> None:
        if not event.metadata.get("success", True):
            self._retrieve_failures.add(1)

    def _on_store_pre_start(self, event: Event) -> None:
        self._store_pre_computed_requests.add(1)

    def _on_store_pre_end(self, event: Event) -> None:
        self._store_pre_computed_chunks.add(event.metadata["stored_chunks"])
        if not event.metadata.get("success", True):
            self._store_pre_computed_failures.add(1)

    def _on_store_final_start(self, event: Event) -> None:
        self._store_final_requests.add(1)

    def _on_store_final_end(self, event: Event) -> None:
        self._store_final_chunks.add(event.metadata["stored_chunks"])
        if not event.metadata.get("success", True):
            self._store_final_failures.add(1)

    def _on_fingerprints_registered(self, event: Event) -> None:
        self._fingerprints_registered.add(event.metadata["num_chunks"])

    def _on_chunks_evicted(self, event: Event) -> None:
        self._chunks_evicted.add(event.metadata["num_chunks"])
