# SPDX-License-Identifier: Apache-2.0
"""Continuous usage reporting: periodic interval counters and histograms."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING
import os
import time

# Third Party
import numpy as np

# First Party
from lmcache.logging import init_logger
from lmcache.usage_telemetry.guard import swallow_telemetry_errors
from lmcache.usage_telemetry.identity import (
    get_usage_identity,
    is_usage_tracking_enabled,
)
from lmcache.usage_telemetry.messages import (
    CacheLifespanMessage,
    ContinuousContextMessage,
    DeploymentMode,
)
from lmcache.usage_telemetry.transport import (
    DEFAULT_SENDER,
    UsageMessageSender,
    build_usage_payload,
    usage_server_url,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.observability import LMCacheStats
    from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)


class ContinuousUsageContext:
    """Periodic reporter of interval cache-usage counters.

    Accumulates hit/stored token counts and cache lifespans via
    :meth:`incr_or_send_stats` and flushes them to the stats server every
    ``LMCACHE_USAGE_TRACK_INTERVAL`` seconds (default 600). Interval data
    is dropped, not retried, when a send fails; gaps in
    ``sequence_number`` mark lost intervals on the backend.
    """

    _instance: ContinuousUsageContext | None = None

    def __init__(
        self,
        metadata: LMCacheMetadata,
        sender: UsageMessageSender | None = None,
        mode: DeploymentMode = DeploymentMode.SINGLE_PROCESS,
    ) -> None:
        """Initialize the reporter.

        Args:
            metadata: The engine metadata; its kv shape/dtype size the
                stored-bytes estimate.
            sender: Message transport; ``None`` selects the default HTTP
                sender.
            mode: Deployment mode stamped on every payload this reporter
                sends.
        """
        self.cache_lifespan_buckets: list[float] = [
            0,
            1,
            5,
            10,
            20,
            40,
            60,
            80,
            100,
            250,
            500,
            750,
            1000,
            2500,
            5000,
        ]
        self.metadata: LMCacheMetadata = metadata
        self.cache_usage_url: str = usage_server_url(ContinuousContextMessage.ENDPOINT)
        self.cache_lifespan_url: str = usage_server_url(CacheLifespanMessage.ENDPOINT)
        try:
            self.min_logging_interval: int = int(
                os.getenv("LMCACHE_USAGE_TRACK_INTERVAL", "600")
            )
        except ValueError:
            # A malformed env value must not raise into engine startup.
            logger.debug("Invalid LMCACHE_USAGE_TRACK_INTERVAL", exc_info=True)
            self.min_logging_interval = 600
        # send the first message immediately after init
        self.last_logged_ts: float = -1

        self.interval_num_hit_tokens: int = 0
        self.interval_num_stored_tokens: int = 0
        try:
            self.kv_sz_per_token_bytes: int = int(
                np.prod(self.metadata.kv_shape)
                * self.metadata.kv_dtype.itemsize
                / self.metadata.kv_shape[2]
            )
        except Exception:
            # 0 marks the stored-bytes estimate as unavailable.
            logger.debug("Cannot derive kv bytes per token", exc_info=True)
            self.kv_sz_per_token_bytes = 0
        self.cache_lifespan_data: list[float] = []
        self._sender = sender if sender is not None else DEFAULT_SENDER
        self._mode = mode
        self._sequence_number: int = 0
        self._start_monotonic: float = time.monotonic()

    @staticmethod
    def GetOrCreate(metadata: LMCacheMetadata) -> ContinuousUsageContext:
        """Return the process-wide instance, creating it on first call.

        Args:
            metadata: The engine metadata used on first creation.

        Returns:
            The singleton continuous usage context.
        """
        if ContinuousUsageContext._instance is None:
            ContinuousUsageContext._instance = ContinuousUsageContext(metadata)
        if ContinuousUsageContext._instance.metadata != metadata:
            logger.error(
                "ContinuousUsageContext instance already created with"
                "different metadata. This should not happen except "
                "in test."
            )
        return ContinuousUsageContext._instance

    def send_caching_message(self) -> None:
        """Flush the interval counters and lifespan histogram, then reset."""
        self._sequence_number += 1
        identity = get_usage_identity()
        uptime_seconds = time.monotonic() - self._start_monotonic

        usage_message = ContinuousContextMessage(
            interval_stored_kv_size=int(
                self.kv_sz_per_token_bytes * self.interval_num_stored_tokens
            ),
            interval_num_hit_tokens=int(self.interval_num_hit_tokens),
            interval_num_stored_tokens=int(self.interval_num_stored_tokens),
            sequence_number=self._sequence_number,
            uptime_seconds=uptime_seconds,
        )
        self._sender.send(
            self.cache_usage_url,
            build_usage_payload(usage_message, identity, self._mode),
        )
        self.interval_num_hit_tokens = 0
        self.interval_num_stored_tokens = 0

        lifespan_message = CacheLifespanMessage(
            cache_lifespan_histogram=self.list_to_histogram(
                self.cache_lifespan_data, self.cache_lifespan_buckets
            ),
            sequence_number=self._sequence_number,
            uptime_seconds=uptime_seconds,
        )
        self._sender.send(
            self.cache_lifespan_url,
            build_usage_payload(lifespan_message, identity, self._mode),
        )
        self.cache_lifespan_data = []

    def list_to_histogram(
        self, data: list[float], buckets: list[float]
    ) -> dict[float, int]:
        """Bucket *data* into a ``{bucket_lower_bound: count}`` histogram.

        Args:
            data: The raw samples.
            buckets: Ascending bucket boundaries.

        Returns:
            Mapping from each bucket boundary to the sample count in it.
        """
        histogram, _ = np.histogram(data, bins=buckets)
        counts = list(histogram)
        counts.insert(0, 0)
        return {
            bucket: int(count) for bucket, count in zip(buckets, counts, strict=False)
        }

    @swallow_telemetry_errors
    def incr_or_send_stats(self, stats: LMCacheStats) -> None:
        """Accumulate interval stats and flush when the interval elapsed.

        No-op when usage tracking is disabled; never raises into the
        stats-logger loop.

        Args:
            stats: The stats snapshot of the elapsed logging tick.
        """
        if not is_usage_tracking_enabled():
            return
        self.cache_lifespan_data.extend(stats.interval_request_cache_lifespan)
        self.interval_num_hit_tokens += stats.interval_hit_tokens
        self.interval_num_stored_tokens += stats.interval_stored_tokens

        cur_ts: float = time.monotonic()
        if cur_ts - self.last_logged_ts >= self.min_logging_interval:
            self.send_caching_message()
            self.last_logged_ts = cur_ts
