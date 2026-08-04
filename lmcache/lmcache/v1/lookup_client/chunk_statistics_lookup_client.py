# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional, Union
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.observability import PrometheusLogger
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.lookup_client.record_strategies import (
    AsyncRecorder,
    create_record_strategy,
)

if TYPE_CHECKING:
    # Third Party
    import torch

    # First Party
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.lookup_client.record_strategies import RecordStrategy
    from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)


class ChunkStatisticsLookupClient(LookupClientInterface):
    """Wrapper client that tracks chunk reuse statistics."""

    def __init__(
        self,
        actual_lookup_client: LookupClientInterface,
        config: "LMCacheEngineConfig",
        metadata: Optional["LMCacheMetadata"] = None,
    ) -> None:
        self.actual_lookup_client = actual_lookup_client
        self.config = config
        self.metadata = metadata
        self.lock = threading.RLock()
        self.chunk_size = config.chunk_size
        self.enabled = False
        self.request_seen: set[str] = set()
        self.lookup_time = 0.0
        self.record_time = 0.0
        self.check_exit_time = 0.0
        self.statistics_start_time = 0.0
        self.timeout_hours = config.chunk_statistics_auto_exit_timeout_hours
        self.target_unique_chunks = (
            config.chunk_statistics_auto_exit_target_unique_chunks
        )
        self.enable_auto_exit = (
            self.timeout_hours > 0.0 or self.target_unique_chunks > 0
        )
        strategy: "RecordStrategy" = create_record_strategy(config)
        self.recorder = AsyncRecorder(
            strategy=strategy,
            queue_capacity=config.get_extra_config_value(
                "chunk_statistics_async_queue_capacity", 100000
            ),
            preprocess_in_caller=config.get_extra_config_value(
                "chunk_statistics_async_preprocess_chunks", False
            ),
        )
        self._setup_metrics()
        if config.chunk_statistics_auto_start_statistics:
            self.start_statistics()

    def lookup_cache(self, lookup_id: str) -> Optional[int]:
        return self.actual_lookup_client.lookup_cache(lookup_id)

    def start_statistics(self) -> None:
        with self.lock:
            self.enabled = True
            # Assign the start time while first recording
            self.statistics_start_time = 0.0

    def stop_statistics(self) -> None:
        with self.lock:
            self.enabled = False

    def reset_statistics(self) -> None:
        self.recorder.wait_for_completion(timeout=5.0)
        with self.lock:
            self.request_seen.clear()
            self.recorder.reset()

    def get_statistics(self) -> dict:
        self.recorder.wait_for_completion(timeout=5.0)
        with self.lock:
            strategy_stats = self.recorder.get_statistics()
            total_time = self.lookup_time + self.record_time + self.check_exit_time
            overhead_time = self.record_time + self.check_exit_time
            overhead_pct = (
                (overhead_time / total_time * 100.0) if total_time > 0 else 0.0
            )
            result = {
                "enabled": self.enabled,
                "total_requests": len(self.request_seen),
                "timing": {
                    "lookup_time_seconds": self.lookup_time,
                    "record_statistics_time_seconds": self.record_time,
                    "check_exit_conditions_time_seconds": self.check_exit_time,
                    "total_time_seconds": total_time,
                    "overhead_time_seconds": overhead_time,
                    "overhead_percentage": overhead_pct,
                },
                "total_chunks": strategy_stats.get("total_chunks", 0),
                "unique_chunks": strategy_stats.get("unique_chunks", 0),
                "duplicate_chunks": strategy_stats.get("duplicate_chunks", 0),
                "reuse_rate": strategy_stats.get("reuse_rate", 0.0),
                **{
                    k: v
                    for k, v in strategy_stats.items()
                    if k in ("bloom_filter", "async_queue", "file_hash")
                },
            }
            return result

    def wait_for_async_processing(self, timeout: float = 5.0) -> bool:
        return self.recorder.wait_for_completion(timeout)

    def lookup(
        self,
        token_ids: Union["torch.Tensor", list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        start_time = time.time()
        result = self.actual_lookup_client.lookup(
            token_ids,
            lookup_id,
            request_configs,
        )
        lookup_elapsed = time.time() - start_time
        with self.lock:
            self.lookup_time += lookup_elapsed

        if not self.enabled:
            return result

        with self.lock:
            if lookup_id in self.request_seen:
                return result
            self.request_seen.add(lookup_id)

        start_time = time.time()
        self.recorder.record_async(token_ids, lookup_id)
        record_elapsed = time.time() - start_time
        with self.lock:
            self.record_time += record_elapsed

        start_time = time.time()
        self._check_exit_conditions()
        check_elapsed = time.time() - start_time
        with self.lock:
            self.check_exit_time += check_elapsed

        return result

    def clear_lookup_status(self, lookup_id: str) -> None:
        self.actual_lookup_client.clear_lookup_status(lookup_id)

    def supports_producer_reuse(self) -> bool:
        return self.actual_lookup_client.supports_producer_reuse()

    def close(self) -> None:
        if self.enabled:
            self.stop_statistics()
        self.recorder.close()
        self.actual_lookup_client.close()

    def _check_exit_conditions(self) -> None:
        if not self.enable_auto_exit:
            return
        if self.statistics_start_time == 0.0:
            self.statistics_start_time = time.time()
        stop_reason = None
        if self.timeout_hours > 0.0:
            elapsed_hours = (time.time() - self.statistics_start_time) / 3600.0
            if elapsed_hours >= self.timeout_hours:
                stop_reason = (
                    f"Timeout: {elapsed_hours:.2f}h >= {self.timeout_hours:.2f}h"
                )
        if self.target_unique_chunks > 0:
            unique = self.recorder.strategy.unique_chunks_count
            if unique >= self.target_unique_chunks:
                stop_reason = f"Target reached: {unique} >= {self.target_unique_chunks}"
        if stop_reason:
            self._trigger_stop(stop_reason)

    def _trigger_stop(self, reason: str) -> None:
        logger.warning("Auto-stop: %s", reason)
        if self.enabled:
            self.stop_statistics()

    def _setup_metrics(self) -> None:
        if self.metadata is None:
            return

        prometheus_logger = PrometheusLogger.GetOrCreate(
            self.metadata,
            config=self.config,
        )
        prometheus_logger.chunk_statistics_enabled.set_function(
            lambda: 1.0 if self.enabled else 0.0
        )
        prometheus_logger.chunk_statistics_total_requests.set_function(
            lambda: len(self.request_seen)
        )
        self.recorder.strategy.setup_metrics(prometheus_logger)
