# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
)
import os
import threading
import time

# Third Party
from prometheus_client import REGISTRY
import prometheus_client

# First Party
from lmcache.logging import init_logger
from lmcache.usage_telemetry import ContinuousUsageContext
from lmcache.utils import thread_safe
from lmcache.v1.metadata import LMCacheMetadata

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)


@dataclass
class LMCacheStats:
    # Counter (Note that these are incremental values,
    # which will accumulate over time in Counter)
    interval_retrieve_requests: int
    interval_store_requests: int
    interval_lookup_requests: int
    interval_requested_tokens: int
    interval_hit_tokens: int
    interval_stored_tokens: int
    interval_lookup_tokens: int
    interval_lookup_hits: int
    interval_vllm_hit_tokens: int
    interval_prompt_tokens: int

    interval_num_slow_retrieval_by_time: int
    interval_num_slow_retrieval_by_speed: int

    interval_remote_read_requests: int
    interval_remote_read_bytes: int
    interval_remote_write_requests: int
    interval_remote_write_bytes: int

    interval_remote_time_to_get: List[float]
    interval_remote_time_to_put: List[float]
    interval_remote_time_to_get_sync: List[float]

    interval_remote_ping_latency: float  # Ping latency in milliseconds
    interval_remote_ping_errors: int  # Number of ping errors
    interval_remote_ping_success: int  # Number of ping successes
    interval_remote_ping_error_code: int  # Latest ping error code

    interval_local_cpu_evict_count: int  # evict count
    interval_local_cpu_evict_keys_count: int  # evict keys count
    interval_local_cpu_evict_failed_count: int  # evict failed count

    interval_forced_unpin_count: int  # forced unpin count due to timeout

    # Real time value measurements (will be reset after each log)
    retrieve_hit_rate: float
    lookup_hit_rate: float

    local_cache_usage_bytes: int  # Size of the used local cache in bytes
    remote_cache_usage_bytes: int  # Size of the used remote cache in bytes
    local_storage_usage_bytes: int  # Size of the used local storage in bytes

    active_memory_objs_count: int  # the number of active memory objects
    pinned_memory_objs_count: int  # the number of pinned memory objects

    # Distribution measurements
    time_to_retrieve: List[float]
    time_to_store: List[float]
    time_to_lookup: List[float]
    retrieve_speed: List[float]  # Tokens per second
    store_speed: List[float]  # Tokens per second

    # Granular profiling measurements
    retrieve_process_tokens_time: List[float]
    retrieve_broadcast_time: List[float]
    retrieve_to_gpu_time: List[float]
    remote_backend_batched_get_blocking_time: List[float]
    instrumented_connector_batched_get_time: List[float]
    store_process_tokens_time: List[float]
    store_from_gpu_time: List[float]
    store_put_time: List[float]

    # P2P transfer metrics
    interval_p2p_requests: int
    interval_p2p_transferred_tokens: int
    p2p_time_to_transfer: List[float]
    p2p_transfer_speed: List[float]  # Tokens per second

    # request lookup hit rates
    # use bucket of interval_lookup_hit_rates to represents non-0 hit requests
    # use interval_lookup_0_hit_requests to represents 0 hit requests
    interval_lookup_hit_rates: List[float]
    interval_lookup_0_hit_requests: int

    interval_request_cache_lifespan: List[float]  # cache lifespan in minutes


@dataclass
class LookupRequestStats:
    request_id: int
    num_tokens: int
    hit_tokens: int
    is_finished: bool
    start_time: float = 0
    end_time: float = 0

    def time_to_lookup(self):
        if self.end_time == 0:
            return 0
        return self.end_time - self.start_time

    def hit_rate(self):
        if self.num_tokens == 0:
            return 0
        return self.hit_tokens / self.num_tokens


@dataclass
class RetrieveRequestStats:
    request_id: int
    num_tokens: int
    local_hit_tokens: int
    remote_hit_tokens: int  # Not used for now
    start_time: float
    end_time: float
    process_tokens_time: float = 0
    broadcast_time: float = 0
    to_gpu_time: float = 0
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)

    def time_to_retrieve(self):
        if self.end_time == 0:
            return 0
        return self.end_time - self.start_time

    def retrieve_speed(self):
        if self.time_to_retrieve() == 0:
            return 0
        return (
            self.local_hit_tokens + self.remote_hit_tokens
        ) / self.time_to_retrieve()

    @contextmanager
    def profile_process_tokens(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.process_tokens_time += time.perf_counter() - start

    @contextmanager
    def profile_broadcast(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.broadcast_time += time.perf_counter() - start

    @contextmanager
    def profile_to_gpu(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.to_gpu_time += time.perf_counter() - start


@dataclass
class StoreRequestStats:
    request_id: int
    num_tokens: int
    start_time: float
    end_time: float
    process_tokens_time: float = 0
    from_gpu_time: float = 0
    put_time: float = 0

    def time_to_store(self):
        if self.end_time == 0:
            return 0
        return self.end_time - self.start_time

    def store_speed(self):
        if self.time_to_store() == 0:
            return 0
        return self.num_tokens / self.time_to_store()

    @contextmanager
    def profile_process_tokens(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.process_tokens_time += time.perf_counter() - start

    @contextmanager
    def profile_from_gpu(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.from_gpu_time += time.perf_counter() - start

    @contextmanager
    def profile_put(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.put_time += time.perf_counter() - start


@dataclass
class P2PTransferRequestStats:
    num_tokens: int
    start_time: float
    end_time: float

    def time_to_transfer(self):
        if self.end_time == 0:
            return 0
        return self.end_time - self.start_time

    def transfer_speed(self):
        if self.time_to_transfer() == 0:
            return 0
        return self.num_tokens / self.time_to_transfer()


class LMCStatsMonitor:
    def __init__(self):
        # Interval metrics that will be reset after each log
        # Accumulate incremental values in the Prometheus Counter
        self.interval_retrieve_requests = 0
        self.interval_store_requests = 0
        self.interval_lookup_requests = 0
        self.interval_requested_tokens = 0  # total requested tokens retrieve
        self.interval_hit_tokens = 0  # total hit tokens retrieve
        self.interval_stored_tokens = 0  # total tokens tored in LMCache
        self.interval_lookup_tokens = 0  # total requested tokens lookup
        self.interval_lookup_hits = 0  # total hit tokens lookup
        self.interval_vllm_hit_tokens = 0  # total hit tokens in vllm
        self.interval_prompt_tokens = 0  # total prompt tokens
        self.interval_lookup_0_hit_requests = 0

        self.interval_num_slow_retrieval_by_time = 0
        self.interval_num_slow_retrieval_by_speed = 0

        # P2P transfer metrics
        self.interval_p2p_requests = 0
        self.interval_p2p_transferred_tokens = 0
        self.p2p_requests: Dict[int, P2PTransferRequestStats] = {}
        self.p2p_request_id = 0

        # remote backends read/write metrics
        self.interval_remote_read_requests = 0
        self.interval_remote_read_bytes = 0
        self.interval_remote_write_requests = 0
        self.interval_remote_write_bytes = 0

        # remote backends get/put cost time metrics
        self.interval_remote_time_to_get: List[float] = []
        self.interval_remote_time_to_put: List[float] = []
        # the time of get value from remote backends synchronously,
        # which includes rpc and schedule time
        self.interval_remote_time_to_get_sync: List[float] = []

        self.interval_remote_ping_latency = 0
        self.interval_remote_ping_errors = 0
        self.interval_remote_ping_success = 0
        self.interval_remote_ping_error_code = 0  # 0 means success

        self.interval_local_cpu_evict_count = 0
        self.interval_local_cpu_evict_keys_count = 0
        self.interval_local_cpu_evict_failed_count = 0

        self.interval_forced_unpin_count = 0

        self.local_cache_usage_bytes = 0
        self.remote_cache_usage_bytes = 0
        self.local_storage_usage_bytes = 0

        self.active_memory_objs_count = 0
        self.pinned_memory_objs_count = 0

        self.retrieve_requests: Dict[int, RetrieveRequestStats] = {}
        self.store_requests: Dict[int, StoreRequestStats] = {}
        self.lookup_requests: Dict[int, LookupRequestStats] = {}

        self.retrieve_request_id = 0
        self.store_request_id = 0
        self.lookup_request_id = 0

        self.interval_request_cache_lifespan: Dict[int, float] = {}
        self.reuse_chunk_id = 0

        self._current_retrieve_stats: Optional[RetrieveRequestStats] = None
        self.retrieve_time_threshold: float = 1e9
        self.retrieve_token_speed_threshold: float = -1.0
        self.last_retrieve_warning_time: float = 0.0
        self.skipped_retrieve_warning_count: int = 0

    def set_current_retrieve_stats(self, stats: RetrieveRequestStats):
        self._current_retrieve_stats = stats

    def get_current_retrieve_stats(self) -> Optional[RetrieveRequestStats]:
        return self._current_retrieve_stats

    def clear_current_retrieve_stats(self):
        self._current_retrieve_stats = None

    @thread_safe
    def on_lookup_request(self, num_tokens: int) -> LookupRequestStats:
        """
        This function is called when a lookup request is sent to the cache.
        It will record the number of tokens requested.
        """
        curr_time = time.perf_counter()
        lookup_stats = LookupRequestStats(
            request_id=self.lookup_request_id,
            num_tokens=num_tokens,
            hit_tokens=0,
            is_finished=False,
            start_time=curr_time,
        )
        self.interval_lookup_requests += 1
        self.interval_lookup_tokens += num_tokens
        self.lookup_requests[self.lookup_request_id] = lookup_stats
        self.lookup_request_id += 1
        return lookup_stats

    @thread_safe
    def on_lookup_finished(
        self,
        stats: LookupRequestStats,
        num_hit_tokens: int,
    ):
        """
        This function is called when a lookup request is finished.
        It will record the number of tokens hit and track by node type.

        Args:
            stats: LookupRequestStats object
            num_hit_tokens: Total number of tokens found in lookup
        """
        curr_time = time.perf_counter()
        assert stats.request_id in self.lookup_requests
        stats.hit_tokens = num_hit_tokens
        stats.is_finished = True
        if stats.end_time == 0:
            stats.end_time = curr_time
        self.interval_lookup_hits += num_hit_tokens
        if num_hit_tokens == 0:
            self.interval_lookup_0_hit_requests += 1

    @thread_safe
    def on_retrieve_request(self, num_tokens: int) -> RetrieveRequestStats:
        """
        Returns the internal "request id" that will be used in
        on_retrieve_finished
        """
        curr_time = time.perf_counter()
        retrieve_stats = RetrieveRequestStats(
            request_id=self.retrieve_request_id,
            num_tokens=num_tokens,
            local_hit_tokens=0,
            remote_hit_tokens=0,
            start_time=curr_time,
            end_time=0,
        )
        self.interval_requested_tokens += num_tokens
        self.interval_retrieve_requests += 1
        self.retrieve_requests[self.retrieve_request_id] = retrieve_stats
        self.retrieve_request_id += 1
        self.set_current_retrieve_stats(retrieve_stats)
        return retrieve_stats

    @thread_safe
    def on_retrieve_finished(
        self,
        retrieve_stats: RetrieveRequestStats,
        num_retrieved_tokens: int,
    ):
        curr_time = time.perf_counter()
        assert retrieve_stats.request_id in self.retrieve_requests
        retrieve_stats.local_hit_tokens = num_retrieved_tokens
        if retrieve_stats.end_time == 0:
            retrieve_stats.end_time = curr_time
        self.interval_hit_tokens += num_retrieved_tokens
        self.clear_current_retrieve_stats()

        time_to_retrieve = retrieve_stats.time_to_retrieve()
        retrieve_speed = retrieve_stats.retrieve_speed()
        if time_to_retrieve > self.retrieve_time_threshold:
            self.interval_num_slow_retrieval_by_time += 1
        if 0 < retrieve_speed < self.retrieve_token_speed_threshold:
            self.interval_num_slow_retrieval_by_speed += 1

        # Log a warning if the retrieval performance is below defined thresholds:
        # 1. Total time taken (time_to_retrieve) exceeds the maximum allowed time.
        # 2. Retrieval speed (retrieve_speed) falls below the minimum required tokens/s.
        # The warnings are rate-limited to once every 10 seconds to avoid log flooding.
        if (
            time_to_retrieve > self.retrieve_time_threshold
            or 0 < retrieve_speed < self.retrieve_token_speed_threshold
        ):
            if curr_time - self.last_retrieve_warning_time > 10.0:
                logger.warning(
                    "Retrieve request %d surpassed threshold: "
                    "time_to_retrieve=%.4f s (threshold=%.4f s), "
                    "retrieve_speed=%.2f tokens/s (threshold=%.2f tokens/s). "
                    "Skipped %d slow retrieval logs in the last %.1f seconds. "
                    "Detailed metrics: "
                    "num_tokens=%d, local_hit_tokens=%d, remote_hit_tokens=%d, "
                    "process_tokens_time=%.5f s, "
                    "broadcast_time=%.5f s, "
                    "to_gpu_time=%.5f s, "
                    "detailed_metrics=%s",
                    retrieve_stats.request_id,
                    time_to_retrieve,
                    self.retrieve_time_threshold,
                    retrieve_speed,
                    self.retrieve_token_speed_threshold,
                    self.skipped_retrieve_warning_count,
                    curr_time - self.last_retrieve_warning_time,
                    retrieve_stats.num_tokens,
                    retrieve_stats.local_hit_tokens,
                    retrieve_stats.remote_hit_tokens,
                    retrieve_stats.process_tokens_time,
                    retrieve_stats.broadcast_time,
                    retrieve_stats.to_gpu_time,
                    retrieve_stats.detailed_metrics,
                )
                self.last_retrieve_warning_time = curr_time
                self.skipped_retrieve_warning_count = 0
            else:
                self.skipped_retrieve_warning_count += 1

    @thread_safe
    def on_store_request(self, num_tokens: int) -> StoreRequestStats:
        """
        Returns the internal "request id" that will be used in on_store_finished
        """
        curr_time = time.perf_counter()
        store_stats = StoreRequestStats(
            request_id=self.store_request_id,
            num_tokens=num_tokens,
            start_time=curr_time,
            end_time=0,
        )
        self.interval_store_requests += 1
        self.interval_stored_tokens += num_tokens
        self.store_requests[self.store_request_id] = store_stats
        self.store_request_id += 1
        return store_stats

    @thread_safe
    def on_store_finished(
        self,
        store_stats: StoreRequestStats,
        num_stored_tokens: int = -1,
    ):
        curr_time = time.perf_counter()
        assert store_stats.request_id in self.store_requests
        if store_stats.end_time == 0:
            store_stats.end_time = curr_time
        if num_stored_tokens >= 0:
            store_stats.num_tokens = num_stored_tokens

    @thread_safe
    def on_p2p_transfer_request(self, num_tokens: int) -> int:
        curr_time = time.time()
        self.interval_p2p_requests += 1
        self.p2p_requests[self.p2p_request_id] = P2PTransferRequestStats(
            num_tokens=num_tokens,
            start_time=curr_time,
            end_time=0,
        )
        self.p2p_request_id += 1
        return self.p2p_request_id - 1

    @thread_safe
    def on_p2p_transfer_finished(self, request_id: int):
        curr_time = time.time()
        assert request_id in self.p2p_requests
        p2p_stats = self.p2p_requests[request_id]
        self.interval_p2p_transferred_tokens += p2p_stats.num_tokens
        p2p_stats.end_time = curr_time

    @thread_safe
    def on_chunk_reuse(self, time_interval: float):
        """
        time_interval: float or int, in seconds
        """
        self.interval_request_cache_lifespan[self.reuse_chunk_id] = time_interval / 60.0
        self.reuse_chunk_id += 1

    @thread_safe
    def update_local_cache_usage(self, usage: int):
        self.local_cache_usage_bytes = usage

    @thread_safe
    def update_remote_cache_usage(self, usage: int):
        self.remote_cache_usage_bytes = usage

    @thread_safe
    def update_local_storage_usage(self, usage: int):
        self.local_storage_usage_bytes = usage

    @thread_safe
    def update_interval_remote_read_metrics(self, read_bytes: int):
        self.interval_remote_read_requests += 1
        self.interval_remote_read_bytes += read_bytes

    @thread_safe
    def update_interval_remote_write_metrics(self, write_bytes: int):
        self.interval_remote_write_requests += 1
        self.interval_remote_write_bytes += write_bytes

    @thread_safe
    def update_interval_remote_time_to_get(self, get_time: float):
        self.interval_remote_time_to_get.append(get_time)

    @thread_safe
    def update_interval_remote_time_to_put(self, put_time: float):
        self.interval_remote_time_to_put.append(put_time)

    @thread_safe
    def update_interval_remote_time_to_get_sync(self, get_time_sync: float):
        self.interval_remote_time_to_get_sync.append(get_time_sync)

    @thread_safe
    def update_remote_ping_latency(self, latency: float):
        self.interval_remote_ping_latency = latency

    @thread_safe
    def update_remote_ping_error_code(self, error_code: int):
        """Update ping error code"""
        self.interval_remote_ping_error_code = error_code
        if error_code != 0:
            self.interval_remote_ping_errors += 1
        else:
            self.interval_remote_ping_success += 1

    @thread_safe
    def update_local_cpu_evict_metrics(self, evict_keys_count: int):
        self.interval_local_cpu_evict_count += 1
        self.interval_local_cpu_evict_keys_count += evict_keys_count

    @thread_safe
    def update_local_cpu_evict_failed_count(self, evict_failed_count: int):
        self.interval_local_cpu_evict_failed_count += evict_failed_count

    @thread_safe
    def update_forced_unpin_count(self, delta: int):
        self.interval_forced_unpin_count += delta

    @thread_safe
    def update_active_memory_objs_count(self, active_memory_objs_count: int):
        self.active_memory_objs_count = active_memory_objs_count

    @thread_safe
    def update_pinned_memory_objs_count(self, delta: int):
        self.pinned_memory_objs_count += delta

    @thread_safe
    def update_interval_vllm_hit_tokens(self, delta: int):
        self.interval_vllm_hit_tokens += delta

    @thread_safe
    def update_interval_prompt_tokens(self, delta: int):
        self.interval_prompt_tokens += delta

    def _clear(self):
        """
        Clear all the distribution stats
        """
        self.interval_retrieve_requests = 0
        self.interval_store_requests = 0
        self.interval_lookup_requests = 0

        self.interval_requested_tokens = 0
        self.interval_hit_tokens = 0
        self.interval_stored_tokens = 0
        self.interval_lookup_tokens = 0
        self.interval_lookup_hits = 0
        self.interval_vllm_hit_tokens = 0
        self.interval_prompt_tokens = 0

        self.interval_num_slow_retrieval_by_time = 0
        self.interval_num_slow_retrieval_by_speed = 0

        self.interval_remote_read_requests = 0
        self.interval_remote_read_bytes = 0
        self.interval_remote_write_requests = 0
        self.interval_remote_write_bytes = 0

        self.interval_remote_time_to_get.clear()
        self.interval_remote_time_to_put.clear()
        self.interval_remote_time_to_get_sync.clear()

        self.interval_remote_ping_latency = 0
        self.interval_remote_ping_errors = 0
        self.interval_remote_ping_success = 0
        self.interval_remote_ping_error_code = 0

        self.interval_local_cpu_evict_count = 0
        self.interval_local_cpu_evict_keys_count = 0
        self.interval_local_cpu_evict_failed_count = 0

        self.interval_forced_unpin_count = 0

        self.interval_p2p_requests = 0
        self.interval_p2p_transferred_tokens = 0

        self.interval_lookup_0_hit_requests = 0

        new_retrieve_requests = {}
        for request_id, retrieve_stats in self.retrieve_requests.items():
            if retrieve_stats.end_time == 0:
                new_retrieve_requests[request_id] = retrieve_stats
        self.retrieve_requests = new_retrieve_requests

        new_store_requests = {}
        for request_id, store_stats in self.store_requests.items():
            if store_stats.end_time == 0:
                new_store_requests[request_id] = store_stats
        self.store_requests = new_store_requests

        new_p2p_requests = {}
        for request_id, p2p_stats in self.p2p_requests.items():
            if p2p_stats.end_time == 0:
                new_p2p_requests[request_id] = p2p_stats
        self.p2p_requests = new_p2p_requests

        new_lookup_requests = {}
        for request_id, lookup_stats in self.lookup_requests.items():
            if not lookup_stats.is_finished:
                new_lookup_requests[request_id] = lookup_stats
        self.lookup_requests = new_lookup_requests

        self.interval_request_cache_lifespan.clear()
        self.reuse_chunk_id = 0

    @thread_safe
    def get_stats_and_clear(self) -> LMCacheStats:
        """
        This function should be called with by prometheus adapter with
        a specific interval.
        The function will return the latest states between the current
        call and the previous call.
        """
        # Calculate retrieve hit rate based on requests finished in this interval
        finished_retrieve_stats = [
            s for s in self.retrieve_requests.values() if s.end_time != 0
        ]
        sum_finished_retrieve_requested_tokens = sum(
            s.num_tokens for s in finished_retrieve_stats
        )
        sum_finished_retrieve_hit_tokens = sum(
            s.local_hit_tokens + s.remote_hit_tokens for s in finished_retrieve_stats
        )
        retrieve_hit_rate = (
            1
            if len(finished_retrieve_stats) == 0
            or sum_finished_retrieve_requested_tokens == 0
            else sum_finished_retrieve_hit_tokens
            / sum_finished_retrieve_requested_tokens
        )

        # Calculate lookup hit rate based on requests finished in this interval
        finished_lookup_stats = [
            s for s in self.lookup_requests.values() if s.is_finished
        ]
        sum_finished_lookup_requested = sum(s.num_tokens for s in finished_lookup_stats)
        sum_finished_lookup_hit = sum(s.hit_tokens for s in finished_lookup_stats)
        lookup_hit_rate = (
            0
            if sum_finished_lookup_requested == 0
            else sum_finished_lookup_hit / sum_finished_lookup_requested
        )

        def filter_out_zeros(stats: Iterable[float]) -> List[float]:
            return [x for x in stats if x != 0]

        time_to_retrieve = filter_out_zeros(
            stats.time_to_retrieve() for stats in self.retrieve_requests.values()
        )

        time_to_store = filter_out_zeros(
            stats.time_to_store() for stats in self.store_requests.values()
        )

        time_to_lookup = filter_out_zeros(
            stats.time_to_lookup() for stats in self.lookup_requests.values()
        )

        retrieve_speed = filter_out_zeros(
            stats.retrieve_speed() for stats in self.retrieve_requests.values()
        )

        store_speed = filter_out_zeros(
            stats.store_speed() for stats in self.store_requests.values()
        )

        # Granular profiling measurements
        retrieve_process_tokens_time = filter_out_zeros(
            stats.process_tokens_time for stats in self.retrieve_requests.values()
        )
        retrieve_broadcast_time = filter_out_zeros(
            stats.broadcast_time for stats in self.retrieve_requests.values()
        )
        retrieve_to_gpu_time = filter_out_zeros(
            stats.to_gpu_time for stats in self.retrieve_requests.values()
        )
        remote_backend_batched_get_blocking_time = filter_out_zeros(
            stats.detailed_metrics.get("remote_backend_batched_get_blocking_time", 0.0)
            for stats in self.retrieve_requests.values()
        )
        instrumented_connector_batched_get_time = filter_out_zeros(
            stats.detailed_metrics.get("instrumented_connector_batched_get_time", 0.0)
            for stats in self.retrieve_requests.values()
        )
        store_process_tokens_time = filter_out_zeros(
            stats.process_tokens_time for stats in self.store_requests.values()
        )
        store_from_gpu_time = filter_out_zeros(
            stats.from_gpu_time for stats in self.store_requests.values()
        )
        store_put_time = filter_out_zeros(
            stats.put_time for stats in self.store_requests.values()
        )

        p2p_time_to_transfer = filter_out_zeros(
            stats.time_to_transfer() for stats in self.p2p_requests.values()
        )

        p2p_transfer_speed = filter_out_zeros(
            stats.transfer_speed() for stats in self.p2p_requests.values()
        )

        request_lookup_hit_rates = filter_out_zeros(
            stats.hit_rate()
            for stats in self.lookup_requests.values()
            if stats.is_finished
        )

        request_lifespan = list(self.interval_request_cache_lifespan.values())

        ret = LMCacheStats(
            interval_retrieve_requests=self.interval_retrieve_requests,
            interval_store_requests=self.interval_store_requests,
            interval_lookup_requests=self.interval_lookup_requests,
            interval_requested_tokens=self.interval_requested_tokens,
            interval_hit_tokens=self.interval_hit_tokens,
            interval_stored_tokens=self.interval_stored_tokens,
            interval_lookup_tokens=self.interval_lookup_tokens,
            interval_lookup_hits=self.interval_lookup_hits,
            interval_remote_read_requests=self.interval_remote_read_requests,
            interval_remote_read_bytes=self.interval_remote_read_bytes,
            interval_remote_write_requests=self.interval_remote_write_requests,
            interval_remote_write_bytes=self.interval_remote_write_bytes,
            interval_remote_time_to_get=self.interval_remote_time_to_get.copy(),
            interval_remote_time_to_put=self.interval_remote_time_to_put.copy(),
            interval_remote_time_to_get_sync=self.interval_remote_time_to_get_sync.copy(),
            interval_remote_ping_latency=self.interval_remote_ping_latency,
            interval_remote_ping_errors=self.interval_remote_ping_errors,
            interval_remote_ping_success=self.interval_remote_ping_success,
            interval_remote_ping_error_code=self.interval_remote_ping_error_code,
            retrieve_hit_rate=retrieve_hit_rate,
            lookup_hit_rate=lookup_hit_rate,
            interval_local_cpu_evict_count=self.interval_local_cpu_evict_count,
            interval_local_cpu_evict_keys_count=self.interval_local_cpu_evict_keys_count,
            interval_local_cpu_evict_failed_count=self.interval_local_cpu_evict_failed_count,
            interval_forced_unpin_count=self.interval_forced_unpin_count,
            local_cache_usage_bytes=self.local_cache_usage_bytes,
            remote_cache_usage_bytes=self.remote_cache_usage_bytes,
            local_storage_usage_bytes=self.local_storage_usage_bytes,
            active_memory_objs_count=self.active_memory_objs_count,
            pinned_memory_objs_count=self.pinned_memory_objs_count,
            time_to_retrieve=time_to_retrieve,
            time_to_store=time_to_store,
            time_to_lookup=time_to_lookup,
            retrieve_speed=retrieve_speed,
            store_speed=store_speed,
            retrieve_process_tokens_time=retrieve_process_tokens_time,
            retrieve_broadcast_time=retrieve_broadcast_time,
            retrieve_to_gpu_time=retrieve_to_gpu_time,
            remote_backend_batched_get_blocking_time=remote_backend_batched_get_blocking_time,  # noqa: E501
            instrumented_connector_batched_get_time=instrumented_connector_batched_get_time,  # noqa: E501
            store_process_tokens_time=store_process_tokens_time,
            store_from_gpu_time=store_from_gpu_time,
            store_put_time=store_put_time,
            interval_vllm_hit_tokens=self.interval_vllm_hit_tokens,
            interval_p2p_requests=self.interval_p2p_requests,
            interval_num_slow_retrieval_by_time=self.interval_num_slow_retrieval_by_time,
            interval_num_slow_retrieval_by_speed=self.interval_num_slow_retrieval_by_speed,
            interval_p2p_transferred_tokens=self.interval_p2p_transferred_tokens,
            p2p_time_to_transfer=p2p_time_to_transfer,
            p2p_transfer_speed=p2p_transfer_speed,
            interval_lookup_hit_rates=request_lookup_hit_rates,
            interval_request_cache_lifespan=request_lifespan,
            interval_prompt_tokens=self.interval_prompt_tokens,
            interval_lookup_0_hit_requests=self.interval_lookup_0_hit_requests,
        )
        self._clear()
        return ret

    _instance = None

    @staticmethod
    def GetOrCreate() -> "LMCStatsMonitor":
        if LMCStatsMonitor._instance is None:
            LMCStatsMonitor._instance = LMCStatsMonitor()
        return LMCStatsMonitor._instance

    @staticmethod
    def DestroyInstance():
        LMCStatsMonitor._instance = None

    @staticmethod
    def unregister_all_metrics():
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            try:
                REGISTRY.unregister(collector)
            except KeyError:
                pass


class PrometheusLogger:
    lmcache_is_healthy: prometheus_client.Gauge
    periodic_threads_total_count: prometheus_client.Gauge
    periodic_threads_running_count: prometheus_client.Gauge
    periodic_threads_active_count: prometheus_client.Gauge

    _gauge_cls = prometheus_client.Gauge
    _counter_cls = prometheus_client.Counter
    _histogram_cls = prometheus_client.Histogram

    def _create_counter(
        self, name: str, documentation: str, labelnames: List[str]
    ) -> prometheus_client.Counter:
        """Create a Counter and register it for reset_counters()."""
        counter = self._counter_cls(
            name=name, documentation=documentation, labelnames=labelnames
        )
        self._counters.append(counter)
        return counter

    def _create_histogram(
        self,
        name: str,
        documentation: str,
        labelnames: List[str],
        buckets: Sequence[float],
    ) -> prometheus_client.Histogram:
        """Create a Histogram and register it for reset_histograms().

        If the ``config`` object's extra_config contains a key
        ``histogram_bucket_<short_name>`` (where ``<short_name>``
        is the metric name without the ``lmcache:`` prefix),
        the value will be used as the bucket list, overriding
        the default *buckets* argument.
        """
        short_name = name.split(":", 1)[-1]
        config_key = "histogram_bucket_%s" % short_name
        custom = (
            self.config.get_extra_config_value(config_key)
            if self.config is not None
            else None
        )
        if custom is not None:
            buckets = custom
            logger.info(
                "Using custom buckets for histogram %s from extra_config key '%s'",
                name,
                config_key,
            )
        histogram = self._histogram_cls(
            name=name,
            documentation=documentation,
            labelnames=labelnames,
            buckets=buckets,
        )
        self._histograms.append(histogram)
        return histogram

    @staticmethod
    def _ensure_multiprocess_dir() -> None:
        multiprocess_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
        if not multiprocess_dir:
            multiprocess_dir = "/tmp/lmcache_prometheus"
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = multiprocess_dir
        os.makedirs(multiprocess_dir, exist_ok=True)

    def __init__(
        self,
        metadata: LMCacheMetadata,
        config: Optional["LMCacheEngineConfig"] = None,
    ):
        # Ensure PROMETHEUS_MULTIPROC_DIR is set before any metric registration
        PrometheusLogger._ensure_multiprocess_dir()

        self.metadata = metadata
        self.config = config

        self.labels = self._metadata_to_labels(metadata)
        labelnames = list(self.labels.keys())

        # List to track all counters/histograms for reset methods
        self._counters: List[prometheus_client.Counter] = []
        self._histograms: List[prometheus_client.Histogram] = []

        self.counter_num_retrieve_requests = self._create_counter(
            name="lmcache:num_retrieve_requests",
            documentation="Total number of retrieve requests sent to lmcache",
            labelnames=labelnames,
        )

        self.counter_num_store_requests = self._create_counter(
            name="lmcache:num_store_requests",
            documentation="Total number of store requests sent to lmcache",
            labelnames=labelnames,
        )

        self.counter_num_lookup_requests = self._create_counter(
            name="lmcache:num_lookup_requests",
            documentation="Total number of lookup requests sent to lmcache",
            labelnames=labelnames,
        )

        self.counter_num_requested_tokens = self._create_counter(
            name="lmcache:num_requested_tokens",
            documentation="Total number of tokens requested from lmcache",
            labelnames=labelnames,
        )

        self.counter_num_hit_tokens = self._create_counter(
            name="lmcache:num_hit_tokens",
            documentation="Total number of tokens hit in lmcache",
            labelnames=labelnames,
        )

        self.counter_num_stored_tokens = self._create_counter(
            name="lmcache:num_stored_tokens",
            documentation=(
                "Total number of tokens stored in lmcache including evicted ones"
            ),
            labelnames=labelnames,
        )

        self.counter_num_lookup_tokens = self._create_counter(
            name="lmcache:num_lookup_tokens",
            documentation="Total number of tokens requested in lookup from lmcache",
            labelnames=labelnames,
        )

        self.counter_num_lookup_hits = self._create_counter(
            name="lmcache:num_lookup_hits",
            documentation="Total number of tokens hit in lookup from lmcache",
            labelnames=labelnames,
        )

        self.counter_num_vllm_hit_tokens = self._create_counter(
            name="lmcache:num_vllm_hit_tokens",
            documentation="Number of hit tokens in vllm",
            labelnames=labelnames,
        )

        self.counter_num_prompt_tokens = self._create_counter(
            name="lmcache:num_prompt_tokens",
            documentation="Number of prompt tokens in lmcache",
            labelnames=labelnames,
        )

        self.counter_num_remote_read_requests = self._create_counter(
            name="lmcache:num_remote_read_requests",
            documentation="Total number of requests read from "
            "remote backends in lmcache",
            labelnames=labelnames,
        )

        self.counter_num_remote_read_bytes = self._create_counter(
            name="lmcache:num_remote_read_bytes",
            documentation="Total number of bytes read from remote backends in lmcache",
            labelnames=labelnames,
        )

        self.counter_num_remote_write_requests = self._create_counter(
            name="lmcache:num_remote_write_requests",
            documentation="Total number of requests write to "
            "remote backends in lmcache",
            labelnames=labelnames,
        )

        self.counter_num_remote_write_bytes = self._create_counter(
            name="lmcache:num_remote_write_bytes",
            documentation="Total number of bytes write to remote backends in lmcache",
            labelnames=labelnames,
        )

        self.counter_local_cpu_evict_count = self._create_counter(
            name="lmcache:local_cpu_evict_count",
            documentation="Total number of evict in local cpu backend",
            labelnames=labelnames,
        )

        self.counter_local_cpu_evict_keys_count = self._create_counter(
            name="lmcache:local_cpu_evict_keys_count",
            documentation="Total number of evict keys in local cpu backend",
            labelnames=labelnames,
        )

        self.counter_local_cpu_evict_failed_count = self._create_counter(
            name="lmcache:local_cpu_evict_failed_count",
            documentation="Total number of failed eviction in local cpu backend",
            labelnames=labelnames,
        )

        self.counter_forced_unpin_count = self._create_counter(
            name="lmcache:forced_unpin_count",
            documentation="Total number of forced unpin due to timeout",
            labelnames=labelnames,
        )

        self.counter_lookup_0_hit_requests = self._create_counter(
            name="lmcache:lookup_0_hit_requests",
            documentation="Total number of 0 hit lookup requests",
            labelnames=labelnames,
        )

        self.counter_num_slow_retrieval_by_time = self._create_counter(
            name="lmcache:num_slow_retrieval_by_time",
            documentation="Total number of slow retrievals by time threshold",
            labelnames=labelnames,
        )

        self.counter_num_slow_retrieval_by_speed = self._create_counter(
            name="lmcache:num_slow_retrieval_by_speed",
            documentation="Total number of slow retrievals by speed threshold",
            labelnames=labelnames,
        )

        self.gauge_retrieve_hit_rate = self._gauge_cls(
            name="lmcache:retrieve_hit_rate",
            documentation="Hit rate of lmcache retrieve requests since last log",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )

        self.gauge_lookup_hit_rate = self._gauge_cls(
            name="lmcache:lookup_hit_rate",
            documentation="Hit rate of lmcache lookup requests since last log",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )

        self.gauge_local_cache_usage = self._gauge_cls(
            name="lmcache:local_cache_usage",
            documentation="Local cache usage (bytes) of lmcache",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )

        self.gauge_remote_cache_usage = self._gauge_cls(
            name="lmcache:remote_cache_usage",
            documentation="Remote cache usage (bytes) of lmcache",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )

        self.gauge_local_storage_usage = self._gauge_cls(
            name="lmcache:local_storage_usage",
            documentation="Local storage usage (bytes) of lmcache",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )

        self.gauge_active_memory_objs_count = self._gauge_cls(
            name="lmcache:active_memory_objs_count",
            documentation="The number of active memory objects",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )

        self.gauge_pinned_memory_objs_count = self._gauge_cls(
            name="lmcache:pinned_memory_objs_count",
            documentation="The number of pinned memory objects",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )

        time_to_retrieve_buckets = [
            0.001,
            0.005,
            0.01,
            0.02,
            0.04,
            0.06,
            0.08,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
        ]
        self.histogram_time_to_retrieve = self._create_histogram(
            name="lmcache:time_to_retrieve",
            documentation="Time to retrieve from lmcache (seconds)",
            labelnames=labelnames,
            buckets=time_to_retrieve_buckets,
        )

        time_to_store_buckets = [
            0.001,
            0.005,
            0.01,
            0.02,
            0.04,
            0.06,
            0.08,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
        ]
        self.histogram_time_to_store = self._create_histogram(
            name="lmcache:time_to_store",
            documentation="Time to store to lmcache (seconds)",
            labelnames=labelnames,
            buckets=time_to_store_buckets,
        )

        time_to_lookup_buckets = [
            0.00001 * 2**i for i in range(20)
        ]  # 0.01 ms to 5000 ms
        self.histogram_time_to_lookup = self._create_histogram(
            name="lmcache:time_to_lookup",
            documentation="Time to lookup in lmcache (seconds)",
            labelnames=labelnames,
            buckets=time_to_lookup_buckets,
        )

        profiling_buckets = [0.00001 * 2**i for i in range(20)]  # 0.01 ms to 5000 ms
        self.histogram_retrieve_process_tokens_time = self._create_histogram(
            name="lmcache:retrieve_process_tokens_time",
            documentation="Time to process tokens in retrieve (seconds)",
            labelnames=labelnames,
            buckets=profiling_buckets,
        )
        self.histogram_retrieve_broadcast_time = self._create_histogram(
            name="lmcache:retrieve_broadcast_time",
            documentation="Time to broadcast memory objects in retrieve (seconds)",
            labelnames=labelnames,
            buckets=profiling_buckets,
        )
        self.histogram_retrieve_to_gpu_time = self._create_histogram(
            name="lmcache:retrieve_to_gpu_time",
            documentation="Time to move data to GPU in retrieve (seconds)",
            labelnames=labelnames,
            buckets=profiling_buckets,
        )
        self.histogram_remote_backend_batched_get_blocking_time = (
            self._create_histogram(
                name="lmcache:remote_backend_batched_get_blocking_time",
                documentation="Time to get data from remote backend (seconds)",
                labelnames=labelnames,
                buckets=profiling_buckets,
            )
        )
        self.histogram_instrumented_connector_batched_get_time = self._create_histogram(
            name="lmcache:instrumented_connector_batched_get_time",
            documentation="Time used by the connector (seconds)",
            labelnames=labelnames,
            buckets=profiling_buckets,
        )
        self.histogram_store_process_tokens_time = self._create_histogram(
            name="lmcache:store_process_tokens_time",
            documentation="Time to process tokens in store (seconds)",
            labelnames=labelnames,
            buckets=profiling_buckets,
        )
        self.histogram_store_from_gpu_time = self._create_histogram(
            name="lmcache:store_from_gpu_time",
            documentation="Time to move data from GPU in store (seconds)",
            labelnames=labelnames,
            buckets=profiling_buckets,
        )
        self.histogram_store_put_time = self._create_histogram(
            name="lmcache:store_put_time",
            documentation="Time to put data to storage in store (seconds)",
            labelnames=labelnames,
            buckets=profiling_buckets,
        )

        retrieve_speed_buckets = [
            1,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        ]
        self.histogram_retrieve_speed = self._create_histogram(
            name="lmcache:retrieve_speed",
            documentation="Retrieve speed of lmcache (tokens per second)",
            labelnames=labelnames,
            buckets=retrieve_speed_buckets,
        )

        store_speed_buckets = [
            1,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        ]
        self.histogram_store_speed = self._create_histogram(
            name="lmcache:store_speed",
            documentation="Store speed of lmcache (tokens per second)",
            labelnames=labelnames,
            buckets=store_speed_buckets,
        )

        # P2P transfer metrics
        p2p_time_buckets = [
            0.001,  # 1ms
            0.005,  # 5ms
            0.01,  # 10ms
            0.02,  # 20ms
            0.04,  # 40ms
            0.06,  # 60ms
            0.08,  # 80ms
            0.1,  # 100ms
            0.25,  # 250ms
            0.5,  # 500ms
            0.75,  # 750ms
            1.0,  # 1s
            2.5,  # 2.5s
            5.0,  # 5s
            7.5,  # 7.5s
            10.0,  # 10s
        ]
        self.histogram_p2p_time_to_transfer = self._create_histogram(
            name="lmcache:p2p_time_to_transfer",
            documentation="Time to transfer via P2P (seconds)",
            labelnames=labelnames,
            buckets=p2p_time_buckets,
        )

        p2p_speed_buckets = [
            1,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        ]
        self.histogram_p2p_transfer_speed = self._create_histogram(
            name="lmcache:p2p_transfer_speed",
            documentation="P2P transfer speed (tokens per second)",
            labelnames=labelnames,
            buckets=p2p_speed_buckets,
        )

        remote_time_to_get = [
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
            7500,
            10000,
        ]
        self.histogram_remote_time_to_get = self._create_histogram(
            name="lmcache:remote_time_to_get",
            documentation="Time to get from remote backends (ms)",
            labelnames=labelnames,
            buckets=remote_time_to_get,
        )

        remote_time_to_put = [
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
            7500,
            10000,
        ]
        self.histogram_remote_time_to_put = self._create_histogram(
            name="lmcache:remote_time_to_put",
            documentation="Time to put to remote backends (ms)",
            labelnames=labelnames,
            buckets=remote_time_to_put,
        )

        remote_time_to_get_sync = [
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
            7500,
            10000,
        ]
        self.histogram_remote_time_to_get_sync = self._create_histogram(
            name="lmcache:remote_time_to_get_sync",
            documentation="Time to get from remote backends synchronously(ms)",
            labelnames=labelnames,
            buckets=remote_time_to_get_sync,
        )

        request_cache_hit_rate = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ]
        self.histogram_request_cache_hit_rate = self._create_histogram(
            name="lmcache:request_cache_hit_rate",
            documentation="Request cache hit rate",
            labelnames=labelnames,
            buckets=request_cache_hit_rate,
        )

        request_cache_lifespan_buckets = [
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
        self.histogram_request_cache_lifespan = self._create_histogram(
            name="lmcache:request_cache_lifespan",
            documentation="Request cache lifespan in minutes",
            labelnames=labelnames,
            buckets=request_cache_lifespan_buckets,
        )

        # Ping latency metrics: use a gauge to record the latest ping latency
        self.gauge_remote_ping_latency = self._gauge_cls(
            name="lmcache:remote_ping_latency",
            documentation="Latest ping latency to remote backends (ms)",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self.counter_remote_ping_errors = self._create_counter(
            name="lmcache:remote_ping_errors",
            documentation="Number of ping errors to remote backends",
            labelnames=labelnames,
        )
        self.counter_remote_ping_successes = self._create_counter(
            name="lmcache:remote_ping_successes",
            documentation="Number of ping successes to remote backends",
            labelnames=labelnames,
        )
        self.gauge_remote_ping_error_code = self._gauge_cls(
            name="lmcache:remote_ping_error_code",
            documentation="Latest ping error code to remote backends",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self._dynamic_metrics(labelnames)

    def _create_dynamic_gauge(
        self,
        name: str,
        documentation: str,
        labelnames: List[str],
        multiprocess_mode: str,
    ) -> None:
        metric_attr = name.removeprefix("lmcache:")
        if metric_attr == name or not metric_attr.isidentifier():
            raise ValueError(f"Invalid dynamic metric name: {name}")

        gauge = self._gauge_cls(
            name=name,
            documentation=documentation,
            labelnames=labelnames,
            multiprocess_mode=multiprocess_mode,
        )
        # Store the shared collector separately from the labeled child.
        # Shallow-copied label views reuse the collector, then call labels()
        # with their own labels so set_function callbacks publish to that view.
        self._dynamic_gauge_collectors[metric_attr] = gauge
        setattr(self, metric_attr, gauge.labels(**self.labels))

    def _bind_dynamic_metric_children(self) -> None:
        PrometheusLogger._ensure_multiprocess_dir()
        for metric_attr, gauge in self._dynamic_gauge_collectors.items():
            setattr(self, metric_attr, gauge.labels(**self.labels))

    def _dynamic_metrics(self, labelnames: List[str]) -> None:
        """
        Dynamically get value by lambda function while capture
        """
        self._dynamic_gauge_collectors: Dict[str, prometheus_client.Gauge] = {}
        self._create_dynamic_gauge(
            name="lmcache:local_cpu_hot_cache_count",
            documentation="The size of the hot_cache",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self._create_dynamic_gauge(
            name="lmcache:local_cpu_keys_in_request_count",
            documentation="The size of the keys_in_request",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self._create_dynamic_gauge(
            name="lmcache:kv_msg_queue_size",
            documentation="The size of the KV message queue in BatchedMessageSender",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self._create_dynamic_gauge(
            name="lmcache:remote_put_task_num",
            documentation="The number of remote put tasks",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self._create_dynamic_gauge(
            name="lmcache:pin_monitor_pinned_objects_count",
            documentation="The number of pinned objects in PinMonitor",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self._create_dynamic_gauge(
            name="lmcache:lmcache_is_healthy",
            documentation="The health status of LMCache (1=healthy, 0=unhealthy)",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self._create_dynamic_gauge(
            name="lmcache:get_blocking_failed_count",
            documentation="The number of get blocking failed",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self._create_dynamic_gauge(
            name="lmcache:put_failed_count",
            documentation="The number of put failed",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )

        event_statuses = ["ongoing", "done", "not_found"]
        for status in event_statuses:
            metric_name = f"storage_events_{status}_count"
            self._create_dynamic_gauge(
                name=f"lmcache:{metric_name}",
                documentation=f"The number of {status.replace('_', ' ')} events",
                labelnames=labelnames,
                multiprocess_mode="sum",
            )

        # Chunk statistics metrics (dynamic)
        self._create_dynamic_gauge(
            name="lmcache:chunk_statistics_enabled",
            documentation="Whether chunk statistics collection is enabled",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self._create_dynamic_gauge(
            name="lmcache:chunk_statistics_total_requests",
            documentation="Total number of requests processed by chunk statistics",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self._create_dynamic_gauge(
            name="lmcache:chunk_statistics_total_chunks",
            documentation="Total number of chunks processed",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self._create_dynamic_gauge(
            name="lmcache:chunk_statistics_unique_chunks",
            documentation="Number of unique chunks (estimated)",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self._create_dynamic_gauge(
            name="lmcache:chunk_statistics_reuse_rate",
            documentation="Chunk reuse rate (0.0 to 1.0)",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self._create_dynamic_gauge(
            name="lmcache:chunk_statistics_bloom_filter_size_mb",
            documentation="Bloom Filter memory usage in MB",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self._create_dynamic_gauge(
            name="lmcache:chunk_statistics_bloom_filter_fill_rate",
            documentation="Bloom Filter fill rate (0.0 to 1.0)",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self._create_dynamic_gauge(
            name="lmcache:chunk_statistics_file_count",
            documentation="Number of files created for file_hash strategy",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self._create_dynamic_gauge(
            name="lmcache:chunk_statistics_current_file_size",
            documentation="Current file size in bytes for file_hash strategy",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )

        # Connector metrics
        connector_metrics = [
            "scheduler_unfinished_requests_count",
            "connector_load_specs_count",
            "connector_request_trackers_count",
            "connector_kv_caches_count",
            "connector_layerwise_retrievers_count",
            "connector_invalid_block_ids_count",
            "connector_requests_priority_count",
        ]

        for metric_name in connector_metrics:
            self._create_dynamic_gauge(
                name=f"lmcache:{metric_name}",
                documentation=f"The count of {metric_name.replace('_', ' ')}",
                labelnames=labelnames,
                multiprocess_mode="livemostrecent",
            )

        # PeriodicThread metrics
        self._create_dynamic_gauge(
            name="lmcache:periodic_threads_total_count",
            documentation="Total number of registered periodic threads",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self._create_dynamic_gauge(
            name="lmcache:periodic_threads_running_count",
            documentation="Number of running periodic threads",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self._create_dynamic_gauge(
            name="lmcache:periodic_threads_active_count",
            documentation="Number of active periodic threads (recently executed)",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )

        # Per-level metrics for periodic threads
        for level_name in ["critical", "high", "medium", "low"]:
            self._create_dynamic_gauge(
                name=f"lmcache:periodic_threads_{level_name}_total",
                documentation=f"Total number of {level_name} level periodic threads",
                labelnames=labelnames,
                multiprocess_mode="livemostrecent",
            )

            self._create_dynamic_gauge(
                name=f"lmcache:periodic_threads_{level_name}_running",
                documentation=f"Number of running {level_name} level periodic threads",
                labelnames=labelnames,
                multiprocess_mode="livemostrecent",
            )

            self._create_dynamic_gauge(
                name=f"lmcache:periodic_threads_{level_name}_active",
                documentation=f"Number of active {level_name} level periodic threads",
                labelnames=labelnames,
                multiprocess_mode="livemostrecent",
            )

    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge.
        gauge.labels(**self.labels).set(data)

    def _log_counter(self, counter, data: Union[int, float]) -> None:
        # Convenience function for logging to counter.
        # Prevent ValueError from negative increment
        if data < 0:
            return
        counter.labels(**self.labels).inc(data)

    def _log_histogram(self, histogram, data: Union[List[int], List[float]]) -> None:
        # Convenience function for logging to histogram.
        for value in data:
            histogram.labels(**self.labels).observe(value)

    def log_prometheus(self, stats: LMCacheStats):
        self._log_counter(
            self.counter_num_retrieve_requests, stats.interval_retrieve_requests
        )
        self._log_counter(
            self.counter_num_store_requests, stats.interval_store_requests
        )
        self._log_counter(
            self.counter_num_lookup_requests, stats.interval_lookup_requests
        )

        self._log_counter(
            self.counter_num_requested_tokens, stats.interval_requested_tokens
        )
        self._log_counter(self.counter_num_hit_tokens, stats.interval_hit_tokens)
        self._log_counter(self.counter_num_stored_tokens, stats.interval_stored_tokens)
        self._log_counter(self.counter_num_lookup_tokens, stats.interval_lookup_tokens)
        self._log_counter(self.counter_num_lookup_hits, stats.interval_lookup_hits)
        self._log_counter(self.counter_num_prompt_tokens, stats.interval_prompt_tokens)
        self._log_counter(
            self.counter_num_vllm_hit_tokens, stats.interval_vllm_hit_tokens
        )

        self._log_counter(
            self.counter_num_remote_read_requests,
            stats.interval_remote_read_requests,
        )
        self._log_counter(
            self.counter_num_remote_read_bytes, stats.interval_remote_read_bytes
        )
        self._log_counter(
            self.counter_num_remote_write_requests,
            stats.interval_remote_write_requests,
        )
        self._log_counter(
            self.counter_num_remote_write_bytes,
            stats.interval_remote_write_bytes,
        )
        self._log_counter(
            self.counter_local_cpu_evict_count,
            stats.interval_local_cpu_evict_count,
        )
        self._log_counter(
            self.counter_local_cpu_evict_keys_count,
            stats.interval_local_cpu_evict_keys_count,
        )
        self._log_counter(
            self.counter_local_cpu_evict_failed_count,
            stats.interval_local_cpu_evict_failed_count,
        )
        self._log_counter(
            self.counter_forced_unpin_count,
            stats.interval_forced_unpin_count,
        )
        self._log_counter(
            self.counter_lookup_0_hit_requests,
            stats.interval_lookup_0_hit_requests,
        )
        self._log_counter(
            self.counter_num_slow_retrieval_by_time,
            stats.interval_num_slow_retrieval_by_time,
        )
        self._log_counter(
            self.counter_num_slow_retrieval_by_speed,
            stats.interval_num_slow_retrieval_by_speed,
        )

        self._log_gauge(self.gauge_retrieve_hit_rate, stats.retrieve_hit_rate)

        self._log_gauge(self.gauge_lookup_hit_rate, stats.lookup_hit_rate)

        self._log_gauge(self.gauge_local_cache_usage, stats.local_cache_usage_bytes)

        self._log_gauge(self.gauge_remote_cache_usage, stats.remote_cache_usage_bytes)

        self._log_gauge(self.gauge_local_storage_usage, stats.local_storage_usage_bytes)

        self._log_histogram(self.histogram_time_to_retrieve, stats.time_to_retrieve)

        self._log_histogram(self.histogram_time_to_store, stats.time_to_store)

        self._log_histogram(self.histogram_time_to_lookup, stats.time_to_lookup)

        self._log_histogram(self.histogram_retrieve_speed, stats.retrieve_speed)

        self._log_histogram(self.histogram_store_speed, stats.store_speed)

        self._log_histogram(
            self.histogram_retrieve_process_tokens_time,
            stats.retrieve_process_tokens_time,
        )
        self._log_histogram(
            self.histogram_retrieve_broadcast_time, stats.retrieve_broadcast_time
        )
        self._log_histogram(
            self.histogram_retrieve_to_gpu_time, stats.retrieve_to_gpu_time
        )
        self._log_histogram(
            self.histogram_remote_backend_batched_get_blocking_time,
            stats.remote_backend_batched_get_blocking_time,
        )
        self._log_histogram(
            self.histogram_instrumented_connector_batched_get_time,
            stats.instrumented_connector_batched_get_time,
        )
        self._log_histogram(
            self.histogram_store_process_tokens_time, stats.store_process_tokens_time
        )
        self._log_histogram(
            self.histogram_store_from_gpu_time, stats.store_from_gpu_time
        )
        self._log_histogram(self.histogram_store_put_time, stats.store_put_time)

        self._log_histogram(
            self.histogram_p2p_time_to_transfer, stats.p2p_time_to_transfer
        )

        self._log_histogram(self.histogram_p2p_transfer_speed, stats.p2p_transfer_speed)

        self._log_histogram(
            self.histogram_remote_time_to_get, stats.interval_remote_time_to_get
        )
        self._log_histogram(
            self.histogram_remote_time_to_put, stats.interval_remote_time_to_put
        )
        self._log_histogram(
            self.histogram_remote_time_to_get_sync,
            stats.interval_remote_time_to_get_sync,
        )
        self._log_histogram(
            self.histogram_request_cache_hit_rate, stats.interval_lookup_hit_rates
        )
        self._log_histogram(
            self.histogram_request_cache_lifespan, stats.interval_request_cache_lifespan
        )
        self._log_gauge(
            self.gauge_remote_ping_latency, stats.interval_remote_ping_latency
        )
        self._log_counter(
            self.counter_remote_ping_errors, stats.interval_remote_ping_errors
        )
        self._log_counter(
            self.counter_remote_ping_successes, stats.interval_remote_ping_success
        )
        self._log_gauge(
            self.gauge_remote_ping_error_code, stats.interval_remote_ping_error_code
        )
        self._log_gauge(
            self.gauge_active_memory_objs_count, stats.active_memory_objs_count
        )
        self._log_gauge(
            self.gauge_pinned_memory_objs_count, stats.pinned_memory_objs_count
        )

    _instances: Dict[tuple[tuple[str, str], ...], "PrometheusLogger"] = {}

    @staticmethod
    @thread_safe
    def GetOrCreate(
        metadata: LMCacheMetadata,
        config: Optional["LMCacheEngineConfig"] = None,
    ) -> "PrometheusLogger":
        metadata_key = PrometheusLogger._metadata_to_key(metadata)
        if metadata_key in PrometheusLogger._instances:
            return PrometheusLogger._instances[metadata_key]

        base_logger = PrometheusLogger._get_base_logger()
        if base_logger is None:
            logger_instance = PrometheusLogger(metadata, config=config)
        else:
            logger_instance = PrometheusLogger._create_label_view(
                base_logger,
                metadata,
            )

        PrometheusLogger._instances[metadata_key] = logger_instance
        return logger_instance

    @staticmethod
    def _get_base_logger() -> Optional["PrometheusLogger"]:
        """
        Return an existing logger to reuse registered Prometheus collectors.
        """
        return next(iter(PrometheusLogger._instances.values()), None)

    @thread_safe
    def reset_counters(self) -> None:
        """
        Reset all Prometheus Counter metrics by calling clear().
        After clearing, re-initialize with labels so metrics remain visible.
        """
        label_views = self._reset_label_views()
        for counter in self._counters:
            counter.clear()
            # Re-initialize all known label views to keep each series visible.
            for label_view in label_views:
                counter.labels(**label_view.labels)

    @thread_safe
    def reset_histograms(self) -> None:
        """
        Reset all Prometheus Histogram metrics by calling clear().
        After clearing, re-initialize with labels so metrics remain visible.
        """
        label_views = self._reset_label_views()
        for histogram in self._histograms:
            histogram.clear()
            # Re-initialize all known label views to keep each series visible.
            for label_view in label_views:
                histogram.labels(**label_view.labels)

    @staticmethod
    def _metadata_to_labels(metadata: LMCacheMetadata) -> Dict[str, Any]:
        labels = {
            "model_name": metadata.model_name,
            "worker_id": metadata.worker_id,
            "role": metadata.role,
            "served_model_name": metadata.served_model_name or "",
        }
        return labels

    @staticmethod
    def _metadata_to_key(metadata: LMCacheMetadata) -> tuple[tuple[str, str], ...]:
        labels = PrometheusLogger._metadata_to_labels(metadata)
        return tuple(sorted((name, str(value)) for name, value in labels.items()))

    @staticmethod
    def _create_label_view(
        base_logger: "PrometheusLogger",
        metadata: LMCacheMetadata,
    ) -> "PrometheusLogger":
        """Reuse registered collectors with a different metadata/label view."""
        label_view = copy(base_logger)
        label_view.metadata = metadata
        label_view.labels = PrometheusLogger._metadata_to_labels(metadata)
        label_view._bind_dynamic_metric_children()
        return label_view

    def _reset_label_views(self) -> List["PrometheusLogger"]:
        """Return label views whose children must be restored after clear().

        Counter.clear() and Histogram.clear() remove every child from the
        shared collector, so reset must recreate children for all known labels.
        """
        views = list(PrometheusLogger._instances.values())
        if self not in views:
            views.append(self)
        return views


def reset_observability_metrics() -> None:
    """
    Reset observability metrics to their initial state.
    """

    prometheus_logger = PrometheusLogger._get_base_logger()
    if prometheus_logger is not None:
        prometheus_logger.reset_counters()
        prometheus_logger.reset_histograms()


class LMCacheStatsLogger:
    def __init__(
        self,
        metadata: LMCacheMetadata,
        log_interval: int,
        config: Optional["LMCacheEngineConfig"] = None,
    ):
        self.metadata = metadata
        self.log_interval = log_interval
        self.monitor = LMCStatsMonitor.GetOrCreate()
        self.prometheus_logger = PrometheusLogger.GetOrCreate(metadata, config=config)
        self.lmc_usage_logger = ContinuousUsageContext.GetOrCreate(metadata)
        self.is_running = True
        # Event for interruptible sleep during shutdown
        self.shutdown_event = threading.Event()

        self.thread = threading.Thread(
            target=self.log_worker, daemon=True, name="stats-logger-thread"
        )
        self.thread.start()

    def log_worker(self):
        while self.is_running:
            stats = self.monitor.get_stats_and_clear()
            self.prometheus_logger.log_prometheus(stats)
            self.lmc_usage_logger.incr_or_send_stats(stats)
            # Use Event.wait() instead of time.sleep() for interruptible sleep
            # Returns True if event was set, False if timeout occurred
            self.shutdown_event.wait(self.log_interval)

    def shutdown(self):
        """Shutdown the stats logger gracefully with immediate wake-up"""
        logger.info("Shutting down LMCacheStatsLogger...")

        # Signal the worker thread to stop
        self.is_running = False

        # Signal the event to wake up the thread immediately from sleep
        self.shutdown_event.set()

        # Wait for thread with a reasonable timeout
        if self.thread.is_alive():
            # Since we wake up the thread immediately, use a shorter timeout
            # Just enough time for the thread to finish its current iteration
            timeout = 5.0
            logger.info(
                f"Waiting for stats logger thread to finish (timeout: {timeout}s)..."
            )

            try:
                self.thread.join(timeout=timeout)

                if self.thread.is_alive():
                    logger.warning(
                        f"Stats logger thread did not terminate "
                        f"within {timeout}s timeout. "
                        "Thread may be blocked in logging operations. "
                        "Proceeding with shutdown anyway."
                    )
                else:
                    logger.info("Stats logger thread terminated successfully")
            except Exception as e:
                logger.error("Error waiting for stats logger thread: %s", e)
        else:
            logger.info("Stats logger thread already stopped")

        logger.info("LMCacheStatsLogger shutdown complete")
