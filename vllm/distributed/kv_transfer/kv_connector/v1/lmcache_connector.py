# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import torch
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorV1Impl as LMCacheConnectorLatestImpl,
)

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class LMCacheConnectorV1(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        assert vllm_config.kv_transfer_config is not None
        use_native = vllm_config.kv_transfer_config.get_from_extra_config(
            "use_native", False
        )
        if use_native:
            logger.info("Initializing native LMCache connector")
            # lazy import
            from vllm.distributed.kv_transfer.kv_connector.v1 import lmcache_integration

            _adapter = lmcache_integration.vllm_v1_adapter

            cls = _adapter.LMCacheConnectorV1Impl
        else:
            logger.info("Initializing latest dev LMCache connector")
            cls = LMCacheConnectorLatestImpl

        self._lmcache_engine = cls(vllm_config, role, self)

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        self._lmcache_engine.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        self._lmcache_engine.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """
        Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        self._lmcache_engine.save_kv_layer(
            layer_name, kv_layer, attn_metadata, **kwargs
        )

    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        self._lmcache_engine.wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        return self._lmcache_engine.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.
        """
        method = getattr(self._lmcache_engine, "get_block_ids_with_load_errors", None)
        if callable(method):
            return method()

        # Fallback for older versions that don't support this method
        return set()

    def get_kv_connector_stats(self) -> "KVConnectorStats | None":
        """
        Get the KV connector stats collected during the last interval.
        Get and clear LMCachestats from LMCStatsMonitor.
        """
        serialized_lm_cache_stats = asdict(
            self._lmcache_engine._stats_monitor.get_stats_and_clear()
        )
        return LMCacheKVConnectorStats().set_data(serialized_lm_cache_stats)

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        return self._lmcache_engine.get_num_new_matched_tokens(
            request, num_computed_tokens
        ), False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        """
        self._lmcache_engine.update_state_after_alloc(request, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        return self._lmcache_engine.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        return self._lmcache_engine.request_finished(request, block_ids)

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> "KVConnectorStats | None":
        """
        KVConnectorStats resolution method. This method allows dynamically
        registered connectors to return their own KVConnectorStats object,
        which can implement custom aggregation logic on the data dict.
        """
        return (
            LMCacheKVConnectorStats(data=data)
            if data is not None
            else LMCacheKVConnectorStats()
        )

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: "VllmConfig",
        metric_types: dict[type["PromMetric"], type["PromMetricT"]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[str]],
    ) -> "KVConnectorPromMetrics | None":
        return LMCachePromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )


@dataclass
class LMCacheKVConnectorStats(KVConnectorStats):
    """
    Container for LMCache telemetry data serialized from LMCacheStats.
    """

    def reset(self):
        """Reset the stats, clear the state."""
        self.data = {}

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        """
        Aggregate stats with another `KVConnectorStats` object.

        self: contains accumulated stats
        other: contains latest stats

        """
        if not other.data:
            return self

        if not self.data:
            # Nothing accumulated yet; take the incoming snapshot.
            self.data = copy.deepcopy(other.data)
            return self

        list_fields = {
            "interval_remote_time_to_get",
            "interval_remote_time_to_put",
            "interval_remote_time_to_get_sync",
            "time_to_retrieve",
            "time_to_store",
            "retrieve_speed",
            "store_speed",
            "p2p_time_to_transfer",
            "p2p_transfer_speed",
            "interval_lookup_hit_rates",
        }
        sum_fields = {
            "interval_retrieve_requests",
            "interval_store_requests",
            "interval_lookup_requests",
            "interval_requested_tokens",
            "interval_hit_tokens",
            "interval_stored_tokens",
            "interval_lookup_tokens",
            "interval_lookup_hits",
            "interval_vllm_hit_tokens",
            "interval_prompt_tokens",
            "interval_remote_read_requests",
            "interval_remote_read_bytes",
            "interval_remote_write_requests",
            "interval_remote_write_bytes",
            "interval_remote_ping_errors",
            "interval_remote_ping_success",
            "interval_local_cpu_evict_count",
            "interval_local_cpu_evict_keys_count",
            "interval_local_cpu_evict_failed_count",
            "local_cache_usage_bytes",
            "remote_cache_usage_bytes",
            "local_storage_usage_bytes",
            "active_memory_objs_count",
            "pinned_memory_objs_count",
            "interval_p2p_requests",
            "interval_p2p_transferred_tokens",
            "interval_lookup_0_hit_requests",
        }

        for key in list_fields:
            self.data.setdefault(key, []).extend(other.data.get(key, []))

        for key in sum_fields:
            self.data[key] = self.data.get(key, 0) + other.data.get(key, 0)

        # Aggregate ping latency using the count of successful pings
        # as the weight. Fall back to zero when no successes.
        self_success = self.data.get("interval_remote_ping_success", 0)
        other_success = other.data.get("interval_remote_ping_success", 0)
        total_success = self_success + other_success
        if total_success > 0:
            weighted_latency = (
                self.data.get("interval_remote_ping_latency", 0) * self_success
                + other.data.get("interval_remote_ping_latency", 0) * other_success
            )
            self.data["interval_remote_ping_latency"] = weighted_latency / total_success
        else:
            self.data["interval_remote_ping_latency"] = 0

        # Keep the latest non-zero error code, if any.
        other_err = other.data.get("interval_remote_ping_error_code", 0)
        if other_err != 0 or "interval_remote_ping_error_code" not in self.data:
            self.data["interval_remote_ping_error_code"] = other_err

        self._recompute_hit_rates()
        return self

    def reduce(self) -> dict[str, int | float]:
        """
        Reduce the observations collected during a time interval to one or
        more representative values (eg avg/median/sum of the series).
        This is meant to be called by the logger to produce a summary of the
        stats for the last time interval.
        """
        if not self.data:
            return {}

        data = self.data

        def _mean(values: list[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        def _p90(values: list[float]) -> float:
            if not values:
                return 0.0
            sorted_vals = sorted(values)
            idx = max(0, math.ceil(0.9 * len(sorted_vals)) - 1)
            return float(sorted_vals[idx])

        retrieve_hit_rate = data.get("retrieve_hit_rate", 0)
        lookup_hit_rate = data.get("lookup_hit_rate", 0)

        time_to_retrieve = data.get("time_to_retrieve", [])
        time_to_store = data.get("time_to_store", [])

        summary: dict[str, int | float] = {
            "retrieve_hit_rate_pct": round(retrieve_hit_rate * 100, 2),
            "lookup_hit_rate_pct": round(lookup_hit_rate * 100, 2),
            "retrieve_requests": data.get("interval_retrieve_requests", 0),
            "store_requests": data.get("interval_store_requests", 0),
            "lookup_requests": data.get("interval_lookup_requests", 0),
            "requested_tokens": data.get("interval_requested_tokens", 0),
            "hit_tokens": data.get("interval_hit_tokens", 0),
            "stored_tokens": data.get("interval_stored_tokens", 0),
            "lookup_tokens": data.get("interval_lookup_tokens", 0),
            "lookup_hits": data.get("interval_lookup_hits", 0),
            "vllm_hit_tokens": data.get("interval_vllm_hit_tokens", 0),
            "prompt_tokens": data.get("interval_prompt_tokens", 0),
            "remote_read_bytes": data.get("interval_remote_read_bytes", 0),
            "remote_write_bytes": data.get("interval_remote_write_bytes", 0),
            "local_cache_usage_bytes": data.get("local_cache_usage_bytes", 0),
            "remote_cache_usage_bytes": data.get("remote_cache_usage_bytes", 0),
            "local_storage_usage_bytes": data.get("local_storage_usage_bytes", 0),
            "active_memory_objs_count": data.get("active_memory_objs_count", 0),
            "pinned_memory_objs_count": data.get("pinned_memory_objs_count", 0),
            "p2p_requests": data.get("interval_p2p_requests", 0),
            "p2p_transferred_tokens": data.get("interval_p2p_transferred_tokens", 0),
            "remote_ping_latency_ms": round(
                data.get("interval_remote_ping_latency", 0) * 1.0, 3
            ),
            "remote_ping_errors": data.get("interval_remote_ping_errors", 0),
            "remote_ping_success": data.get("interval_remote_ping_success", 0),
            "local_cpu_evict_count": data.get("interval_local_cpu_evict_count", 0),
            "local_cpu_evict_keys_count": data.get(
                "interval_local_cpu_evict_keys_count", 0
            ),
            "local_cpu_evict_failed_count": data.get(
                "interval_local_cpu_evict_failed_count", 0
            ),
            "lookup_0_hit_requests": data.get("interval_lookup_0_hit_requests", 0),
        }

        if time_to_retrieve:
            summary["avg_time_to_retrieve_ms"] = round(_mean(time_to_retrieve) * 1e3, 3)
            summary["p90_time_to_retrieve_ms"] = round(_p90(time_to_retrieve) * 1e3, 3)
        if time_to_store:
            summary["avg_time_to_store_ms"] = round(_mean(time_to_store) * 1e3, 3)
            summary["p90_time_to_store_ms"] = round(_p90(time_to_store) * 1e3, 3)

        retrieve_speed = data.get("retrieve_speed", [])
        store_speed = data.get("store_speed", [])
        p2p_speed = data.get("p2p_transfer_speed", [])
        if retrieve_speed:
            summary["avg_retrieve_speed_tps"] = round(_mean(retrieve_speed), 3)
        if store_speed:
            summary["avg_store_speed_tps"] = round(_mean(store_speed), 3)
        if p2p_speed:
            summary["avg_p2p_speed_tps"] = round(_mean(p2p_speed), 3)

        return summary

    def is_empty(self) -> bool:
        return not self.data

    def set_data(self, data: dict[str, Any] | None) -> KVConnectorStats:
        if data:
            self.data = data
        return self

    def _recompute_hit_rates(self) -> None:
        requested = self.data.get("interval_requested_tokens", 0)
        hit = self.data.get("interval_hit_tokens", 0)
        lookup_tokens = self.data.get("interval_lookup_tokens", 0)
        lookup_hits = self.data.get("interval_lookup_hits", 0)
        self.data["retrieve_hit_rate"] = hit / requested if requested else 0
        self.data["lookup_hit_rate"] = (
            lookup_hits / lookup_tokens if lookup_tokens else 0
        )


class LMCachePromMetrics(KVConnectorPromMetrics):
    """
    A base class for per-connector Prometheus metric registration
    and recording.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[str]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)

        counter_num_retrieve_requests = self._counter_cls(
            name="lmcache:num_retrieve_requests",
            documentation="Total number of retrieve requests sent to lmcache",
            labelnames=labelnames,
        )
        self.counter_num_retrieve_requests = self.make_per_engine(
            counter_num_retrieve_requests
        )

        counter_num_store_requests = self._counter_cls(
            name="lmcache:num_store_requests",
            documentation="Total number of store requests sent to lmcache",
            labelnames=labelnames,
        )
        self.counter_num_store_requests = self.make_per_engine(
            counter_num_store_requests
        )

        counter_num_lookup_requests = self._counter_cls(
            name="lmcache:num_lookup_requests",
            documentation="Total number of lookup requests sent to lmcache",
            labelnames=labelnames,
        )
        self.counter_num_lookup_requests = self.make_per_engine(
            counter_num_lookup_requests
        )

        counter_num_requested_tokens = self._counter_cls(
            name="lmcache:num_requested_tokens",
            documentation="Total number of tokens requested from lmcache",
            labelnames=labelnames,
        )
        self.counter_num_requested_tokens = self.make_per_engine(
            counter_num_requested_tokens
        )

        counter_num_hit_tokens = self._counter_cls(
            name="lmcache:num_hit_tokens",
            documentation="Total number of tokens hit in lmcache",
            labelnames=labelnames,
        )
        self.counter_num_hit_tokens = self.make_per_engine(counter_num_hit_tokens)

        counter_num_stored_tokens = self._counter_cls(
            name="lmcache:num_stored_tokens",
            documentation=(
                "Total number of tokens stored in lmcache including evicted ones"
            ),
            labelnames=labelnames,
        )
        self.counter_num_stored_tokens = self.make_per_engine(counter_num_stored_tokens)

        counter_num_lookup_tokens = self._counter_cls(
            name="lmcache:num_lookup_tokens",
            documentation="Total number of tokens requested in lookup from lmcache",
            labelnames=labelnames,
        )
        self.counter_num_lookup_tokens = self.make_per_engine(counter_num_lookup_tokens)

        counter_num_lookup_hits = self._counter_cls(
            name="lmcache:num_lookup_hits",
            documentation="Total number of tokens hit in lookup from lmcache",
            labelnames=labelnames,
        )
        self.counter_num_lookup_hits = self.make_per_engine(counter_num_lookup_hits)

        counter_num_vllm_hit_tokens = self._counter_cls(
            name="lmcache:num_vllm_hit_tokens",
            documentation="Number of hit tokens in vllm",
            labelnames=labelnames,
        )
        self.counter_num_vllm_hit_tokens = self.make_per_engine(
            counter_num_vllm_hit_tokens
        )

        counter_num_prompt_tokens = self._counter_cls(
            name="lmcache:num_prompt_tokens",
            documentation="Number of prompt tokens in lmcache",
            labelnames=labelnames,
        )
        self.counter_num_prompt_tokens = self.make_per_engine(counter_num_prompt_tokens)

        counter_num_remote_read_requests = self._counter_cls(
            name="lmcache:num_remote_read_requests",
            documentation="Total number of requests read from "
            "remote backends in lmcache",
            labelnames=labelnames,
        )
        self.counter_num_remote_read_requests = self.make_per_engine(
            counter_num_remote_read_requests
        )

        counter_num_remote_read_bytes = self._counter_cls(
            name="lmcache:num_remote_read_bytes",
            documentation="Total number of bytes read from remote backends in lmcache",
            labelnames=labelnames,
        )
        self.counter_num_remote_read_bytes = self.make_per_engine(
            counter_num_remote_read_bytes
        )

        counter_num_remote_write_requests = self._counter_cls(
            name="lmcache:num_remote_write_requests",
            documentation="Total number of requests write to "
            "remote backends in lmcache",
            labelnames=labelnames,
        )
        self.counter_num_remote_write_requests = self.make_per_engine(
            counter_num_remote_write_requests
        )

        counter_num_remote_write_bytes = self._counter_cls(
            name="lmcache:num_remote_write_bytes",
            documentation="Total number of bytes write to remote backends in lmcache",
            labelnames=labelnames,
        )
        self.counter_num_remote_write_bytes = self.make_per_engine(
            counter_num_remote_write_bytes
        )

        counter_local_cpu_evict_count = self._counter_cls(
            name="lmcache:local_cpu_evict_count",
            documentation="Total number of evict in local cpu backend",
            labelnames=labelnames,
        )
        self.counter_local_cpu_evict_count = self.make_per_engine(
            counter_local_cpu_evict_count
        )

        counter_local_cpu_evict_keys_count = self._counter_cls(
            name="lmcache:local_cpu_evict_keys_count",
            documentation="Total number of evict keys in local cpu backend",
            labelnames=labelnames,
        )
        self.counter_local_cpu_evict_keys_count = self.make_per_engine(
            counter_local_cpu_evict_keys_count
        )

        counter_local_cpu_evict_failed_count = self._counter_cls(
            name="lmcache:local_cpu_evict_failed_count",
            documentation="Total number of failed eviction in local cpu backend",
            labelnames=labelnames,
        )
        self.counter_local_cpu_evict_failed_count = self.make_per_engine(
            counter_local_cpu_evict_failed_count
        )

        counter_lookup_0_hit_requests = self._counter_cls(
            name="lmcache:lookup_0_hit_requests",
            documentation="Total number of 0 hit lookup requests",
            labelnames=labelnames,
        )
        self.counter_lookup_0_hit_requests = self.make_per_engine(
            counter_lookup_0_hit_requests
        )

        gauge_retrieve_hit_rate = self._gauge_cls(
            name="lmcache:retrieve_hit_rate",
            documentation="Hit rate of lmcache retrieve requests since last log",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self.gauge_retrieve_hit_rate = self.make_per_engine(gauge_retrieve_hit_rate)

        gauge_lookup_hit_rate = self._gauge_cls(
            name="lmcache:lookup_hit_rate",
            documentation="Hit rate of lmcache lookup requests since last log",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self.gauge_lookup_hit_rate = self.make_per_engine(gauge_lookup_hit_rate)

        gauge_local_cache_usage = self._gauge_cls(
            name="lmcache:local_cache_usage",
            documentation="Local cache usage (bytes) of lmcache",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.gauge_local_cache_usage = self.make_per_engine(gauge_local_cache_usage)

        gauge_remote_cache_usage = self._gauge_cls(
            name="lmcache:remote_cache_usage",
            documentation="Remote cache usage (bytes) of lmcache",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.gauge_remote_cache_usage = self.make_per_engine(gauge_remote_cache_usage)

        gauge_local_storage_usage = self._gauge_cls(
            name="lmcache:local_storage_usage",
            documentation="Local storage usage (bytes) of lmcache",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.gauge_local_storage_usage = self.make_per_engine(gauge_local_storage_usage)

        gauge_active_memory_objs_count = self._gauge_cls(
            name="lmcache:active_memory_objs_count",
            documentation="The number of active memory objects",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.gauge_active_memory_objs_count = self.make_per_engine(
            gauge_active_memory_objs_count
        )

        gauge_pinned_memory_objs_count = self._gauge_cls(
            name="lmcache:pinned_memory_objs_count",
            documentation="The number of pinned memory objects",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.gauge_pinned_memory_objs_count = self.make_per_engine(
            gauge_pinned_memory_objs_count
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
        histogram_time_to_retrieve = self._histogram_cls(
            name="lmcache:time_to_retrieve",
            documentation="Time to retrieve from lmcache (seconds)",
            labelnames=labelnames,
            buckets=time_to_retrieve_buckets,
        )
        self.histogram_time_to_retrieve = self.make_per_engine(
            histogram_time_to_retrieve
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
        histogram_time_to_store = self._histogram_cls(
            name="lmcache:time_to_store",
            documentation="Time to store to lmcache (seconds)",
            labelnames=labelnames,
            buckets=time_to_store_buckets,
        )
        self.histogram_time_to_store = self.make_per_engine(histogram_time_to_store)

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
        histogram_retrieve_speed = self._histogram_cls(
            name="lmcache:retrieve_speed",
            documentation="Retrieve speed of lmcache (tokens per second)",
            labelnames=labelnames,
            buckets=retrieve_speed_buckets,
        )
        self.histogram_retrieve_speed = self.make_per_engine(histogram_retrieve_speed)

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
        histogram_store_speed = self._histogram_cls(
            name="lmcache:store_speed",
            documentation="Store speed of lmcache (tokens per second)",
            labelnames=labelnames,
            buckets=store_speed_buckets,
        )
        self.histogram_store_speed = self.make_per_engine(histogram_store_speed)

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
        histogram_p2p_time_to_transfer = self._histogram_cls(
            name="lmcache:p2p_time_to_transfer",
            documentation="Time to transfer via P2P (seconds)",
            labelnames=labelnames,
            buckets=p2p_time_buckets,
        )
        self.histogram_p2p_time_to_transfer = self.make_per_engine(
            histogram_p2p_time_to_transfer
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
        histogram_p2p_transfer_speed = self._histogram_cls(
            name="lmcache:p2p_transfer_speed",
            documentation="P2P transfer speed (tokens per second)",
            labelnames=labelnames,
            buckets=p2p_speed_buckets,
        )
        self.histogram_p2p_transfer_speed = self.make_per_engine(
            histogram_p2p_transfer_speed
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
        histogram_remote_time_to_get = self._histogram_cls(
            name="lmcache:remote_time_to_get",
            documentation="Time to get from remote backends (ms)",
            labelnames=labelnames,
            buckets=remote_time_to_get,
        )
        self.histogram_remote_time_to_get = self.make_per_engine(
            histogram_remote_time_to_get
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
        histogram_remote_time_to_put = self._histogram_cls(
            name="lmcache:remote_time_to_put",
            documentation="Time to put to remote backends (ms)",
            labelnames=labelnames,
            buckets=remote_time_to_put,
        )
        self.histogram_remote_time_to_put = self.make_per_engine(
            histogram_remote_time_to_put
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
        histogram_remote_time_to_get_sync = self._histogram_cls(
            name="lmcache:remote_time_to_get_sync",
            documentation="Time to get from remote backends synchronously(ms)",
            labelnames=labelnames,
            buckets=remote_time_to_get_sync,
        )
        self.histogram_remote_time_to_get_sync = self.make_per_engine(
            histogram_remote_time_to_get_sync
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
        histogram_request_cache_hit_rate = self._histogram_cls(
            name="lmcache:request_cache_hit_rate",
            documentation="Request cache hit rate",
            labelnames=labelnames,
            buckets=request_cache_hit_rate,
        )
        self.histogram_request_cache_hit_rate = self.make_per_engine(
            histogram_request_cache_hit_rate
        )

        # Ping latency metrics: use a gauge to record the latest ping latency
        gauge_remote_ping_latency = self._gauge_cls(
            name="lmcache:remote_ping_latency",
            documentation="Latest ping latency to remote backends (ms)",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self.gauge_remote_ping_latency = self.make_per_engine(gauge_remote_ping_latency)
        counter_remote_ping_errors = self._counter_cls(
            name="lmcache:remote_ping_errors",
            documentation="Number of ping errors to remote backends",
            labelnames=labelnames,
        )
        self.counter_remote_ping_errors = self.make_per_engine(
            counter_remote_ping_errors
        )
        counter_remote_ping_successes = self._counter_cls(
            name="lmcache:remote_ping_successes",
            documentation="Number of ping successes to remote backends",
            labelnames=labelnames,
        )
        self.counter_remote_ping_successes = self.make_per_engine(
            counter_remote_ping_successes
        )
        gauge_remote_ping_error_code = self._gauge_cls(
            name="lmcache:remote_ping_error_code",
            documentation="Latest ping error code to remote backends",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self.gauge_remote_ping_error_code = self.make_per_engine(
            gauge_remote_ping_error_code
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """
        Record the supplied transfer statistics to Prometheus metrics. These
        statistics are engine-specific, and should be recorded to a metric
        with the appropriate 'engine' label. These metric instances can be
        created using the make_per_engine() helper method.
        """
        if not transfer_stats_data:
            return

        for counter_obj, counter_key in zip(
            [
                self.counter_num_retrieve_requests,
                self.counter_num_store_requests,
                self.counter_num_lookup_requests,
                self.counter_num_requested_tokens,
                self.counter_num_hit_tokens,
                self.counter_num_stored_tokens,
                self.counter_num_lookup_tokens,
                self.counter_num_lookup_hits,
                self.counter_num_prompt_tokens,
                self.counter_num_vllm_hit_tokens,
                self.counter_num_remote_read_requests,
                self.counter_num_remote_read_bytes,
                self.counter_num_remote_write_requests,
                self.counter_num_remote_write_bytes,
                self.counter_local_cpu_evict_count,
                self.counter_local_cpu_evict_keys_count,
                self.counter_local_cpu_evict_failed_count,
                self.counter_lookup_0_hit_requests,
                self.counter_remote_ping_errors,
                self.counter_remote_ping_successes,
            ],
            [
                "interval_retrieve_requests",
                "interval_store_requests",
                "interval_lookup_requests",
                "interval_requested_tokens",
                "interval_hit_tokens",
                "interval_stored_tokens",
                "interval_lookup_tokens",
                "interval_lookup_hits",
                "interval_prompt_tokens",
                "interval_vllm_hit_tokens",
                "interval_remote_read_requests",
                "interval_remote_read_bytes",
                "interval_remote_write_requests",
                "interval_remote_write_bytes",
                "interval_local_cpu_evict_count",
                "interval_local_cpu_evict_keys_count",
                "interval_local_cpu_evict_failed_count",
                "interval_lookup_0_hit_requests",
                "interval_remote_ping_errors",
                "interval_remote_ping_success",
            ],
        ):
            value = transfer_stats_data.get(counter_key)
            if value is not None:
                counter_obj[engine_idx].inc(value)

        for gauge_obj, gauge_key in zip(
            [
                self.gauge_retrieve_hit_rate,
                self.gauge_lookup_hit_rate,
                self.gauge_local_cache_usage,
                self.gauge_remote_cache_usage,
                self.gauge_local_storage_usage,
                self.gauge_remote_ping_latency,
                self.gauge_remote_ping_error_code,
                self.gauge_active_memory_objs_count,
                self.gauge_pinned_memory_objs_count,
            ],
            [
                "retrieve_hit_rate",
                "lookup_hit_rate",
                "local_cache_usage_bytes",
                "remote_cache_usage_bytes",
                "local_storage_usage_bytes",
                "interval_remote_ping_latency",
                "interval_remote_ping_error_code",
                "active_memory_objs_count",
                "pinned_memory_objs_count",
            ],
        ):
            value = transfer_stats_data.get(gauge_key)
            if value is not None:
                gauge_obj[engine_idx].set(value)

        for histogram_obj, list_item_key in zip(
            [
                self.histogram_time_to_retrieve,
                self.histogram_time_to_store,
                self.histogram_retrieve_speed,
                self.histogram_store_speed,
                self.histogram_p2p_time_to_transfer,
                self.histogram_p2p_transfer_speed,
                self.histogram_remote_time_to_get,
                self.histogram_remote_time_to_put,
                self.histogram_remote_time_to_get_sync,
                self.histogram_request_cache_hit_rate,
            ],
            [
                "time_to_retrieve",
                "time_to_store",
                "retrieve_speed",
                "store_speed",
                "p2p_time_to_transfer",
                "p2p_transfer_speed",
                "interval_remote_time_to_get",
                "interval_remote_time_to_put",
                "interval_remote_time_to_get_sync",
                "interval_lookup_hit_rates",
            ],
        ):
            for list_item in transfer_stats_data.get(list_item_key, []):
                histogram_obj[engine_idx].observe(list_item)
