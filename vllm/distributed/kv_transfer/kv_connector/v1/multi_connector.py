# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBaseType
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    CopyBlocksOp,
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_events import KVCacheEvent
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Key used in MultiKVConnectorStats.data for per-connector selection metrics.
_SELECTION_KEY = "__selection__"


@dataclass
class _SelectionStats(KVConnectorStats):
    """Per-connector selection statistics accumulated by MultiConnector.

    Tracks how often each child connector is queried, wins the weighted
    selection, contributes matched tokens, and misses.  Flows through the
    existing KVConnectorStats pipeline so that counters are registered in
    the APIServer process (the one that serves /metrics) rather than in the
    EngineCore subprocess.
    """

    def __post_init__(self):
        if not self.data:
            self.reset()

    def reset(self):
        # {connector_name: {"queries": int, "hits": int,
        #                    "hit_tokens": int, "misses": int}}
        self.data = {}

    def _ensure(self, name: str):
        if name not in self.data:
            self.data[name] = {
                "queries": 0,
                "hits": 0,
                "hit_tokens": 0,
                "misses": 0,
            }

    def record_query(self, name: str):
        self._ensure(name)
        self.data[name]["queries"] += 1

    def record_hit(self, name: str, tokens: int):
        self._ensure(name)
        self.data[name]["hits"] += 1
        self.data[name]["hit_tokens"] += tokens

    def record_miss(self, name: str):
        self._ensure(name)
        self.data[name]["misses"] += 1

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        if not isinstance(other, _SelectionStats):
            return self
        for name, counts in other.data.items():
            if name not in self.data:
                self.data[name] = dict(counts)
            else:
                for k, v in counts.items():
                    self.data[name][k] = self.data[name].get(k, 0) + v
        return self

    def reduce(self) -> dict[str, Any]:
        # Return a shallow copy so the caller can't mutate our state.
        return dict(self.data)

    def is_empty(self) -> bool:
        return not self.data


@dataclass
class MultiKVConnectorMetadata(KVConnectorMetadata):
    metadata: tuple[KVConnectorMetadata, ...]
    extra_async_saves: dict[str, int] | None = None


@dataclass
class MultiKVConnectorWorkerMetadata(KVConnectorWorkerMetadata):
    metadata: tuple[KVConnectorWorkerMetadata | None, ...]

    def aggregate(self, other: KVConnectorWorkerMetadata) -> KVConnectorWorkerMetadata:
        assert isinstance(other, MultiKVConnectorWorkerMetadata)

        assert len(self.metadata) == len(other.metadata)
        metadata_list = []
        for metadata1, metadata2 in zip(self.metadata, other.metadata):
            if metadata1 is None:
                metadata_list.append(metadata2)
            elif metadata2 is None:
                metadata_list.append(metadata1)
            else:
                metadata_list.append(metadata1.aggregate(metadata2))

        return MultiKVConnectorWorkerMetadata(metadata=tuple(metadata_list))


@dataclass
class MultiKVConnectorStats(KVConnectorStats):
    """
    Maintain a dict of KVConnectorStats objects, one for each connector.
    This is used to aggregate the stats from all connectors separately.
    """

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        for connector_id, stats in other.data.items():
            if connector_id not in self.data:
                self[connector_id] = stats
            else:
                assert isinstance(stats, type(self.data[connector_id]))
                self[connector_id] = self[connector_id].aggregate(stats)
        return self

    def reset(self):
        for stats in self.data.values():
            stats.reset()

    def reduce(self) -> dict[str, Any]:
        # TODO (NickLucche) Adjust for logging on separate lines
        return {
            connector_id: stats.reduce() for connector_id, stats in self.data.items()
        }

    def is_empty(self) -> bool:
        return all(stats.is_empty() for stats in self.data.values())

    def __getitem__(self, connector_id: str) -> KVConnectorStats:
        return self.data[connector_id]

    def __setitem__(self, connector_id: str, stats: KVConnectorStats):
        self.data[connector_id] = stats


class MultiKVConnectorPromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
        prom_metrics: dict[str, KVConnectorPromMetrics],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        self._prom_metrics = prom_metrics

        # Per-connector selection counters.  Labels: model_name, engine,
        # connector.  Registered here (APIServer process) so they appear in
        # the /metrics endpoint.
        _sel_labels = labelnames + ["connector"]
        self._counter_mc_queries = self._counter_cls(
            name="vllm:kv_connector_mc_queries_total",
            documentation=(
                "Total cache-lookup queries issued to each child connector "
                "by MultiConnector."
            ),
            labelnames=_sel_labels,
        )
        self._counter_mc_hits = self._counter_cls(
            name="vllm:kv_connector_mc_hits_total",
            documentation=(
                "Number of times each child connector won the weighted "
                "selection and will serve the load."
            ),
            labelnames=_sel_labels,
        )
        self._counter_mc_hit_tokens = self._counter_cls(
            name="vllm:kv_connector_mc_hit_tokens_total",
            documentation="Total tokens matched by the winning child connector.",
            labelnames=_sel_labels,
        )
        self._counter_mc_misses = self._counter_cls(
            name="vllm:kv_connector_mc_misses_total",
            documentation=(
                "Number of requests where each child connector had no cache hit."
            ),
            labelnames=_sel_labels,
        )
        # Cache of labeled metric instances keyed by (engine_idx, connector_name).
        self._mc_queries: dict[tuple[int, str], Any] = {}
        self._mc_hits: dict[tuple[int, str], Any] = {}
        self._mc_hit_tokens: dict[tuple[int, str], Any] = {}
        self._mc_misses: dict[tuple[int, str], Any] = {}

    def _observe_selection(
        self,
        per_connector: dict[str, dict[str, int]],
        engine_idx: int,
    ) -> None:
        """Update per-connector selection counters from a _SelectionStats dict."""
        for conn_name, counts in per_connector.items():
            key = (engine_idx, conn_name)
            if key not in self._mc_queries:
                label_vals = self.per_engine_labelvalues[engine_idx] + [conn_name]
                self._mc_queries[key] = self._counter_mc_queries.labels(*label_vals)
                self._mc_hits[key] = self._counter_mc_hits.labels(*label_vals)
                self._mc_hit_tokens[key] = self._counter_mc_hit_tokens.labels(
                    *label_vals
                )
                self._mc_misses[key] = self._counter_mc_misses.labels(*label_vals)
            self._mc_queries[key].inc(counts.get("queries", 0))
            self._mc_hits[key].inc(counts.get("hits", 0))
            self._mc_hit_tokens[key].inc(counts.get("hit_tokens", 0))
            self._mc_misses[key].inc(counts.get("misses", 0))

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        # Handle MultiConnector's own selection metrics first.
        selection_data = transfer_stats_data.get(_SELECTION_KEY)
        if selection_data is not None:
            # Cross-process: msgspec serialises the dataclass as a dict with a
            # "data" field.  Same-process: the _SelectionStats object itself.
            per_conn = (
                selection_data["data"]
                if isinstance(selection_data, dict)
                else selection_data.data
            )
            self._observe_selection(per_conn, engine_idx)

        # Route child-connector stats.
        for connector_id, stats_data in transfer_stats_data.items():
            if connector_id == _SELECTION_KEY:
                continue
            assert connector_id in self._prom_metrics, (
                f"{connector_id} is not contained in the list of registered "
                f"connectors with Prometheus metrics support: "
                f"{self._prom_metrics.keys()}"
            )
            self._prom_metrics[connector_id].observe(stats_data["data"], engine_idx)


class MultiConnector(KVConnectorBase_V1, SupportsHMA):
    """
    A wrapper for using multiple KVConnectors at the same time.

    The current logic is:
    - Load KV from the first connector that advertises available tokens from
      get_num_new_matched_tokens(), based on the order in the config.
    - Save to all connectors.
    """

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        """
        MultiConnector requires PIECEWISE CUDA graph mode if any of its
        child connectors require it.
        """
        connectors_config = extra_config.get("connectors", [])
        for conn_config in connectors_config:
            temp_ktc = KVTransferConfig(**conn_config)
            connector_cls = KVConnectorFactory.get_connector_class(temp_ktc)
            child_extra_config = conn_config.get("kv_connector_extra_config", {})
            if connector_cls.requires_piecewise_for_cudagraph(child_extra_config):
                return True
        return False

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )

        self._connectors: list[KVConnectorBase_V1] = []
        self._ktc_kv_transfer_config = []
        for connector_cls, temp_config in self._get_connector_classes_and_configs(
            vllm_config
        ):
            self._connectors.append(connector_cls(temp_config, role, kv_cache_config))
            self._ktc_kv_transfer_config.append(temp_config.kv_transfer_config)

        # Per-connector load weights for weighted selection.
        # Higher weight means a connector's hit is preferred even if
        # another offers more tokens (e.g. fast CPU cache vs slow disk).
        # Configured via "load_weight" in each connector's config.
        assert vllm_config.kv_transfer_config is not None
        connectors_cfg = (
            vllm_config.kv_transfer_config.kv_connector_extra_config
            .get("connectors", [])
        )
        self._load_weights: list[float] = [
            float(cfg.get("kv_connector_extra_config", {})
                      .get("load_weight", 1.0))
            for cfg in connectors_cfg
        ]
        # Pad if config is shorter than connectors (shouldn't happen)
        while len(self._load_weights) < len(self._connectors):
            self._load_weights.append(1.0)

        # Validate HMA: MultiConnector advertises SupportsHMA, but this
        # only works if all children also support it.
        if not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
            non_hma = [
                type(c).__name__ for c in self._connectors
                if not isinstance(c, SupportsHMA)
            ]
            if non_hma:
                raise TypeError(
                    f"MultiConnector has HMA enabled but these child "
                    f"connectors do not support it: {non_hma}. Either "
                    f"use --disable-hybrid-kv-cache-manager or replace "
                    f"the non-HMA connectors."
                )

        # Human-readable names for per-connector Prometheus labels.
        self._connector_names: list[str] = [
            type(c).__name__ for c in self._connectors
        ]

        # Per-connector selection stats; flushed via get_kv_connector_stats().
        self._selection_stats = _SelectionStats()

        # A mapping from request id to the index of the connector chosen to
        # load the request from (if any).
        self._requests_to_connector: dict[str, int] = {}

        # Keeps track of *additional* remaining async saves (beyond 1) to be
        # finished per request. Not needed for async loads since we only allow
        # a single connector to load.
        # Propagated from scheduler to worker side via the connector metadata.
        self._extra_async_saves: dict[str, int] = {}

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        if not self._connectors:
            return False
        return all(c.prefer_cross_layer_blocks for c in self._connectors)

    @classmethod
    def _get_connector_classes_and_configs(
        cls, vllm_config: "VllmConfig"
    ) -> list[tuple[type[KVConnectorBaseType], "VllmConfig"]]:
        assert vllm_config.kv_transfer_config is not None
        ktcs = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "connectors"
        )
        assert ktcs is not None
        ret: list[tuple[type[KVConnectorBaseType], VllmConfig]] = []
        for ktc in ktcs:
            temp_config = copy.copy(vllm_config)
            engine_id = ktc.get("engine_id", vllm_config.kv_transfer_config.engine_id)
            temp_config.kv_transfer_config = KVTransferConfig(
                **ktc, engine_id=engine_id
            )
            ret.append(
                (
                    KVConnectorFactory.get_connector_class(
                        temp_config.kv_transfer_config
                    ),
                    temp_config,
                )
            )
        return ret

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        # Register on all connectors
        for c in self._connectors:
            c.register_cross_layers_kv_cache(kv_cache, attn_backend)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        for c in self._connectors:
            c.register_kv_caches(kv_caches)

    # We must override the base class method here because we need to bind
    # the metadata to each connector in the order of the connectors in the
    # MultiKVConnectorMetadata.
    #
    # Note: Call the base class method to ensure metadata is also set on the
    # MultiConnector instance itself; otherwise, `has_connector_metadata()` will
    # always return False.
    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        assert isinstance(connector_metadata, MultiKVConnectorMetadata)
        if connector_metadata.extra_async_saves:
            self._extra_async_saves.update(connector_metadata.extra_async_saves)
        for c, cm in zip(self._connectors, connector_metadata.metadata):
            c.bind_connector_metadata(cm)
        super().bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        for c in self._connectors:
            c.clear_connector_metadata()
        super().clear_connector_metadata()

    def shutdown(self):
        exception: Exception | None = None
        for c in self._connectors:
            try:
                c.shutdown()
            except Exception as e:
                logger.exception(
                    "Exception during connector %s shutdown.", c.__class__.__name__
                )
                exception = e
        if exception:
            raise exception

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        for c in self._connectors:
            c.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        for c in self._connectors:
            c.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        for c in self._connectors:
            c.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self):
        for c in self._connectors:
            c.wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        finished_sending: set[str] = set()
        finished_recving: set[str] = set()
        for c in self._connectors:
            sending, recving = c.get_finished(finished_req_ids)
            if not recving and not sending:
                continue
            # Aggregate finished recving request ids.
            finished_recving.update(recving or ())
            # Aggregate finished sending request ids - only include
            # once we've drained the "extra" count (for cases where
            # more than one connector is async-saving the same request).
            for req_id in sending or ():
                extra_pending = self._extra_async_saves.get(req_id)
                if extra_pending is None:
                    finished_sending.add(req_id)
                    continue
                assert extra_pending > 0
                if extra_pending == 1:
                    del self._extra_async_saves[req_id]
                else:
                    self._extra_async_saves[req_id] = extra_pending - 1

        return finished_sending or None, finished_recving or None

    def get_block_ids_with_load_errors(self) -> set[int]:
        agg_block_ids: set[int] = set()
        for c in self._connectors:
            agg_block_ids |= c.get_block_ids_with_load_errors()
        return agg_block_ids

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        """Set xPU-specific copy ops for all sub-connectors."""
        for c in self._connectors:
            c.set_host_xfer_buffer_ops(copy_operation)

    def handle_preemptions(
        self,
        kv_connector_metadata: KVConnectorMetadata | set[str],
    ):
        """Handle preempted requests for all sub-connectors.

        Stock vLLM 0.18.0 passes ``set[str]`` (preempted request IDs),
        while the MultiConnector metadata path passes
        ``MultiKVConnectorMetadata``.  Accept both.
        """
        if isinstance(kv_connector_metadata, set):
            # Stock vLLM path — forward the raw set to every child.
            for c in self._connectors:
                c.handle_preemptions(kv_connector_metadata)
        else:
            assert isinstance(kv_connector_metadata,
                              MultiKVConnectorMetadata)
            for c, cm in zip(self._connectors,
                             kv_connector_metadata.metadata):
                c.handle_preemptions(cm)

    def get_finished_count(self) -> int | None:
        # TODO(https://github.com/vllm-project/vllm/issues/33400)
        # Currently no connectors return non-None
        return None

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        metadata_list: list[KVConnectorWorkerMetadata | None] | None = None
        for i, c in enumerate(self._connectors):
            kv_connector_worker_meta = c.build_connector_worker_meta()
            if metadata_list is None and kv_connector_worker_meta is not None:
                metadata_list = [None] * i
            if metadata_list is not None:
                metadata_list.append(kv_connector_worker_meta)
        if metadata_list is None:
            return None
        return MultiKVConnectorWorkerMetadata(metadata=tuple(metadata_list))

    # TODO: Add a generic implementation of 'get_kv_connector_kv_cache_events'
    # method for the MultiConnector. It should be able to get events from
    # multiple connectors, handling the case where only a subset of the
    # requested connectors implements the 'get_kv_connector_kv_cache_events'
    # WIP: https://github.com/vllm-project/vllm/pull/31811

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        # Weighted selection: each connector's hit is scored as
        # tokens * load_weight.  The connector with the highest
        # weighted score wins.  This lets a fast CPU cache (high
        # weight) beat a slow disk cache unless the disk hit is
        # substantially larger.
        best_idx = -1
        best_score = 0.0
        best_result: tuple[int, bool] = (0, False)

        # Track which connectors gave a definitive answer vs deferred.
        per_connector_results: list[tuple[int, bool] | None] = []
        any_resolved = False
        for i, c in enumerate(self._connectors):
            toks, load_async = c.get_num_new_matched_tokens(
                request, num_computed_tokens
            )
            if toks is None:
                # Connector is still resolving (e.g. backpressured).
                # Skip it — don't block connectors that already answered.
                per_connector_results.append(None)
                continue
            any_resolved = True
            name = self._connector_names[i]
            self._selection_stats.record_query(name)
            per_connector_results.append((toks, load_async))
            if toks > 0:
                score = toks * self._load_weights[i]
                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_result = (toks, load_async)

        # Only defer if ALL connectors returned None.
        if not any_resolved:
            return (None, False)

        if best_idx >= 0:
            winner_name = self._connector_names[best_idx]
            self._requests_to_connector[request.request_id] = best_idx
            self._selection_stats.record_hit(winner_name, best_result[0])
            for i, result in enumerate(per_connector_results):
                if i != best_idx and result is not None and result[0] == 0:
                    self._selection_stats.record_miss(self._connector_names[i])
        else:
            # No connector had a hit (resolved ones all returned 0).
            for i, result in enumerate(per_connector_results):
                if result is not None:
                    self._selection_stats.record_miss(self._connector_names[i])

        return best_result

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        chosen_connector = self._requests_to_connector.get(request.request_id, -1)
        empty_blocks = blocks.new_empty()
        for i, c in enumerate(self._connectors):
            if i == chosen_connector:
                # Forward call to the chosen connector (if any).
                c.update_state_after_alloc(request, blocks, num_external_tokens)
            else:
                # Call with empty blocks for other connectors.
                c.update_state_after_alloc(request, empty_blocks, 0)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> MultiKVConnectorMetadata:
        metadata = MultiKVConnectorMetadata(
            metadata=tuple(
                c.build_connector_meta(scheduler_output) for c in self._connectors
            )
        )
        if self._extra_async_saves:
            metadata.extra_async_saves = self._extra_async_saves
            self._extra_async_saves = {}
        return metadata

    def update_connector_output(self, connector_output: KVConnectorOutput):
        multi_connector_worker_meta: MultiKVConnectorWorkerMetadata | None = None
        if connector_output.kv_connector_worker_meta is not None:
            assert isinstance(
                connector_output.kv_connector_worker_meta,
                MultiKVConnectorWorkerMetadata,
            )
            multi_connector_worker_meta = connector_output.kv_connector_worker_meta

        try:
            for i, c in enumerate(self._connectors):
                if multi_connector_worker_meta is not None:
                    # set the connector-specific worker metadata
                    connector_output.kv_connector_worker_meta = (
                        multi_connector_worker_meta.metadata[i]
                    )
                c.update_connector_output(connector_output)
        finally:
            # restore kv_connector_worker_meta
            connector_output.kv_connector_worker_meta = multi_connector_worker_meta

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        """
        Get the KVConnector handshake metadata from sub-connectors.
        Returns the first non-None metadata from sub-connectors.
        """
        for c in self._connectors:
            metadata = c.get_handshake_metadata()
            if metadata is not None:
                return metadata
        return None

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        """
        Set the KV connector handshake metadata for all sub-connectors.
        This is needed to start the NIXL listener thread for NixlConnector.
        """
        for c in self._connectors:
            c.set_xfer_handshake_metadata(metadata)

    def request_finished(
        self,
        request: "Request",
        blocks: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished_all_groups(request, (blocks,))

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        async_saves = 0
        kv_txfer_params = None
        for c in self._connectors:
            if isinstance(c, SupportsHMA):
                async_save, txfer_params = c.request_finished_all_groups(
                    request, block_ids
                )
            else:
                # Flatten block_ids for non-HMA connectors
                flat = [bid for group in block_ids for bid in group]
                async_save, txfer_params = c.request_finished(request, flat)
            if async_save:
                async_saves += 1
            if txfer_params is not None:
                if kv_txfer_params is not None:
                    # TODO we can probably change this to merge the dicts here,
                    # checking for key clashes.
                    raise RuntimeError(
                        "Only one connector can produce KV transfer params"
                    )
                kv_txfer_params = txfer_params
        if async_saves > 1:
            self._extra_async_saves[request.request_id] = async_saves - 1

        # Clean up other state for this request.
        self._requests_to_connector.pop(request.request_id, None)

        return async_saves > 0, kv_txfer_params

    def take_events(self) -> Iterable["KVCacheEvent"]:
        for c in self._connectors:
            yield from c.take_events()

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        """
        Get the required KV cache layout for this connector.
        Args:
            vllm_config (VllmConfig): the vllm config.

        Returns:
            str: the required KV cache layout. e.g. HND, or NHD.
            None if the connector does not require a specific layout.
        """
        assert vllm_config.kv_transfer_config is not None
        layouts: set[str] = set()
        for connector_cls, temp_config in cls._get_connector_classes_and_configs(
            vllm_config
        ):
            required_kvcache_layout = connector_cls.get_required_kvcache_layout(
                temp_config
            )
            if required_kvcache_layout is not None:
                layouts.add(required_kvcache_layout)

        if len(layouts) > 1:
            raise ValueError(
                f"KV cache layout mismatch: "
                f"found {len(layouts)} different layouts "
                f"({', '.join(layouts)})."
                f"All connectors must use the same layout."
            )
        return next(iter(layouts), None)

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        if data is None:
            return MultiKVConnectorStats()

        # data is a dict mapping connector name to their stats data.
        # The stats data can be either:
        # 1. Already-instantiated KVConnectorStats objects (same process)
        # 2. Serialized dicts (cross-process after serialization)
        # We need to reconstruct proper KVConnectorStats objects from dicts
        reconstructed_data = {}
        for connector_name, stats_value in data.items():
            # Selection stats are internal to MultiConnector — reconstruct
            # directly without going through KVConnectorFactory.
            if connector_name == _SELECTION_KEY:
                if isinstance(stats_value, _SelectionStats):
                    reconstructed_data[connector_name] = stats_value
                else:
                    assert isinstance(stats_value, dict) and "data" in stats_value, (
                        f"Expected a dict with a 'data' field for "
                        f"{_SELECTION_KEY!r}, got {stats_value!r}"
                    )
                    reconstructed_data[connector_name] = _SelectionStats(
                        data=stats_value["data"]
                    )
                continue

            # If already a KVConnectorStats object, use it directly
            if isinstance(stats_value, KVConnectorStats):
                reconstructed_data[connector_name] = stats_value
                continue

            # Otherwise, reconstruct from serialized dict
            # Get the connector class to reconstruct its stats
            connector_cls = KVConnectorFactory.get_connector_class_by_name(
                connector_name
            )

            # stats_value is the serialized dataclass which contains {'data': {...}}
            # We need to extract the inner 'data' field to avoid double-nesting
            assert isinstance(stats_value, dict) and "data" in stats_value, (
                f"Expected a dict with a 'data' field, got {stats_value}"
            )
            inner_data = stats_value["data"]

            # Use the connector's build_kv_connector_stats to reconstruct
            if reconstructed_stats := connector_cls.build_kv_connector_stats(
                data=inner_data
            ):
                reconstructed_data[connector_name] = reconstructed_stats

        return MultiKVConnectorStats(data=reconstructed_data)

    def get_kv_connector_stats(self) -> MultiKVConnectorStats | None:
        # Group connector stats by connector type.
        stats_by_connector: MultiKVConnectorStats | None = None
        for c in self._connectors:
            stats = c.get_kv_connector_stats()
            if stats is None:
                continue
            if stats_by_connector is None:
                # Lazy init to allow optional return value.
                stats_by_connector = MultiKVConnectorStats()
            stats_by_connector[c.__class__.__name__] = stats

        # Attach accumulated selection metrics, then reset for the next window.
        if not self._selection_stats.is_empty():
            if stats_by_connector is None:
                stats_by_connector = MultiKVConnectorStats()
            stats_by_connector[_SELECTION_KEY] = self._selection_stats
            self._selection_stats = _SelectionStats()

        return stats_by_connector

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: "VllmConfig",
        metric_types: dict[type["PromMetric"], type["PromMetricT"]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        prom_metrics: dict[str, KVConnectorPromMetrics] = {}
        for connector_cls, temp_config in cls._get_connector_classes_and_configs(
            vllm_config
        ):
            connector_prom = connector_cls.build_prom_metrics(
                temp_config, metric_types, labelnames, per_engine_labelvalues
            )
            if connector_prom is not None:
                prom_metrics[connector_cls.__name__] = connector_prom
        return MultiKVConnectorPromMetrics(
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
            prom_metrics,
        )

    def reset_cache(self) -> bool:
        results = [c.reset_cache() is not False for c in self._connectors]
        return all(results)
