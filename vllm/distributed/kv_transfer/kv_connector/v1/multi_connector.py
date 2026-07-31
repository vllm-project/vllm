# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

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
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class MultiKVConnectorMetadata(KVConnectorMetadata):
    metadata: tuple[KVConnectorMetadata, ...]
    extra_async_saves: dict[str, int] | None = None
    extra_async_loads: dict[str, int] | None = None


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

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for connector_id, stats_data in transfer_stats_data.items():
            assert connector_id in self._prom_metrics, (
                f"{connector_id} is not contained in the list of registered connectors "
                f"with Prometheus metrics support: {self._prom_metrics.keys()}"
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

    @classmethod
    def all_children_support_hma(cls, kv_transfer_config: "KVTransferConfig") -> bool:
        """Return True only if every configured child connector supports HMA."""
        connectors_config = kv_transfer_config.kv_connector_extra_config.get(
            "connectors", []
        )
        if not connectors_config:
            return False
        for conn_config in connectors_config:
            child_config = KVTransferConfig(
                **{"engine_id": kv_transfer_config.engine_id, **conn_config}
            )
            if not KVConnectorFactory.supports_hma_config(child_config):
                return False
        return True

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

        assert vllm_config.kv_transfer_config is not None
        self._all_support_hma = MultiConnector.all_children_support_hma(
            vllm_config.kv_transfer_config
        )
        assert (
            vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            or self._all_support_hma
        ), "HMA should not be enabled unless all sub-connectors support it"

        # Whether to let sub-connectors *compose* their hits for one request:
        # each connector is offered the tokens the earlier ones did not cover,
        # instead of the first one with a hit taking the whole request. Opt-in
        # because it changes how many tokens each sub-connector is asked to
        # load; see `get_num_new_matched_tokens`.
        self._compose_connectors = bool(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "compose_connectors", False
            )
        )

        # A mapping from request id to {connector index: number of tokens that
        # connector is loading}. Without composition this holds at most one
        # entry -- the connector chosen to load the request.
        self._requests_to_connector: dict[str, dict[int, int]] = {}

        # Keeps track of *additional* remaining async saves (beyond 1) to be
        # finished per request.
        # Propagated from scheduler to worker side via the connector metadata.
        self._extra_async_saves: dict[str, int] = {}

        # Same, for async loads: with composition more than one connector can
        # be loading a single request, and the scheduler must not resume it
        # until all of them are done. Without composition this stays empty.
        # Propagated from scheduler to worker side via the connector metadata.
        self._extra_async_loads: dict[str, int] = {}

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        if not self._connectors:
            return False
        return all(c.prefer_cross_layer_blocks for c in self._connectors)

    @property
    def requires_kv_delivery(self) -> bool:
        return any(c.requires_kv_delivery for c in self._connectors)

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

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        for c in self._connectors:
            c.bind_gpu_block_pool(gpu_block_pool)

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
        if connector_metadata.extra_async_loads:
            self._extra_async_loads.update(connector_metadata.extra_async_loads)
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
            # Aggregate finished recving request ids - only include once we've
            # drained the "extra" count, exactly as for sends below. With
            # composition more than one connector loads a single request, and
            # the scheduler must not be told the load is complete until the
            # last of them reports: the first report moves the request out of
            # WAITING_FOR_REMOTE_KVS, so a second one would hit
            # `assert RequestStatus.is_finished(req.status)` in
            # `Scheduler._update_from_kv_xfer_finished` and kill EngineCore.
            for req_id in recving or ():
                if self._drain_extra(self._extra_async_loads, req_id):
                    finished_recving.add(req_id)
            # Aggregate finished sending request ids - only include
            # once we've drained the "extra" count (for cases where
            # more than one connector is async-saving the same request).
            for req_id in sending or ():
                if self._drain_extra(self._extra_async_saves, req_id):
                    finished_sending.add(req_id)

        return finished_sending or None, finished_recving or None

    @staticmethod
    def _drain_extra(extra: dict[str, int], req_id: str) -> bool:
        """Whether this is the last connector to report `req_id`.

        `extra` holds the number of reports still expected *beyond* this one.
        A missing entry means no other connector is working on the request.
        """
        extra_pending = extra.get(req_id)
        if extra_pending is None:
            return True
        assert extra_pending > 0
        if extra_pending == 1:
            del extra[req_id]
        else:
            extra[req_id] = extra_pending - 1
        return False

    def get_block_ids_with_load_errors(self) -> set[int]:
        agg_block_ids: set[int] = set()
        for c in self._connectors:
            agg_block_ids |= c.get_block_ids_with_load_errors()
        return agg_block_ids

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        """Set xPU-specific copy ops for all sub-connectors."""
        for c in self._connectors:
            c.set_host_xfer_buffer_ops(copy_operation)

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata):
        """Handle preempted requests for all sub-connectors."""
        assert isinstance(kv_connector_metadata, MultiKVConnectorMetadata)
        for c, cm in zip(self._connectors, kv_connector_metadata.metadata):
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
        """Ask each sub-connector how many tokens it can supply.

        Default (`compose_connectors` off): the first connector reporting a
        non-zero count is assigned the whole request and the rest contribute
        nothing, even when they hold tokens the winner does not.

        With `compose_connectors` on, each connector is instead offered the
        tokens the earlier ones did not cover, by advancing the
        `num_computed_tokens` we pass down. That is the same "everything up to
        here is already covered" contract the scheduler uses, so a connector
        needs no changes to participate -- it simply reports the prefix it can
        serve beyond what it is told exists, and returning 0 opts it out.

        Composition is restricted to connectors that agree with the first
        contributor on `load_async`: the scheduler tracks one load mode per
        request, so mixing a synchronous loader (which loads inside the forward
        pass) with an asynchronous one (which loads between steps) is not
        expressible. Connectors that disagree are offered nothing.
        """
        per_connector: dict[int, int] = {}
        total = 0
        load_async_mode: bool | None = None
        for i, c in enumerate(self._connectors):
            # Only advance the offer when composing. Otherwise every connector
            # is asked exactly the question the scheduler asked us, so the
            # first-hit-takes-all path is bit-for-bit unchanged.
            covered = total if self._compose_connectors else 0
            toks, load_async = c.get_num_new_matched_tokens(
                request, num_computed_tokens + covered
            )
            # If there is a connector still looking up the matches,
            # we return None to indicate that we are not done yet.
            if toks is None:
                return (None, False)
            if toks == 0:
                continue
            if load_async_mode is None:
                # First contributor decides the load mode for this request.
                load_async_mode = load_async
            elif not self._compose_connectors or load_async != load_async_mode:
                # Either we are not composing (first hit takes everything), or
                # this connector cannot be combined with the ones before it.
                continue
            per_connector[i] = toks
            total += toks

        if not per_connector:
            return (0, False)

        self._requests_to_connector[request.request_id] = per_connector
        if len(per_connector) > 1:
            self._extra_async_loads[request.request_id] = len(per_connector) - 1
        assert load_async_mode is not None
        return (total, load_async_mode)

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        per_connector = self._requests_to_connector.get(request.request_id) or {}
        # `num_external_tokens` is the scheduler's final say and can be lower
        # than what we asked for; hand it out in connector order so the tail
        # connectors are the ones that lose tokens, and pass 0 to a connector
        # once the budget runs out so it skips its load. It is also 0 on the
        # second call for async-load requests, which correctly tells every
        # sub-connector there is nothing more to do.
        budget = num_external_tokens
        for i, c in enumerate(self._connectors):
            share = min(per_connector.get(i, 0), budget)
            budget -= share
            # Every connector still receives the request's real blocks.
            c.update_state_after_alloc(request, blocks, share)

    def on_new_request(self, request: "Request") -> None:
        for c in self._connectors:
            c.on_new_request(request)

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
        if self._extra_async_loads:
            metadata.extra_async_loads = self._extra_async_loads
            self._extra_async_loads = {}
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

    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: dict[tuple[int, int], KVConnectorHandshakeMetadata]
    ) -> None:
        for c in self._connectors:
            c.set_xfer_handshake_metadata_pp_aware(metadata)

    def _aggregate_request_finished(
        self,
        request: "Request",
        per_connector_fn: Callable[
            [KVConnectorBase_V1], tuple[bool, dict[str, Any] | None]
        ],
    ) -> tuple[bool, dict[str, Any] | None]:
        async_saves = 0
        kv_txfer_params = None
        for c in self._connectors:
            async_save, txfer_params = per_connector_fn(c)
            if async_save:
                async_saves += 1
            if txfer_params is not None:
                if kv_txfer_params is not None:
                    clashes = set(kv_txfer_params) & set(txfer_params)
                    if clashes:
                        raise RuntimeError(
                            "Key clash in kv_transfer_params from multiple "
                            f"connectors: {clashes}"
                        )
                    kv_txfer_params.update(txfer_params)
                else:
                    kv_txfer_params = txfer_params
        if async_saves > 1:
            self._extra_async_saves[request.request_id] = async_saves - 1

        self._requests_to_connector.pop(request.request_id, None)

        return async_saves > 0, kv_txfer_params

    def request_finished(
        self,
        request: "Request",
        blocks: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self._aggregate_request_finished(
            request,
            lambda c: c.request_finished(request, blocks),
        )

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if not self._all_support_hma:
            assert len(block_ids) == 1, (
                "HMA with multiple kv_cache_groups requires all "
                "sub-connectors to support HMA"
            )
            return self.request_finished(request, block_ids[0])

        return self._aggregate_request_finished(
            request,
            lambda c: cast(SupportsHMA, c).request_finished_all_groups(
                request, block_ids
            ),
        )

    def take_events(self) -> Iterable["KVCacheEvent"]:
        for c in self._connectors:
            yield from c.take_events()

    def has_pending_push_work(self) -> bool:
        return any(c.has_pending_push_work() for c in self._connectors)

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
            connector_id = c.__class__.__name__
            if connector_id in stats_by_connector.data:
                stats_by_connector[connector_id] = stats_by_connector[
                    connector_id
                ].aggregate(stats)
            else:
                stats_by_connector[connector_id] = stats
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
        seen_classes: set[type] = set()
        for connector_cls, temp_config in cls._get_connector_classes_and_configs(
            vllm_config
        ):
            if connector_cls in seen_classes:
                continue
            seen_classes.add(connector_cls)
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
