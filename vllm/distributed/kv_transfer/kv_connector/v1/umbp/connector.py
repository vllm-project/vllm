# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-facing shared UMBP connector."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent, KVConnectorKVEvents
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    PromMetric,
    PromMetricT,
)
from vllm.forward_context import ForwardContext
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

from .data import (
    BlockIdentityCodec,
    KVLayoutPlanner,
    RankTopology,
    UMBPConnectorMetadata,
    UMBPConnectorWorkerMetadata,
    UMBPNamespace,
)
from .runtime import UMBPRuntimeConfig, UMBPRuntimeFactory
from .scheduler import UMBPStoreConnectorScheduler
from .stats import UMBPStoreConnectorStats, UMBPStorePromMetrics
from .worker import UMBPStoreConnectorWorker

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


class UMBPStoreKVEvents(KVConnectorKVEvents):
    """UMBP events, released once every worker has reported them.

    Each rank stores its own shard of a block and emits the same event for it,
    possibly in different steps, so an event waits until all ranks sent it.
    Every worker returns a container each step, even an empty one, so the
    worker count of a step's combined container is the world size.
    """

    def __init__(self, events: list[KVCacheEvent] | None = None) -> None:
        self._event_counter: Counter[KVCacheEvent] = Counter(events or [])
        self._num_workers = 1

    def add_events(self, events: list[KVCacheEvent]) -> None:
        self._event_counter.update(events)

    def aggregate(self) -> UMBPStoreKVEvents:
        common_events = self.pop_common_events()
        self._event_counter = Counter(common_events)
        self._num_workers = 1
        return self

    def pop_common_events(self) -> list[KVCacheEvent]:
        common_events = [
            event
            for event, count in self._event_counter.items()
            if count >= self._num_workers
        ]
        for event in common_events:
            del self._event_counter[event]
        return common_events

    def has_events(self) -> bool:
        return bool(self._event_counter)

    def increment_workers(self, count: int = 1) -> None:
        if count <= 0:
            raise ValueError("count must be positive")
        self._num_workers += count

    def get_all_events(self) -> list[KVCacheEvent]:
        return list(self._event_counter.elements())

    def get_number_of_workers(self) -> int:
        return self._num_workers

    def clear_events(self) -> None:
        self._event_counter.clear()
        self._num_workers = 1


class UMBPStoreConnector(KVConnectorBase_V1, SupportsHMA):
    """vLLM connector lifecycle shared by every UMBP runtime mode."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        topology = RankTopology.from_vllm_config(vllm_config)
        runtime_config = UMBPRuntimeConfig.from_vllm(
            vllm_config
        ).resolve_for_rank_count(topology.rank_count)
        if topology.pp_size > 1:
            raise NotImplementedError(
                f"{runtime_config.mode} UMBP does not support pipeline parallelism"
            )
        # A restored prefix skips the draft model's prefill too, so its KV must
        # come from the pool along with the target's.
        cacheable = set(kv_cache_config.prefix_cacheable_group_ids)
        skipped_drafts = [
            group_id
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if getattr(group, "is_eagle_group", False) and group_id not in cacheable
        ]
        if skipped_drafts:
            raise NotImplementedError(
                f"{runtime_config.mode} UMBP cannot restore draft-model KV cache "
                f"groups {skipped_drafts}: they are not prefix-cacheable"
            )
        runtime = UMBPRuntimeFactory.build(runtime_config)
        namespace = UMBPNamespace.from_vllm_config(vllm_config, kv_cache_config)
        codec = BlockIdentityCodec(
            namespace=namespace,
            tp_rank=topology.tp_rank,
            pp_rank=topology.pp_rank,
            pcp_rank=topology.pcp_rank,
            dcp_rank=topology.dcp_rank,
        )
        layout = KVLayoutPlanner.from_kv_cache_config(kv_cache_config)
        layout_descriptor = layout.describe(topology)
        self._runtime = runtime
        self._store_submitted = False
        self._kv_cache_events: UMBPStoreKVEvents | None = None
        self.connector_scheduler: UMBPStoreConnectorScheduler | None = None
        self.connector_worker: UMBPStoreConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = UMBPStoreConnectorScheduler(
                vllm_config,
                kv_cache_config,
                runtime.create_scheduler_handle(
                    namespace.value, topology, layout_descriptor
                ),
                codec,
                topology,
            )
        else:
            self.connector_worker = UMBPStoreConnectorWorker(
                runtime.create_worker_handle(
                    namespace.value, topology, layout_descriptor
                ),
                layout,
                codec=codec,
                report_failed_requests=len(kv_cache_config.kv_cache_groups) > 1,
                enable_kv_cache_events=bool(
                    vllm_config.kv_events_config
                    and vllm_config.kv_events_config.enable_kv_cache_events
                ),
            )

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def bind_gpu_block_pool(self, gpu_block_pool: Any) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.bind_gpu_block_pool(gpu_block_pool)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, (block_ids,))

    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def register_finished_partial_tail(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> bool:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.register_finished_partial_tail(
            request, block_ids, partial_tail_offloads
        )

    def update_connector_output(self, connector_output: Any) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)
        events = connector_output.kv_cache_events
        if not isinstance(events, UMBPStoreKVEvents):
            return
        if self._kv_cache_events is None:
            self._kv_cache_events = events
        else:
            self._kv_cache_events.add_events(events.get_all_events())

    def take_events(self) -> Iterable[KVCacheEvent]:
        if self._kv_cache_events is None:
            return
        yield from self._kv_cache_events.pop_common_events()
        if not self._kv_cache_events.has_events():
            self._kv_cache_events = None

    def has_pending_push_work(self) -> bool:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.has_pending_push_work()

    def reset_cache(self) -> bool | None:
        if self.connector_scheduler is None:
            return None
        return self.connector_scheduler.reset_store()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        super().bind_connector_metadata(connector_metadata)
        self._store_submitted = False

    def clear_connector_metadata(self) -> None:
        # No-forward steps skip the model runner's wait_for_save hook.
        if self.connector_worker is not None and self.has_connector_metadata():
            self.wait_for_save()
        super().clear_connector_metadata()

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        assert self.connector_worker is not None
        assert isinstance(kv_connector_metadata, UMBPConnectorMetadata)
        self.connector_worker.handle_preemptions(kv_connector_metadata)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        del kwargs
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UMBPConnectorMetadata)
        self.connector_worker.start_load_kv(forward_context, metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        assert self.connector_worker is not None
        self.connector_worker.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        # Stores are submitted once per step, from wait_for_save.
        return

    def wait_for_save(self) -> None:
        if self._store_submitted:
            return
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UMBPConnectorMetadata)
        self.connector_worker.enqueue_stores(metadata)
        self.connector_worker.wait_for_save()
        self._store_submitted = True

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        assert self.connector_worker is not None
        finished_sending, finished_recving = self.connector_worker.get_finished(
            finished_req_ids
        )
        return KVConnectorTransferResults(
            finished_sending=set(finished_sending or ()),
            finished_recving=set(finished_recving or ()),
            failed_recving=self.connector_worker.get_failed_recving(),
        )

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def build_connector_worker_meta(self) -> UMBPConnectorWorkerMetadata:
        assert self.connector_worker is not None
        return self.connector_worker.build_connector_worker_meta()

    def get_kv_connector_stats(self) -> UMBPStoreConnectorStats | None:
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_kv_connector_stats()

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> UMBPStoreConnectorStats | None:
        return UMBPStoreConnectorStats(data=data or {})

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> UMBPStorePromMetrics:
        return UMBPStorePromMetrics(
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
        )

    def get_kv_connector_kv_cache_events(self) -> UMBPStoreKVEvents | None:
        assert self.connector_worker is not None
        events = self.connector_worker.get_kv_events()
        if not self.connector_worker.enable_kv_cache_events:
            return None
        # An empty container still counts this worker toward the quorum.
        return UMBPStoreKVEvents(events)

    def shutdown(self) -> None:
        if self.connector_scheduler is not None:
            self.connector_scheduler.close()
        if self.connector_worker is not None:
            self.connector_worker.close()
