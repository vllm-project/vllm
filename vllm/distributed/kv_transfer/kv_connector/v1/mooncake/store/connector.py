# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from vllm-project/vllm-ascend
# (vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/).
"""MooncakeStoreConnector - KV cache connector using MooncakeDistributedStore.

Unlike MooncakeConnector which does direct P2P transfer, this connector
uses MooncakeDistributedStore as a shared KV cache pool. Both producer
and consumer instances read/write KV to/from the store independently,
enabling prefix caching via hash-based deduplication.
"""

from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockStored,
    KVCacheEvent,
    KVConnectorKVEvents,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from .data import (
    BlockKey,
    BoundaryStoreStats,
    MooncakeStoreConnectorMetadata,
    StoreResidency,
    store_block_key,
)
from .metrics import MooncakeStoreConnectorStats, MooncakeStorePromMetrics
from .scheduler import MooncakeStoreScheduler
from .worker import MooncakeStoreWorker

logger = init_logger(__name__)


@dataclass
class _PendingResidency:
    """One logical block waiting for its group's lookup namespaces to report."""

    event: BlockStored
    covered: set[str]


class MooncakeStoreKVEvents(KVConnectorKVEvents):
    """Cross-rank aggregation of Mooncake Store residency events.

    A ``BlockStored`` from this connector claims the Store holds a complete copy
    of one logical block for one cache group. Complete means every rank namespace
    a later lookup probes for that group holds it, which is the prefix set
    ``MooncakeStoreWorker._lookup_key_prefixes`` builds from the group's TP
    replication factor and the engine's PCP/DCP/PP layout. Each rank therefore
    reports the namespaces its committed Store objects cover, and a block is
    released once their union spans its group's probe set.

    Coverage accumulates across engine steps and is retained until the block is
    released, so stores that land in different steps still announce the block
    once, after the last namespace reports. Repeated reports of a block add
    coverage only: the payload announced is the first one seen, and the block is
    announced once per residency epoch, which ``AllBlocksCleared`` ends.

    An instance is reached by one thread at a time: a worker fills a container of
    its own per poll, and the engine's aggregation and the scheduler's step run
    the containers they own serially.
    """

    def __init__(self, num_workers: int = 1) -> None:
        if num_workers <= 0:
            raise ValueError("num_workers must be greater than zero.")
        # Bookkeeping for the connector framework: a block is released on the
        # namespaces that cover it, never on a count of reporting workers.
        self._num_workers = num_workers
        # Blocks whose coverage is still short of their group's probe set.
        self._pending: dict[BlockKey, _PendingResidency] = {}
        # Blocks released in this residency epoch, drained or not.
        self._released: set[BlockKey] = set()
        # Released blocks and clear events awaiting the publisher.
        self._ready: list[KVCacheEvent] = []
        # Per group, the namespaces a later lookup probes.
        self._required: dict[int, frozenset[str]] = {}

    def add_events(self, events: list[KVCacheEvent]) -> None:
        """Record events that carry no cross-rank coverage contract.

        ``AllBlocksCleared`` is the one event that qualifies: it ends the
        residency epoch, dropping what was learned before the Store was wiped so
        the same block is announced again once it is stored into the fresh Store.
        A ``BlockStored`` is released on the namespaces its rank committed, which
        a bare event cannot express; ranks report blocks through
        :meth:`add_residency`.
        """
        if not isinstance(events, list):
            raise TypeError("events must be a list of KVCacheEvent.")
        for event in events:
            if not isinstance(event, AllBlocksCleared):
                raise ValueError(
                    f"{type(event).__name__} must carry the Store namespaces "
                    f"that cover it; report blocks through add_residency()"
                )
            self._retire_epoch()
            self._ready.append(event)

    def add_residency(
        self,
        residency: StoreResidency,
        required: Mapping[int, frozenset[str]],
    ) -> None:
        """Record the blocks one rank committed, and the namespaces covering them.

        Args:
            residency: The blocks this rank finished writing, and per block the
                Store key namespaces its own objects occupy.
            required: Per cache group, the namespaces a later lookup probes.

        """
        self._merge_required(required)
        for event in residency.events:
            key = store_block_key(event)
            namespaces = residency.covered.get(key)
            if namespaces is None:
                raise ValueError(
                    f"residency for group {key[0]} block {key[1]} reports no "
                    f"covered Store namespaces"
                )
            self._record(key, event, namespaces)

    def merge(self, other: "KVConnectorKVEvents") -> "MooncakeStoreKVEvents":
        """Fold another container's contributions into this one."""
        if not isinstance(other, MooncakeStoreKVEvents):
            raise TypeError(
                f"cannot merge {type(other).__name__} into MooncakeStoreKVEvents"
            )
        self._merge_required(other._required)
        # `other` queues the blocks it released before it retired its own epoch
        # ahead of the clear itself, so replaying them in order keeps pre-clear
        # releases ahead of the clear while the coverage it accumulated
        # afterwards lands in the new epoch.
        for event in other._ready:
            if isinstance(event, AllBlocksCleared):
                self._retire_epoch()
                self._ready.append(event)
                continue
            self._release(store_block_key(event), event)
        self._released |= other._released
        for key, pending in other._pending.items():
            self._record(key, pending.event, frozenset(pending.covered))
        return self

    def _merge_required(self, required: Mapping[int, frozenset[str]]) -> None:
        for group_idx, namespaces in required.items():
            known = self._required.get(group_idx)
            if known is not None and known != namespaces:
                raise ValueError(
                    f"cache group {group_idx} lookup namespaces differ between "
                    f"ranks: {sorted(known)} != {sorted(namespaces)}"
                )
            self._required[group_idx] = namespaces

    def _record(
        self,
        key: BlockKey,
        event: BlockStored,
        namespaces: frozenset[str],
    ) -> None:
        """Fold one rank's coverage of one logical block into the pending state."""
        if key in self._released:
            # Already announced this epoch; a late report adds nothing.
            return
        probe_set = self._required.get(key[0])
        if probe_set is None:
            raise ValueError(f"no lookup namespaces reported for cache group {key[0]}")
        unknown = namespaces - probe_set
        if unknown:
            raise ValueError(
                f"residency for group {key[0]} covers {sorted(unknown)}, which a "
                f"lookup for that group never probes"
            )
        pending = self._pending.get(key)
        if pending is None:
            pending = _PendingResidency(event, set())
            self._pending[key] = pending
        pending.covered |= namespaces
        if probe_set <= pending.covered:
            del self._pending[key]
            self._release(key, pending.event)

    def _release(self, key: BlockKey, event: BlockStored) -> None:
        """Announce one block for the rest of this residency epoch."""
        if key in self._released:
            return
        self._released.add(key)
        self._pending.pop(key, None)
        self._ready.append(event)

    def _retire_epoch(self) -> None:
        """Forget everything learned before the current residency epoch ended.

        Pending coverage goes, so a store that was still incomplete when the
        Store was wiped is not announced afterwards, and the released ledger
        goes, so a block stored into the fresh Store is announced again.
        """
        self._pending.clear()
        self._released.clear()

    def pop_ready_events(self) -> list[KVCacheEvent]:
        """Remove and return the events ready for the publisher."""
        events = self._ready
        self._ready = []
        return events

    def aggregate(self) -> "MooncakeStoreKVEvents":
        """Return self; a block is released as soon as its coverage is complete."""
        return self

    def increment_workers(self, count: int = 1) -> None:
        if count <= 0:
            raise ValueError("count must be positive.")
        self._num_workers += count

    def get_all_events(self) -> list[KVCacheEvent]:
        events: list[KVCacheEvent] = [p.event for p in self._pending.values()]
        events.extend(self._ready)
        return events

    def get_number_of_workers(self) -> int:
        return self._num_workers

    def clear_events(self) -> None:
        """Retire the residency epoch and drop every queued event."""
        self._retire_epoch()
        self._ready.clear()

    def __repr__(self) -> str:
        return (
            f"<MooncakeStoreKVEvents workers={self._num_workers} "
            f"pending={len(self._pending)} released={len(self._released)} "
            f"ready={len(self._ready)}>"
        )


class MooncakeStoreConnector(KVConnectorBase_V1, SupportsHMA):
    """KV connector using MooncakeDistributedStore as shared KV pool."""

    @staticmethod
    def _validate_kv_cache_config(
        vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
    ) -> None:
        from vllm.v1.kv_cache_interface import CrossAttentionSpec, MambaSpec

        unsupported: list[str] = []
        store_group_ids = kv_cache_config.prefix_cacheable_group_ids
        if not store_group_ids:
            raise ValueError(
                "MooncakeStore requires at least one prefix-cacheable KV cache group"
            )
        for group_id in store_group_ids:
            spec = kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
            if isinstance(spec, CrossAttentionSpec):
                unsupported.append(f"group {group_id}: CrossAttentionSpec")
            if isinstance(spec, MambaSpec) and spec.mamba_cache_mode != "align":
                unsupported.append(
                    f"group {group_id}: mamba_cache_mode="
                    f"{spec.mamba_cache_mode!r} != 'align'"
                )
        pcp = vllm_config.parallel_config.prefill_context_parallel_size
        if len(store_group_ids) > 1 and pcp > 1:
            unsupported.append(f"PCP > 1 (pcp={pcp}) with hybrid attention")
        if unsupported:
            raise ValueError(
                "MooncakeStoreConnector does not support: " + "; ".join(unsupported)
            )

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,  # type: ignore[arg-type]
        )
        assert vllm_config.kv_transfer_config is not None
        assert kv_cache_config is not None, "kv_cache_config is required"
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        save_decode_cache = extra_config.get("save_decode_cache", False)
        # Capacity-only: contributes its segment to the store pool but transfers
        # no KV, so the KV-cache-shape invariants below cannot be reached.
        self._capacity_only = (
            self.kv_role == "kv_consumer"
            and not extra_config.get("enable_lookup", True)
            and not save_decode_cache
        )
        if not self._capacity_only:
            self._validate_kv_cache_config(vllm_config, kv_cache_config)
        self._kv_cache_config = kv_cache_config
        self._kv_cache_events: MooncakeStoreKVEvents | None = None

        self.connector_scheduler: MooncakeStoreScheduler | None = None
        self.connector_worker: MooncakeStoreWorker | None = None

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = MooncakeStoreScheduler(
                vllm_config, kv_cache_config
            )
        else:
            self.connector_worker = MooncakeStoreWorker(vllm_config, kv_cache_config)

    def shutdown(self):
        """Release connector resources on teardown.

        Closes the worker's MooncakeDistributedStore handle so its
        TransferEngine and RDMA registrations are released. Invoked from the
        engine's explicit shutdown path and as a backstop from ``__del__``;
        a no-op on the scheduler role, which holds no store handle.
        """
        worker = getattr(self, "connector_worker", None)
        if worker is not None:
            worker.close()

    def __del__(self):
        self.shutdown()

    # ============================================================
    # Scheduler-side methods
    # ============================================================

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
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
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.bind_gpu_block_pool(gpu_block_pool)

    def has_pending_push_work(self) -> bool:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.has_pending_push_work()

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        assert self.connector_worker is not None
        return self.connector_worker.build_connector_worker_meta()

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished_all_groups(request, (block_ids,))

    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        # An in-flight store job holds its own reference on the blocks it reads,
        # so a finishing request never has to defer freeing them.
        return False, None

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

    def reset_cache(self) -> bool | None:
        """Reset the external Mooncake store on prefix-cache reset.

        Drains the worker send queue, then runs ``remove_all`` on the
        Mooncake master. Caller must first pause generation (e.g.
        ``pause_generation``) so no new puts are enqueued during drain.

        Returns True on ack, False on failure, None for the worker role.
        """
        if self.role == KVConnectorRole.SCHEDULER:
            assert self.connector_scheduler is not None
            # Clear local references to keys we're about to wipe.
            self.connector_scheduler.load_specs.clear()
            reset_ok = self.connector_scheduler.reset_store()
            if reset_ok:
                # A wiped Store opens a new residency epoch: coverage that was
                # still pending when it was wiped must not be announced, and a
                # block stored again afterwards is announced again. A failed
                # reset leaves the accumulator alone, since the Store still
                # holds what it announced.
                self._kv_cache_events = None
            return reset_ok
        return None

    def update_connector_output(self, connector_output: KVConnectorOutput):
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)

        kv_cache_events = connector_output.kv_cache_events
        if not kv_cache_events or not isinstance(
            kv_cache_events, MooncakeStoreKVEvents
        ):
            return

        if self._kv_cache_events is None:
            # The accumulator outlives the step that first feeds it: a block's
            # coverage is only complete once every rank namespace has reported,
            # which can take several engine steps.
            self._kv_cache_events = MooncakeStoreKVEvents()
        self._kv_cache_events.merge(kv_cache_events)

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Drain the Store blocks whose cross-rank coverage is complete."""
        if self._kv_cache_events is None:
            return ()
        return self._kv_cache_events.pop_ready_events()

    def get_boundary_store_stats(self) -> BoundaryStoreStats | None:
        """Return a snapshot of the mamba boundary hand-off counters."""
        if self.connector_scheduler is not None:
            return self.connector_scheduler.get_boundary_store_stats()
        return None

    # ============================================================
    # Worker-side methods
    # ============================================================

    def get_mem_pool_context(self) -> AbstractContextManager | None:
        """Return a context manager for the custom MemPool, or None.

        Called by the Worker before ``initialize_kv_cache`` so that KV
        cache is allocated from the Mooncake-managed pool when
        ``custom_mem_pool`` is set in ``kv_connector_extra_config``.
        """
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_mem_pool_context()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MooncakeStoreConnectorMetadata)
        self.connector_worker.start_load_kv(metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        # No layerwise support - no-op
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        # No layerwise support - no-op
        return

    def wait_for_save(self):
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MooncakeStoreConnectorMetadata)
        self.connector_worker.wait_for_save(metadata)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MooncakeStoreConnectorMetadata)
        return self.connector_worker.get_finished(finished_req_ids, metadata)

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MooncakeStoreConnectorMetadata)
        return self.connector_worker.get_transfer_results(finished_req_ids, metadata)

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def get_kv_connector_kv_cache_events(
        self,
    ) -> MooncakeStoreKVEvents | None:
        assert self.connector_worker is not None
        if (
            not self.connector_worker.enable_kv_events
            or self.connector_worker.kv_send_thread is None
        ):
            return None
        # A worker with nothing to report still polls an empty container: with
        # coverage, a step that contributes no namespace neither completes nor
        # resets another rank's contribution.
        kv_events = MooncakeStoreKVEvents(num_workers=1)
        kv_events.add_residency(
            self.connector_worker.drain_residency(),
            {
                group_idx: frozenset(prefixes)
                for group_idx, prefixes in enumerate(
                    self.connector_worker.lookup_key_prefixes
                )
            },
        )
        return kv_events

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_kv_connector_stats()

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        return (
            MooncakeStoreConnectorStats(data=data)
            if data is not None
            else MooncakeStoreConnectorStats()
        )

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return MooncakeStorePromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )
