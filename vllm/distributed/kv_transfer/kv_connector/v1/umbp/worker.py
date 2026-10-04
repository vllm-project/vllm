# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared worker-side UMBP connector behavior."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

import torch

from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash

from .data import (
    BlockIdentityCodec,
    BlockLoadBatch,
    BlockTransferPlan,
    KVLayoutPlanner,
    StoreEventResult,
    TransferJobState,
    TransferJobStatus,
    UMBPConnectorMetadata,
    UMBPConnectorWorkerMetadata,
)
from .runtime import UMBPWorkerHandle
from .stats import UMBPStoreConnectorStats

logger = init_logger(__name__)


@dataclass
class _StoreBatch:
    event_id: int
    jobs: dict[str, TransferJobState]
    failed_tokens: set[tuple[str, int]] = field(default_factory=set)


class UMBPStoreConnectorWorker:
    """Translate shared metadata into runtime load/store jobs."""

    def __init__(
        self,
        runtime: UMBPWorkerHandle,
        layout: KVLayoutPlanner | None = None,
        *,
        codec: BlockIdentityCodec | None = None,
        layerwise_load: bool = True,
        layerwise_store: bool = False,
        enable_kv_cache_events: bool = False,
    ) -> None:
        self.runtime = runtime
        self.layout = layout
        self.codec = codec
        self.layerwise_load = layerwise_load
        self.layerwise_store = layerwise_store
        self.enable_kv_cache_events = enable_kv_cache_events
        self._load_jobs: dict[str, dict[str | None, TransferJobState]] = {}
        self._store_jobs: dict[str, TransferJobState] = {}
        self._pending_stores: list[_StoreBatch] = []
        self._layer_store_jobs: dict[str, TransferJobState] = {}
        self._layer_store_plans: dict[str, BlockTransferPlan] = {}
        self._submitted_store_layers: set[str] = set()
        self._worker_meta = UMBPConnectorWorkerMetadata()
        self._kv_events: list[KVCacheEvent] = []
        self._finished_recving: set[str] = set()
        self._failed_recving: set[str] = set()
        self._report_load_completions = False
        self._active_store_event = -1
        self._store_failed_tokens: set[tuple[str, int]] = set()
        self._stats = UMBPStoreConnectorStats()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self.layout is not None and self.layout.regions:
            self.layout.register_kv_caches(kv_caches)
        self.runtime.register_buffers(kv_caches)

    def _materialize_plans(
        self, plans: list[BlockTransferPlan]
    ) -> list[BlockTransferPlan]:
        plans = [self._localize_plan(plan) for plan in plans]
        if self.layout is None or not self.layout.regions:
            return plans
        materialized: list[BlockTransferPlan] = []
        for plan in plans:
            if plan.ranges:
                materialized.append(plan)
                continue
            materialized.append(self.layout.materialize(plan))
        return materialized

    def _localize_plan(self, plan: BlockTransferPlan) -> BlockTransferPlan:
        if self.codec is None or plan.group_id is None:
            return plan
        key = self._localize_key(plan.key, plan.group_id)
        if key == plan.key:
            return plan
        return replace(
            plan,
            key=key,
            logical_key=plan.logical_key or plan.key,
        )

    def _localize_key(self, key: str, group_id: int) -> str:
        if self.codec is None:
            return key
        try:
            block_hash = bytes.fromhex(key.rsplit(":", 1)[1])
        except (IndexError, ValueError):
            return key
        return self.codec.key(block_hash, group_id)

    @staticmethod
    def _completion_token(plan: BlockTransferPlan) -> tuple[str, int]:
        return (plan.logical_key or plan.key, plan.generation)

    def start_load_kv(
        self, forward_context: ForwardContext, metadata: UMBPConnectorMetadata
    ) -> None:
        del forward_context
        self._report_load_completions = metadata.async_load
        for request_id, plans in metadata.load_requests.items():
            if plans:
                self._submit_layer_loads(request_id, plans)

    def _submit_layer_loads(
        self, request_id: str, plans: list[BlockTransferPlan] | BlockLoadBatch
    ) -> None:
        if self._report_load_completions or not self.layerwise_load:
            load_blocks = getattr(self.runtime, "load_blocks", None)
            if load_blocks is not None:
                localized: list[BlockTransferPlan] | BlockLoadBatch
                if isinstance(plans, BlockLoadBatch):
                    localized = plans
                    if self.codec is not None:
                        localized = replace(
                            plans,
                            keys=[
                                self._localize_key(key, group)
                                for key, group in zip(
                                    plans.keys, plans.group_ids, strict=True
                                )
                            ],
                        )
                else:
                    localized = [self._localize_plan(plan) for plan in plans]
                job = load_blocks(localized)
                if job is not None:
                    self._stats.record("load", submitted=len(plans))
                    self._load_jobs[request_id] = {None: job}
                    return
        materialized = self._materialize_plans(list(plans))
        if self._report_load_completions or not self.layerwise_load:
            logger.debug(
                "UMBP bulk load submitted request=%s plans=%d ranges=%d bytes=%d",
                request_id,
                len(materialized),
                sum(len(plan.ranges) for plan in materialized),
                sum(item.length for plan in materialized for item in plan.ranges),
            )
            self._stats.record(
                "load",
                submitted=len(materialized),
            )
            self._load_jobs[request_id] = {None: self.runtime.load(materialized)}
            return
        plans_by_layer: dict[str, list[BlockTransferPlan]] = {}
        for plan in materialized:
            layer_names = {item.layer_name for item in plan.ranges}
            if not layer_names:
                plans_by_layer.setdefault("__bulk__", []).append(plan)
                continue
            for layer_name in layer_names:
                plans_by_layer.setdefault(layer_name, []).append(
                    replace(
                        plan,
                        ranges=tuple(
                            item
                            for item in plan.ranges
                            if item.layer_name == layer_name
                        ),
                    )
                )
        jobs = self._load_jobs.setdefault(request_id, {})
        for layer_name, layer_plans in plans_by_layer.items():
            self._stats.record(
                "load",
                submitted=len(layer_plans),
            )
            jobs[layer_name] = self.runtime.load(layer_plans)

    def _finish_job(
        self,
        request_id: str,
        result: TransferJobState,
        *,
        is_load: bool,
        publish: bool = True,
        record_bytes: bool = True,
    ) -> bool:
        operation = "load" if is_load else "store"
        self._stats.record(
            operation,
            completed=len(result.completed_keys),
            failed=len(result.failed_keys),
            num_bytes=result.completed_bytes if record_bytes else 0,
        )
        succeeded = result.status == TransferJobStatus.COMPLETED
        if not succeeded:
            logger.debug(
                "UMBP %s failed request=%s status=%s error=%s",
                operation,
                request_id,
                result.status.value,
                result.error,
            )
        if is_load:
            if not succeeded:
                self._failed_recving.add(request_id)
            self._worker_meta.failed_block_ids.update(result.failed_block_ids)
        elif succeeded:
            if publish:
                self.runtime.publish(result)
            if self.enable_kv_cache_events:
                for plan in result.plans:
                    if plan.key not in result.completed_keys or plan.block_hash is None:
                        continue
                    self._kv_events.append(
                        BlockStored(
                            block_hashes=[
                                maybe_convert_block_hash(
                                    cast(BlockHash, plan.block_hash)
                                )
                            ],
                            parent_block_hash=(
                                maybe_convert_block_hash(
                                    cast(BlockHash, plan.parent_block_hash)
                                )
                                if plan.parent_block_hash is not None
                                else None
                            ),
                            token_ids=list(plan.token_ids),
                            block_size=plan.block_size,
                            lora_id=None,
                            medium=plan.medium,
                            lora_name=None,
                            group_idx=plan.group_id,
                            locality="LOCAL",
                        )
                    )
        return succeeded

    def wait_for_layer_load(self, layer_name: str) -> None:
        self._drain_load_jobs(wait=True, layer_name=layer_name)

    def save_kv_layer(
        self,
        metadata: UMBPConnectorMetadata,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        del kv_layer, attn_metadata, kwargs
        if not self.layerwise_store:
            return
        layer_plans = metadata.store_plans_by_layer.get(layer_name, ())
        if not layer_plans:
            return
        materialized = self._materialize_layer_plans(layer_plans, layer_name)
        if not materialized:
            return
        for plan in layer_plans:
            self._layer_store_plans.setdefault(plan.key, plan)
        self._stats.record(
            "store",
            submitted=len(materialized),
        )
        self._layer_store_jobs[layer_name] = self.runtime.store(materialized)
        self._submitted_store_layers.add(layer_name)

    def wait_for_save(self) -> None:
        if not self.layerwise_store:
            if self._store_jobs or self._active_store_event >= 0:
                self._pending_stores.append(
                    _StoreBatch(self._active_store_event, self._store_jobs)
                )
                self._store_jobs = {}
                self._active_store_event = -1
            self._drain_store_jobs(wait=False)
            return

        layer_results = [
            self.runtime.wait(job) for job in self._layer_store_jobs.values()
        ]
        if layer_results:
            self._stats.record(
                "store",
                num_bytes=sum(
                    item.length
                    for result in layer_results
                    for plan in result.plans
                    if plan.key in result.completed_keys
                    for item in plan.ranges
                ),
            )
        layers_succeeded = all(
            result.status == TransferJobStatus.COMPLETED for result in layer_results
        )
        if layers_succeeded:
            for result in layer_results:
                self.runtime.publish(result)
        if self._layer_store_plans:
            aggregate = TransferJobState(tuple(self._layer_store_plans.values()))
            aggregate.start()
            if layers_succeeded:
                aggregate.complete()
            else:
                aggregate.fail(list(self._layer_store_plans), "layer-wise store failed")
            if not self._finish_job(
                "__layerwise_store__",
                aggregate,
                is_load=False,
                publish=False,
                record_bytes=False,
            ):
                self._store_failed_tokens.update(
                    self._completion_token(plan) for plan in aggregate.plans
                )
        self._layer_store_jobs.clear()
        self._layer_store_plans.clear()
        self._submitted_store_layers.clear()
        for request_id, job in self._store_jobs.items():
            result = self.runtime.wait(job)
            if not self._finish_job(request_id, result, is_load=False):
                self._store_failed_tokens.update(
                    self._completion_token(plan) for plan in result.plans
                )
        self._store_jobs.clear()
        if self._active_store_event >= 0:
            self._worker_meta.store_events[self._active_store_event] = StoreEventResult(
                completed_workers=1, failed_tokens=self._store_failed_tokens
            )
            self._active_store_event = -1
        self._store_failed_tokens = set()

    def _drain_store_jobs(self, *, wait: bool) -> None:
        for batch in list(self._pending_stores):
            for request_id, job in list(batch.jobs.items()):
                result = self.runtime.wait(job) if wait else self.runtime.poll(job)
                if result is None:
                    continue
                if not self._finish_job(request_id, result, is_load=False):
                    # Partial jobs are not published, including successful keys.
                    batch.failed_tokens.update(
                        self._completion_token(plan) for plan in result.plans
                    )
                del batch.jobs[request_id]
            if not batch.jobs:
                if batch.event_id >= 0:
                    self._worker_meta.store_events[batch.event_id] = StoreEventResult(
                        completed_workers=1, failed_tokens=batch.failed_tokens
                    )
                self._pending_stores.remove(batch)

    def _materialize_layer_plans(
        self, plans: Sequence[BlockTransferPlan], layer_name: str
    ) -> list[BlockTransferPlan]:
        materialized: list[BlockTransferPlan] = []
        for full_plan in self._materialize_plans(list(plans)):
            layer_ranges = tuple(
                item for item in full_plan.ranges if item.layer_name == layer_name
            )
            if layer_ranges:
                materialized.append(replace(full_plan, ranges=layer_ranges))
        return materialized

    def _store_was_submitted_layerwise(self, plan: BlockTransferPlan) -> bool:
        materialized = self._materialize_plans([plan])
        required_layers = {
            item.layer_name for full_plan in materialized for item in full_plan.ranges
        }
        return bool(required_layers) and required_layers.issubset(
            self._submitted_store_layers
        )

    def _submit_store_plans(
        self, plans: list[BlockTransferPlan], request_id: str
    ) -> None:
        if not plans:
            return
        materialized = self._materialize_plans(plans)
        self._stats.record(
            "store",
            submitted=len(materialized),
        )
        self._store_jobs[request_id] = self.runtime.store(materialized)

    def enqueue_stores(self, metadata: UMBPConnectorMetadata) -> None:
        self._active_store_event = metadata.store_event
        store_requests = {
            request_id: list(plans)
            for request_id, plans in metadata.store_requests.items()
        }
        grouped = {plan for plans in store_requests.values() for plan in plans}
        for plan in metadata.store_plans:
            if plan not in grouped:
                store_requests.setdefault(
                    plan.request_id or "__umbp_batch__", []
                ).append(plan)
                grouped.add(plan)
        for request_id, plans in store_requests.items():
            if self.layerwise_store:
                plans = [
                    plan
                    for plan in plans
                    if not self._store_was_submitted_layerwise(plan)
                ]
            self._submit_store_plans(plans, request_id)

    def handle_preemptions(self, metadata: UMBPConnectorMetadata) -> None:
        """Cancel request-local jobs before vLLM reuses their GPU blocks."""
        if not (metadata.preempted_block_ids or metadata.preempted_request_ids):
            return
        if not metadata.preempted_request_ids:
            self.wait_for_layer_load("")
            self.wait_for_save()
            self._drain_store_jobs(wait=True)
            return

        preempted = metadata.preempted_request_ids
        for request_id in preempted:
            for job in self._load_jobs.pop(request_id, {}).values():
                self._cancel_job(job)
        for jobs in [self._store_jobs, *(batch.jobs for batch in self._pending_stores)]:
            for request_id, job in jobs.items():
                if request_id in preempted or any(
                    plan.request_id in preempted for plan in job.plans
                ):
                    jobs[request_id] = self._cancel_job(job)
        for layer_name, job in self._layer_store_jobs.items():
            if any(plan.request_id in preempted for plan in job.plans):
                self._layer_store_jobs[layer_name] = self._cancel_job(job)
        self._submitted_store_layers.clear()

    def _cancel_job(self, job: TransferJobState) -> TransferJobState:
        cancel = getattr(self.runtime, "cancel", None)
        if callable(cancel):
            return cancel(job)
        return self.runtime.wait(job)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        del finished_req_ids
        self._drain_load_jobs(wait=False)
        self._drain_store_jobs(wait=False)
        recving = set(self._finished_recving)
        self._finished_recving.difference_update(recving)
        return None, recving or None

    def _drain_load_jobs(self, *, wait: bool, layer_name: str = "") -> None:
        layers = {layer for jobs in self._load_jobs.values() for layer in jobs}
        selected = layers
        if wait and layer_name:
            if layer_name in layers:
                selected = {None, layer_name}
            elif "__bulk__" in layers:
                selected = {None, "__bulk__"}
        for request_id, jobs in list(self._load_jobs.items()):
            for layer, job in list(jobs.items()):
                if layer not in selected:
                    continue
                result = self.runtime.wait(job) if wait else self.runtime.poll(job)
                if result is None:
                    continue
                del jobs[layer]
                self._finish_job(request_id, result, is_load=True)
            if not jobs:
                del self._load_jobs[request_id]
                if self._report_load_completions:
                    self._finished_recving.add(request_id)

    def get_failed_recving(self) -> set[str]:
        failed = self._failed_recving - self._load_jobs.keys()
        self._failed_recving.difference_update(failed)
        return failed

    def build_connector_worker_meta(self) -> UMBPConnectorWorkerMetadata:
        result = self._worker_meta
        self._worker_meta = UMBPConnectorWorkerMetadata()
        return result

    def get_kv_events(self) -> list[KVCacheEvent]:
        take_evicted = getattr(self.runtime, "take_evicted_keys", None)
        if callable(take_evicted):
            for key in take_evicted():
                if not self.enable_kv_cache_events:
                    continue
                parsed = self._parse_key(key)
                if parsed is None:
                    continue
                group_id, block_hash = parsed
                self._kv_events.append(
                    BlockRemoved(
                        block_hashes=[
                            maybe_convert_block_hash(cast(BlockHash, block_hash))
                        ],
                        medium="CPU",
                        group_idx=group_id,
                        locality="LOCAL",
                    )
                )
        events = self._kv_events
        self._kv_events = []
        return events

    @staticmethod
    def _parse_key(key: str) -> tuple[int, bytes] | None:
        try:
            group_part, hash_hex = key.rsplit(":", 1)
            group_id = int(group_part.rsplit(":g", 1)[1])
            return group_id, bytes.fromhex(hash_hex)
        except (IndexError, ValueError):
            return None

    def get_block_ids_with_load_errors(self) -> set[int]:
        failed = self._worker_meta.failed_block_ids
        self._worker_meta.failed_block_ids = set()
        return failed

    def get_kv_connector_stats(self) -> UMBPStoreConnectorStats | None:
        if self._stats.is_empty():
            return None
        result = UMBPStoreConnectorStats(data=dict(self._stats.data))
        self._stats.reset()
        return result

    def close(self) -> None:
        self.wait_for_save()
        self._drain_store_jobs(wait=True)
        self._drain_load_jobs(wait=True)
        self.runtime.close()
