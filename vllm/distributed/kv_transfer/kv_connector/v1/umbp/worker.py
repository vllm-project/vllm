# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared worker-side UMBP connector behavior."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast

import torch

from vllm.distributed.kv_events import BlockRemoved, BlockStored
from vllm.forward_context import ForwardContext
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash

from .data import (
    BlockTransferPlan,
    KVLayoutPlanner,
    PartialTailPlan,
    TransferJobState,
    TransferJobStatus,
    UMBPConnectorMetadata,
    UMBPConnectorWorkerMetadata,
)
from .runtime import UMBPWorkerHandle
from .stats import UMBPStoreConnectorStats


class UMBPStoreConnectorWorker:
    """Translate shared metadata into runtime load/store jobs."""

    def __init__(
        self,
        runtime: UMBPWorkerHandle,
        layout: KVLayoutPlanner | None = None,
        *,
        layerwise_load: bool = True,
        layerwise_store: bool = False,
    ) -> None:
        self.runtime = runtime
        self.layout = layout
        self.layerwise_load = layerwise_load
        self.layerwise_store = layerwise_store
        self._load_jobs: dict[str, TransferJobState] = {}
        self._layer_load_jobs: dict[str, dict[str, TransferJobState]] = {}
        self._pending_load_layers: dict[str, set[str]] = {}
        self._store_jobs: dict[str, TransferJobState] = {}
        self._layer_store_jobs: dict[str, TransferJobState] = {}
        self._layer_store_plans: dict[str, BlockTransferPlan] = {}
        self._submitted_store_layers: set[str] = set()
        self._worker_meta = UMBPConnectorWorkerMetadata()
        self._finished_sending: set[str] = set()
        self._finished_recving: set[str] = set()
        self._failed_recving: set[str] = set()
        self._report_load_completions = False
        self._report_store_completions: set[str] = set()
        self._stats = UMBPStoreConnectorStats()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self.layout is not None and self.layout.regions:
            self.layout.register_kv_caches(kv_caches)
        self.runtime.register_buffers(kv_caches)

    def _materialize_plans(
        self, plans: list[BlockTransferPlan]
    ) -> list[BlockTransferPlan]:
        if self.layout is None or not self.layout.regions:
            return plans
        return [
            plan
            if plan.ranges
            else self.layout.plan_registered_block(
                plan.key,
                plan.block_id,
                request_id=plan.request_id,
                generation=plan.generation,
                token_start=plan.token_start,
                token_end=plan.token_end,
            )
            for plan in plans
        ]

    def start_load_kv(
        self, forward_context: ForwardContext, metadata: UMBPConnectorMetadata
    ) -> None:
        del forward_context
        self._report_load_completions = metadata.async_load
        for request_id, plans in metadata.load_requests.items():
            if plans:
                self._submit_layer_loads(request_id, plans)
        if metadata.load_plans and not metadata.load_requests:
            self._submit_layer_loads("__umbp_batch__", metadata.load_plans)

    def _submit_layer_loads(
        self, request_id: str, plans: list[BlockTransferPlan]
    ) -> None:
        materialized = self._materialize_plans(plans)
        if not self.layerwise_load:
            self._load_jobs[request_id] = self.runtime.load(materialized)
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
        self._pending_load_layers[request_id] = set(plans_by_layer)
        for layer_name, layer_plans in plans_by_layer.items():
            self._stats.record(
                "load",
                submitted=len(layer_plans),
                num_bytes=sum(
                    item.length for plan in layer_plans for item in plan.ranges
                ),
            )
            self._layer_load_jobs.setdefault(layer_name, {})[request_id] = (
                self.runtime.load(layer_plans)
            )

    def _finish_job(
        self,
        request_id: str,
        job: TransferJobState,
        *,
        is_load: bool,
        mark_finished: bool = True,
        wait: bool = True,
        publish: bool = True,
    ) -> None:
        result = self.runtime.wait(job) if wait else job
        if is_load:
            self._stats.record(
                "load",
                completed=len(result.completed_keys),
                failed=len(result.failed_keys),
                num_bytes=sum(
                    item.length
                    for plan in result.plans
                    for item in plan.ranges
                    if plan.key in result.completed_keys
                ),
            )
            self._worker_meta.completed_loads.update(result.completed_keys)
            if mark_finished and self._report_load_completions:
                self._finished_recving.add(request_id)
            if result.status != TransferJobStatus.COMPLETED:
                self._failed_recving.add(request_id)
                for key in result.failed_keys:
                    self._worker_meta.failed_loads[key] = result.error or "load failed"
        else:
            self._stats.record(
                "store",
                completed=len(result.completed_keys),
                failed=len(result.failed_keys),
                num_bytes=sum(
                    item.length
                    for plan in result.plans
                    for item in plan.ranges
                    if plan.key in result.completed_keys
                ),
            )
            if publish and result.status == TransferJobStatus.COMPLETED:
                self.runtime.publish(result)
            self._worker_meta.completed_stores.update(result.completed_keys)
            for plan in result.plans:
                if plan.key not in result.completed_keys or plan.block_hash is None:
                    continue
                self._worker_meta.kv_events.append(
                    BlockStored(
                        block_hashes=[
                            maybe_convert_block_hash(cast(BlockHash, plan.block_hash))
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
            for key in result.completed_keys:
                self._worker_meta.completed_store_counts[key] = (
                    self._worker_meta.completed_store_counts.get(key, 0) + 1
                )
            for plan in result.plans:
                token = (plan.key, plan.generation)
                if plan.key in result.completed_keys:
                    self._worker_meta.completed_store_tokens[token] = (
                        self._worker_meta.completed_store_tokens.get(token, 0) + 1
                    )
            if result.status != TransferJobStatus.COMPLETED:
                failed_keys = {
                    plan.key
                    for plan in result.plans
                    if plan.key not in result.completed_keys
                }
                self._worker_meta.failed_stores.update(failed_keys)
                for key in failed_keys:
                    self._worker_meta.failed_store_counts[key] = (
                        self._worker_meta.failed_store_counts.get(key, 0) + 1
                    )
                for plan in result.plans:
                    if plan.key in failed_keys:
                        token = (plan.key, plan.generation)
                        self._worker_meta.failed_store_tokens[token] = (
                            self._worker_meta.failed_store_tokens.get(token, 0) + 1
                        )
                for key in failed_keys:
                    self._worker_meta.failed_store_errors[key] = (
                        result.error or "store failed"
                    )
            if mark_finished and request_id in self._report_store_completions:
                self._finished_sending.add(request_id)
        self._worker_meta.failed_block_ids.update(result.failed_block_ids)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self._layer_load_jobs:
            selected_layers = (
                set(self._layer_load_jobs)
                if not layer_name
                else (
                    {layer_name}
                    if layer_name in self._layer_load_jobs
                    else (
                        {"__bulk__"}
                        if "__bulk__" in self._layer_load_jobs
                        else set(self._layer_load_jobs)
                    )
                )
            )
            affected: set[str] = set()
            for selected_layer in selected_layers:
                jobs = self._layer_load_jobs.pop(selected_layer, {})
                for request_id, job in jobs.items():
                    affected.add(request_id)
                    self._finish_job(
                        request_id,
                        job,
                        is_load=True,
                        mark_finished=False,
                    )
                    pending = self._pending_load_layers.get(request_id)
                    if pending is not None:
                        pending.discard(selected_layer)
            for request_id in affected:
                if not self._pending_load_layers.get(request_id):
                    self._pending_load_layers.pop(request_id, None)
                    if self._report_load_completions:
                        self._finished_recving.add(request_id)
            return

        for request_id, job in self._load_jobs.items():
            self._finish_job(request_id, job, is_load=True)
        self._load_jobs.clear()

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
            num_bytes=sum(item.length for plan in materialized for item in plan.ranges),
        )
        self._layer_store_jobs[layer_name] = self.runtime.store(materialized)
        self._submitted_store_layers.add(layer_name)

    def wait_for_save(self) -> None:
        finished_requests: set[str] = set()
        layer_results = [
            self.runtime.wait(job) for job in self._layer_store_jobs.values()
        ]
        all_keys = set(self._layer_store_plans)
        failed_keys = {key for result in layer_results for key in result.failed_keys}
        completed_keys = all_keys - failed_keys
        if layer_results and not failed_keys:
            for result in layer_results:
                self.runtime.publish(result)
        if self._layer_store_plans:
            aggregate = TransferJobState(tuple(self._layer_store_plans.values()))
            aggregate.start()
            if completed_keys:
                aggregate.complete(list(completed_keys))
            if failed_keys:
                aggregate.fail(list(failed_keys), "layer-wise store failed")
            self._finish_job(
                "__layerwise_store__",
                aggregate,
                is_load=False,
                mark_finished=False,
                wait=False,
                publish=False,
            )
            finished_requests.update(
                plan.request_id
                for plan in aggregate.plans
                if plan.request_id is not None
            )
        self._layer_store_jobs.clear()
        self._layer_store_plans.clear()
        self._submitted_store_layers.clear()
        for request_id, job in self._store_jobs.items():
            self._finish_job(request_id, job, is_load=False)
            if request_id != "__umbp_batch__":
                finished_requests.add(request_id)
            else:
                finished_requests.update(
                    plan.request_id for plan in job.plans if plan.request_id is not None
                )
        self._store_jobs.clear()
        self._finished_sending.update(
            finished_requests & self._report_store_completions
        )
        self._report_store_completions.clear()

    def _materialize_layer_plans(
        self, plans: Sequence[BlockTransferPlan], layer_name: str
    ) -> list[BlockTransferPlan]:
        materialized: list[BlockTransferPlan] = []
        for plan in plans:
            for full_plan in self._materialize_plans([plan]):
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
            num_bytes=sum(item.length for plan in materialized for item in plan.ranges),
        )
        self._store_jobs[request_id] = self.runtime.store(materialized)

    def enqueue_stores(self, metadata: UMBPConnectorMetadata) -> None:
        self._report_store_completions = set(
            metadata.deferred_store_requests
        )
        store_requests = {
            request_id: list(plans)
            for request_id, plans in metadata.store_requests.items()
        }
        for partial_plan in metadata.partial_tail_plans:
            store_requests.setdefault(partial_plan.request_id, []).append(
                self._partial_tail_to_plan(partial_plan)
            )
        if self.layerwise_store:
            for request_id, plans in store_requests.items():
                remaining = [
                    plan
                    for plan in plans
                    if not self._store_was_submitted_layerwise(plan)
                ]
                self._submit_store_plans(remaining, request_id)
            if metadata.store_plans and not store_requests:
                remaining = [
                    plan
                    for plan in metadata.store_plans
                    if not self._store_was_submitted_layerwise(plan)
                ]
                self._submit_store_plans(remaining, "__umbp_batch__")
            return

        for request_id, plans in store_requests.items():
            self._submit_store_plans(plans, request_id)
        if metadata.store_plans and not store_requests:
            self._submit_store_plans(metadata.store_plans, "__umbp_batch__")

    @staticmethod
    def _partial_tail_to_plan(
        plan: PartialTailPlan,
    ) -> BlockTransferPlan:
        return BlockTransferPlan(
            key=plan.key,
            block_id=plan.block_id,
            request_id=plan.request_id,
            generation=plan.generation,
            group_id=plan.group_id,
            block_size=plan.block_size,
            token_start=plan.start_token % plan.block_size,
            token_end=plan.end_token % plan.block_size,
        )

    def handle_preemptions(self, metadata: UMBPConnectorMetadata) -> None:
        """Cancel request-local jobs before vLLM reuses their GPU blocks."""
        if not (metadata.preempted_block_ids or metadata.preempted_request_ids):
            return
        if not metadata.preempted_request_ids:
            self.wait_for_layer_load("")
            self.wait_for_save()
            return

        preempted = metadata.preempted_request_ids
        for layer_name in list(self._layer_load_jobs):
            jobs = self._layer_load_jobs[layer_name]
            for request_id in preempted & jobs.keys():
                job = jobs.pop(request_id)
                self._cancel_job(job)
            if not jobs:
                del self._layer_load_jobs[layer_name]
        for request_id in preempted:
            self._pending_load_layers.pop(request_id, None)
            load_job = self._load_jobs.pop(request_id, None)
            if load_job is not None:
                self._cancel_job(load_job)
            store_job = self._store_jobs.pop(request_id, None)
            if store_job is not None:
                result = self._cancel_job(store_job)
                self._finish_job(request_id, result, is_load=False)
        for layer_name, job in list(self._layer_store_jobs.items()):
            if any(
                plan.request_id in preempted
                for plan in job.plans
                if plan.request_id is not None
            ):
                result = self._cancel_job(self._layer_store_jobs.pop(layer_name))
                self._finish_job(
                    "__layerwise_store__",
                    result,
                    is_load=False,
                    mark_finished=False,
                )
        self._submitted_store_layers.clear()

    def _cancel_job(self, job: TransferJobState) -> TransferJobState:
        cancel = getattr(self.runtime, "cancel", None)
        if callable(cancel):
            return cancel(job)
        return self.runtime.wait(job)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        allowed = finished_req_ids | set()
        sending = self._finished_sending & allowed
        recving = set(self._finished_recving)
        self._finished_sending.difference_update(sending)
        self._finished_recving.difference_update(recving)
        self._failed_recving.difference_update(recving)
        return sending or None, recving or None

    def get_failed_recving(self) -> set[str]:
        failed = set(self._failed_recving)
        self._failed_recving.clear()
        return failed

    def build_connector_worker_meta(self) -> UMBPConnectorWorkerMetadata:
        result = self._worker_meta
        self._worker_meta = UMBPConnectorWorkerMetadata()
        return result

    def get_kv_events(self) -> list[Any]:
        take_evicted = getattr(self.runtime, "take_evicted_keys", None)
        if callable(take_evicted):
            for key in take_evicted():
                parsed = self._parse_key(key)
                if parsed is None:
                    continue
                group_id, block_hash = parsed
                self._worker_meta.kv_events.append(
                    BlockRemoved(
                        block_hashes=[
                            maybe_convert_block_hash(cast(BlockHash, block_hash))
                        ],
                        medium="CPU",
                        group_idx=group_id,
                        locality="LOCAL",
                    )
                )
        events = self._worker_meta.kv_events
        self._worker_meta.kv_events = []
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
        self.runtime.close()
