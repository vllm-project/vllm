# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared worker-side UMBP connector behavior."""

from __future__ import annotations

from typing import Any

import torch

from vllm.forward_context import ForwardContext
from vllm.v1.attention.backend import AttentionMetadata

from .data import (
    BlockTransferPlan,
    KVLayoutPlanner,
    TransferJobState,
    TransferJobStatus,
    UMBPConnectorMetadata,
    UMBPConnectorWorkerMetadata,
)
from .runtime import UMBPWorkerHandle


class UMBPStoreConnectorWorker:
    """Translate shared metadata into runtime load/store jobs."""

    def __init__(
        self, runtime: UMBPWorkerHandle, layout: KVLayoutPlanner | None = None
    ) -> None:
        self.runtime = runtime
        self.layout = layout
        self._load_jobs: dict[str, TransferJobState] = {}
        self._store_jobs: dict[str, TransferJobState] = {}
        self._worker_meta = UMBPConnectorWorkerMetadata()
        self._finished_sending: set[str] = set()
        self._finished_recving: set[str] = set()
        self._failed_recving: set[str] = set()

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
            )
            for plan in plans
        ]

    def start_load_kv(
        self, forward_context: ForwardContext, metadata: UMBPConnectorMetadata
    ) -> None:
        del forward_context
        for request_id, plans in metadata.load_requests.items():
            if plans:
                self._load_jobs[request_id] = self.runtime.load(
                    self._materialize_plans(plans)
                )
        if metadata.load_plans and not metadata.load_requests:
            self._load_jobs["__umbp_batch__"] = self.runtime.load(
                self._materialize_plans(metadata.load_plans)
            )

    def _finish_job(
        self,
        request_id: str,
        job: TransferJobState,
        *,
        is_load: bool,
    ) -> None:
        result = self.runtime.wait(job)
        if is_load:
            self._worker_meta.completed_loads.update(result.completed_keys)
            self._finished_recving.add(request_id)
            if result.status != TransferJobStatus.COMPLETED:
                self._failed_recving.add(request_id)
                for key in result.failed_keys:
                    self._worker_meta.failed_loads[key] = (
                        result.error or "load failed"
                    )
        else:
            if result.status == TransferJobStatus.COMPLETED:
                self.runtime.publish(result)
            self._worker_meta.completed_stores.update(result.completed_keys)
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
                            self._worker_meta.failed_store_tokens.get(token, 0)
                            + 1
                        )
                for key in failed_keys:
                    self._worker_meta.failed_store_errors[key] = (
                        result.error or "store failed"
                    )
            self._finished_sending.add(request_id)
        self._worker_meta.failed_block_ids.update(result.failed_block_ids)

    def wait_for_layer_load(self, layer_name: str) -> None:
        del layer_name
        for request_id, job in self._load_jobs.items():
            self._finish_job(request_id, job, is_load=True)
        self._load_jobs.clear()

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        del layer_name, kv_layer, attn_metadata, kwargs

    def wait_for_save(self) -> None:
        for request_id, job in self._store_jobs.items():
            self._finish_job(request_id, job, is_load=False)
        self._store_jobs.clear()

    def enqueue_stores(self, metadata: UMBPConnectorMetadata) -> None:
        for request_id, plans in metadata.store_requests.items():
            if plans:
                self._store_jobs[request_id] = self.runtime.store(
                    self._materialize_plans(plans)
                )
        if metadata.store_plans and not metadata.store_requests:
            self._store_jobs["__umbp_batch__"] = self.runtime.store(
                self._materialize_plans(metadata.store_plans)
            )

    def handle_preemptions(self, metadata: UMBPConnectorMetadata) -> None:
        """Drain jobs before vLLM reuses preempted GPU block IDs."""
        if not (
            metadata.preempted_block_ids or metadata.preempted_request_ids
        ):
            return
        self.wait_for_layer_load("")
        self.wait_for_save()

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

    def get_block_ids_with_load_errors(self) -> set[int]:
        failed = self._worker_meta.failed_block_ids
        self._worker_meta.failed_block_ids = set()
        return failed

    def close(self) -> None:
        self.runtime.close()
