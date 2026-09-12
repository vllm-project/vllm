# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connector boundary for HiSparse scheduler and worker state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
    HiSparseConnectorWorker,
)
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.hisparse.coordinator import get_hisparse_coordinator
from vllm.v1.hisparse.types import SparseKVOffloadCommand, SparseKVRowMirror
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
    from vllm.v1.hisparse.coordinator import HiSparseCoordinator
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


@dataclass
class HiSparseConnectorMetadata(KVConnectorMetadata):
    command: SparseKVOffloadCommand | None
    host_block_copies: tuple[KVCacheBlockCopy, ...]
    source_block_ids: tuple[int, ...]
    row_mirrors: dict[str, tuple[SparseKVRowMirror, ...]]
    all_context_pages_resident: bool


@dataclass
class HiSparseConnectorWorkerMetadata(KVConnectorWorkerMetadata):
    enqueued_transfer_counts: dict[int, int]
    completed_transfer_counts: dict[int, int]
    completed_host_copy_dst_ids: tuple[int, ...] = ()

    def aggregate(self, other: KVConnectorWorkerMetadata) -> KVConnectorWorkerMetadata:
        assert isinstance(other, HiSparseConnectorWorkerMetadata)

        def add_counts(first: dict[int, int], second: dict[int, int]) -> dict[int, int]:
            combined = first.copy()
            for transfer_id, count in second.items():
                combined[transfer_id] = combined.get(transfer_id, 0) + count
            return combined

        return HiSparseConnectorWorkerMetadata(
            enqueued_transfer_counts=add_counts(
                self.enqueued_transfer_counts, other.enqueued_transfer_counts
            ),
            completed_transfer_counts=add_counts(
                self.completed_transfer_counts, other.completed_transfer_counts
            ),
            # Every worker runs the same copies, so a union is the ack.
            completed_host_copy_dst_ids=tuple(
                {*self.completed_host_copy_dst_ids, *other.completed_host_copy_dst_ids}
            ),
        )


class HiSparseConnectorScheduler:
    def __init__(
        self,
        *,
        async_speculative: bool,
        draft_kv_lookahead: int = 0,
    ) -> None:
        self.coordinator: HiSparseCoordinator | None = None
        self.async_speculative = async_speculative
        self.draft_kv_lookahead = draft_kv_lookahead
        self.requests: dict[str, Request] = {}

    def bind_coordinator(self, coordinator: HiSparseCoordinator) -> None:
        assert self.coordinator is None
        self.coordinator = coordinator

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> HiSparseConnectorMetadata:
        assert self.coordinator is not None
        # HiSparse rebinds worker state every step, so the load must start
        # before the forward rather than being deferred to post-forward.
        scheduler_output.has_sync_kv_loads = True
        scheduler_output.block_table_updates = (
            self.coordinator.take_block_table_updates() or None
        )
        command = self.coordinator.build_offload_command()
        host_block_copies = self.coordinator.take_host_block_copies()
        source_group_id = self.coordinator.host_group_id
        assert source_group_id is not None
        source_block_ids = [
            block_id
            for request in scheduler_output.scheduled_new_reqs
            for block_id in request.block_ids[source_group_id]
        ]
        for new_block_ids in scheduler_output.scheduled_cached_reqs.new_block_ids:
            if new_block_ids is not None:
                source_block_ids.extend(new_block_ids[source_group_id])
        num_computed_tokens = {
            request.req_id: request.num_computed_tokens
            for request in scheduler_output.scheduled_new_reqs
        }
        num_computed_tokens.update(
            zip(
                scheduler_output.scheduled_cached_reqs.req_ids,
                scheduler_output.scheduled_cached_reqs.num_computed_tokens,
            )
        )
        scheduled_requests = tuple(
            (
                request_id,
                num_computed_tokens[request_id],
                scheduled_count,
            )
            for request_id, scheduled_count in (
                scheduler_output.num_scheduled_tokens.items()
            )
        )
        row_mirrors = {}
        for request_id, scheduled_start, scheduled_count in scheduled_requests:
            mirror_start = scheduled_start
            if self.async_speculative:
                mirror_start = max(
                    0,
                    scheduled_start - self.requests[request_id].num_output_placeholders,
                )
            row_mirrors[request_id] = self.coordinator.build_row_mirrors(
                (
                    (
                        request_id,
                        mirror_start,
                        scheduled_count
                        + scheduled_start
                        - mirror_start
                        + self.draft_kv_lookahead,
                    ),
                ),
            )
        return HiSparseConnectorMetadata(
            command,
            host_block_copies,
            tuple(source_block_ids),
            row_mirrors,
            self.coordinator.all_context_pages_resident(scheduled_requests),
        )

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        assert self.coordinator is not None
        for request_id in connector_output.finished_recving or ():
            self.coordinator.finish_host_import(
                request_id, failed=request_id in connector_output.failed_recving
            )
        metadata = connector_output.kv_connector_worker_meta
        if metadata is None:
            return
        assert isinstance(metadata, HiSparseConnectorWorkerMetadata)
        self.coordinator.release_completed_host_copies(
            metadata.completed_host_copy_dst_ids
        )
        self.coordinator.update_spills(
            metadata.enqueued_transfer_counts,
            metadata.completed_transfer_counts,
        )


class HiSparseConnector(KVConnectorBase_V1, SupportsHMA):
    """Join the scheduler coordinator to the worker's transfer engine."""

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str:
        return "BLHNC"

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        if kv_cache_config.hisparse_host_num_blocks is None:
            raise ValueError("HiSparseConnector requires a HiSparse host pool")
        self.connector_scheduler: HiSparseConnectorScheduler | None = None
        self.connector_worker: HiSparseConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            speculative_config = vllm_config.speculative_config
            self.connector_scheduler = HiSparseConnectorScheduler(
                async_speculative=bool(
                    vllm_config.scheduler_config.async_scheduling
                    and speculative_config is not None
                ),
                draft_kv_lookahead=vllm_config.num_lookahead_tokens,
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = HiSparseConnectorWorker(
                vllm_config, kv_cache_config
            )
        else:
            raise ValueError(f"Unsupported KV connector role: {role}")

    def bind_kv_cache_manager(self, kv_cache_manager: KVCacheManager) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.bind_coordinator(
            get_hisparse_coordinator(kv_cache_manager)
        )

    @property
    def requires_kv_delivery(self) -> bool:
        return False

    def has_pending_push_work(self) -> bool:
        assert self.connector_scheduler is not None
        assert self.connector_scheduler.coordinator is not None
        return self.connector_scheduler.coordinator.has_pending_work()

    def has_pending_block_frees(self) -> bool:
        assert self.connector_scheduler is not None
        assert self.connector_scheduler.coordinator is not None
        return self.connector_scheduler.coordinator.has_pending_reclamation()

    def finish_forward(self) -> None:
        assert self.connector_worker is not None
        self.connector_worker.finish_forward()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def reset_capture_state(self) -> None:
        assert self.connector_worker is not None
        self.connector_worker.reset_hot_state()

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, HiSparseConnectorMetadata)
        request_state_indices = kwargs.get("request_state_indices")
        assert request_state_indices is None or isinstance(
            request_state_indices, torch.Tensor
        )
        request_ids = kwargs.get("request_ids")
        assert request_ids is None or isinstance(request_ids, list)
        attn_metadata = kwargs.get("attn_metadata")
        assert attn_metadata is None or isinstance(attn_metadata, dict)
        self.connector_worker.start_step(
            metadata,
            request_state_indices,
            request_ids,
            num_tokens=int(kwargs.get("num_tokens") or 0),
        )
        self.connector_worker.prepare_forward(attn_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        return

    def wait_for_save(self) -> None:
        return

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        assert self.connector_worker is not None
        enqueued, completed = self.connector_worker.take_transfer_updates()
        host_copies = self.connector_worker.take_completed_host_copies()
        if not enqueued and not completed and not host_copies:
            return None
        return HiSparseConnectorWorkerMetadata(
            enqueued_transfer_counts={transfer_id: 1 for transfer_id in enqueued},
            completed_transfer_counts={transfer_id: 1 for transfer_id in completed},
            completed_host_copy_dst_ids=tuple(host_copies),
        )

    def shutdown(self) -> None:
        if self.connector_worker is not None:
            self.connector_worker.shutdown()

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.requests[request.request_id] = request

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)

    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        self.connector_scheduler.requests.pop(request.request_id, None)
        return False, None
