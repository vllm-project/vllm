# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connector boundary for HiSparse scheduler and worker state."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.config.kv_transfer import KVTransferConfig
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
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.hisparse.types import SparseKVOffloadCommand
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.hisparse_coordinator import HiSparseCoordinator
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


@dataclass
class HiSparseConnectorMetadata(KVConnectorMetadata):
    command: SparseKVOffloadCommand | None
    host_block_copies: tuple[KVCacheBlockCopy, ...]
    source_block_ids: tuple[int, ...]


@dataclass
class HiSparseConnectorWorkerMetadata(KVConnectorWorkerMetadata):
    enqueued_transfer_counts: dict[int, int]
    completed_transfer_counts: dict[int, int]

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
        )


class HiSparseConnectorScheduler:
    def __init__(self, coordinator: HiSparseCoordinator) -> None:
        self.coordinator = coordinator

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> HiSparseConnectorMetadata:
        scheduler_output.block_table_updates = (
            self.coordinator.take_block_table_updates() or None
        )
        command = self.coordinator.build_offload_command()
        host_block_copies = tuple(
            copy
            for copy in scheduler_output.kv_cache_block_copies or ()
            if copy.block_pool_id is None
        )
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
        return HiSparseConnectorMetadata(
            command, host_block_copies, tuple(source_block_ids)
        )

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        metadata = connector_output.kv_connector_worker_meta
        if metadata is None:
            return
        assert isinstance(metadata, HiSparseConnectorWorkerMetadata)
        self.coordinator.update_spills(
            metadata.enqueued_transfer_counts,
            metadata.completed_transfer_counts,
        )


def _hisparse_config(vllm_config: VllmConfig) -> VllmConfig:
    config = copy.copy(vllm_config)
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="HiSparseConnector", kv_role="kv_both"
    )
    return config


class HiSparseConnector(KVConnectorBase_V1, SupportsHMA):
    """Join the scheduler coordinator to the worker's transfer engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
        coordinator: HiSparseCoordinator | None = None,
    ) -> None:
        super().__init__(_hisparse_config(vllm_config), role, kv_cache_config)
        self.connector_scheduler: HiSparseConnectorScheduler | None = None
        self.connector_worker: HiSparseConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            if coordinator is None:
                raise ValueError("HiSparse scheduler requires a coordinator.")
            self.connector_scheduler = HiSparseConnectorScheduler(coordinator)
        elif role == KVConnectorRole.WORKER:
            if coordinator is not None:
                raise ValueError("HiSparse worker cannot receive a coordinator.")
            self.connector_worker = HiSparseConnectorWorker()
        else:
            raise ValueError(f"Unsupported KV connector role: {role}")

    @property
    def requires_kv_delivery(self) -> bool:
        return False

    def finish_forward(self) -> None:
        assert self.connector_worker is not None
        self.connector_worker.finish_forward()

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
        self.connector_worker.start_step(metadata, request_state_indices)

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
        if not enqueued and not completed:
            return None
        return HiSparseConnectorWorkerMetadata(
            enqueued_transfer_counts={transfer_id: 1 for transfer_id in enqueued},
            completed_transfer_counts={transfer_id: 1 for transfer_id in completed},
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
        return

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
        return False, None


class _HiSparseMultiConnector(MultiConnector):
    """Compose HiSparse without changing the configured connector's telemetry."""

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        return self._connectors[0].get_kv_connector_stats()


def attach_hisparse_connector(
    connector: KVConnectorBase_V1 | None,
    vllm_config: VllmConfig,
    role: KVConnectorRole,
    kv_cache_config: KVCacheConfig,
    coordinator: HiSparseCoordinator | None = None,
) -> KVConnectorBase_V1:
    hisparse_connector = HiSparseConnector(
        vllm_config, role, kv_cache_config, coordinator
    )
    if connector is None:
        return hisparse_connector
    return _HiSparseMultiConnector.from_connectors(
        vllm_config, role, kv_cache_config, [connector, hisparse_connector]
    )


def get_hisparse_worker(
    connector: KVConnectorBase_V1,
) -> HiSparseConnectorWorker:
    if isinstance(connector, HiSparseConnector):
        assert connector.connector_worker is not None
        return connector.connector_worker
    if isinstance(connector, MultiConnector):
        for child in connector._connectors:
            if isinstance(child, HiSparseConnector):
                assert child.connector_worker is not None
                return child.connector_worker
    raise RuntimeError("HiSparse connector worker is not initialized.")
