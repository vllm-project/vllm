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
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorMetadata,
)
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_offload.sparse.base import SparseKVOffloadCommand
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.hisparse_coordinator import HiSparseCoordinator
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.kv_offload.sparse.hisparse_worker import HiSparseWorker
    from vllm.v1.request import Request


@dataclass
class HiSparseConnectorMetadata(KVConnectorMetadata):
    command: SparseKVOffloadCommand | None


@dataclass
class HiSparseConnectorWorkerMetadata(KVConnectorWorkerMetadata):
    completed_transfer_ids: list[int]

    def aggregate(self, other: KVConnectorWorkerMetadata) -> KVConnectorWorkerMetadata:
        assert isinstance(other, HiSparseConnectorWorkerMetadata)
        return HiSparseConnectorWorkerMetadata(
            completed_transfer_ids=list(
                dict.fromkeys(
                    self.completed_transfer_ids + other.completed_transfer_ids
                )
            )
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
        self._coordinator = coordinator
        self._worker: HiSparseWorker | None = None

    @property
    def requires_kv_delivery(self) -> bool:
        return False

    def bind_worker(self, worker: HiSparseWorker) -> None:
        self._worker = worker

    def prepare_step(self, scheduler_output: SchedulerOutput) -> None:
        assert self._worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, HiSparseConnectorMetadata)
        self._worker.prepare_step(metadata.command, scheduler_output)

    def finish_forward(self) -> None:
        assert self._worker is not None
        self._worker.finish_forward()

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        return

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
        assert self._worker is not None
        completed = self._worker.take_completed_transfer_ids()
        if completed is None:
            return None
        return HiSparseConnectorWorkerMetadata(completed)

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.shutdown()
            self._worker = None

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
        assert self._coordinator is not None
        return HiSparseConnectorMetadata(
            self._coordinator.build_offload_command(
                list(scheduler_output.num_scheduled_tokens)
            )
        )

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        assert self._coordinator is not None
        metadata = connector_output.kv_connector_worker_meta
        if metadata is None:
            return
        assert isinstance(metadata, HiSparseConnectorWorkerMetadata)
        self._coordinator.complete_spills(metadata.completed_transfer_ids)

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


def bind_hisparse_worker(connector: KVConnectorBase_V1, worker: HiSparseWorker) -> None:
    if isinstance(connector, HiSparseConnector):
        connector.bind_worker(worker)
        return
    if isinstance(connector, MultiConnector):
        for child in connector._connectors:
            bind_hisparse_worker(child, worker)


def get_hisparse_connector_metadata(
    metadata: KVConnectorMetadata | None,
) -> HiSparseConnectorMetadata | None:
    if isinstance(metadata, HiSparseConnectorMetadata):
        return metadata
    if isinstance(metadata, MultiKVConnectorMetadata):
        for child in metadata.metadata:
            if (result := get_hisparse_connector_metadata(child)) is not None:
                return result
    return None
