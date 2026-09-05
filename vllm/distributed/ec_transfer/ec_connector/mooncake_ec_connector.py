# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder-cache connector backed by Mooncake TransferEngine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
    ECConnectorWorkerMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
    ECMooncakeConnectorMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.scheduler import (
    ECMooncakeScheduler,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.worker import ECMooncakeWorker

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ECConnectorOutput
    from vllm.v1.request import Request


class ECMooncakeConnector(ECConnectorBase):
    """Preserve the public API while delegating to one process-role component."""

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)

        self._scheduler: ECMooncakeScheduler | None = None
        self._worker: ECMooncakeWorker | None = None
        self._closed = False

        if role == ECConnectorRole.SCHEDULER:
            self._scheduler = ECMooncakeScheduler(vllm_config)
        elif role == ECConnectorRole.WORKER:
            self._worker = ECMooncakeWorker(vllm_config)
        else:
            raise ValueError(f"Unknown EC connector role: {role}")

    def start_worker_services(self) -> None:
        assert self._worker is not None
        self._worker.start_services()

    def start_save_caches(self, **kwargs: Any) -> None:
        assert self._worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECMooncakeConnectorMetadata)
        self._worker.start_save_caches(metadata, **kwargs)

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs: Any
    ) -> None:
        assert self._worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECMooncakeConnectorMetadata)
        self._worker.start_load_caches(metadata, encoder_cache, **kwargs)

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs: Any
    ) -> None:
        assert self._worker is not None
        self._worker.save_caches(encoder_cache, mm_hash, **kwargs)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        assert self._worker is not None
        return self._worker.get_finished(finished_req_ids)

    def build_connector_worker_meta(self) -> ECConnectorWorkerMetadata | None:
        assert self._worker is not None
        return self._worker.build_connector_worker_meta()

    def take_unavailable_requests(self) -> set[str]:
        assert self._scheduler is not None
        return self._scheduler.take_unavailable_requests()

    def has_cache_item(self, identifier: str) -> bool:
        assert self._scheduler is not None
        return self._scheduler.has_cache_item(identifier)

    def ensure_cache_available(
        self, request: Request, num_computed_tokens: int
    ) -> bool:
        assert self._scheduler is not None
        return self._scheduler.ensure_cache_available(request, num_computed_tokens)

    def update_state_after_alloc(self, request: Request, index: int) -> None:
        assert self._scheduler is not None
        self._scheduler.update_state_after_alloc(request, index)

    def update_state_after_free(self, request: Request, index: int) -> None:
        assert self._scheduler is not None
        self._scheduler.update_state_after_free(request, index)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        assert self._scheduler is not None
        return self._scheduler.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: ECConnectorOutput) -> None:
        assert self._scheduler is not None
        self._scheduler.update_connector_output(connector_output)

    def has_pending_push_work(self) -> bool:
        assert self._scheduler is not None
        return self._scheduler.has_pending_push_work()

    def request_finished(self, request: Request) -> tuple[bool, dict[str, Any] | None]:
        assert self._scheduler is not None
        return self._scheduler.request_finished(request)

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._scheduler is not None:
            self._scheduler.close()
        if self._worker is not None:
            self._worker.close()
