# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ECCPUConnector — CPU encoder-cache offloading.

A role-routed shell: one instance per process. The scheduler delegate owns the
mmap region and offload bookkeeping; the worker delegate owns the mmap view and
GPU copy plumbing. An ec_both instance reuses encoder outputs it has already
offloaded to CPU instead of recomputing them.
"""

from typing import TYPE_CHECKING

import torch

from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


class ECCPUConnector(ECConnectorBase):
    """EC connector that offloads encoder cache to a shared CPU mmap region."""

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole) -> None:
        super().__init__(vllm_config=vllm_config, role=role)

        self.connector_worker = None
        self.connector_scheduler = None

        if role == ECConnectorRole.WORKER:
            self.connector_worker = self._make_worker(vllm_config)
        elif role == ECConnectorRole.SCHEDULER:
            self.connector_scheduler = self._make_scheduler(vllm_config)
        else:
            raise ValueError(f"Unknown ECConnectorRole: {role}")

    # Construction seams.
    def _make_worker(self, vllm_config: "VllmConfig"):
        # Deferred import: the worker module touches torch/CUDA at import time
        # via the region, so keep that cost off the scheduler path.
        from vllm.distributed.ec_transfer.ec_connector.cpu.worker import (
            ECCPUWorker,
        )

        return ECCPUWorker(vllm_config)

    def _make_scheduler(self, vllm_config: "VllmConfig"):
        from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import (
            ECCPUScheduler,
        )

        return ECCPUScheduler(vllm_config)

    # Worker-side forwarders.
    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs
    ) -> None:
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECCPUConnectorMetadata)
        self.connector_worker.start_load_caches(
            encoder_cache, connector_metadata=metadata
        )

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECCPUConnectorMetadata)
        self.connector_worker.save_caches(
            encoder_cache, mm_hash, connector_metadata=metadata
        )

    # Scheduler-side forwarders.
    def has_cache_item(self, identifier: str) -> bool:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.has_cache_item(identifier)

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.ensure_cache_available(
            request, num_computed_tokens
        )

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_state_after_alloc(request, index)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> ECCPUConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        if self.connector_worker is not None:
            self.connector_worker.flush_saves()
        return None, None

    # Shared.
    def shutdown(self) -> None:
        if self.connector_scheduler is not None:
            self.connector_scheduler.shutdown()
        if self.connector_worker is not None:
            self.connector_worker.shutdown()
