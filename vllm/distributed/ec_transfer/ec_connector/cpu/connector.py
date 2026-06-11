# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ECCPUConnector entry point.

A thin role-routed shell: one instance lives in each process (scheduler
or worker) and forwards every call to the appropriate delegate. The
scheduler delegate owns NIXL + ZMQ state; the worker delegate owns the
mmap view and GPU copy plumbing. `ec_both` deployments run both
producer and consumer branches inside each delegate.
"""

from typing import TYPE_CHECKING, Any

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
    """EC connector for E-PD disaggregation with NIXL CPU cache.

    See `ec-nixl-transfer-v4.md` for the full design. This class only
    performs role dispatch; all state and logic live in the delegates.
    """

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole) -> None:
        super().__init__(vllm_config=vllm_config, role=role)

        self.connector_worker = None
        self.connector_scheduler = None

        if role == ECConnectorRole.WORKER:
            # Deferred import: the worker module touches torch/CUDA at
            # module level via the region, so keep import cost off the
            # scheduler path.
            from vllm.distributed.ec_transfer.ec_connector.cpu.worker import (
                ECCPUWorker,
            )

            self.connector_worker = ECCPUWorker(vllm_config)
        elif role == ECConnectorRole.SCHEDULER:
            from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import (
                ECCPUScheduler,
            )

            self.connector_scheduler = ECCPUScheduler(vllm_config)
        else:
            raise ValueError(f"Unknown ECConnectorRole: {role}")

    # ==============================
    # Worker-side forwarders
    # ==============================

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
        self,
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
        **kwargs,
    ) -> None:
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECCPUConnectorMetadata)
        self.connector_worker.save_caches(
            encoder_cache, mm_hash, connector_metadata=metadata
        )

    # ==============================
    # Scheduler-side forwarders
    # ==============================

    def has_cache_item(self, identifier: str) -> bool:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.has_cache_item(identifier)

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        """Consumer-side admission hook; see design §4.4.

        This method is referenced by the design's consumer flow. The ABC
        surface it plugs into (on `ECConnectorBase`) lands in a separate
        PR; until then, callers that need it can reach it through the
        concrete connector.
        """
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

    def request_finished(
        self, request: "Request"
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request)

    # ==============================
    # Shared
    # ==============================

    def shutdown(self) -> None:
        if self.connector_scheduler is not None:
            self.connector_scheduler.shutdown()
        if self.connector_worker is not None:
            self.connector_worker.shutdown()
