# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ECZmqConnector -- push encoder outputs straight to the consumer over ZMQ.

A role-routed shell: one instance per process. The scheduler delegate decides
what to push where and tracks what has arrived; the worker delegate owns the
sockets, the host-side staging area and the GPU copies.

Unlike the storage-backed connectors, delivery here is one-shot: an embedding is
handed over exactly once, to the consumers named for the request. Reuse across
requests is left to the consumer's own encoder cache.
"""

from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorRole,
    ECConnectorWorkerMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.utils import build_ec_items
from vllm.distributed.ec_transfer.ec_connector.zmq.common import (
    ECZmqConnectorMetadata,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ECConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


class ECZmqConnector(ECConnectorBase):
    """EC connector that pushes encoder outputs over ZMQ."""

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole) -> None:
        super().__init__(vllm_config=vllm_config, role=role)

        self.connector_worker = None
        self.connector_scheduler = None

        if role == ECConnectorRole.WORKER:
            self.connector_worker = self._make_worker(vllm_config)
        elif role == ECConnectorRole.SCHEDULER:
            self.connector_scheduler = self._make_scheduler(vllm_config)
            self._model_config = vllm_config.model_config
            self._metadata_fields_cache: dict[str, set[str]] = {}
        else:
            raise ValueError(f"Unknown ECConnectorRole: {role}")

    # Construction seams.
    def _make_worker(self, vllm_config: "VllmConfig"):
        # Deferred import: the worker module binds sockets and touches the
        # accelerator, neither of which belongs on the scheduler path.
        from vllm.distributed.ec_transfer.ec_connector.zmq.worker import ECZmqWorker

        return ECZmqWorker(vllm_config)

    def _make_scheduler(self, vllm_config: "VllmConfig"):
        from vllm.distributed.ec_transfer.ec_connector.zmq.scheduler import (
            ECZmqScheduler,
        )

        return ECZmqScheduler(vllm_config)

    # Worker-side forwarders.
    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs
    ) -> None:
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECZmqConnectorMetadata)
        self.connector_worker.start_load_caches(
            encoder_cache, connector_metadata=metadata
        )

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECZmqConnectorMetadata)
        self.connector_worker.save_caches(
            encoder_cache, mm_hash, connector_metadata=metadata
        )

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        if self.connector_worker is None:
            return None, None
        return self.connector_worker.get_finished(), None

    def build_connector_worker_meta(self) -> ECConnectorWorkerMetadata | None:
        if self.connector_worker is None:
            return None
        return self.connector_worker.build_worker_meta()

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
    ) -> ECZmqConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: "ECConnectorOutput") -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)

    def has_pending_push_work(self) -> bool:
        if self.connector_scheduler is None:
            return False
        return self.connector_scheduler.has_pending_push_work()

    def request_finished(
        self, request: "Request"
    ) -> tuple[bool, dict[str, Any] | None]:
        """Report each item's cache key and grid in the response body.

        `save_caches` already copied the embedding out of the encoder cache, so
        nothing has to be kept alive past the request.
        """
        if self.connector_scheduler is None or not self.is_producer:
            return False, None

        items = build_ec_items(request, self._model_config, self._metadata_fields_cache)
        return False, {"ec_items": items} if items else None

    # Shared.
    def shutdown(self) -> None:
        if self.connector_scheduler is not None:
            self.connector_scheduler.shutdown()
        if self.connector_worker is not None:
            self.connector_worker.shutdown()
