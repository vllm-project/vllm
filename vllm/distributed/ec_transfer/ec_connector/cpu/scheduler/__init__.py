# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ECCPUScheduler — CPU offload scheduler delegate.

Owns the mmap region and the shared offload bookkeeping dicts, and routes
scheduler-side calls to the producer and consumer. Construction is factored
behind _build_producer/_build_consumer so a later NIXL subclass can supply
richer delegates.
"""

import threading
from typing import TYPE_CHECKING

from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
    ECRegionContext,
    setup_ec_region,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.consumer import (
    ECCPUConsumer,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.producer import (
    ECCPUProducer,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


class ECCPUScheduler:
    """Scheduler delegate for the ECCPUConnector."""

    def __init__(self, vllm_config: "VllmConfig") -> None:
        ec_config = vllm_config.ec_transfer_config
        assert ec_config is not None
        self._is_producer: bool = ec_config.is_ec_producer
        self._is_consumer: bool = ec_config.is_ec_consumer

        self._memory_context: ECRegionContext = setup_ec_region(vllm_config)

        # Shared bookkeeping: producer and consumer hold references to the same
        # dicts so an ec_both instance reuses its own offloaded encodings.
        self._local_encodings: dict[str, None] = {}
        self._blocks: dict[str, list[int]] = {}
        self._shared_lock = threading.Lock()

        self._producer: ECCPUProducer | None = None
        self._consumer: ECCPUConsumer | None = None
        if self._is_producer:
            self._producer = self._build_producer()
        if self._is_consumer:
            self._consumer = self._build_consumer()

    # Construction seams.
    def _build_producer(self) -> ECCPUProducer:
        return ECCPUProducer(
            memory_context=self._memory_context,
            local_encodings=self._local_encodings,
            blocks=self._blocks,
            lock=self._shared_lock,
        )

    def _build_consumer(self) -> ECCPUConsumer:
        return ECCPUConsumer(
            memory_context=self._memory_context,
            local_encodings=self._local_encodings,
            blocks=self._blocks,
            lock=self._shared_lock,
        )

    def has_cache_item(self, identifier: str) -> bool:
        if not self._is_consumer:
            return False
        assert self._consumer is not None
        return self._consumer.has_cache_item(identifier)

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        if self._is_producer:
            assert self._producer is not None
            if not self._producer.ensure_cache_available(request, num_computed_tokens):
                return False
        if self._is_consumer:
            assert self._consumer is not None
            if not self._consumer.ensure_cache_available(request, num_computed_tokens):
                return False
        return True

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        if self._is_producer:
            assert self._producer is not None
            self._producer.update_state_after_alloc(request, index)
        if self._is_consumer:
            assert self._consumer is not None
            self._consumer.update_state_after_alloc(request, index)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> ECCPUConnectorMetadata:
        meta = ECCPUConnectorMetadata()
        if self._is_producer:
            assert self._producer is not None
            meta.saves.update(self._producer.build_saves())
        if self._is_consumer:
            assert self._consumer is not None
            meta.loads.update(self._consumer.build_loads())
        return meta

    def shutdown(self) -> None:
        if self._producer is not None:
            self._producer.shutdown()
        if self._consumer is not None:
            self._consumer.shutdown()
        try:
            self._memory_context.region.cleanup()
        except Exception:
            logger.debug("ec: region cleanup failed", exc_info=True)
