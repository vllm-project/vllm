# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ECCPUScheduler — composition shell."""

import threading
from typing import TYPE_CHECKING, Any

from vllm import envs
from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
    ECRegionContext,
    setup_ec_region,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.consumer import (
    ECCPUConsumer,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq import (
    ZmqClientTransport,
    ZmqServerTransport,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.data.nixl import (
    NixlDataTransport,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.producer import (
    ECCPUProducer,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.protocol import (
    compute_ec_compatibility_hash,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.session import (
    ProducerSession,
)
from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config  # noqa: F401
from vllm.logger import init_logger
from vllm.version import __version__ as VLLM_VERSION

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
        self._ec_config = ec_config
        self._is_producer: bool = ec_config.is_ec_producer
        self._is_consumer: bool = ec_config.is_ec_consumer

        assert ec_config.engine_id is not None
        self._engine_id: str = ec_config.engine_id

        if NixlWrapper is None or nixl_agent_config is None:
            raise RuntimeError(
                "ECCPUConnector requires NIXL; "
                "install the `nixl` package or set a different ec_connector."
            )
        self._memory_context: ECRegionContext = setup_ec_region(vllm_config)
        self._data = NixlDataTransport(
            agent_name=self._engine_id,
            base_ptr=self._memory_context.region.base_ptr,
            num_blocks=self._memory_context.num_blocks,
            block_size_bytes=self._memory_context.block_size_bytes,
            total_size_bytes=self._memory_context.region.total_size_bytes,
        )
        self._compat_hash: str = compute_ec_compatibility_hash(
            vllm_version=VLLM_VERSION,
            model=str(vllm_config.model_config.model),
            dtype=str(self._memory_context.dtype),
            block_size_bytes=self._memory_context.block_size_bytes,
        )

        # Shared state: producer and consumer hold references.
        self._local_encodings: dict[str, None] = {}
        self._blocks: dict[str, list[int]] = {}
        self._shared_lock = threading.Lock()

        # True at the start of each scheduling step; passed to ensure_cache_available
        # so the consumer calls transport.poll() exactly once per step.
        self._first_in_batch: bool = True

        self._producer: ECCPUProducer | None = None
        self._producer_session: ProducerSession | None = None
        self._consumer: ECCPUConsumer | None = None

        if self._is_producer:
            self._peer_host = envs.VLLM_EC_SIDE_CHANNEL_HOST
            self._peer_port = envs.VLLM_EC_SIDE_CHANNEL_PORT

            self._producer = ECCPUProducer(
                memory_context=self._memory_context,
                compat_hash=self._compat_hash,
                addr=(self._peer_host, self._peer_port),
                local_encodings=self._local_encodings,
                blocks=self._blocks,
                lock=self._shared_lock,
            )
            server_transport = ZmqServerTransport(
                host=self._peer_host,
                port=self._peer_port,
            )
            self._producer_session = ProducerSession(
                transport=server_transport,
                data=self._data,
                region=self._memory_context.region,
                local_encodings=self._local_encodings,
                blocks=self._blocks,
                lock=self._shared_lock,
                compat_hash=self._compat_hash,
            )
            self._producer_session.start()

        if self._is_consumer:
            consumer_transport = ZmqClientTransport()
            self._consumer = ECCPUConsumer(
                memory_context=self._memory_context,
                transport=consumer_transport,
                data=self._data,
                compat_hash=self._compat_hash,
                local_encodings=self._local_encodings,
                blocks=self._blocks,
                lock=self._shared_lock,
            )

    # ── public API ────────────────────────────────────────────────────────────

    def has_cache_item(self, identifier: str) -> bool:
        if not self._is_consumer:
            return False
        assert self._consumer is not None
        return self._consumer.has_cache_item(identifier)

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        first = self._first_in_batch
        self._first_in_batch = False
        if self._is_producer:
            assert self._producer is not None
            if not self._producer.ensure_cache_available(
                request, num_computed_tokens, first
            ):
                return False
        if self._is_consumer:
            assert self._consumer is not None
            if not self._consumer.ensure_cache_available(
                request, num_computed_tokens, first
            ):
                return False
        return True

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        if self._is_producer:
            assert self._producer is not None
            self._producer.update_state_after_alloc(request, index)
        if self._is_consumer:
            assert self._consumer is not None
            self._consumer.update_state_after_alloc(request, index)

    def request_finished(
        self, request: "Request"
    ) -> tuple[bool, dict[str, Any] | None]:
        if not self._is_producer:
            return False, None
        assert self._producer is not None
        return False, self._producer.request_finished(request)

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
        # Reset for the next scheduling step.
        self._first_in_batch = True
        return meta

    def shutdown(self) -> None:
        if self._producer_session is not None:
            self._producer_session.stop()
        if self._producer is not None:
            self._producer.shutdown()
        if self._consumer is not None:
            self._consumer.shutdown()
        try:
            self._data.deregister()
        except Exception:
            logger.debug("ec: deregister failed", exc_info=True)
        try:
            self._memory_context.region.cleanup()
        except Exception:
            logger.debug("ec: region cleanup failed", exc_info=True)
