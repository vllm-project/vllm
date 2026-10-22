# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in asynchronous initialization for KV connectors (Model Runner V2)."""

import time
from abc import abstractmethod
from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorInitStatus,
    KVConnectorMetadata,
    KVConnectorRankInitStatus,
    KVConnectorRole,
    NoOpKVConnectorMetadata,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class SupportsAsyncInit(KVConnectorBase_V1):
    """Connectors that initialize asynchronously with Model Runner V2.

    Inherit this instead of KVConnectorBase_V1 and implement is_init_done(),
    which reports the current worker's setup; how it runs is up to the connector.
    Serving must work without this connector so vllm can hide its startup
    latency. Readiness is monotonic; activation waits for all workers' reports
    during ordinary engine steps. Idle engines do not poll. Metadata reporting,
    output updates, stats, reset and shutdown must remain safe while
    initializing. Requests admitted before readiness are never shown to this
    connector, except as finished request IDs on the worker, which must be
    ignored. Callers reach these hooks through get_request_handler(), so other
    connectors carry no readiness API.
    """

    # Class-level so instances built without __init__ default to synchronous.
    _async_init_enabled = False
    _connector_ready = True

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self.enable_async_init()

    def enable_async_init(self) -> None:
        """Validate the deployment and start waiting for worker readiness."""
        if not self._vllm_config.use_v2_model_runner:
            raise ValueError(
                "Asynchronous connector initialization requires Model Runner V2"
            )
        parallel_config = self._vllm_config.parallel_config
        backend = parallel_config.distributed_executor_backend
        if (backend == "external_launcher" and parallel_config.world_size > 1) or (
            backend == "ray" and parallel_config.pipeline_parallel_size > 1
        ):
            raise ValueError(
                "Asynchronous connector initialization requires readiness reports "
                "from every worker; external_launcher with multiple ranks and "
                "Ray pipeline parallelism are not supported"
            )
        self._async_init_enabled = True
        self._connector_ready = False
        self._ready_connector_ranks: set[int] = set()
        self._init_start_time = time.monotonic()

    @abstractmethod
    def is_init_done(self) -> bool:
        """Whether the current worker's setup has finished; raise on failure.

        Polled on workers only, once per engine step until it returns True.
        Raises TimeoutError if still False after VLLM_KV_CONNECTOR_INIT_TIMEOUT.
        """
        raise NotImplementedError

    def is_connector_ready(self) -> bool:
        """Whether this connector can accept new requests.

        Workers latch their own is_init_done(); the scheduler waits for every
        worker's report.
        """
        if not self._async_init_enabled:
            return True
        if self._role == KVConnectorRole.WORKER and not self._connector_ready:
            self._connector_ready = self.is_init_done()
            elapsed = time.monotonic() - self._init_start_time
            if not self._connector_ready and (
                elapsed > envs.VLLM_KV_CONNECTOR_INIT_TIMEOUT
            ):
                raise TimeoutError(
                    f"{type(self).__name__} initialization did not finish "
                    f"within {envs.VLLM_KV_CONNECTOR_INIT_TIMEOUT}s"
                )
        return self._connector_ready

    def requires_allocation_tracking(self) -> bool:
        """Whether initialization can hide block reuse."""
        return not self.is_connector_ready()

    def build_connector_init_status(self) -> KVConnectorInitStatus | None:
        """Report initialization until this step has real connector metadata."""
        if (
            not self._async_init_enabled
            or self._role != KVConnectorRole.WORKER
            or self.has_connector_metadata()
        ):
            return None
        rank = self._vllm_config.parallel_config.rank
        return KVConnectorRankInitStatus(
            ready_ranks={rank} if self.is_connector_ready() else set(),
        )

    def update_connector_init_status(
        self, status: KVConnectorInitStatus | None
    ) -> None:
        """Apply aggregated worker initialization status on the scheduler."""
        if not self._async_init_enabled or self._role != KVConnectorRole.SCHEDULER:
            return
        if status is None:
            return
        assert isinstance(status, KVConnectorRankInitStatus)
        if self._connector_ready:
            return
        self._ready_connector_ranks |= status.ready_ranks
        if (
            len(self._ready_connector_ranks)
            == self._vllm_config.parallel_config.world_size
        ):
            logger.info(
                "Asynchronous connector initialization completed on all workers"
            )
            self._connector_ready = True


class KVConnectorRequestHandler:
    """Resolved once per connector by get_request_handler().

    Connectors without SupportsAsyncInit accept every request, so this handler
    forwards to them directly and reports no initialization status.
    """

    def __init__(self, connector: KVConnectorBase_V1) -> None:
        self._connector = connector

    def is_connector_ready(self) -> bool:
        return True

    def is_eligible(self, request_id: str) -> bool:
        return True

    def requires_allocation_tracking(self) -> bool:
        return False

    def on_new_request(self, request: "Request") -> None:
        self._connector.on_new_request(request)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return self._connector.build_connector_meta(scheduler_output)

    def build_connector_init_status(self) -> KVConnectorInitStatus | None:
        return None

    def update_connector_init_status(
        self, status: KVConnectorInitStatus | None
    ) -> None:
        pass


class AsyncInitRequestHandler(KVConnectorRequestHandler):
    """Own lifetime exclusions for an asynchronously initialized connector.

    Requests admitted before readiness stay excluded through completion.
    Callers must check is_eligible() before invoking per-request hooks.
    """

    _connector: SupportsAsyncInit

    def __init__(self, connector: SupportsAsyncInit) -> None:
        super().__init__(connector)
        self._excluded_request_ids: set[str] = set()

    def is_connector_ready(self) -> bool:
        return self._connector.is_connector_ready()

    def is_eligible(self, request_id: str) -> bool:
        return request_id not in self._excluded_request_ids

    def requires_allocation_tracking(self) -> bool:
        return (
            bool(self._excluded_request_ids)
            or self._connector.requires_allocation_tracking()
        )

    def on_new_request(self, request: "Request") -> None:
        if self._connector.is_connector_ready():
            self._connector.on_new_request(request)
        else:
            self._excluded_request_ids.add(request.request_id)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        if self._connector.is_connector_ready():
            metadata = self._connector.build_connector_meta(
                scheduler_output.without_requests(self._excluded_request_ids)
            )
        else:
            metadata = NoOpKVConnectorMetadata()
        self._excluded_request_ids.difference_update(scheduler_output.finished_req_ids)
        return metadata

    def build_connector_init_status(self) -> KVConnectorInitStatus | None:
        return self._connector.build_connector_init_status()

    def update_connector_init_status(
        self, status: KVConnectorInitStatus | None
    ) -> None:
        self._connector.update_connector_init_status(status)


def get_request_handler(connector: KVConnectorBase_V1) -> KVConnectorRequestHandler:
    if isinstance(connector, SupportsAsyncInit):
        return AsyncInitRequestHandler(connector)
    return KVConnectorRequestHandler(connector)
