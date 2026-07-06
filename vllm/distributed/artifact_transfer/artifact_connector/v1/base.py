# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base interfaces for vLLM artifact transfer connectors.

Artifact transfer is modeled after the v1 KV connector split: scheduler-side
connectors produce serializable metadata and worker-side connectors move the
large payloads through a backend data plane. Unlike KV transfer, artifacts are
request outputs rather than cache blocks, so the base API intentionally avoids
KV-cache block concepts.
"""

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


class ArtifactConnectorRole(enum.Enum):
    # Connector running in the scheduler process.
    SCHEDULER = 0

    # Connector running in the worker process.
    WORKER = 1


@dataclass
class ArtifactHandle:
    """Serializable reference to artifact payloads in an external backend."""

    backend: str
    artifact_id: str
    location: dict[str, Any] = field(default_factory=dict)
    fields: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "artifact_id": self.artifact_id,
            "location": self.location,
            "fields": self.fields,
            "metadata": self.metadata,
        }


class ArtifactConnectorMetadata(ABC):  # noqa: B024
    """Metadata sent from scheduler connector to worker connector."""

    pass


class ArtifactConnectorWorkerMetadata(ABC):
    """Metadata sent from worker connector back to scheduler connector."""

    @abstractmethod
    def aggregate(
        self, other: "ArtifactConnectorWorkerMetadata"
    ) -> "ArtifactConnectorWorkerMetadata":
        pass


@dataclass
class ArtifactConnectorOutput:
    """Worker-side artifact connector output for one engine step."""

    finished_sending: set[str] | None = None
    handles: dict[str, ArtifactHandle] = field(default_factory=dict)
    worker_meta: ArtifactConnectorWorkerMetadata | None = None
    errors: dict[str, str] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return (
            not self.finished_sending
            and not self.handles
            and self.worker_meta is None
            and not self.errors
        )


class ArtifactConnectorBase_V1(ABC):
    """Base class for artifact transfer connectors."""

    def __init__(self, vllm_config: "VllmConfig", role: ArtifactConnectorRole):
        logger.warning(
            "Initializing ArtifactConnectorBase_V1. This API is experimental "
            "and subject to change while artifact transfer is developed."
        )
        if vllm_config.artifact_transfer_config is None:
            raise ValueError(
                "artifact_transfer_config must be set for ArtifactConnectorBase_V1"
            )
        self._vllm_config = vllm_config
        self._artifact_transfer_config = vllm_config.artifact_transfer_config
        self._role = role
        self._connector_metadata: ArtifactConnectorMetadata | None = None

    @property
    def role(self) -> ArtifactConnectorRole:
        return self._role

    @property
    def is_producer(self) -> bool:
        return self._artifact_transfer_config.is_artifact_producer

    @property
    def is_consumer(self) -> bool:
        return self._artifact_transfer_config.is_artifact_consumer

    # ==============================
    # Worker-side methods
    # ==============================

    def bind_connector_metadata(
        self, connector_metadata: ArtifactConnectorMetadata
    ) -> None:
        self._connector_metadata = connector_metadata

    def clear_connector_metadata(self) -> None:
        self._connector_metadata = None

    def _get_connector_metadata(self) -> ArtifactConnectorMetadata:
        assert self._connector_metadata is not None
        return self._connector_metadata

    def has_connector_metadata(self) -> bool:
        return self._connector_metadata is not None

    def start_artifact_transfers(self) -> None:
        """Start worker-side artifact movement for the current step."""
        return

    def record_step_artifacts(
        self,
        *,
        request_ids: list[str],
        artifacts: dict[str, Any],
    ) -> None:
        """Record artifacts produced by a model-runner step.

        Concrete connectors may stage payloads here and return handles later via
        :meth:`get_finished` or :meth:`build_connector_worker_meta`.
        """
        return

    def get_finished(self, finished_req_ids: set[str]) -> set[str] | None:
        return None

    def build_connector_worker_meta(self) -> ArtifactConnectorWorkerMetadata | None:
        return None

    def get_artifact_connector_output(self) -> ArtifactConnectorOutput | None:
        return None

    def shutdown(self) -> None:
        return None

    # ==============================
    # Scheduler-side methods
    # ==============================

    def on_new_request(self, request: "Request") -> None:
        return

    def record_request_output(
        self,
        request: "Request",
        token_ids: list[int],
        logprobs: Any | None,
    ) -> None:
        """Record accepted output tokens and their sampled-token logprobs."""
        return

    @abstractmethod
    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> ArtifactConnectorMetadata:
        pass

    def update_connector_output(
        self, connector_output: ArtifactConnectorOutput
    ) -> None:
        return

    def request_finished(self, request: "Request") -> ArtifactHandle | None:
        return None
