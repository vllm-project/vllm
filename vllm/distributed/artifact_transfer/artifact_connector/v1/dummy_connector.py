# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op artifact connector used to validate artifact-transfer plumbing."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.distributed.artifact_transfer.artifact_connector.v1.base import (
    ArtifactConnectorBase_V1,
    ArtifactConnectorMetadata,
    ArtifactConnectorRole,
    ArtifactConnectorWorkerMetadata,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput


@dataclass
class DummyArtifactConnectorMetadata(ArtifactConnectorMetadata):
    scheduled_request_ids: list[str] = field(default_factory=list)
    export_fields: list[str] = field(default_factory=list)


@dataclass
class DummyArtifactConnectorWorkerMetadata(ArtifactConnectorWorkerMetadata):
    num_seen_requests: int = 0

    def aggregate(
        self, other: ArtifactConnectorWorkerMetadata
    ) -> ArtifactConnectorWorkerMetadata:
        assert isinstance(other, DummyArtifactConnectorWorkerMetadata)
        return DummyArtifactConnectorWorkerMetadata(
            num_seen_requests=self.num_seen_requests + other.num_seen_requests
        )


class DummyArtifactConnector(ArtifactConnectorBase_V1):
    """A no-op connector that exposes the scheduler/worker lifecycle."""

    def __init__(self, vllm_config: "VllmConfig", role: ArtifactConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._num_seen_requests = 0

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> ArtifactConnectorMetadata:
        req_ids = list(scheduler_output.num_scheduled_tokens.keys())
        return DummyArtifactConnectorMetadata(
            scheduled_request_ids=req_ids,
            export_fields=list(self._artifact_transfer_config.export_fields),
        )

    def start_artifact_transfers(self) -> None:
        if not self.has_connector_metadata():
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, DummyArtifactConnectorMetadata)
        self._num_seen_requests += len(metadata.scheduled_request_ids)

    def build_connector_worker_meta(self) -> ArtifactConnectorWorkerMetadata | None:
        if self._num_seen_requests == 0:
            return None
        worker_meta = DummyArtifactConnectorWorkerMetadata(
            num_seen_requests=self._num_seen_requests
        )
        self._num_seen_requests = 0
        return worker_meta
