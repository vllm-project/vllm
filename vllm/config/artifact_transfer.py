# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import uuid
from dataclasses import field
from typing import Any, Literal, get_args

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash

ArtifactProducer = Literal["artifact_producer", "artifact_both"]
ArtifactConsumer = Literal["artifact_consumer", "artifact_both"]
ArtifactRole = Literal[ArtifactProducer, ArtifactConsumer]
ArtifactTransferMode = Literal["final", "streaming", "chunked"]
ArtifactFailurePolicy = Literal[
    "fail_request",
    "fallback_to_request_output",
    "ignore",
]


@config
class ArtifactTransferConfig:
    """Configuration for rollout artifact transfer."""

    artifact_connector: str | None = None
    """The artifact connector used to export rollout artifacts."""

    engine_id: str | None = None
    """The engine id for artifact transfers."""

    artifact_role: ArtifactRole | None = None
    """Whether this vLLM instance produces, consumes, or both."""

    transfer_mode: ArtifactTransferMode = "final"
    """When artifacts are exported to the connector backend."""

    export_fields: list[str] = field(default_factory=list)
    """Artifact fields to export, for example token_ids or logprobs."""

    failure_policy: ArtifactFailurePolicy = "fail_request"
    """How vLLM should handle artifact connector export failures."""

    artifact_connector_extra_config: dict[str, Any] = field(default_factory=dict)
    """Any extra config that the connector may need."""

    artifact_connector_module_path: str | None = None
    """Python module path to dynamically load an artifact connector."""

    def compute_hash(self) -> str:
        """Return a hash for graph-affecting configuration.

        Artifact transfer runs after model execution and does not affect the
        model computation graph, so no factors are included here.
        """
        factors: list[Any] = []
        return safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()

    def __post_init__(self) -> None:
        if self.engine_id is None:
            self.engine_id = str(uuid.uuid4())

        if self.artifact_role is not None and self.artifact_role not in get_args(
            ArtifactRole
        ):
            raise ValueError(
                f"Unsupported artifact_role: {self.artifact_role}. "
                f"Supported roles are {get_args(ArtifactRole)}"
            )

        if self.artifact_connector is not None and self.artifact_role is None:
            raise ValueError(
                "Please specify artifact_role when artifact_connector "
                f"is set, supported roles are {get_args(ArtifactRole)}"
            )

        if self.transfer_mode not in get_args(ArtifactTransferMode):
            raise ValueError(
                f"Unsupported transfer_mode: {self.transfer_mode}. "
                f"Supported modes are {get_args(ArtifactTransferMode)}"
            )

        if self.failure_policy not in get_args(ArtifactFailurePolicy):
            raise ValueError(
                f"Unsupported failure_policy: {self.failure_policy}. "
                f"Supported policies are {get_args(ArtifactFailurePolicy)}"
            )

    @property
    def is_artifact_transfer_instance(self) -> bool:
        return self.artifact_connector is not None and self.artifact_role in get_args(
            ArtifactRole
        )

    @property
    def is_artifact_producer(self) -> bool:
        return self.artifact_connector is not None and self.artifact_role in get_args(
            ArtifactProducer
        )

    @property
    def is_artifact_consumer(self) -> bool:
        return self.artifact_connector is not None and self.artifact_role in get_args(
            ArtifactConsumer
        )

    def get_from_extra_config(self, key: str, default: Any) -> Any:
        return self.artifact_connector_extra_config.get(key, default)
