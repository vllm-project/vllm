# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed.artifact_transfer.artifact_connector.v1 import (
    ArtifactConnectorBase_V1,
)

ArtifactConnectorBase = ArtifactConnectorBase_V1
ArtifactConnectorBaseType = ArtifactConnectorBase_V1

__all__ = ["ArtifactConnectorBase", "ArtifactConnectorBaseType"]
