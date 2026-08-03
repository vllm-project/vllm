# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for execution artifacts."""

from typing import Literal

from pydantic import Field

from vllm.config.utils import config


@config
class ArtifactConfig:
    """Configuration for execution-artifact delivery."""

    enable_return_routed_experts: bool = False
    """Capture and return routed-experts artifacts."""

    backend: Literal["shm"] = "shm"
    """Artifact delivery backend."""

    shm_dir: str = "/dev/shm/vllm-artifacts"
    """Trusted root for immutable artifact objects."""

    max_shm_bytes: int = Field(default=8 << 30, gt=0)
    """Hard capacity limit for one live engine and DP rank."""

    shm_ttl_seconds: int = Field(default=3600, gt=0)
    """Grace period before an inactive engine store is removed.

    Inactive stores may coexist and consume node SHM during this period.
    """

    @property
    def enabled(self) -> bool:
        """Whether any execution artifact is enabled."""
        return self.enable_return_routed_experts
