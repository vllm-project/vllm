# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for execution artifacts."""

from pydantic import Field

from vllm.config.utils import config


@config
class ArtifactConfig:
    """Configuration for execution-artifact delivery."""

    enable_return_routed_experts: bool = False
    """Capture and return routed-experts artifacts."""

    shm_dir: str = "/dev/shm/vllm-artifacts"
    """Trusted root for immutable artifact objects."""

    max_shm_bytes: int | None = Field(default=None, gt=0)
    """LRU capacity, or ``None`` to derive it from the KV cache capacity."""

    shm_ttl_seconds: int = Field(default=3600, gt=0)
    """Grace period before an inactive engine store is removed.

    Inactive stores may coexist and consume node SHM during this period.
    """

    @property
    def enabled(self) -> bool:
        """Whether any execution artifact is enabled."""
        return self.enable_return_routed_experts
