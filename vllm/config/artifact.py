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

    max_bytes: int | None = Field(default=None, gt=0)
    """LRU capacity, or ``None`` to derive it from the KV cache capacity."""

    @property
    def enabled(self) -> bool:
        """Whether any execution artifact is enabled."""
        return self.enable_return_routed_experts

    def compute_hash(self) -> str:
        """Hash Artifact settings that alter the model forward graph."""
        from vllm.config.utils import hash_factors

        return hash_factors(
            {"enable_return_routed_experts": self.enable_return_routed_experts}
        )
