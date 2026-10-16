# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for execution auxiliary outputs."""

from pydantic import Field

from vllm.config.utils import config


@config
class AuxOutputConfig:
    """Configuration for auxiliary-output delivery."""

    enable_return_routed_experts: bool = False
    """Capture and return routed-experts auxiliary outputs."""

    max_bytes: int | None = Field(default=None, gt=0)
    """LRU capacity, or ``None`` to derive it from the KV cache capacity."""

    @property
    def enabled(self) -> bool:
        """Whether any execution auxiliary output is enabled."""
        return self.enable_return_routed_experts

    def compute_hash(self) -> str:
        """Hash AuxOutput settings that alter the model forward graph."""
        from vllm.config.utils import hash_factors

        return hash_factors(
            {"enable_return_routed_experts": self.enable_return_routed_experts}
        )
