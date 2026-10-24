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

    enable_return_indexer_topk: bool = False
    """Capture and return sparse-attention index selections."""

    max_bytes: int | None = Field(default=None, gt=0)
    """Initial capacity, or ``None`` to derive it from the KV cache capacity.
    Grows when every object is referenced."""

    @property
    def enabled(self) -> bool:
        """Whether any execution auxiliary output is enabled."""
        return self.enable_return_routed_experts or self.enable_return_indexer_topk

    def compute_hash(self) -> str:
        """Hash AuxOutput settings that alter the model forward graph."""
        from vllm.config.utils import hash_factors

        factors: dict[str, object] = {
            "enable_return_routed_experts": self.enable_return_routed_experts
        }
        if self.enable_return_indexer_topk:
            factors.update(
                enable_return_indexer_topk=True,
            )
        return hash_factors(factors)
