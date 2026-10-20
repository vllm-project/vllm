# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for execution auxiliary outputs."""

from typing import Literal

from pydantic import Field

from vllm.config.utils import config


@config
class AuxOutputConfig:
    """Configuration for auxiliary-output delivery."""

    enable_return_routed_experts: bool = False
    """Capture and return routed-experts auxiliary outputs."""

    enable_return_indexer_topk: bool = False
    """Capture and return sparse-attention index selections."""

    indexer_output_format: Literal["raw", "packed"] = "raw"
    """Return decoded int32 selections or compact payload with a profile."""

    indexer_encoding: Literal["raw", "compact"] = "compact"
    """Use model-specific lossless encoding when available."""

    max_bytes: int | None = Field(default=None, gt=0)
    """LRU capacity, or ``None`` to derive it from the KV cache capacity."""

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
                indexer_encoding=self.indexer_encoding,
            )
        return hash_factors(factors)
