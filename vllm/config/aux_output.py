# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for execution auxiliary outputs."""

from pydantic import Field

from vllm.config.utils import CompileFactors, config


@config
class AuxOutputConfig:
    """Configuration for auxiliary-output delivery."""

    enable_return_routed_experts: bool = False
    """Capture and return routed-experts auxiliary outputs."""

    max_bytes: int | None = Field(default=None, gt=0)
    """Initial capacity, or ``None`` to derive it from the KV cache capacity.
    Grows when every object is referenced."""

    @property
    def enabled(self) -> bool:
        """Whether any execution auxiliary output is enabled."""
        return self.enable_return_routed_experts

    def compile_factors(self) -> CompileFactors:
        """AuxOutput settings that alter the model forward graph."""
        return {"enable_return_routed_experts": self.enable_return_routed_experts}
