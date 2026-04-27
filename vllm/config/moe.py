# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import model_validator

from vllm.config.utils import config


@config
class MoEOffloadConfig:
    """Configuration for sparse-MoE CPU offload expert paging mode."""

    enabled: bool = False
    """Enable sparse-MoE CPU offload expert paging mode."""

    mode: str = "disabled"
    """Selected MoE offload mode: disabled, passive, or prefetch."""

    gpu_prefetch: int | None = None
    """Requested Case 2 GPU active expert residency target."""

    effective_gpu_prefetch: int | None = None
    """Effective Case 2 GPU active expert residency target."""

    @model_validator(mode="after")
    def _validate_enabled_relationships(self) -> "MoEOffloadConfig":
        if self.gpu_prefetch is not None and self.gpu_prefetch <= 0:
            raise ValueError("gpu_prefetch must be a positive integer")
        if (
            self.effective_gpu_prefetch is not None
            and self.effective_gpu_prefetch <= 0
        ):
            raise ValueError("effective_gpu_prefetch must be a positive integer")
        if self.mode not in {"disabled", "passive", "prefetch"}:
            raise ValueError("mode must be one of: disabled, passive, prefetch")
        if self.enabled and self.mode == "disabled":
            self.mode = "prefetch" if self.gpu_prefetch is not None else "passive"
        if not self.enabled:
            self.mode = "disabled"
            self.gpu_prefetch = None
            self.effective_gpu_prefetch = None
        return self

    def metrics_info(self) -> dict[str, str]:
        return {
            "enabled": str(self.enabled),
            "mode": self.mode,
            "gpu_prefetch": str(self.gpu_prefetch),
            "effective_gpu_prefetch": str(self.effective_gpu_prefetch),
        }
