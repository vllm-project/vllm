# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import Field, model_validator

from vllm.config.utils import config


@config
class MoEOffloadConfig:
    """Configuration for sparse-MoE CPU offload expert paging mode."""

    enabled: bool = False
    """Enable sparse-MoE CPU offload expert paging mode."""

    gpu_limit: float = Field(default=0.4, gt=0, le=1)
    """Maximum fraction of GPU memory for the offload-managed working set."""

    active_expert_budget: int = Field(default=2, ge=1)
    """Maximum number of active expert models resident on GPU."""

    max_pipeline_depth: int = Field(default=4, ge=1)
    """Maximum routed expert bucket pipeline depth for expert reuse."""

    @model_validator(mode="after")
    def _validate_enabled_relationships(self) -> "MoEOffloadConfig":
        # Keep Stage 1 conservative: values are always valid individually, but
        # only interpreted by runtime when enabled=True.
        return self

    def metrics_info(self) -> dict[str, str]:
        return {
            "enabled": str(self.enabled),
            "gpu_limit": str(self.gpu_limit),
            "active_expert_budget": str(self.active_expert_budget),
            "max_pipeline_depth": str(self.max_pipeline_depth),
        }
