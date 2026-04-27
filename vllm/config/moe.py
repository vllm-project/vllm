# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import model_validator

from vllm.config.utils import config


@config
class MoEOffloadConfig:
    """Configuration for sparse-MoE CPU offload expert paging mode."""

    enabled: bool = False
    """Enable sparse-MoE CPU offload expert paging mode."""

    @model_validator(mode="after")
    def _validate_enabled_relationships(self) -> "MoEOffloadConfig":
        # Keep Stage 1 conservative: values are always valid individually, but
        # only interpreted by runtime when enabled=True.
        return self

    def metrics_info(self) -> dict[str, str]:
        return {
            "enabled": str(self.enabled),
        }
