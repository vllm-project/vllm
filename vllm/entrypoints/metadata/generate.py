# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from pydantic import Field

from vllm.entrypoints.metadata.base import (BriefMetadata, HfConfigMetadata,
                                            Metadata)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class GenerateBrief(BriefMetadata):
    enable_prefix_caching: Optional[bool] = Field(
        ..., title="Enable prefix caching")

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> "BriefMetadata":
        return cls(
            task=vllm_config.model_config.task,
            served_model_name=vllm_config.model_config.served_model_name,
            max_model_len=vllm_config.model_config.max_model_len,
            enable_prefix_caching=vllm_config.cache_config.
            enable_prefix_caching)


@dataclass
class GenerateMetadata(Metadata):
    brief: GenerateBrief
    hf_config: HfConfigMetadata
