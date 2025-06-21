# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import Field

from vllm.entrypoints.metadata.base import (BriefMetadata, DetailMetadata,
                                            HfConfigMetadata, Metadata)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class GenerateBrief(BriefMetadata):
    enable_prefix_caching: bool = Field(..., title="Enable prefix caching")

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> "BriefMetadata":
        return cls(
            task=vllm_config.model_config.task,
            served_model_name=vllm_config.model_config.served_model_name,
            architectures=vllm_config.model_config.architectures,
            max_model_len=vllm_config.model_config.max_model_len,
            enable_prefix_caching=vllm_config.cache_config.
            enable_prefix_caching)


class GenerateDetail(DetailMetadata):
    enable_prefix_caching: bool = Field(..., title="Enable prefix caching")

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> "GenerateDetail":
        return cls(
            task=vllm_config.model_config.task,
            served_model_name=vllm_config.model_config.served_model_name,
            architectures=vllm_config.model_config.architectures,
            max_model_len=vllm_config.model_config.max_model_len,
            enable_prefix_caching=vllm_config.cache_config.
            enable_prefix_caching)


@dataclass
class GenerateMetadata(Metadata):
    brief: GenerateBrief
    detail: GenerateDetail
    hf_config: HfConfigMetadata
