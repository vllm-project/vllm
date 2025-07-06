# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import Field

from vllm.entrypoints.metadata.base import (BriefMetadata, HfConfigMetadata,
                                            Metadata, PoolerConfigMetadata)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class ClassifyBrief(BriefMetadata):
    num_labels: int = Field(..., title="Num labels")

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> "ClassifyBrief":
        return cls(
            task=vllm_config.model_config.task,
            served_model_name=vllm_config.model_config.served_model_name,
            max_model_len=vllm_config.model_config.max_model_len,
            num_labels=vllm_config.model_config.hf_config.num_labels,
        )


@dataclass
class ClassifyMetadata(Metadata):
    brief: ClassifyBrief
    hf_config: HfConfigMetadata
    pooler_config: PoolerConfigMetadata
