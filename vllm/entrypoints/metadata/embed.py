# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from pydantic import Field

from vllm.entrypoints.metadata.base import (BriefMetadata, DetailMetadata,
                                            HfConfigMetadata, Metadata,
                                            PoolerConfigMetadata)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class EmbedBrief(BriefMetadata):
    embedding_dim: int = Field(..., title="Embedding dimension")
    is_matryoshka: bool = Field(..., title="Is matryoshka model")
    matryoshka_dimensions: Optional[int] = Field(...,
                                                 title="Matryoshka dimensions")
    truncation_side: str = Field(..., title="Truncation side")

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> "EmbedBrief":
        return cls(
            task=vllm_config.model_config.task,
            served_model_name=vllm_config.model_config.served_model_name,
            architectures=vllm_config.model_config.architectures,
            embedding_dim=vllm_config.model_config.hf_config.hidden_size,
            max_model_len=vllm_config.model_config.max_model_len,
            is_matryoshka=vllm_config.model_config.is_matryoshka,
            matryoshka_dimensions=vllm_config.model_config.
            matryoshka_dimensions,
            truncation_side=vllm_config.model_config.truncation_side,
        )


class EmbedDetail(DetailMetadata):
    embedding_dim: int = Field(..., title="Embedding dimension")
    is_matryoshka: bool = Field(..., title="Is matryoshka model")
    matryoshka_dimensions: Optional[int] = Field(...,
                                                 title="Matryoshka dimensions")
    truncation_side: str = Field(..., title="Truncation side")

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> "EmbedDetail":
        return cls(
            task=vllm_config.model_config.task,
            served_model_name=vllm_config.model_config.served_model_name,
            architectures=vllm_config.model_config.architectures,
            embedding_dim=vllm_config.model_config.hf_config.hidden_size,
            max_model_len=vllm_config.model_config.max_model_len,
            is_matryoshka=vllm_config.model_config.is_matryoshka,
            matryoshka_dimensions=vllm_config.model_config.
            matryoshka_dimensions,
            truncation_side=vllm_config.model_config.truncation_side,
        )


@dataclass
class EmbedMetadata(Metadata):
    brief: EmbedBrief
    detail: EmbedDetail
    hf_config: HfConfigMetadata
    pooler_config: PoolerConfigMetadata
