# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from fastapi import APIRouter

    from vllm.config import PoolerConfig, VllmConfig


class BriefMetadata(BaseModel):
    task: str = Field(..., title="Task")
    served_model_name: str = Field(..., title="Model name")
    architectures: Optional[list[str]] = Field(...,
                                               title="Model architectures")
    max_model_len: int = Field(..., title="Maximum model length")

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> "BriefMetadata":
        return cls(
            task=vllm_config.model_config.task,
            served_model_name=vllm_config.model_config.served_model_name,
            architectures=vllm_config.model_config.architectures,
            max_model_len=vllm_config.model_config.max_model_len,
        )


class DetailMetadata(BaseModel):
    task: str = Field(..., title="Task")
    served_model_name: str = Field(..., title="Model name")
    architectures: Optional[list[str]] = Field(...,
                                               title="Model architectures")
    max_model_len: int = Field(..., title="Maximum model length")

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> "DetailMetadata":
        return cls(
            task=vllm_config.model_config.task,
            served_model_name=vllm_config.model_config.served_model_name,
            architectures=vllm_config.model_config.architectures,
            max_model_len=vllm_config.model_config.max_model_len,
        )


class HfConfigMetadata:

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig"):
        return vllm_config.model_config.hf_config.to_dict()


class PoolerConfigMetadata:

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> "PoolerConfig":
        return vllm_config.model_config.pooler_config


@dataclass
class Metadata:
    brief: BriefMetadata
    detail: DetailMetadata
    hf_config: HfConfigMetadata

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> "Metadata":
        return cls(
            **{
                key: metadata_class.from_vllm_config(vllm_config)
                for key, metadata_class in cls.__annotations__.items()
            })

    @classmethod
    def get_router(cls) -> "APIRouter":
        from fastapi import APIRouter, Request
        router = APIRouter()

        def get_func(metadata_class):
            response_field = metadata_class if issubclass(
                metadata_class, BaseModel) else dict

            async def func(raw_request: Request) -> response_field:
                out = metadata_class.from_vllm_config(
                    raw_request.app.state.vllm_config)
                if isinstance(out, (BaseModel, dict)):
                    return out
                else:
                    return vars(out)

            return func

        for key, metadata_class in cls.__annotations__.items():
            router.get(f"/metadata/{key}")(get_func(metadata_class))
        return router
