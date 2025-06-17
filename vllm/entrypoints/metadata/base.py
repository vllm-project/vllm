# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field

from vllm.config import PoolerConfig, VllmConfig

if TYPE_CHECKING:
    from fastapi import APIRouter


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
        pooler_config = vllm_config.model_config.pooler_config
        assert isinstance(pooler_config, PoolerConfig)

        return pooler_config


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
    def get_router(cls, brief_metadata_only) -> "APIRouter":
        from fastapi import APIRouter, HTTPException, Request
        router = APIRouter()

        def add_api_route(metadata_class):

            def get_metadata(metadata_class):
                response_field = metadata_class if issubclass(
                    metadata_class, BaseModel) else dict

                async def func(raw_request: Request) -> response_field:
                    metadata = metadata_class.from_vllm_config(
                        raw_request.app.state.vllm_config)
                    if isinstance(metadata, (BaseModel, dict)):
                        return metadata
                    else:
                        return vars(metadata)

                return func

            def quick_access(metadata_class):

                async def func(key: str,
                               raw_request: Request) -> dict[str, Any]:
                    metadata = metadata_class.from_vllm_config(
                        raw_request.app.state.vllm_config)

                    value = getattr(metadata, key, None)

                    if value is not None:
                        return {key: value}
                    else:
                        raise HTTPException(status_code=404)

                return func

            router.get("/metadata/brief")(get_metadata(metadata_class))

            if issubclass(metadata_class, BaseModel):
                router.get("/metadata/brief/{key}")(
                    quick_access(metadata_class))

        if brief_metadata_only:
            add_api_route(cls.__annotations__["brief"])
        else:
            for key, metadata_class in cls.__annotations__.items():
                add_api_route(metadata_class)
        return router
