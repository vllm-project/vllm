# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field

from vllm.config import ModelConfig, PoolerConfig, VllmConfig, get_attr_docs

if TYPE_CHECKING:
    from fastapi import APIRouter

model_config_docs = get_attr_docs(ModelConfig)
pooler_config_docs = get_attr_docs(PoolerConfig)


class ReadOnlyBaseModel(BaseModel):

    class Config:
        frozen = True


class BriefMetadata(ReadOnlyBaseModel):
    # Hacky way to get the defined docs.
    task: str = Field(..., description=model_config_docs["task"])
    served_model_name: str = Field(
        ..., description=model_config_docs["served_model_name"])
    max_model_len: int = Field(...,
                               description=model_config_docs["max_model_len"])

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> "BriefMetadata":
        return cls(
            task=vllm_config.model_config.task,
            served_model_name=vllm_config.model_config.served_model_name,
            max_model_len=vllm_config.model_config.max_model_len,
        )


class HfConfigMetadata:

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig"):
        return vllm_config.model_config.hf_config.to_dict()


class PoolerConfigMetadata(ReadOnlyBaseModel):
    # Hacky way to get the defined docs.
    pooling_type: Optional[str] = Field(
        ..., description=pooler_config_docs["pooling_type"])
    normalize: Optional[bool] = Field(
        ..., description=pooler_config_docs["normalize"])
    softmax: Optional[bool] = Field(...,
                                    description=pooler_config_docs["softmax"])
    step_tag_id: Optional[int] = Field(
        ..., description=pooler_config_docs["step_tag_id"])
    returned_token_ids: Optional[list[int]] = Field(
        ..., description=pooler_config_docs["returned_token_ids"])

    @classmethod
    def from_vllm_config(cls,
                         vllm_config: "VllmConfig") -> "PoolerConfigMetadata":
        pooler_config = vllm_config.model_config.pooler_config
        assert isinstance(pooler_config, PoolerConfig)

        return cls(**asdict(pooler_config))


@dataclass
class Metadata:
    brief: BriefMetadata
    hf_config: HfConfigMetadata

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> "Metadata":
        return cls(
            **{
                key: metadata_class.from_vllm_config(vllm_config)
                for key, metadata_class in cls.__annotations__.items()
            })

    @classmethod
    def get_router(cls, metadata_dev_mode) -> "APIRouter":
        from fastapi import APIRouter, HTTPException, Request
        router = APIRouter()

        def add_api_route(name, metadata_class):

            def get_metadata(metadata_class):
                response_field = metadata_class if issubclass(
                    metadata_class, BaseModel) else dict

                async def func(
                        raw_request: Request
                ) -> response_field:  # type: ignore
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

            router.get(f"/metadata/{name}")(get_metadata(metadata_class))
            if issubclass(metadata_class, BaseModel):
                router.get(f"/metadata/{name}/" + "{key}")(
                    quick_access(metadata_class))

        if metadata_dev_mode:
            # metadata will show more information in dev mode.
            for name, metadata_class in cls.__annotations__.items():
                add_api_route(name, metadata_class)
        else:
            # metadata only shows brief when default mode.
            add_api_route("brief", cls.__annotations__["brief"])

        return router
