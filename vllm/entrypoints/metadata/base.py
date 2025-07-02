# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="valid-type, name-defined, attr-defined"

from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from vllm.config import ModelConfig, PoolerConfig, VllmConfig, get_attr_docs

if TYPE_CHECKING:
    from fastapi import APIRouter


def get_attr_typing(cls: type[Any]):
    out = {}
    for f in fields(cls):
        out[f.name] = f.type
    return out


model_config_docs = get_attr_docs(ModelConfig)
model_config_typing = get_attr_typing(ModelConfig)

pooler_config_docs = get_attr_docs(PoolerConfig)
pooler_config_typing = get_attr_typing(PoolerConfig)


class ReadOnlyBaseModel(BaseModel):

    class Config:
        frozen = True


class BriefMetadata(ReadOnlyBaseModel):
    # Hacky way to get the defined docs and typing.
    # Getting the corresponding typing from the dictionary returned
    # by get_attr_typing is not a valid mypy expression,
    # but it can make Swagger: API Documentation work.
    task: model_config_typing["task"] = Field(
        ..., description=model_config_docs["task"])
    served_model_name: model_config_typing["served_model_name"] = Field(
        ..., description=model_config_docs["served_model_name"])
    max_model_len: model_config_typing["max_model_len"] = Field(
        ..., description=model_config_docs["max_model_len"])

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
    pooling_type: pooler_config_typing["pooling_type"] = Field(
        ..., description=pooler_config_docs["pooling_type"])
    normalize: pooler_config_typing["normalize"] = Field(
        ..., description=pooler_config_docs["normalize"])
    softmax: pooler_config_typing["softmax"] = Field(
        ..., description=pooler_config_docs["softmax"])
    step_tag_id: pooler_config_typing["step_tag_id"] = Field(
        ..., description=pooler_config_docs["step_tag_id"])
    returned_token_ids: pooler_config_typing["returned_token_ids"] = Field(
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
