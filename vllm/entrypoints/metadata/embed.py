from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Self

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from vllm.config import PoolerConfig, VllmConfig


class EmbedOverview(BaseModel):
    task: str = Field(..., title="Task")
    served_model_name: str = Field(..., title="Model name")
    architectures: Optional[list[str]] = Field(...,
                                               title="Model architectures")
    embedding_dim: int = Field(..., title="Embedding dimension")
    max_model_len: int = Field(..., title="Maximum model length")
    is_matryoshka: bool = Field(..., title="Is matryoshka model")
    matryoshka_dimensions: Optional[int] = Field(...,
                                                 title="Matryoshka dimensions")
    truncation_side: str = Field(..., title="Truncation side")

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> Self:
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


class EmbedHfConfig:

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig"):
        return vllm_config.model_config.hf_config.to_dict()


class EmbedPoolerConfig:

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> "PoolerConfig":
        return vllm_config.model_config.pooler_config


@dataclass
class EmbedMetadata:
    overview: EmbedOverview
    hf_config: EmbedHfConfig
    pooler_config: EmbedPoolerConfig

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> Self:
        return cls(
            **{
                key: metadata_class.from_vllm_config(vllm_config)
                for key, metadata_class in cls.__annotations__.items()
            })

    @classmethod
    def get_router(cls) -> Any:
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
