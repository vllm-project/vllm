from typing import (TYPE_CHECKING, List, Optional, Protocol, Type, Union,
                    overload, runtime_checkable)

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from typing_extensions import TypeIs, TypeVar

from vllm.logger import init_logger
from vllm.utils import supports_kw

if TYPE_CHECKING:
    from vllm.attention import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.pooler import PoolerOutput
    from vllm.model_executor.layers.sampler import SamplerOutput
    from vllm.model_executor.pooling_metadata import PoolingMetadata
    from vllm.model_executor.sampling_metadata import SamplingMetadata

logger = init_logger(__name__)

# The type of HF config
C_co = TypeVar("C_co", bound=PretrainedConfig, covariant=True)

# The type of hidden states
# Currently, T = torch.Tensor for all models except for Medusa
# which has T = List[torch.Tensor]
T = TypeVar("T", default=torch.Tensor)
T_co = TypeVar("T_co", default=torch.Tensor, covariant=True)

# NOTE: Unlike those in `interfaces.py`, we don't define `ClassVar` tags
# for the base interfaces to avoid breaking OOT registration for existing models
# that don't inherit from the base interface classes


@runtime_checkable
class VllmModel(Protocol[C_co, T_co]):

    def __init__(
        self,
        vllm_config: "VllmConfig",
        prefix: str = "",
    ) -> None:
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: "AttentionMetadata",
    ) -> T_co:
        ...


def _check_vllm_model_init(model: Union[Type[object], object]) -> bool:
    model_init = model.__init__
    return supports_kw(model_init, "vllm_config")


def _check_vllm_model_forward(model: Union[Type[object], object]) -> bool:
    model_forward = getattr(model, "forward", None)
    if not callable(model_forward):
        return False

    vllm_kws = ("input_ids", "positions", "kv_caches", "attn_metadata")
    missing_kws = tuple(kw for kw in vllm_kws
                        if not supports_kw(model_forward, kw))

    if missing_kws and (isinstance(model, type)
                        and issubclass(model, nn.Module)):
        logger.warning(
            "The model (%s) is missing "
            "vLLM-specific keywords from its initializer: %s",
            model,
            missing_kws,
        )

    return len(missing_kws) == 0


@overload
def is_vllm_model(model: Type[object]) -> TypeIs[Type[VllmModel]]:
    ...


@overload
def is_vllm_model(model: object) -> TypeIs[VllmModel]:
    ...


def is_vllm_model(
    model: Union[Type[object], object],
) -> Union[TypeIs[Type[VllmModel]], TypeIs[VllmModel]]:
    return _check_vllm_model_init(model) and _check_vllm_model_forward(model)


@runtime_checkable
class VllmModelForTextGeneration(VllmModel[C_co, T], Protocol[C_co, T]):

    def compute_logits(
        self,
        hidden_states: T,
        sampling_metadata: "SamplingMetadata",
    ) -> Optional[T]:
        """Return `None` if TP rank > 0."""
        ...

    def sample(
        self,
        logits: T,
        sampling_metadata: "SamplingMetadata",
    ) -> "SamplerOutput":
        """Only called on TP rank 0."""
        ...


@overload
def is_text_generation_model(
        model: Type[object]) -> TypeIs[Type[VllmModelForTextGeneration]]:
    ...


@overload
def is_text_generation_model(
        model: object) -> TypeIs[VllmModelForTextGeneration]:
    ...


def is_text_generation_model(
    model: Union[Type[object], object],
) -> Union[TypeIs[Type[VllmModelForTextGeneration]],
           TypeIs[VllmModelForTextGeneration]]:
    if not is_vllm_model(model):
        return False

    if isinstance(model, type):
        return isinstance(model, VllmModelForTextGeneration)

    return isinstance(model, VllmModelForTextGeneration)


@runtime_checkable
class VllmModelForEmbedding(VllmModel[C_co, T], Protocol[C_co, T]):

    def pooler(
        self,
        hidden_states: T,
        pooling_metadata: "PoolingMetadata",
    ) -> "PoolerOutput":
        """Only called on TP rank 0."""
        ...


@overload
def is_embedding_model(
        model: Type[object]) -> TypeIs[Type[VllmModelForEmbedding]]:
    ...


@overload
def is_embedding_model(model: object) -> TypeIs[VllmModelForEmbedding]:
    ...


def is_embedding_model(
    model: Union[Type[object], object],
) -> Union[TypeIs[Type[VllmModelForEmbedding]], TypeIs[VllmModelForEmbedding]]:
    if not is_vllm_model(model):
        return False

    if isinstance(model, type):
        return isinstance(model, VllmModelForEmbedding)

    return isinstance(model, VllmModelForEmbedding)
