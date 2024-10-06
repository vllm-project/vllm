from typing import (TYPE_CHECKING, List, Optional, Protocol, Type, Union,
                    overload, runtime_checkable)

import torch
from transformers import PretrainedConfig
from typing_extensions import TypeIs, TypeVar

if TYPE_CHECKING:
    from vllm.attention import AttentionMetadata
    from vllm.config import CacheConfig
    from vllm.model_executor.layers.pooler import PoolerOutput
    from vllm.model_executor.layers.quantization import QuantizationConfig
    from vllm.model_executor.layers.sampler import SamplerOutput
    from vllm.model_executor.pooling_metadata import PoolingMetadata
    from vllm.model_executor.sampling_metadata import SamplingMetadata

# NOTE: Unlike those in `interfaces.py`, we don't define `ClassVar` tags
# for the base interfaces to avoid breaking OOT registration

# The type of hidden states
# Currently, T = torch.Tensor for all models except for Medusa
# which has T = List[torch.Tensor]
T = TypeVar("T", default=torch.Tensor)


@runtime_checkable
class VllmModelForTextGeneration(Protocol[T]):

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        cache_config: Optional["CacheConfig"],
        quant_config: Optional["QuantizationConfig"],
    ) -> None:
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: "AttentionMetadata",
    ) -> T:
        ...

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
def supports_text_generation(
        model: Type[object]) -> TypeIs[Type[VllmModelForTextGeneration]]:
    ...


@overload
def supports_text_generation(
        model: object) -> TypeIs[VllmModelForTextGeneration]:
    ...


def supports_text_generation(
    model: Union[Type[object], object],
) -> Union[TypeIs[Type[VllmModelForTextGeneration]],
           TypeIs[VllmModelForTextGeneration]]:
    if isinstance(model, type):
        return isinstance(model, VllmModelForTextGeneration)

    return isinstance(model, VllmModelForTextGeneration)


@runtime_checkable
class VllmModelForEmbedding(Protocol[T]):

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        cache_config: Optional["CacheConfig"],
        quant_config: Optional["QuantizationConfig"],
    ) -> None:
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: "AttentionMetadata",
    ) -> T:
        ...

    def pooler(
        self,
        hidden_states: T,
        pooling_metadata: "PoolingMetadata",
    ) -> "PoolerOutput":
        """Only called on TP rank 0."""
        ...


@overload
def supports_embedding(
        model: Type[object]) -> TypeIs[Type[VllmModelForEmbedding]]:
    ...


@overload
def supports_embedding(model: object) -> TypeIs[VllmModelForEmbedding]:
    ...


def supports_embedding(
    model: Union[Type[object], object],
) -> Union[TypeIs[Type[VllmModelForEmbedding]], TypeIs[VllmModelForEmbedding]]:
    if isinstance(model, type):
        return isinstance(model, VllmModelForEmbedding)

    return isinstance(model, VllmModelForEmbedding)
