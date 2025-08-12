# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import (TYPE_CHECKING, Any, ClassVar, Literal, Optional, Protocol,
                    Union, overload, runtime_checkable)

import torch
import torch.nn as nn
from typing_extensions import TypeIs, TypeVar

from vllm.logger import init_logger
from vllm.utils import supports_kw

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.pooler import Pooler
    from vllm.model_executor.sampling_metadata import SamplingMetadata
else:
    VllmConfig = Any
    Pooler = Any
    SamplingMetadata = Any

logger = init_logger(__name__)

# The type of hidden states
# Currently, T = torch.Tensor for all models except for Medusa
# which has T = list[torch.Tensor]
T = TypeVar("T", default=torch.Tensor)
T_co = TypeVar("T_co", default=torch.Tensor, covariant=True)

# NOTE: Unlike those in `interfaces.py`, we don't define `ClassVar` tags
# for the base interfaces to avoid breaking OOT registration for existing models
# that don't inherit from the base interface classes


@runtime_checkable
class VllmModel(Protocol[T_co]):
    """The interface required for all models in vLLM."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> T_co:
        ...


def _check_vllm_model_init(model: Union[type[object], object]) -> bool:
    model_init = model.__init__
    return supports_kw(model_init, "vllm_config")


def _check_vllm_model_forward(model: Union[type[object], object]) -> bool:
    model_forward = getattr(model, "forward", None)
    if not callable(model_forward):
        return False

    vllm_kws = ("input_ids", "positions")
    missing_kws = tuple(kw for kw in vllm_kws
                        if not supports_kw(model_forward, kw))

    if missing_kws and (isinstance(model, type)
                        and issubclass(model, nn.Module)):
        logger.warning(
            "The model (%s) is missing "
            "vLLM-specific keywords from its `forward` method: %s",
            model,
            missing_kws,
        )

    return len(missing_kws) == 0


@overload
def is_vllm_model(model: type[object]) -> TypeIs[type[VllmModel]]:
    ...


@overload
def is_vllm_model(model: object) -> TypeIs[VllmModel]:
    ...


def is_vllm_model(
    model: Union[type[object], object],
) -> Union[TypeIs[type[VllmModel]], TypeIs[VllmModel]]:
    return _check_vllm_model_init(model) and _check_vllm_model_forward(model)


@runtime_checkable
class VllmModelForTextGeneration(VllmModel[T], Protocol[T]):
    """The interface required for all generative models in vLLM."""

    def compute_logits(
        self,
        hidden_states: T,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[T]:
        """Return `None` if TP rank > 0."""
        ...


@overload
def is_text_generation_model(
        model: type[object]) -> TypeIs[type[VllmModelForTextGeneration]]:
    ...


@overload
def is_text_generation_model(
        model: object) -> TypeIs[VllmModelForTextGeneration]:
    ...


def is_text_generation_model(
    model: Union[type[object], object],
) -> Union[TypeIs[type[VllmModelForTextGeneration]],
           TypeIs[VllmModelForTextGeneration]]:
    if not is_vllm_model(model):
        return False

    if isinstance(model, type):
        return isinstance(model, VllmModelForTextGeneration)

    return isinstance(model, VllmModelForTextGeneration)


@runtime_checkable
class VllmModelForPooling(VllmModel[T_co], Protocol[T_co]):
    """The interface required for all pooling models in vLLM."""

    is_pooling_model: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports pooling.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    pooler: Pooler
    """The pooler is only called on TP rank 0."""


@overload
def is_pooling_model(model: type[object]) -> TypeIs[type[VllmModelForPooling]]:
    ...


@overload
def is_pooling_model(model: object) -> TypeIs[VllmModelForPooling]:
    ...


def is_pooling_model(
    model: Union[type[object], object],
) -> Union[TypeIs[type[VllmModelForPooling]], TypeIs[VllmModelForPooling]]:
    if not is_vllm_model(model):
        return False

    return getattr(model, "is_pooling_model", False)
