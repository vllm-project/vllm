# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Protocol,
    overload,
    runtime_checkable,
)

import torch
import torch.nn as nn
from typing_extensions import TypeIs, TypeVar

from vllm.logger import init_logger
from vllm.utils.func_utils import supports_kw

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.model import AttnTypeStr
    from vllm.config.pooler import PoolingTypeStr
    from vllm.model_executor.layers.pooler import Pooler
else:
    VllmConfig = Any
    Pooler = Any
    PoolingTypeStr = Any
    AttnTypeStr = Any

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

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply token embeddings to `input_ids`."""
        if hasattr(self, "get_input_embeddings"):
            logger.warning_once(
                "`get_input_embeddings` for vLLM models is deprecated and will be "
                "removed in v0.13.0 or v1.0.0, whichever is earlier. Please rename "
                "this method to `embed_input_ids`."
            )
            return self.get_input_embeddings(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> T_co: ...


def _check_vllm_model_init(model: type[object] | object) -> bool:
    model_init = model.__init__
    return supports_kw(model_init, "vllm_config")


def _check_vllm_model_embed_input_ids(model: type[object] | object) -> bool:
    model_embed_input_ids = getattr(model, "embed_input_ids", None)
    if not callable(model_embed_input_ids):
        logger.warning(
            "The model (%s) is missing the `embed_input_ids` method.",
            model,
        )
        return False

    return True


def _check_vllm_model_forward(model: type[object] | object) -> bool:
    model_forward = getattr(model, "forward", None)
    if not callable(model_forward):
        return False

    vllm_kws = ("input_ids", "positions")
    missing_kws = tuple(kw for kw in vllm_kws if not supports_kw(model_forward, kw))

    if missing_kws and (isinstance(model, type) and issubclass(model, nn.Module)):
        logger.warning(
            "The model (%s) is missing "
            "vLLM-specific keywords from its `forward` method: %s",
            model,
            missing_kws,
        )

    return len(missing_kws) == 0


@overload
def is_vllm_model(model: type[object]) -> TypeIs[type[VllmModel]]: ...


@overload
def is_vllm_model(model: object) -> TypeIs[VllmModel]: ...


def is_vllm_model(
    model: type[object] | object,
) -> TypeIs[type[VllmModel]] | TypeIs[VllmModel]:
    return (
        _check_vllm_model_init(model)
        and _check_vllm_model_embed_input_ids(model)
        and _check_vllm_model_forward(model)
    )


@runtime_checkable
class VllmModelForTextGeneration(VllmModel[T], Protocol[T]):
    """The interface required for all generative models in vLLM."""

    def compute_logits(
        self,
        hidden_states: T,
    ) -> T | None:
        """Return `None` if TP rank > 0."""
        ...


@overload
def is_text_generation_model(
    model: type[object],
) -> TypeIs[type[VllmModelForTextGeneration]]: ...


@overload
def is_text_generation_model(model: object) -> TypeIs[VllmModelForTextGeneration]: ...


def is_text_generation_model(
    model: type[object] | object,
) -> TypeIs[type[VllmModelForTextGeneration]] | TypeIs[VllmModelForTextGeneration]:
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

    default_pooling_type: ClassVar[PoolingTypeStr] = "LAST"
    """
    Indicates the [vllm.config.pooler.PoolerConfig.pooling_type][]
    to use by default.

    You can use the
    [vllm.model_executor.models.interfaces_base.default_pooling_type][]
    decorator to conveniently set this field.
    """

    attn_type: ClassVar[AttnTypeStr] = "decoder"
    """
    Indicates the
    [vllm.config.model.ModelConfig.attn_type][]
    to use by default.

    You can use the
    [vllm.model_executor.models.interfaces_base.attn_type][]
    decorator to conveniently set this field.
    """

    pooler: Pooler
    """The pooler is only called on TP rank 0."""


@overload
def is_pooling_model(model: type[object]) -> TypeIs[type[VllmModelForPooling]]: ...


@overload
def is_pooling_model(model: object) -> TypeIs[VllmModelForPooling]: ...


def is_pooling_model(
    model: type[object] | object,
) -> TypeIs[type[VllmModelForPooling]] | TypeIs[VllmModelForPooling]:
    if not is_vllm_model(model):
        return False

    return getattr(model, "is_pooling_model", False)


_T = TypeVar("_T", bound=type[nn.Module])


def default_pooling_type(pooling_type: PoolingTypeStr):
    """Decorator to set `VllmModelForPooling.default_pooling_type`."""

    def func(model: _T) -> _T:
        model.default_pooling_type = pooling_type  # type: ignore
        return model

    return func


def get_default_pooling_type(model: type[object] | object) -> PoolingTypeStr:
    return getattr(model, "default_pooling_type", "LAST")


def attn_type(attn_type: AttnTypeStr):
    """Decorator to set `VllmModelForPooling.attn_type`."""

    def func(model: _T) -> _T:
        model.attn_type = attn_type  # type: ignore
        return model

    return func


def get_attn_type(model: type[object] | object) -> AttnTypeStr:
    return getattr(model, "attn_type", "decoder")
