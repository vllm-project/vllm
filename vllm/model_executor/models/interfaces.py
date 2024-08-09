from typing import (ClassVar, Dict, List, Literal, Optional, Protocol, Type,
                    Union, overload, runtime_checkable)

from typing_extensions import TypeIs

from vllm.config import LoRAConfig, MultiModalConfig, SchedulerConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


@runtime_checkable
class SupportsVision(Protocol):
    """The interface required for all vision language models (VLMs)."""

    supports_vision: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports vision inputs.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    def __init__(self, *, multimodal_config: MultiModalConfig) -> None:
        ...


# We can't use runtime_checkable with ClassVar for issubclass checks
# so we need to treat the class as an instance and use isinstance instead
@runtime_checkable
class _SupportsVisionType(Protocol):
    supports_vision: Literal[True]

    def __call__(self, *, multimodal_config: MultiModalConfig) -> None:
        ...


@overload
def supports_vision(model: Type[object]) -> TypeIs[Type[SupportsVision]]:
    ...


@overload
def supports_vision(model: object) -> TypeIs[SupportsVision]:
    ...


def supports_vision(
    model: Union[Type[object], object],
) -> Union[TypeIs[Type[SupportsVision]], TypeIs[SupportsVision]]:
    if isinstance(model, type):
        return isinstance(model, _SupportsVisionType)

    return isinstance(model, SupportsVision)


@runtime_checkable
class SupportsLoRA(Protocol):
    """The interface required for all models that support LoRA."""

    supports_lora: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports LoRA.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    packed_modules_mapping: ClassVar[Dict[str, List[str]]]
    supported_lora_modules: ClassVar[List[str]]
    embedding_modules: ClassVar[Dict[str, str]]
    embedding_padding_modules: ClassVar[List[str]]

    # lora_config is None when LoRA is not enabled
    def __init__(self, *, lora_config: Optional[LoRAConfig] = None) -> None:
        ...


# We can't use runtime_checkable with ClassVar for issubclass checks
# so we need to treat the class as an instance and use isinstance instead
@runtime_checkable
class _SupportsLoRAType(Protocol):
    supports_lora: Literal[True]

    packed_modules_mapping: Dict[str, List[str]]
    supported_lora_modules: List[str]
    embedding_modules: Dict[str, str]
    embedding_padding_modules: List[str]

    def __call__(self, *, lora_config: Optional[LoRAConfig] = None) -> None:
        ...


@overload
def supports_lora(model: Type[object]) -> TypeIs[Type[SupportsLoRA]]:
    ...


@overload
def supports_lora(model: object) -> TypeIs[SupportsLoRA]:
    ...


def supports_lora(
    model: Union[Type[object], object],
) -> Union[TypeIs[Type[SupportsLoRA]], TypeIs[SupportsLoRA]]:
    result = _supports_lora(model)

    if not result:
        lora_attrs = (
            "packed_modules_mapping",
            "supported_lora_modules",
            "embedding_modules",
            "embedding_padding_modules",
        )
        missing_attrs = tuple(attr for attr in lora_attrs
                              if not hasattr(model, attr))

        if getattr(model, "supports_lora", False):
            if missing_attrs:
                logger.warning(
                    "The model (%s) sets `supports_lora=True`, "
                    "but is missing LoRA-specific attributes: %s",
                    model,
                    missing_attrs,
                )
        else:
            if not missing_attrs:
                logger.warning(
                    "The model (%s) contains all LoRA-specific attributes, "
                    "but does not set `supports_lora=True`.", model)

    return result


def _supports_lora(
    model: Union[Type[object], object],
) -> Union[TypeIs[Type[SupportsLoRA]], TypeIs[SupportsLoRA]]:
    if isinstance(model, type):
        return isinstance(model, _SupportsLoRAType)

    return isinstance(model, SupportsLoRA)


@runtime_checkable
class HasInnerState(Protocol):
    """The interface required for all models that has inner state."""

    has_inner_state: ClassVar[Literal[True]] = True
    """
        A flag that indicates this model has inner state.
        Models that has inner state usually need access to the scheduler_config
        for max_num_seqs ,etc... (Currently only used by Jamba)
    """

    def __init__(self,
                 *,
                 scheduler_config: Optional[SchedulerConfig] = None) -> None:
        ...


@runtime_checkable
class _HasInnerStateType(Protocol):
    has_inner_state: ClassVar[Literal[True]]

    def __init__(self,
                 *,
                 scheduler_config: Optional[SchedulerConfig] = None) -> None:
        ...


@overload
def has_inner_state(model: object) -> TypeIs[HasInnerState]:
    ...


@overload
def has_inner_state(model: Type[object]) -> TypeIs[Type[HasInnerState]]:
    ...


def has_inner_state(
    model: Union[Type[object], object]
) -> Union[TypeIs[Type[HasInnerState]], TypeIs[HasInnerState]]:
    if isinstance(model, type):
        return isinstance(model, _HasInnerStateType)

    return isinstance(model, HasInnerState)
